from neo4j import GraphDatabase
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from typing import List, Dict, Any, Optional
import asyncio
from graph_rag.services.syncer import BaseSyncProcessor
from graph_rag.core.interfaces import IGraphStore

class KGManager(BaseSyncProcessor, IGraphStore):
    """
    负责 Neo4j 连接、CRUD、图谱推理以及自动化同步
    实现 IGraphStore 接口
    """
    def __init__(self, llm_client: Any):
        self.driver = GraphDatabase.driver(
            settings.neo4j.URI,
            auth=(settings.neo4j.USERNAME, settings.neo4j.PASSWORD)
        )
        self.llm_client = llm_client

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行 Cypher 查询
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"执行 Cypher 查询失败: {e}\nQuery: {query}")
                return []

    async def clear(self):
        """实现 IGraphStore.clear，清空数据库所有节点和关系"""
        query = "MATCH (n) DETACH DELETE n"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.execute_query, query)
        logger.info("Neo4j: 数据库已清空")

    def add_nodes_and_relationships(self, nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """
        批量添加节点和关系 (使用 UNWIND 优化)
        
        Args:
            nodes: 节点列表，每个节点包含 type, properties
            relationships: 关系列表，每个关系包含 source, target, type, properties
        """
        with self.driver.session() as session:
            session.execute_write(self._add_nodes_and_relationships_tx, nodes, relationships)

    @staticmethod
    def _add_nodes_and_relationships_tx(tx, nodes, relationships):
        # 按类型分组节点
        nodes_by_type = {}
        for node in nodes:
            node_type = node.get("type", "ENTITY")
            name = node.get("name")
            if not name:
                continue
            
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            
            # 准备 UNWIND 数据
            nodes_by_type[node_type].append({
                "name": name,
                "properties": node.get("properties", {})
            })
        
        # 批量写入节点
        for node_type, batch_data in nodes_by_type.items():
            query = (
                f"UNWIND $batch_data AS row "
                f"MERGE (n:{node_type} {{name: row.name}}) "
                f"SET n += row.properties"
            )
            tx.run(query, batch_data=batch_data)

        # 按类型分组关系
        rels_by_type = {}
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            
            if not (source and target and rel_type):
                continue
            
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            
            rels_by_type[rel_type].append({
                "source": source,
                "target": target,
                "properties": rel.get("properties", {})
            })
            
        # 批量写入关系
        for rel_type, batch_data in rels_by_type.items():
            query = (
                f"UNWIND $batch_data AS row "
                f"MATCH (a {{name: row.source}}) "
                f"MATCH (b {{name: row.target}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"SET r += row.properties"
            )
            tx.run(query, batch_data=batch_data)

    def graph_reasoning(self, entities: List[str]) -> Dict[str, Any]:
        """
        图谱推理：
        1. 1-Hop 检索 (置信度 1.0)
        2. 2-Hop 检索 (置信度 0.8)
        返回相关实体和关系的并集
        实现 IGraphStore.graph_reasoning
        """
        if not entities:
            return {"entities": [], "relationships": [], "confidence": 0.0}

        # 1-Hop
        query_1hop = (
            "MATCH (a)-[r]->(b) "
            "WHERE a.name IN $names OR b.name IN $names "
            "RETURN a.name as source, type(r) as relation, b.name as target, 1.0 as confidence"
        )
        
        # 2-Hop
        query_2hop = (
            "MATCH (a)-[r1]->(b)-[r2]->(c) "
            "WHERE a.name IN $names OR c.name IN $names "
            "RETURN a.name as source, type(r1) + '->' + type(r2) as relation, c.name as target, 0.8 as confidence"
        )
        
        logger.debug(f"Cypher 1-Hop: {query_1hop} Params: {entities}")
        results_1hop = self.execute_query(query_1hop, {"names": entities})
        logger.debug(f"Cypher 2-Hop: {query_2hop} Params: {entities}")
        results_2hop = self.execute_query(query_2hop, {"names": entities})
        
        logger.debug(f"Graph Reasoning Results: 1-Hop={len(results_1hop)}, 2-Hop={len(results_2hop)}")
        
        all_results = results_1hop + results_2hop
        
        # 去重与聚合
        unique_rels = []
        seen = set()
        all_entities = set(entities)
        
        total_conf = 0.0
        
        for res in all_results:
            key = (res['source'], res['relation'], res['target'])
            if key not in seen:
                seen.add(key)
                unique_rels.append(res)
                all_entities.add(res['source'])
                all_entities.add(res['target'])
                total_conf += res['confidence']
                
        avg_conf = (total_conf / len(unique_rels)) if unique_rels else 0.0
        
        return {
            "entities": list(all_entities),
            "relationships": unique_rels,
            "confidence": avg_conf
        }



    def process_file_content_safe(self, content: str):
        """
        处理文件内容：抽取 -> 校验 -> 写入
        """
        from graph_rag.core.schema import get_extraction_prompt, validate_extraction
        import json
        
        prompt = get_extraction_prompt(content)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self.llm_client.chat(messages, response_format={"type": "json_object"})
            data = json.loads(response_text)
            
            # Schema 校验
            if not validate_extraction(data):
                logger.warning("抽取结果未通过 Schema 校验，已忽略")
                return
            
            nodes = data.get("nodes", [])
            relationships = data.get("relationships", [])
            
            self.add_nodes_and_relationships(nodes, relationships)
            logger.info(f"成功更新 KG: 增加了 {len(nodes)} 个节点和 {len(relationships)} 条关系")
        except Exception as e:
            logger.error(f"提取实体关系并更新 KG 失败: {e}")

    async def process_added(self, file_paths: List[str]):
        """
        处理新增文件：
        1. 并发读取文件并抽取实体关系
        2. 校验 Schema
        3. 批量写入 Neo4j
        """
        logger.info(f"KG 开始处理新增文件: {len(file_paths)} 个")
        loop = asyncio.get_running_loop()
        
        async def process_file(file_path: str):
            if not file_path.endswith(('.txt', '.md')):
                return
                
            try:
                # 读文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 抽取和写入（包含 Schema 校验）
                await loop.run_in_executor(None, self.process_file_content_safe, content)
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")

        tasks = [process_file(fp) for fp in file_paths]
        await asyncio.gather(*tasks)
        logger.info("KG 新增文件处理完成")
    async def process_modified(self, file_paths: List[str]):
        """KG 不支持自动增量更新"""
        logger.warning(f"KG 检测到修改文件 {file_paths}，但不执行自动更新。请手动调用 /kg/rebuild 进行全量重建以保证一致性。")

    async def process_deleted(self, file_paths: List[str]):
        """KG 不支持自动增量更新"""
        logger.warning(f"KG 检测到删除文件 {file_paths}，但不执行自动更新。请手动调用 /kg/rebuild 进行全量重建以保证一致性。")

