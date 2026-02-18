from typing import List, Dict, Any
import asyncio
import re
import json
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from graph_rag.core.interfaces import IVectorStore, IKeywordStore, IGraphStore, IRanker
from graph_rag.services.ranker import WeightedRanker
from graph_rag.core.llm_client import llm_client
from graph_rag.core.schema import NODE_TYPES, Entity

# 定义常见的无用词/停用词，用于兜底提取
STOPWORDS = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "这", "那", "之", "与", "和", "或", "请问", "告诉我", "查询", "搜索", "关于", "谁是", "什么是", "哪里有"}

class HybridRetriever:
    """
    混合检索器 (Async)
    融合向量检索、关键词检索和图谱推理
    """
    def __init__(self, 
                 milvus_mgr: IVectorStore, 
                 es_mgr: IKeywordStore, 
                 kg_mgr: IGraphStore,
                 ranker: IRanker = None):
        self.milvus_mgr = milvus_mgr
        self.es_mgr = es_mgr
        self.kg_mgr = kg_mgr
        self.weights = settings.rag
        self.ranker = ranker or WeightedRanker()

    def _rule_based_fallback(self, query: str) -> List[Entity]:
        """基于规则的实体提取兜底逻辑"""
        potential_entities = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
        entities = []
        for word in potential_entities:
            if word not in STOPWORDS:
                entities.append(Entity(name=word, type="COMPANY"))
        return entities

    async def _extract_entities_async(self, query: str) -> List[Entity]:
        """异步从查询中提取实体"""
        return await asyncio.to_thread(self._extract_entities, query)

    def _extract_entities(self, query: str) -> List[Entity]:
        """从查询中提取结构化实体（含重试和兜底逻辑）"""
        node_types_str = ", ".join(NODE_TYPES.keys())
        prompt = f"""
        你是一个实体提取专家。请从用户的查询中提取出相关的实体。
        
        查询：{query}
        
        可用的实体类型：{node_types_str}
        
        示例 1：
        查询：A集团的董事长是谁？
        输出：[
            {{"name": "A集团", "type": "GROUP", "properties": {{}}}}
        ]
        
        示例 2：
        查询：查询C基金投资了哪些公司
        输出：[
            {{"name": "C基金", "type": "FUND", "properties": {{}}}}
        ]
        
        输出格式要求（必须是有效的 JSON 数组）：
        [
            {{"name": "实体名称", "type": "实体类型", "properties": {{}}}}
        ]
        
        仅输出 JSON 结果，不要包含任何解释。
        """
        messages = [{"role": "user", "content": prompt}]
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = llm_client.chat(messages, temperature=0.0)
                logger.debug(f"LLM 原始响应: {response}")
                if isinstance(response, str):
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                else:
                    match = None
                
                if match:
                    data = json.loads(match.group())
                    entities = []
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            # 验证实体类型
                            entity_type = item.get("type", "UNKNOWN")
                            if entity_type not in NODE_TYPES:
                                logger.warning(f"发现未知实体类型: {entity_type}，实体: {item['name']}")
                                entity_type = "UNKNOWN"
                                
                            entities.append(Entity(
                                name=item["name"],
                                type=entity_type,
                                properties=item.get("properties", {})
                            ))
                    if entities: 
                        logger.info(f"解析出实体: {[e.name for e in entities]}")
                        return entities
                
            except Exception as e:
                logger.error(f"实体提取失败 ({attempt}): {e}")
        
        logger.info("LLM 提取失败，使用规则兜底")
        return self._rule_based_fallback(query)

    async def _kg_retrieval_chain(self, query: str) -> Dict[str, Any]:
        """
        KG 检索链：实体提取 -> 图谱推理
        """
        try:
            # 1. 实体提取 (Async)
            entities = await self._extract_entities_async(query)
            entity_names = [e.name for e in entities]
            logger.info(f"提取实体: {entity_names}")
            
            if not entity_names:
                return {"entities": [], "confidence": 0.0}

            # 2. 图谱推理 (Sync -> Thread)
            # 调用接口方法 graph_reasoning
            kg_res = await asyncio.to_thread(self.kg_mgr.graph_reasoning, entity_names)
            logger.info(f"图谱推理结果: 实体数={len(kg_res.get('entities', []))}, 关系数={len(kg_res.get('relationships', []))}, 置信度={kg_res.get('confidence', 0.0)}")
            return kg_res
            
        except Exception as e:
            logger.error(f"KG Retrieval Chain Failed: {e}")
            return {"entities": [], "confidence": 0.0}

    async def retrieve(self, query: str) -> Dict[str, Any]:
        """
        并发执行混合检索与联合评分
        """
        # 1. 并发三路召回
        # Vector Search (Sync -> Thread)
        task_vec = asyncio.to_thread(self.milvus_mgr.search, query, self.weights.TOP_K)
        # Keyword Search (Async)
        task_kw = self.es_mgr.search(query, self.weights.TOP_K)
        # KG Chain (Async) - 包含实体提取和推理
        task_kg = self._kg_retrieval_chain(query)

        results = await asyncio.gather(task_vec, task_kw, task_kg, return_exceptions=True)
        
        vec_res = results[0] if not isinstance(results[0], Exception) else []
        kw_res = results[1] if not isinstance(results[1], Exception) else []
        kg_res = results[2] if not isinstance(results[2], Exception) else {"entities": [], "confidence": 0.0}

        if kw_res:
            logger.info(f"ES 关键词搜索召回: {len(kw_res)} 条结果")
        else:
            logger.debug(f"ES 搜索关键词: {query} 未召回任何结果")

        if isinstance(results[0], Exception): logger.error(f"Vector Search Failed: {results[0]}")
        if isinstance(results[1], Exception): logger.error(f"Keyword Search Failed: {results[1]}")
        # KG Chain 内部已处理异常

        # 2. 图谱增强准备 (Graph Boost Calculation)
        # 这部分逻辑可以放在 Ranker 内部，或者在这里准备好 boost map
        # 为了保持 Ranker 的纯粹性 (只做算分)，我们将 I/O 操作 (search_by_entity) 留在这里执行
        # Ranker 接收一个 boost_map: {doc_id: score}
        
        kg_conf = kg_res.get("confidence", 0.0)
        kg_entities = kg_res.get("entities", [])
        graph_boost_map = {}
        
        if kg_conf > 0 and kg_entities:
            try:
                logger.info(f"准备计算 Graph Boost: 实体={kg_entities}, 基础置信度={kg_conf}")
                # 并发查找包含这些实体的 Chunk
                boost_tasks = [self.es_mgr.search_by_entity(entity) for entity in kg_entities]
                boost_results = await asyncio.gather(*boost_tasks, return_exceptions=True)
                
                # 计算每个文档的最大增益
                current_boost = kg_conf * self.weights.GRAPH_WEIGHT
                for i, chunk_ids in enumerate(boost_results):
                    if isinstance(chunk_ids, list):
                        logger.debug(f"实体 '{kg_entities[i]}' 命中 {len(chunk_ids)} 个文档")
                        for doc_id in chunk_ids:
                            graph_boost_map[doc_id] = max(graph_boost_map.get(doc_id, 0.0), current_boost)
                
                logger.info(f"Graph Boost 覆盖文档数: {len(graph_boost_map)}")
            except Exception as e:
                logger.error(f"Graph Boost Calculation Failed: {e}")

        # 3. 调用 Ranker 进行评分与融合
        final_result = self.ranker.rank(vec_res, kw_res, graph_boost_map, kg_res)
        
        return final_result
