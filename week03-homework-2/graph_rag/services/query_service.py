from typing import Dict, Any, List
from graph_rag.core.interfaces import IQueryService, IGenerator
from graph_rag.services.retriever import HybridRetriever
from graph_rag.core.logger import logger

class QueryService(IQueryService):
    """
    RAG 查询服务实现
    """
    def __init__(self, retriever: HybridRetriever, generator: IGenerator):
        self.retriever = retriever
        self.generator = generator

    async def query(self, question: str) -> Dict[str, Any]:
        """执行混合 RAG 查询"""
        # 1. 执行联合检索
        retrieval_results = await self.retriever.retrieve(question)
        
        # 2. 构建上下文
        context_parts = []
        joint_results = retrieval_results.get("joint_results", [])
        
        # 尝试构建推理路径，即使失败也不影响主流程
        reasoning_log = []
        try:
            debug_info = retrieval_results.get("debug_info", {})
            
            # Helper function to log both to list and logger
            def add_log(msg):
                reasoning_log.append(msg)
                logger.info(f"[Reasoning] {msg}")

            add_log(f"1. 接收查询: {question}")
            add_log(f"2. 执行混合检索:")
            add_log(f"   - Vector Search: 检索到 {debug_info.get('vector_count', 0)} 个相关文档")
            add_log(f"   - Keyword Search (ES): 检索到 {debug_info.get('keyword_count', 0)} 个相关文档")
            
            kg_entities = debug_info.get("graph_entities", [])
            kg_conf = debug_info.get("graph_confidence", 0.0)
            add_log(f"   - KG Reasoning: 提取实体 {kg_entities}, 置信度 {kg_conf}")
            
            add_log(f"3. 召回合并与评分 (Rerank):")
            component_scores = debug_info.get("component_scores", {})
            if component_scores:
                for doc_id, scores in component_scores.items():
                    vec = scores.get('vector_weighted', 0)
                    kw = scores.get('keyword_weighted', 0)
                    gb = scores.get('graph_boost', 0)
                    final = scores.get('final_score', 0)
                    add_log(f"   - Doc {doc_id}: Vector={vec:.4f}, Keyword={kw:.4f}, GraphBoost={gb:.4f} -> Final={final:.4f}")
            else:
                add_log("   - 无评分详情")
                
            add_log(f"4. 筛选 Top-K 文档:")
            chunk_ids = [res.get('id') for res in joint_results]
            add_log(f"   - 选中 Chunk IDs: {chunk_ids}")
            
        except Exception as e:
            logger.error(f"构建推理日志失败: {e}")
            reasoning_log.append(f"构建推理日志时发生错误: {e}")

        for res in joint_results:
            context_parts.append(f"Content: {res.get('content', '')}")
            
        context_str = "\n---\n".join(context_parts)
        
        # 4. 生成回答
        logger.debug(f"Calling Generator with context length: {len(context_str)}")
        if reasoning_log:
            reasoning_log.append(f"5. 生成回答 (Context Length: {len(context_str)})")
            logger.info(f"[Reasoning] 5. 生成回答 (Context Length: {len(context_str)})")
            
        answer = await self.generator.generate(context_str, question)
        
        if reasoning_log:
            reasoning_log.append(f"6. 完成生成")
            logger.info(f"[Reasoning] 6. 完成生成")
        
        return {
            "answer": answer,
            "retrieval_results": joint_results,
            "reasoning_log": reasoning_log
        }
