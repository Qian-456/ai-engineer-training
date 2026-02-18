from typing import List, Dict, Any
from graph_rag.core.interfaces import IRanker
from graph_rag.core.config import settings
from graph_rag.core.logger import logger

class WeightedRanker(IRanker):
    """
    加权排序器
    基于配置的权重和阈值对多路召回结果进行融合
    """
    def __init__(self, weights=None):
        self.weights = weights or settings.rag

    def rank(self, 
             vec_res: List[Dict[str, Any]], 
             kw_res: List[Dict[str, Any]], 
             graph_boost_map: Dict[str, float],
             kg_debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行评分逻辑
        
        Args:
            vec_res: 向量检索结果
            kw_res: 关键词检索结果
            graph_boost_map: {doc_id: boost_score} 图谱增益映射
            kg_debug_info: KG 调试信息
            
        Returns:
            Dict: 包含 joint_results 和 debug_info
        """
        scores: Dict[str, float] = {}
        docs_map: Dict[str, Dict] = {}
        vec_raw_scores: Dict[str, float] = {}
        kw_raw_scores: Dict[str, float] = {}
        vec_weighted_scores: Dict[str, float] = {}
        kw_weighted_scores: Dict[str, float] = {}
        boost_scores: Dict[str, float] = {}

        # 1. 向量分数
        for item in vec_res:
            doc_id = item.get("id")
            if not doc_id: continue
            base_score = item.get("score", 0.0)
            vec_raw_scores[doc_id] = base_score
            weighted = base_score * self.weights.VECTOR_WEIGHT
            vec_weighted_scores[doc_id] = vec_weighted_scores.get(doc_id, 0.0) + weighted
            scores[doc_id] = scores.get(doc_id, 0.0) + weighted
            docs_map[doc_id] = item

        # 2. 关键词分数 (归一化处理)
        # BM25 分数没有固定上限，通常在 0~10+ 之间，需要归一化到 0~1 区间以便与向量分数融合
        # 简单的归一化策略：除以最大分数（如果有）或者使用 Sigmoid 函数
        
        max_kw_score = 0.0
        if kw_res:
            max_kw_score = max([item.get("score", 0.0) for item in kw_res])
        
        for item in kw_res:
            doc_id = item.get("id")
            if not doc_id: continue
            
            raw_score = item.get("score", 0.0)
            # 归一化: score / (max_score + epsilon)
            # 这样最高分接近 1.0
            norm_score = raw_score / (max_kw_score + 1e-6) if max_kw_score > 0 else 0.0
            
            kw_raw_scores[doc_id] = raw_score
            
            weighted = norm_score * self.weights.KEYWORD_WEIGHT
            kw_weighted_scores[doc_id] = kw_weighted_scores.get(doc_id, 0.0) + weighted
            scores[doc_id] = scores.get(doc_id, 0.0) + weighted
            if doc_id not in docs_map: docs_map[doc_id] = item

        # 3. 应用图谱增益
        # 策略：只增强已召回的文档
        for doc_id, boost in graph_boost_map.items():
            if doc_id in scores:
                scores[doc_id] += boost
            else:
                scores[doc_id] = boost
            boost_scores[doc_id] = boost


        # 4. 排序与过滤
        component_scores: Dict[str, Dict[str, float]] = {}
        final_results = []
        if scores:
            max_score = max(scores.values())
            
            for doc_id, score in scores.items():
                # 硬阈值过滤
                if score < self.weights.MIN_SCORE_THRESHOLD:
                    continue
                # 动态阈值过滤
                if score < (max_score * 0.5):
                    continue
                
                doc = docs_map.get(doc_id, {})
                component_scores[doc_id] = {
                    "vector_raw": vec_raw_scores.get(doc_id, 0.0),
                    "vector_weighted": vec_weighted_scores.get(doc_id, 0.0),
                    "keyword_raw": kw_raw_scores.get(doc_id, 0.0),
                    "keyword_weighted": kw_weighted_scores.get(doc_id, 0.0),
                    "graph_boost": boost_scores.get(doc_id, 0.0),
                    "final_score": score,
                }

                final_results.append({
                    "id": doc_id,
                    "content": doc.get("content", ""),
                    "score": score,
                    "metadata": doc.get("metadata", {})
                })
        
        # 排序
        final_results.sort(key=lambda x: x["score"], reverse=True)

        for item in final_results:
            doc_id = item["id"]
            comp = component_scores.get(doc_id, {})
            logger.info(
                f"Doc {doc_id} scores -> "
                f"Milvus(vector_raw)={comp.get('vector_raw', 0.0):.4f}, "
                f"ES(keyword_raw)={comp.get('keyword_raw', 0.0):.4f}, "
                f"GraphBoost={comp.get('graph_boost', 0.0):.4f}, "
                f"Final={item['score']:.4f}"
            )
        
        return {
            "joint_results": final_results,
            "debug_info": {
                "vector_count": len(vec_res),
                "keyword_count": len(kw_res),
                "graph_entities": kg_debug_info.get("entities", []),
                "graph_confidence": kg_debug_info.get("confidence", 0.0),
                "component_scores": component_scores
            }
        }
