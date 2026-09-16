from typing import List, Dict, Any
from graph_rag.core.interfaces import IRanker
from loguru import logger
from graph_rag.services.fusion import fuse_results


class WeightedRanker(IRanker):
    """共享可离线验证的融合逻辑，默认保留原有加权策略。"""
    def __init__(self, weights=None):
        if weights is None:
            from graph_rag.core.config import settings
            from graph_rag.core.logger import logger as configured_logger
            self.weights = settings.rag
        else:
            self.weights = weights

    def rank(self, vec_res: List[Dict[str, Any]], kw_res: List[Dict[str, Any]],
             graph_boost_map: Dict[str, float], kg_debug_info: Dict[str, Any]) -> Dict[str, Any]:
        ranked, components = fuse_results(vec_res, kw_res, graph_boost_map, self.weights)
        for doc in ranked:
            logger.info(f"Doc {doc['id']} scores -> {components[doc['id']]}")
        return {'joint_results': ranked, 'debug_info': {
            'vector_count': len(vec_res), 'keyword_count': len(kw_res),
            'graph_entities': kg_debug_info.get('entities', []),
            'graph_confidence': kg_debug_info.get('confidence', 0.0),
            'component_scores': components,
        }}
