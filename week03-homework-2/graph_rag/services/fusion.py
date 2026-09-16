"""无服务依赖的融合评分；线上 Ranker 与离线实验共用，避免评测另一套算法。"""
import math


def fuse_results(vec_res, kw_res, graph_boost_map, weights):
    method = getattr(weights, 'FUSION_METHOD', 'weighted')
    rrf_k = getattr(weights, 'RRF_K', 60)
    relative = getattr(weights, 'RELATIVE_THRESHOLD', 0.5)
    if method not in ('weighted', 'rrf') or rrf_k <= 0 or not 0 <= relative <= 1:
        raise ValueError('Invalid fusion method, RRF_K or relative threshold')
    docs, components = {}, {}
    for results, channel, weight in (
        (vec_res, 'vector', weights.VECTOR_WEIGHT),
        (kw_res, 'keyword', weights.KEYWORD_WEIGHT),
    ):
        valid = [d for d in results if d.get('id') and math.isfinite(float(d.get('score', 0)))]
        maximum = max((d.get('score', 0) for d in valid), default=0)
        seen = set()
        for position, doc in enumerate(valid, 1):
            key = doc['id']
            if key in seen:
                continue
            seen.add(key)
            docs.setdefault(key, doc)
            comp = components.setdefault(key, {'vector_raw': 0, 'keyword_raw': 0,
                'vector_weighted': 0, 'keyword_weighted': 0, 'graph_boost': 0, 'final_score': 0})
            raw = float(doc.get('score', 0))
            if method == 'rrf':
                # 乘公共常数 k+1 保持单路最高分为 1，使图谱权重有稳定量级。
                score = (rrf_k + 1) / (rrf_k + position)
            else:
                score = raw if channel == 'vector' else raw / (maximum + 1e-6) if maximum > 0 else 0
            comp[channel + '_raw'] = raw
            comp[channel + '_weighted'] = score * weight
            comp['final_score'] += score * weight
    # 图谱只能增强已有正文的候选；不返回无法提供引用内容的幽灵文档。
    for key, boost in graph_boost_map.items():
        if key in components and math.isfinite(float(boost)):
            components[key]['graph_boost'] = float(boost)
            components[key]['final_score'] += float(boost)
    maximum = max((c['final_score'] for c in components.values()), default=0)
    kept = {key: c for key, c in components.items()
            if c['final_score'] >= weights.MIN_SCORE_THRESHOLD and c['final_score'] >= maximum * relative}
    ranked = [{'id': key, 'content': docs[key].get('content', ''),
               'metadata': docs[key].get('metadata', {}), 'score': c['final_score']} for key, c in kept.items()]
    ranked.sort(key=lambda d: (-d['score'], str(d['id'])))
    return ranked, kept
