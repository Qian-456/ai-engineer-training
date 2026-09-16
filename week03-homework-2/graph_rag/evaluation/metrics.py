"""文档级检索指标；空标注不计作召回成功。"""


def measure(ranked, relevant, k=5):
    if k < 1:
        raise ValueError('k must be positive')
    ids = list(dict.fromkeys(ranked))[:k]
    relevant = set(relevant)
    if not relevant:
        return {'recall': None, 'mrr': None, 'all_support': None, 'empty_return': int(not ids)}
    hits = len(set(ids) & relevant)
    return {'recall': hits / len(relevant),
            'mrr': next((1 / rank for rank, key in enumerate(ids, 1) if key in relevant), 0),
            'all_support': int(relevant.issubset(ids)), 'empty_return': int(not ids)}


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot evaluate an empty split')
    return {key: sum(r[key] for r in rows if r[key] is not None) / sum(r[key] is not None for r in rows)
            if any(r[key] is not None for r in rows) else None for key in rows[0]}
