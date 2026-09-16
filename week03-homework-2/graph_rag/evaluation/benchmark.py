"""HotpotQA pooled-distractor 文档检索：先调参冻结，再读独立测试标签。"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from rank_bm25 import BM25Okapi
import pyarrow.parquet as pq

from graph_rag.services.fusion import fuse_results
from .metrics import measure, aggregate
from .prepare import ARTIFACTS, DATA_SHA256, DATA_REVISION, MODEL_REVISION, MODEL_NAME

SEED = 20260916


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


def split_rows(rows, tune_count, test_count, seed=SEED):
    if tune_count <= 0 or test_count <= 0 or tune_count + test_count > len(rows):
        raise ValueError('Invalid split sizes')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate query IDs')
    ordered = sorted(rows, key=lambda r: r['id'])
    random.Random(seed).shuffle(ordered)
    return ordered[:tune_count], ordered[tune_count:tune_count + test_count]


def build_corpus(rows):
    # 只使用所有 context，绝不按 supporting_facts 过滤或添加边。
    passages = defaultdict(set)
    for row in rows:
        for title, sentences in zip(row['context']['title'], row['context']['sentences']):
            passages[title].add(''.join(sentences))
    return [{'id': title, 'content': title + '\n' + '\n'.join(sorted(texts)), 'metadata': {}}
            for title, texts in sorted(passages.items())]


def title_graph(corpus):
    """仅从正文中的标题提及建有向边，不使用问题答案或支持事实。"""
    aliases = defaultdict(set)
    for doc in corpus:
        alias = tuple(tokens(re.sub(r'\([^)]*\)', '', doc['id'])))
        if 1 <= len(alias) <= 10 and (len(alias) > 1 or len(alias[0]) >= 5):
            aliases[alias].add(doc['id'])
    aliases = {k: next(iter(v)) for k, v in aliases.items() if len(v) == 1}
    lengths = sorted({len(k) for k in aliases})

    def match(text):
        words = tokens(text)
        return {aliases[phrase] for length in lengths for i in range(len(words) - length + 1)
                if (phrase := tuple(words[i:i + length])) in aliases}

    edges = {doc['id']: match(doc['content']) - {doc['id']} for doc in corpus}
    return edges, match


def weights(method='weighted', dense=0.5, relative=0, rrf_k=60, graph=0, minimum=0):
    return {'FUSION_METHOD': method, 'VECTOR_WEIGHT': dense, 'KEYWORD_WEIGHT': 1 - dense,
            'RELATIVE_THRESHOLD': relative, 'RRF_K': rrf_k,
            'GRAPH_WEIGHT': graph, 'MIN_SCORE_THRESHOLD': minimum}


def ranking(feature, config):
    if config['FUSION_METHOD'] == 'dense':
        return [d['id'] for d in feature['dense']]
    if config['FUSION_METHOD'] == 'bm25':
        return [d['id'] for d in feature['bm25']]
    boost = {key: value * config['GRAPH_WEIGHT'] for key, value in feature['graph'].items()}
    ranked, _ = fuse_results(feature['dense'], feature['bm25'], boost, SimpleNamespace(**config))
    return [d['id'] for d in ranked]


def query_score(row, ranked, k):
    return measure(ranked, row['supporting_facts']['title'], k)


def choose(features, rows, candidates, k):
    trials = []
    for config in candidates:
        scores = [query_score(row, ranking(features[row['id']], config), k) for row in rows]
        trials.append({'config': config, 'metrics': aggregate(scores)})
    # 相同得分优先较小图谱权重，再按固定候选顺序；不读取测试标签。
    best = max(trials, key=lambda t: (t['metrics']['recall'], t['metrics']['all_support'],
                                      t['metrics']['mrr'], -t['config']['GRAPH_WEIGHT']))
    return best['config'], trials


def build_features(corpus, rows, target):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(str(ARTIFACTS / 'model'), device='cuda' if torch.cuda.is_available() else 'cpu')
    model.max_seq_length = 256
    print('encode', len(corpus), 'documents', len(rows), 'queries', 'device', model.device, flush=True)
    document_vectors = model.encode([d['content'] for d in corpus], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    query_vectors = model.encode([r['question'] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.savez_compressed(target / 'embeddings.npz', documents=document_vectors, queries=query_vectors)
    bm25 = BM25Okapi([tokens(d['content']) for d in corpus], k1=1.5, b=0.75)
    edges, match = title_graph(corpus)
    features = {}
    for row, vector in zip(rows, query_vectors):
        dense_scores = document_vectors @ vector
        keyword_scores = bm25.get_scores(tokens(row['question']))

        def top(scores, only_positive=False):
            indices = np.argsort(-scores, kind='stable')[:40]
            return [{**corpus[i], 'score': float(scores[i])} for i in indices
                    if not only_positive or scores[i] > 0]

        dense, keyword = top(dense_scores), top(keyword_scores, True)
        seed_docs = match(row['question'])
        graph = {key: 1.0 for key in seed_docs}
        for seed in sorted(seed_docs):
            for neighbor in edges[seed]:
                graph[neighbor] = max(graph.get(neighbor, 0), 0.8)
        features[row['id']] = {'dense': dense, 'bm25': keyword, 'graph': graph}
    return features, {'nodes': len(edges), 'edges': sum(map(len, edges.values())),
                      'query_seeded': sum(bool(f['graph']) for f in features.values()),
                      'device': str(model.device), 'max_seq_length': model.max_seq_length}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def paired_comparison(detailed, baseline, variant='tuned_graph'):
    """固定测试题的配对 bootstrap 区间，不用于再选参数。"""
    deltas = np.array([row['methods'][variant]['metrics']['recall'] -
                       row['methods'][baseline]['metrics']['recall'] for row in detailed])
    rng = np.random.default_rng(SEED)
    means = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1)
    return {'recall_delta': float(deltas.mean()), 'bootstrap_95_percent': np.quantile(means, [0.025, 0.975]).tolist(),
            'improved': int((deltas > 0).sum()), 'regressed': int((deltas < 0).sum()),
            'unchanged': int((deltas == 0).sum()), 'resamples': 2000}


def render(summary, detailed):
    esc = html.escape
    table = ''.join(f"<tr><td>{esc(name)}</td><td>{m['recall']:.4f}</td><td>{m['mrr']:.4f}</td><td>{m['all_support']:.4f}</td></tr>"
                    for name, m in summary['test_metrics'].items())
    cases = []
    for row in detailed:
        cases.append(f"<details><summary>{esc(row['id'])} · {esc(row['type'])} · {esc(row['question'])}</summary>"
                     f"<pre>{esc(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details>")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GraphRAG 真实数据检索评测</title><style>body{{font:16px/1.7 system-ui;max-width:1080px;margin:32px auto;padding:16px}}
td,th{{padding:10px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
details{{border:1px solid #ddd;padding:12px;margin:8px 0}}summary{{cursor:pointer}}</style>
<h1>HotpotQA · 检索对照实验</h1><p>调参 {summary['tune_count']} 题 / 独立测试 {summary['test_count']} 题 / 全局候选 {summary['corpus_count']} 篇。</p>
<p>由 distractor contexts 汇集成固定语料库；不是每题只检索 10 篇，也不是 full-wiki 官方评测。
图为标题提及图，不代表线上 Neo4j 企业关系抽取质量。未生成答案，未评测答案准确率。</p>
<table><tr><th>方法</th><th>Recall@{summary['k']}</th><th>MRR@{summary['k']}</th><th>全部支持文档命中率</th></tr>{table}</table>
<p><a href="summary.json">指标与版本</a> · <a href="selection.json">调参记录/冻结配置</a> · <a href="per_query.json">逐题原始结果</a></p>
<h2>逐题检查（含改善与退化）</h2>{''.join(cases)}</html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune-count', type=int, default=120)
    parser.add_argument('--test-count', type=int, default=240)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--selection-file', type=Path, help='复用既有冻结配置；核对相同题目划分和指标口径，不重新调参')
    args = parser.parse_args()
    if not 1 <= args.k <= 40:
        parser.error('k must be between 1 and 40')
    path = ARTIFACTS / 'hotpot-validation.parquet'
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError('Data checksum mismatch; run prepare first')
    rows = pq.read_table(path).to_pylist()
    tune, test = split_rows(rows, args.tune_count, args.test_count)
    target = ARTIFACTS / ('run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6])
    target.mkdir()
    corpus = build_corpus(tune + test)
    known = {d['id'] for d in corpus}
    if any(not set(r['supporting_facts']['title']).issubset(known) for r in tune + test):
        raise ValueError('Gold documents missing from corpus')
    manifest = {'seed': SEED, 'tune_ids': [r['id'] for r in tune], 'test_ids': [r['id'] for r in test],
                'corpus_sha256': hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    write_json(target / 'manifest.json', manifest)
    start = time.perf_counter()
    features, graph_info = build_features(corpus, tune + test, target)
    if args.selection_file:
        selection = json.loads(args.selection_file.read_text(encoding='utf-8'))
        old_manifest = json.loads((args.selection_file.parent / 'manifest.json').read_text(encoding='utf-8'))
        old_summary = json.loads((args.selection_file.parent / 'summary.json').read_text(encoding='utf-8'))
        if old_manifest != manifest or old_summary['k'] != args.k or old_summary['data_sha256'] != DATA_SHA256 or old_summary['model_revision'] != MODEL_REVISION:
            raise ValueError('Frozen selection does not match this dataset/split/model/k')
        selection['replayed_from_sha256'] = hashlib.sha256(args.selection_file.read_bytes()).hexdigest()
        selected = selection['selected']
    else:
        hybrid_candidates = [weights(method, dense, relative, rrf_k)
            for dense in (0.25, 0.5, 0.75) for method, relative, rrf_k in
            (('weighted', 0, 60), ('weighted', 0.5, 60), ('rrf', 0, 20), ('rrf', 0, 60))]
        hybrid, hybrid_trials = choose(features, tune, hybrid_candidates, args.k)
        graph_candidates = [{**config, 'GRAPH_WEIGHT': boost} for config in hybrid_candidates for boost in (0, 0.1, 0.2)]
        graph, graph_trials = choose(features, tune, graph_candidates, args.k)
        selected = {'dense': {'FUSION_METHOD': 'dense'}, 'bm25': {'FUSION_METHOD': 'bm25'},
                    'legacy_weighted': {**weights(relative=0.5, minimum=0.2), 'KEYWORD_WEIGHT': 0.3},
                    'tuned_hybrid': hybrid, 'tuned_graph': graph}
        selection = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'selected': selected,
                     'hybrid_trials': hybrid_trials, 'graph_trials': graph_trials}
    selected['graph_ablation_no_boost'] = {**selected['tuned_graph'], 'GRAPH_WEIGHT': 0}
    write_json(target / 'selection.json', selection)
    # 配置已落盘冻结后，才使用测试集标签计算指标。
    detailed = []
    for row in test:
        methods = {}
        for name, config in selected.items():
            ranked = ranking(features[row['id']], config)
            methods[name] = {'metrics': query_score(row, ranked, args.k), 'top_docs': ranked[:args.k]}
        detailed.append({'id': row['id'], 'question': row['question'], 'type': row['type'],
                         'relevant_docs': sorted(set(row['supporting_facts']['title'])), 'methods': methods})
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip())
    source_hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (Path(__file__).parent, Path(__file__).parents[1] / 'services') for p in folder.glob('*.py')}
    summary = {'schema_version': '1.0', 'dataset': 'HotpotQA distractor validation', 'data_revision': DATA_REVISION,
        'data_sha256': DATA_SHA256, 'model': MODEL_NAME, 'model_revision': MODEL_REVISION,
        'tune_count': len(tune), 'test_count': len(test), 'corpus_count': len(corpus), 'k': args.k,
        'test_metrics': {name: aggregate([row['methods'][name]['metrics'] for row in detailed]) for name in selected},
        'paired_comparisons': {name: paired_comparison(detailed, name)
                               for name in ('dense', 'legacy_weighted', 'tuned_hybrid', 'graph_ablation_no_boost')},
        'by_type': {kind: {name: aggregate([row['methods'][name]['metrics'] for row in detailed if row['type'] == kind])
                          for name in selected} for kind in sorted({row['type'] for row in detailed})},
        'graph': graph_info, 'seconds': time.perf_counter() - start, 'revision': revision, 'dirty': dirty,
        'source_sha256': source_hashes, 'python': platform.python_version(),
        'dependencies': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'sentence-transformers', 'rank-bm25', 'pyarrow')}}
    write_json(target / 'summary.json', summary)
    write_json(target / 'per_query.json', detailed)
    (target / 'index.html').write_text(render(summary, detailed), encoding='utf-8')
    print(json.dumps(summary['test_metrics'], indent=2), flush=True)
    print(target / 'index.html', flush=True)


if __name__ == '__main__':
    main()
