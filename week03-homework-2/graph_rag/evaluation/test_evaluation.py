import copy
import unittest
from types import SimpleNamespace

from graph_rag.services.fusion import fuse_results
from graph_rag.services.ranker import WeightedRanker
from .benchmark import split_rows, build_corpus, title_graph, weights, choose, paired_comparison
from .metrics import measure, aggregate


class EvaluationTests(unittest.TestCase):
    def test_paired_comparison_preserves_regressions(self):
        rows = [{'methods': {'tuned_graph': {'metrics': {'recall': a}},
                             'dense': {'metrics': {'recall': b}}}}
                for a, b in [(1, 0.5), (0, 0.5), (1, 1)]]
        result = paired_comparison(rows, 'dense')
        self.assertEqual(result['recall_delta'], 0)
        self.assertEqual((result['improved'], result['regressed'], result['unchanged']), (1, 1, 1))
        self.assertEqual(result, paired_comparison(rows, 'dense'))

    def test_metrics_duplicates_and_all_support(self):
        result = measure(['a', 'a', 'b'], ['a', 'c'], 2)
        self.assertEqual(result['recall'], 0.5)
        self.assertEqual(result['mrr'], 1)
        self.assertEqual(result['all_support'], 0)
        self.assertEqual(measure(['b', 'a'], ['a'], 2)['mrr'], 0.5)

    def test_empty_gold_is_not_perfect_recall(self):
        self.assertIsNone(measure([], [], 5)['recall'])
        self.assertEqual(measure(['a'], [], 5)['empty_return'], 0)
        with self.assertRaises(ValueError):
            aggregate([])

    def test_split_is_deterministic_disjoint_and_input_order_independent(self):
        rows = [{'id': str(i)} for i in range(30)]
        tune, test = split_rows(rows, 10, 10)
        self.assertFalse({r['id'] for r in tune} & {r['id'] for r in test})
        self.assertEqual((tune, test), split_rows(list(reversed(rows)), 10, 10))
        with self.assertRaises(ValueError):
            split_rows(rows + [rows[0]], 10, 10)

    def test_graph_and_corpus_do_not_consume_labels(self):
        rows = [{'context': {'title': ['Alpha Company', 'Beta Company'],
                 'sentences': [['Alpha Company owns Beta Company.'], ['Beta Company is small.']]},
                 'answer': 'secret', 'supporting_facts': {'title': ['Alpha Company']}}]
        corpus = build_corpus(rows)
        changed = copy.deepcopy(rows)
        changed[0]['answer'] = 'different'
        changed[0]['supporting_facts'] = {'title': ['invented']}
        self.assertEqual(corpus, build_corpus(changed))
        edges, match = title_graph(corpus)
        self.assertIn('Beta Company', edges['Alpha Company'])
        self.assertEqual(match('Who owns Alpha Company?'), {'Alpha Company'})

    def test_no_graph_only_empty_document(self):
        ranked, _ = fuse_results([{'id':'a','content':'real','score':0.8}], [],
                                 {'ghost': 10}, SimpleNamespace(**weights()))
        self.assertEqual([r['id'] for r in ranked], ['a'])
        self.assertEqual(ranked[0]['content'], 'real')

    def test_default_weighted_score_and_runtime_wrapper(self):
        config = SimpleNamespace(**{**weights(relative=0.5, minimum=0.2), 'KEYWORD_WEIGHT':0.3})
        vec = [{'id':'a','content':'a','score':0.8}, {'id':'b','content':'b','score':0.4}]
        kw = [{'id':'a','content':'a','score':0.5}]
        result = WeightedRanker(config).rank(vec, kw, {'a':0.2}, {})
        self.assertEqual([r['id'] for r in result['joint_results']], ['a'])
        self.assertAlmostEqual(result['joint_results'][0]['score'], 0.9, places=5)
        self.assertIn('vector_raw', result['debug_info']['component_scores']['a'])

    def test_rrf_is_score_scale_invariant(self):
        config = SimpleNamespace(**weights('rrf'))
        def result(scale):
            return fuse_results([{'id':'a','score':0.9}, {'id':'b','score':0.8}],
                                [{'id':'b','score':9*scale}, {'id':'a','score':3*scale}], {}, config)[0]
        self.assertEqual(result(1), result(1000))

    def test_tuning_uses_only_supplied_tune_rows(self):
        features = {'tune': {'dense':[{'id':'a','score':1}], 'bm25':[{'id':'b','score':1}], 'graph':{}},
                    'test': {'dense':[{'id':'b','score':1}], 'bm25':[{'id':'a','score':1}], 'graph':{}}}
        rows = [{'id':'tune','supporting_facts':{'title':['a']}}]
        chosen, _ = choose(features, rows, [weights(dense=1), weights(dense=0)], 1)
        self.assertEqual(chosen['VECTOR_WEIGHT'], 1)


if __name__ == '__main__':
    unittest.main()
