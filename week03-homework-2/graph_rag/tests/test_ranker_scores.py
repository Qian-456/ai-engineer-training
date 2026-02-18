import pytest

from graph_rag.services.ranker import WeightedRanker


def test_ranker_component_scores_and_logging(caplog):
    vec_res = [
        {"id": "doc1", "content": "v1", "score": 0.8, "metadata": {}},
        {"id": "doc2", "content": "v2", "score": 0.4, "metadata": {}},
    ]
    kw_res = [
        {"id": "doc1", "content": "k1", "score": 0.5, "metadata": {}},
    ]
    graph_boost_map = {"doc1": 0.2}
    kg_debug_info = {"entities": ["A 公司"], "confidence": 1.0}

    ranker = WeightedRanker()

    with caplog.at_level("INFO"):
        result = ranker.rank(vec_res, kw_res, graph_boost_map, kg_debug_info)

    joint = result["joint_results"]
    debug = result["debug_info"]

    assert len(joint) >= 1
    assert "component_scores" in debug
    assert "doc1" in debug["component_scores"]

    comp = debug["component_scores"]["doc1"]
    assert "vector_raw" in comp
    assert "keyword_raw" in comp
    assert "graph_boost" in comp
    assert "final_score" in comp

