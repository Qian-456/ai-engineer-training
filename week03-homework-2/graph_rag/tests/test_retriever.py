import pytest
from unittest.mock import MagicMock, patch
from graph_rag.retriever import HybridRetriever

@pytest.fixture
def mock_managers():
    milvus_mgr = MagicMock()
    kg_mgr = MagicMock()
    return milvus_mgr, kg_mgr

def test_extract_entities(mock_managers):
    milvus_mgr, kg_mgr = mock_managers
    retriever = HybridRetriever(milvus_mgr, kg_mgr)
    
    with patch("graph_rag.retriever.llm_client") as mock_llm:
        mock_llm.chat.return_value = '["A 公司"]'
        entities = retriever._extract_entities("A 公司的股东是谁？")
        assert entities == ["A 公司"]

def test_retrieve_and_score(mock_managers):
    milvus_mgr, kg_mgr = mock_managers
    retriever = HybridRetriever(milvus_mgr, kg_mgr)
    
    # Mock Milvus search
    milvus_mgr.search.return_value = [
        {"content": "A 公司的主要业务是...", "score": 0.9}
    ]
    
    # Mock KG query
    kg_mgr.query_kg.return_value = "A 公司 --[INVESTS_IN]--> B 公司"
    
    with patch.object(retriever, "_extract_entities", return_value=["A 公司"]):
        with patch.object(retriever, "_evaluate_consistency", return_value=0.8):
            results = retriever.retrieve_and_score("A 公司的股东是谁？")
            
            assert len(results) == 1
            assert results[0]["score"] > 0
            assert "A 公司" in results[0]["kg_context"]
