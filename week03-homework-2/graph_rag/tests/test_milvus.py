import pytest
from unittest.mock import MagicMock, patch
from graph_rag.milvus_manager import MilvusManager

@pytest.fixture
def mock_milvus_deps():
    with patch("graph_rag.milvus_manager.MilvusVectorStore") as mock_store, \
         patch("graph_rag.milvus_manager.VectorStoreIndex") as mock_index, \
         patch("graph_rag.milvus_manager.HuggingFaceEmbedding") as mock_embed, \
         patch("graph_rag.milvus_manager.Settings") as mock_settings:
        yield mock_store, mock_index, mock_embed, mock_settings

def test_milvus_search(mock_milvus_deps):
    """测试向量检索"""
    mock_store, mock_index, mock_embed, mock_settings = mock_milvus_deps
    
    # Mock index.as_retriever().retrieve()
    mock_retriever = MagicMock()
    mock_index.from_vector_store.return_value.as_retriever.return_value = mock_retriever
    
    mock_node = MagicMock()
    mock_node.get_content.return_value = "Test content"
    mock_node.get_score.return_value = 0.95
    mock_node.metadata = {"file": "test.txt"}
    mock_retriever.retrieve.return_value = [mock_node]
    
    mgr = MilvusManager()
    results = mgr.search("query", top_k=1)
    
    assert len(results) == 1
    assert results[0]["content"] == "Test content"
    assert results[0]["score"] == 0.95
