import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from graph_rag.services.retriever import HybridRetriever
from graph_rag.core.schema import Entity

@pytest.fixture
def mock_managers():
    milvus = MagicMock()
    es = AsyncMock()
    kg = MagicMock()
    
    # Milvus 模拟 (Vector)
    milvus.search.return_value = [
        {"id": "doc1", "content": "Content 1", "score": 0.8},
        {"id": "doc2", "content": "Content 2", "score": 0.6}
    ]
    
    # ES 模拟 (Keyword)
    es.search.return_value = [
        {"id": "doc2", "content": "Content 2", "score": 2.0}, # High BM25 score
        {"id": "doc3", "content": "Content 3", "score": 1.0}
    ]
    es.search_by_entity.side_effect = lambda entity: ["doc1"] if entity == "EntityA" else []
    
    # KG 模拟 (Graph Reasoning)
    # KGManager.graph_reasoning is sync, but we might wrap it
    # 注意：代码中调用的是 kg_mgr.graph_reasoning (现在是接口方法)
    # 确保 mock 返回的字典包含正确的数据类型
    kg.graph_reasoning.return_value = {
        "entities": ["EntityA"],
        "confidence": 0.9
    }
    
    return milvus, es, kg

@pytest.mark.asyncio
async def test_retrieve_concurrency(mock_managers):
    milvus, es, kg = mock_managers
    retriever = HybridRetriever(milvus, es, kg)
    
    # Mock LLM entity extraction
    # 注意：现在 _extract_entities_async 是在 _kg_retrieval_chain 内部调用的
    # 如果我们直接测试 retrieve，它会调用 _kg_retrieval_chain -> _extract_entities_async
    # 所以 mock _extract_entities_async 依然有效
    with patch.object(retriever, '_extract_entities_async', return_value=[Entity(name="EntityA", type="ORG")]) as mock_extract:
        results = await retriever.retrieve("test query")
        
        # 验证结果不为空
        assert len(results["joint_results"]) > 0
        
        # 验证 doc1 分数: 
        # Vector: 0.8 * 0.5 = 0.4
        # Keyword: 0
        # Graph: 0.9 * 0.2 = 0.18 (EntityA -> doc1)
        # Total: 0.58
        
        # 验证 doc2 分数:
        # Vector: 0.6 * 0.5 = 0.3
        # Keyword: 2.0 * 0.3 = 0.6
        # Graph: 0
        # Total: 0.9
        
        # doc2 should be first
        assert results["joint_results"][0]["id"] == "doc2"
        assert results["joint_results"][1]["id"] == "doc1"
        
        # 验证并发调用
        es.search.assert_called_once()
        # 验证 _kg_retrieval_chain 逻辑被执行：提取实体 -> 推理
        mock_extract.assert_called_once()
        kg.graph_reasoning.assert_called_once_with(["EntityA"])

@pytest.mark.asyncio
async def test_retrieve_fallback(mock_managers):
    milvus, es, kg = mock_managers
    retriever = HybridRetriever(milvus, es, kg)
    
    # 模拟 ES 挂了
    es.search.side_effect = Exception("ES Down")
    
    with patch.object(retriever, '_extract_entities_async', return_value=[]):
        results = await retriever.retrieve("test query")
        
        # 应该只返回 Milvus 结果
        assert len(results["joint_results"]) == 2
        # doc1: 0.4, doc2: 0.3
        assert results["joint_results"][0]["id"] == "doc1"

@pytest.mark.asyncio
async def test_filtering(mock_managers):
    milvus, es, kg = mock_managers
    retriever = HybridRetriever(milvus, es, kg)
    
    # Mock scores to trigger filtering
    milvus.search.return_value = [
        {"id": "doc1", "score": 0.1}, # Very low -> 0.05
    ]
    es.search.return_value = []
    kg.graph_reasoning.return_value = {"entities": [], "confidence": 0.0}
    
    with patch.object(retriever, '_extract_entities_async', return_value=[]):
        results = await retriever.retrieve("test query")
        
        # 0.05 < 0.2 (Threshold), should be empty
        assert len(results["joint_results"]) == 0
