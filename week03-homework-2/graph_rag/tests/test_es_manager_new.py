import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from graph_rag.storage.es_manager import ESManager

@pytest.fixture
def mock_es_client():
    with patch("graph_rag.storage.es_manager.AsyncElasticsearch") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        
        # 默认 ping 成功
        mock_instance.ping.return_value = True
        
        # 默认 indices.exists 返回 False (需要创建)
        mock_instance.indices.exists.return_value = False
        
        yield mock_instance

@pytest.mark.asyncio
async def test_es_initialization_creates_index(mock_es_client):
    manager = ESManager()
    await manager.initialize()
    
    # 验证创建索引时是否设置了 number_of_replicas=0
    mock_es_client.indices.create.assert_awaited()
    call_args = mock_es_client.indices.create.call_args[1]
    
    assert "settings" in call_args["body"]
    assert call_args["body"]["settings"]["number_of_replicas"] == 0

@pytest.mark.asyncio
async def test_es_search_logic(mock_es_client):
    manager = ESManager()
    manager.client = mock_es_client
    manager.is_available = True
    
    # Mock search response
    mock_es_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "doc1", 
                    "_score": 1.5, 
                    "_source": {"content": "test content", "meta": "data"}
                }
            ]
        }
    }
    
    results = await manager.search("test query")
    
    # 验证查询构建
    mock_es_client.search.assert_awaited()
    call_args = mock_es_client.search.call_args[1]
    query_body = call_args["body"]
    
    # 验证使用了 match 查询
    assert "query" in query_body
    assert "match" in query_body["query"]
    assert query_body["query"]["match"]["content"] == "test query"
    
    # 验证结果解析
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["content"] == "test content"

@pytest.mark.asyncio
async def test_es_search_by_entity_logic(mock_es_client):
    manager = ESManager()
    manager.client = mock_es_client
    manager.is_available = True
    
    # Mock response
    mock_es_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "chunk1"},
                {"_id": "chunk2"}
            ]
        }
    }
    
    ids = await manager.search_by_entity("Entity A")
    
    # 验证使用了 match_phrase
    mock_es_client.search.assert_awaited()
    call_args = mock_es_client.search.call_args[1]
    query_body = call_args["body"]
    
    assert "match_phrase" in query_body["query"]
    assert query_body["query"]["match_phrase"]["content"] == "Entity A"
    
    # 验证返回 ID 列表
    assert ids == ["chunk1", "chunk2"]
