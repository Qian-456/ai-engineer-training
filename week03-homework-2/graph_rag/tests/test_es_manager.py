import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from graph_rag.storage.es_manager import ESManager
from graph_rag.core.config import settings

@pytest.fixture
def mock_es_client():
    with patch('graph_rag.storage.es_manager.AsyncElasticsearch') as mock_client:
        # 模拟 ES 客户端实例
        client_instance = AsyncMock()
        mock_client.return_value = client_instance
        
        # 模拟 ping 返回 True
        client_instance.ping.return_value = True
        
        # 模拟 search 响应
        client_instance.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 1.5,
                        "_source": {"content": "Test content", "file_path": "test.txt"}
                    }
                ]
            }
        }
        
        yield client_instance

@pytest.mark.asyncio
async def test_es_manager_initialization(mock_es_client):
    """测试 ESManager 初始化和连接检查"""
    manager = ESManager()
    await manager.initialize()
    
    assert manager.client is not None
    mock_es_client.ping.assert_called_once()
    assert manager.is_available is True

@pytest.mark.asyncio
async def test_es_index_document(mock_es_client):
    """测试文档索引"""
    manager = ESManager()
    await manager.initialize()
    
    await manager.index_document(
        doc_id="doc1",
        content="Test content",
        metadata={"file_path": "test.txt"}
    )
    
    mock_es_client.index.assert_called_once_with(
        index=settings.es.INDEX_NAME,
        id="doc1",
        document={
            "content": "Test content",
            "file_path": "test.txt"
        }
    )

@pytest.mark.asyncio
async def test_es_search_keyword(mock_es_client):
    """测试关键词搜索"""
    manager = ESManager()
    await manager.initialize()
    
    results = await manager.search("test query")
    
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] == 1.5
    
    mock_es_client.search.assert_called_once()
    args, kwargs = mock_es_client.search.call_args
    assert kwargs["index"] == settings.es.INDEX_NAME
    assert "query" in kwargs["body"]
    assert "match" in kwargs["body"]["query"]

@pytest.mark.asyncio
async def test_es_search_by_entity(mock_es_client):
    """测试实体精确匹配"""
    manager = ESManager()
    await manager.initialize()
    
    # 修改 mock 返回以匹配 match_phrase
    mock_es_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "doc2",
                    "_source": {"content": "Content with Entity A", "file_path": "test.txt"}
                }
            ]
        }
    }
    
    results = await manager.search_by_entity("Entity A")
    
    assert len(results) == 1
    assert results[0] == "doc2"
    
    args, kwargs = mock_es_client.search.call_args
    assert "match_phrase" in kwargs["body"]["query"]

@pytest.mark.asyncio
async def test_es_fallback(mock_es_client):
    """测试连接失败时的降级"""
    mock_es_client.ping.return_value = False
    mock_es_client.ping.side_effect = Exception("Connection Refused")
    
    manager = ESManager()
    await manager.initialize()
    
    assert manager.is_available is False
    
    # 在不可用状态下调用方法应直接返回空或不报错
    results = await manager.search("test")
    assert results == []
    
    await manager.index_document("id", "content")
    mock_es_client.index.assert_not_called()
