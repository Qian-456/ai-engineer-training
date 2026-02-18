import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graph_rag.services.rebuild_service import RebuildService
from graph_rag.core.interfaces import IDataSyncer, IVectorStore, IKeywordStore, IGraphStore
import asyncio

@pytest.fixture
def mock_dependencies():
    # 使用 AsyncMock 模拟异步行为，即便是现在的同步接口，我们预期它会变成异步
    vector_syncer = AsyncMock(spec=IDataSyncer)
    keyword_syncer = AsyncMock(spec=IDataSyncer)
    kg_syncer = AsyncMock(spec=IDataSyncer)
    
    vector_store = AsyncMock(spec=IVectorStore)
    keyword_store = AsyncMock(spec=IKeywordStore)
    graph_store = AsyncMock(spec=IGraphStore)
    
    # 手动设置我们期望的异步方法，绕过 spec 检查（因为接口还没改）
    vector_store.clear = AsyncMock()
    keyword_store.clear = AsyncMock()
    graph_store.clear = AsyncMock()
    
    vector_syncer.rebuild = AsyncMock()
    keyword_syncer.rebuild = AsyncMock()
    kg_syncer.rebuild = AsyncMock()

    return {
        "vector_syncer": vector_syncer,
        "keyword_syncer": keyword_syncer,
        "kg_syncer": kg_syncer,
        "vector_store": vector_store,
        "keyword_store": keyword_store,
        "graph_store": graph_store
    }

@pytest.mark.asyncio
async def test_rebuild_vector_store(mock_dependencies):
    deps = mock_dependencies
    service = RebuildService(**deps)
    
    await service.rebuild_vector_store()
    
    # 验证是否调用了异步的 clear 和 rebuild
    deps["vector_store"].clear.assert_awaited_once()
    deps["vector_syncer"].rebuild.assert_awaited_once()

@pytest.mark.asyncio
async def test_rebuild_keyword_store(mock_dependencies):
    deps = mock_dependencies
    service = RebuildService(**deps)
    
    await service.rebuild_keyword_store()
    
    deps["keyword_store"].clear.assert_awaited_once()
    deps["keyword_syncer"].rebuild.assert_awaited_once()

@pytest.mark.asyncio
async def test_rebuild_knowledge_graph(mock_dependencies):
    deps = mock_dependencies
    service = RebuildService(**deps)
    
    await service.rebuild_knowledge_graph()
    
    deps["graph_store"].clear.assert_awaited_once()
    deps["kg_syncer"].rebuild.assert_awaited_once()

@pytest.mark.asyncio
async def test_rebuild_all_concurrency(mock_dependencies):
    deps = mock_dependencies
    service = RebuildService(**deps)
    
    # Mock asyncio.gather to verify concurrency
    # 注意：RebuildService 需要导入 asyncio，并且调用 asyncio.gather
    with patch("graph_rag.services.rebuild_service.asyncio.gather", new_callable=AsyncMock) as mock_gather:
        await service.rebuild_all()
        
        # 验证 gather 被调用了一次
        assert mock_gather.call_count == 1
        
        # 验证它被传入了3个参数（协程）
        args, _ = mock_gather.call_args
        assert len(args) == 3
