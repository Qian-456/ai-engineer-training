import pytest
from unittest.mock import AsyncMock, patch
from graph_rag.services.rebuild_service import RebuildService
from graph_rag.core.interfaces import IDataSyncer

@pytest.mark.asyncio
async def test_rebuild_service_delegates_to_syncer():
    # Mock Syncers
    rag_syncer = AsyncMock(spec=IDataSyncer)
    es_syncer = AsyncMock(spec=IDataSyncer)
    kg_syncer = AsyncMock(spec=IDataSyncer)
    
    # Initialize Service
    service = RebuildService(rag_syncer, es_syncer, kg_syncer)
    
    # Test rebuild_vector_store
    await service.rebuild_vector_store()
    rag_syncer.rebuild.assert_awaited_once()
    
    # Test rebuild_keyword_store
    await service.rebuild_keyword_store()
    es_syncer.rebuild.assert_awaited_once()
    
    # Test rebuild_knowledge_graph
    await service.rebuild_knowledge_graph()
    kg_syncer.rebuild.assert_awaited_once()
    
    # Test rebuild_all (concurrency)
    # Reset mocks
    rag_syncer.rebuild.reset_mock()
    es_syncer.rebuild.reset_mock()
    kg_syncer.rebuild.reset_mock()
    
    with patch("graph_rag.services.rebuild_service.asyncio.gather", new_callable=AsyncMock) as mock_gather:
        await service.rebuild_all()
        assert mock_gather.call_count == 1
        args, _ = mock_gather.call_args
        assert len(args) == 3
