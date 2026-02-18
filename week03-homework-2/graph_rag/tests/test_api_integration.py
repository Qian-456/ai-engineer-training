import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from graph_rag.api.api_server import app

@pytest.fixture
def client_with_mock_pipeline():
    # Mock Container instead of GraphRAGPipeline class
    with patch("graph_rag.api.api_server.container") as mock_container:
        mock_pipeline = AsyncMock()
        
        # Setup async methods
        mock_pipeline.run = AsyncMock(return_value={
            "answer": "Integration Test Answer",
            "retrieval_results": {"joint_results": []},
            "confidence": 0.95,
            "metrics": {}
        })
        mock_pipeline.sync_data = AsyncMock()
        
        mock_pipeline.rebuild_vector = AsyncMock()
        mock_pipeline.rebuild_keyword = AsyncMock()
        mock_pipeline.rebuild_kg = AsyncMock()
        mock_pipeline.rebuild_all = AsyncMock()

        # Configure container to return this mock pipeline
        mock_container.graph_rag_pipeline = mock_pipeline
        mock_container.initialize_resources = AsyncMock()
        mock_container.close_resources = MagicMock()

        # Patch threading.Thread to prevent background sync from actually running and cluttering logs
        with patch("threading.Thread"):
            with TestClient(app) as client:
                yield client, mock_pipeline

def test_query_integration(client_with_mock_pipeline):
    client, mock_pipeline = client_with_mock_pipeline
    
    response = client.post("/query", json={"text": "Integration Test Query"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Integration Test Answer"
    
    # Verify pipeline.run was called
    mock_pipeline.run.assert_awaited_once()

def test_rebuild_vector_endpoint(client_with_mock_pipeline):
    client, mock_pipeline = client_with_mock_pipeline
    
    response = client.post("/rebuild/vector")
    
    assert response.status_code == 200
    assert "任务已启动" in response.json()["message"]
    
    # TestClient usually waits for background tasks
    mock_pipeline.rebuild_vector.assert_awaited_once()

def test_rebuild_keyword_endpoint(client_with_mock_pipeline):
    client, mock_pipeline = client_with_mock_pipeline
    
    response = client.post("/rebuild/keyword")
    
    assert response.status_code == 200
    mock_pipeline.rebuild_keyword.assert_awaited_once()

def test_rebuild_kg_endpoint(client_with_mock_pipeline):
    client, mock_pipeline = client_with_mock_pipeline
    
    response = client.post("/rebuild/kg")
    
    assert response.status_code == 200
    mock_pipeline.rebuild_kg.assert_awaited_once()

def test_rebuild_all_endpoint(client_with_mock_pipeline):
    client, mock_pipeline = client_with_mock_pipeline
    
    response = client.post("/rebuild/all")
    
    assert response.status_code == 200
    assert "全量重建任务已启动" in response.json()["message"]
    
    mock_pipeline.rebuild_all.assert_awaited_once()
