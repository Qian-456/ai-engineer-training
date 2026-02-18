import pytest
from unittest.mock import MagicMock, patch
from graph_rag.storage.neo4j_manager import KGManager

@pytest.fixture
def mock_neo4j():
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        yield mock_driver

def test_neo4j_execute_query(mock_neo4j):
    """测试 Cypher 查询执行"""
    mock_session = MagicMock()
    mock_neo4j.return_value.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [
        MagicMock(data=lambda: {"name": "A 公司"})
    ]
    
    mgr = KGManager(llm_client=MagicMock())
    result = mgr.execute_query("MATCH (n) RETURN n")
    assert len(result) == 1
    assert result[0]["name"] == "A 公司"

def test_neo4j_add_nodes(mock_neo4j):
    """测试添加节点和关系"""
    mock_session = MagicMock()
    mock_neo4j.return_value.session.return_value.__enter__.return_value = mock_session
    
    mgr = KGManager(llm_client=MagicMock())
    nodes = [{"type": "COMPANY", "name": "A 公司", "properties": {"code": "123"}}]
    relationships = [{"source": "B 公司", "target": "A 公司", "type": "INVESTS_IN"}]
    
    mgr.add_nodes_and_relationships(nodes, relationships)
    assert mock_session.execute_write.called
