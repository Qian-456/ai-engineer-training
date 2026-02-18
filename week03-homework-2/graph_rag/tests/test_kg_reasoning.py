import pytest
from unittest.mock import MagicMock, patch
from graph_rag.storage.neo4j_manager import KGManager

@pytest.fixture
def mock_neo4j_driver():
    with patch('graph_rag.storage.neo4j_manager.GraphDatabase') as mock_gd:
        mock_driver = MagicMock()
        mock_gd.driver.return_value = mock_driver
        yield mock_driver

def test_graph_reasoning(mock_neo4j_driver):
    """测试图谱推理逻辑"""
    manager = KGManager(llm_client=MagicMock())
    
    # 模拟 session.run 返回的数据
    # 1-hop 结果
    mock_result_1hop = [
        {"source": "A", "relation": "invest", "target": "B", "confidence": 1.0},
    ]
    # 2-hop 结果
    mock_result_2hop = [
        {"source": "A", "relation": "invest->control", "target": "C", "confidence": 0.8},
    ]
    
    # 模拟 execute_query 的返回值
    # 假设 graph_reasoning 内部调用了两次 execute_query
    # 这里的 mock 比较 trick，因为 execute_query 被多次调用
    # 我们可以 mock execute_query 方法本身
    
    with patch.object(manager, 'execute_query') as mock_exec:
        mock_exec.side_effect = [mock_result_1hop, mock_result_2hop]
        
        entities = ["A"]
        result = manager.graph_reasoning(entities)
        
        # 验证返回结构
        # 返回的实体应包含输入实体以及所有关联出的实体
        expected_entities = {"A", "B", "C"}
        assert set(result["entities"]) == expected_entities
        assert len(result["relationships"]) == 2
        assert result["confidence"] == (1.0 + 0.8) / 2
        
        # 验证是否调用了两次查询
        assert mock_exec.call_count == 2
