import pytest
from unittest.mock import MagicMock, patch
from graph_rag.core.llm_client import LLMClient
import openai

@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mock Response"))]
        )
        yield client

def test_llm_chat_default_text_temperature(mock_openai_client):
    """测试普通对话使用默认 temperature=1.0"""
    llm = LLMClient()
    llm.chat([{"role": "user", "content": "Hi"}])
    
    # 验证调用参数
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 1.0

def test_llm_chat_json_temperature(mock_openai_client):
    """测试 JSON 模式使用默认 temperature=0.0"""
    llm = LLMClient()
    llm.chat(
        [{"role": "user", "content": "Hi"}],
        response_format={"type": "json_object"}
    )
    
    # 验证调用参数
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.0
    assert kwargs["response_format"] == {"type": "json_object"}

def test_llm_chat_explicit_temperature(mock_openai_client):
    """测试显式指定 temperature 覆盖默认值"""
    llm = LLMClient()
    
    # 覆盖 text 模式默认值
    llm.chat([{"role": "user", "content": "Hi"}], temperature=0.7)
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.7
    
    # 覆盖 json 模式默认值
    llm.chat(
        [{"role": "user", "content": "Hi"}], 
        response_format={"type": "json_object"},
        temperature=0.5
    )
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.5
