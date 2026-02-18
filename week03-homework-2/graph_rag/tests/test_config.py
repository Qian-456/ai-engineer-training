import os
import pytest
from graph_rag.config import settings

def test_settings_load():
    """测试配置加载"""
    assert settings.PROJECT_NAME == "Graph RAG System"
    assert settings.llm.MODEL == "deepseek-chat"
    assert settings.rag.VECTOR_WEIGHT == 0.3

def test_env_override(monkeypatch):
    """测试环境变量覆盖配置"""
    monkeypatch.setenv("LLM__MODEL", "gpt-4")
    # 重新加载配置是不太容易的，因为 settings 是单例
    # 但我们可以检查环境变量是否能被 pydantic settings 识别
    from graph_rag.config import AppConfig
    new_settings = AppConfig()
    assert new_settings.llm.MODEL == "gpt-4"
