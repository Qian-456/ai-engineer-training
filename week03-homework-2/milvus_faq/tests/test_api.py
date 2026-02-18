"""API 接口单元测试模块。

通过 Mock 掉重型依赖，实现对 FastAPI 路由逻辑的纯粹验证。
"""
import sys
import os

# 为了支持 IDE 直接点击运行按钮，将项目根目录加入 sys.path
# 确保在 milvus_faq 包外部运行也能正确识别包路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from unittest.mock import MagicMock

# 1. 拦截重型依赖，防止测试时加载模型或连接数据库
mock_core = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.node_parser'] = MagicMock()
sys.modules['llama_index.vector_stores.milvus'] = MagicMock()
sys.modules['llama_index.llms.openai_like'] = MagicMock()
sys.modules['llama_index.embeddings.huggingface'] = MagicMock()
sys.modules['milvus_faq.logger'] = MagicMock()
# 关键：拦截 core 模块，使 api.py 导入的 rag_manager 成为 Mock 对象
sys.modules['milvus_faq.core'] = mock_core

# 导入响应模型以便进行数据断言
from milvus_faq.models import QueryResponse, SourceInfo

class TestAPI(unittest.TestCase):
    """FastAPI 接口测试类。"""

    def setUp(self):
        """测试前置初始化。
        
        设置 FastAPI TestClient 并手动注入 Mock 后的 RAG 管理器。
        """
        # 延迟导入以确保模块拦截已生效
        from milvus_faq.api import app
        import milvus_faq.api as api
        from fastapi.testclient import TestClient
        
        # 手动注入 Mock 实例，跳过 lifespan 的初始化
        api.rag_manager = mock_core.rag_manager
        self.client = TestClient(app)
        self.mock_rag = api.rag_manager

    def test_query_endpoint(self):
        """测试 /query 接口的正常查询流程。
        
        验证接口是否能正确接收参数、调用底层的 RAG 管理器，并返回符合 QueryResponse 模型的 JSON 响应。
        """
        # 预定义 Mock 行为：模拟 RAG 管理器的返回结果
        self.mock_rag.query.return_value = QueryResponse(
            answer="这是一条模拟的测试回答",
            sources=[SourceInfo(content="测试文档内容", score=0.95)]
        )
        
        # 发送 GET 请求
        response = self.client.get("/query", params={"query": "什么是 RAG？"})
        
        # 断言响应状态码和内容
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "这是一条模拟的测试回答")
        self.assertEqual(data["sources"][0]["score"], 0.95)
        
        # 验证底层方法调用是否符合预期
        self.mock_rag.query.assert_called_with("什么是 RAG？")

if __name__ == '__main__':
    unittest.main()
