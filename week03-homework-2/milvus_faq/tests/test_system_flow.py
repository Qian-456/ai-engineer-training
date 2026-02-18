import sys
import os

# 为了支持 IDE 直接点击运行按钮，将项目根目录加入 sys.path
# 确保在 milvus_faq 包外部运行也能正确识别包路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.node_parser'] = MagicMock()
sys.modules['llama_index.vector_stores.milvus'] = MagicMock()
sys.modules['llama_index.llms.openai_like'] = MagicMock()
sys.modules['llama_index.embeddings.huggingface'] = MagicMock()
sys.modules['milvus_faq.logger'] = MagicMock()
# sys.modules['milvus_faq.core'] = MagicMock() # Removed to avoid pollution

from milvus_faq.api import app
from milvus_faq.models import QueryResponse, SourceInfo

class TestSystemIntegration(unittest.TestCase):
    """系统集成测试模拟类。
    
    验证 API 路由与业务逻辑的协同工作。
    由于测试环境限制，对底层重型依赖进行了 Mock 处理。
    """
    
    def setUp(self):
        """测试前置准备。"""
        self.client = TestClient(app)
        # 拦截 api 模块中的 rag_manager 全局变量
        self.rag_patcher = patch('milvus_faq.api.rag_manager')
        self.mock_rag = self.rag_patcher.start()
        
    def tearDown(self):
        """测试后置清理。"""
        self.rag_patcher.stop()

    def test_1_query_empty_kb(self):
        """测试场景：知识库初始化后的查询流程。"""
        # 模拟 RAG 返回空结果或通用回答
        self.mock_rag.query.return_value = QueryResponse(
            answer="抱歉，知识库中没有相关信息。",
            sources=[]
        )
        
        response = self.client.get("/query", params={"query": "什么是 Milvus？"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['answer'], "抱歉，知识库中没有相关信息。")
        self.assertEqual(len(data['sources']), 0)

if __name__ == "__main__":
    unittest.main()
