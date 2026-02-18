"""核心业务逻辑（RAGManager）单元测试模块。

通过大量 Mock 模拟 LlamaIndex 组件及文件系统，验证知识库同步与查询逻辑。
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
from unittest.mock import MagicMock, patch, ANY

from milvus_faq.core import RAGManager
from milvus_faq.models import QueryResponse

class TestRAGManager(unittest.TestCase):
    """RAG 管理器测试类。"""

    def setUp(self):
        """测试前置准备。
        
        通过 patch 拦截所有重型组件（LLM, Embedding, VectorStore, Splitters）并配置 Mock 行为。
        """
        # 1. 设置所有需要的 Patcher
        # 使用 start() 启动并注册 stop() 清理，确保在测试方法执行期间 Mock 一直有效
        
        # Mock Settings
        self.patcher_settings = patch('milvus_faq.core.Settings')
        self.MockSettings = self.patcher_settings.start()
        self.addCleanup(self.patcher_settings.stop)
        
        # 配置 Settings Mock
        self.MockSettings.llm = MagicMock()
        self.MockSettings.embed_model = MagicMock()

        # Mock 其他依赖
        self.patcher_milvus = patch('milvus_faq.core.MilvusVectorStore')
        self.MockMilvus = self.patcher_milvus.start()
        self.addCleanup(self.patcher_milvus.stop)

        self.patcher_index = patch('milvus_faq.core.VectorStoreIndex')
        self.MockIndex = self.patcher_index.start()
        self.addCleanup(self.patcher_index.stop)

        self.patcher_storage = patch('milvus_faq.core.StorageContext')
        self.MockStorage = self.patcher_storage.start()
        self.addCleanup(self.patcher_storage.stop)

        self.patcher_openai = patch('milvus_faq.core.OpenAILike')
        self.MockOpenAI = self.patcher_openai.start()
        self.addCleanup(self.patcher_openai.stop)

        self.patcher_hf = patch('milvus_faq.core.HuggingFaceEmbedding')
        self.MockHF = self.patcher_hf.start()
        self.addCleanup(self.patcher_hf.stop)

        self.patcher_file_manager = patch('milvus_faq.core.FileStateManager')
        self.MockFileManager = self.patcher_file_manager.start()
        self.addCleanup(self.patcher_file_manager.stop)

        self.patcher_semantic = patch('milvus_faq.core.SemanticSplitterNodeParser')
        self.MockSemanticSplitter = self.patcher_semantic.start()
        self.addCleanup(self.patcher_semantic.stop)

        self.patcher_sentence = patch('milvus_faq.core.SentenceSplitter')
        self.MockSentenceSplitter = self.patcher_sentence.start()
        self.addCleanup(self.patcher_sentence.stop)
        

        # 2. 配置 Mock 行为
        # File Manager
        self.mock_file_manager_instance = self.MockFileManager.return_value
        self.mock_file_manager_instance.scan_changes.return_value = ([], [], [])
        
        # Splitters
        self.mock_semantic_splitter_instance = self.MockSemanticSplitter.return_value
        self.mock_sentence_splitter_instance = self.MockSentenceSplitter.return_value

        # 3. 初始化 RAGManager
        # 禁止后台线程启动
        with patch.object(RAGManager, '_start_watcher'):
             self.manager = RAGManager()
        
        # 获取 manager 内部引用的对象，方便断言
        self.mock_index_instance = self.manager.index
        # 此时 self.manager.semantic_splitter 应该是我们的 Mock 对象

    @patch('milvus_faq.core.SimpleDirectoryReader')
    def test_sync_data_directory_added(self, MockReader):
        """测试新增文件同步。
        
        验证当文件管理器检测到新文件时，系统能正确加载文档、切分节点并插入索引。
        """
        # 模拟文件变更：新增一个文件
        fake_file = r'C:\test\data\new.txt'
        self.mock_file_manager_instance.scan_changes.return_value = ([fake_file], [], [])
        
        # 模拟 Document 加载
        mock_doc = MagicMock()
        mock_doc.id_ = fake_file
        mock_reader_instance = MockReader.return_value
        mock_reader_instance.load_data.return_value = [mock_doc]
        
        # 模拟切分器返回节点
        self.manager.semantic_splitter.get_nodes_from_documents.return_value = ["node1"]
        
        # 执行同步
        with patch('os.path.exists', return_value=True):
            self.manager.sync_data_directory()
        
        # 验证逻辑
        # 1. Reader 是否被正确调用
        MockReader.assert_called_with(input_files=[fake_file], filename_as_id=True)
        
        # 2. 路径归一化验证 (Windows 下反斜杠应转为正斜杠)
        expected_id = fake_file.replace("\\", "/")
        self.assertEqual(mock_doc.id_, expected_id)
        
        # 3. 是否调用了 index.insert_nodes
        self.mock_index_instance.insert_nodes.assert_called_with(["node1"])
        
        # 4. 状态是否保存
        self.mock_file_manager_instance.save_state.assert_called()

    def test_sync_data_directory_deleted(self):
        """测试文件删除同步。
        
        验证当文件被删除时，系统能根据文件路径 ID 从向量索引中移除对应引用。
        """
        # 模拟文件变更：删除一个文件
        fake_file = r'C:\test\data\deleted.txt'
        self.mock_file_manager_instance.scan_changes.return_value = ([], [], [fake_file])
        
        # 执行同步
        self.manager.sync_data_directory()
        
        # 验证索引删除操作被调用 (Windows 路径兼容处理)
        expected_id = fake_file.replace('\\', '/')
        self.mock_index_instance.delete_ref_doc.assert_called_with(expected_id, delete_from_docstore=True)

    def test_sync_data_directory_modified(self):
        """测试文件修改同步。
        
        验证当文件修改时，系统先执行旧节点删除，再加载新内容并插入。
        """
        # 模拟文件变更：修改一个文件
        fake_file = r'C:\test\data\mod.txt'
        self.mock_file_manager_instance.scan_changes.return_value = ([], [fake_file], [])
        
        # 模拟 Document 加载逻辑
        with patch('milvus_faq.core.SimpleDirectoryReader') as MockReader:
            mock_doc = MagicMock()
            mock_doc.id_ = fake_file
            MockReader.return_value.load_data.return_value = [mock_doc]
            
            self.manager.sync_data_directory()
            
            # 验证先删后增逻辑
            expected_id = fake_file.replace('\\', '/')
            self.mock_index_instance.delete_ref_doc.assert_called_with(expected_id, delete_from_docstore=True)
            self.mock_index_instance.insert_nodes.assert_called()

    def test_query_success(self):
        """测试查询成功场景。
        
        验证查询请求能正确透传至索引查询引擎，并封装为 QueryResponse 对象。
        """
        # 模拟查询引擎返回
        mock_engine = self.mock_index_instance.as_query_engine.return_value
        
        # 模拟 LlamaIndex 的 Response 对象
        mock_response = MagicMock()
        mock_response.__str__.return_value = "Found answer"
        
        # 模拟 NodeWithScore 对象
        mock_node_with_score = MagicMock()
        mock_node_with_score.score = 0.8
        # 模拟内部的 TextNode 对象
        mock_node_with_score.node.get_text.return_value = "source content"
        
        mock_response.source_nodes = [mock_node_with_score]
        mock_engine.query.return_value = mock_response
        
        response = self.manager.query("Hello")
        
        self.assertIsInstance(response, QueryResponse)
        self.assertEqual(response.answer, "Found answer")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].content, "source content")
        self.assertEqual(response.sources[0].score, 0.8)

    def test_query_empty(self):
        """测试空查询返回。
        
        验证当索引未初始化或查询无结果时，系统能优雅处理并返回默认响应。
        """
        self.manager.index = None
        response = self.manager.query("Hello")
        self.assertEqual(response.answer, "Empty Response")

if __name__ == "__main__":
    unittest.main()
