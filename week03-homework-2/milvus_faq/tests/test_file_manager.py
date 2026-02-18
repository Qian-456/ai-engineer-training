"""文件管理器单元测试模块。

测试文件状态监控逻辑，包括文件的增、删、改及持久化。
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
import shutil
import tempfile
from milvus_faq.file_manager import FileStateManager

class TestFileManager(unittest.TestCase):
    """文件状态管理器测试类。"""

    def setUp(self):
        """测试前置准备。
        
        创建一个临时目录并初始化文件状态管理器。
        """
        self.test_dir = tempfile.mkdtemp()
        # 定义状态持久化文件路径
        self.state_file = os.path.join(self.test_dir, ".file_state.json")
        self.manager = FileStateManager(self.state_file)

    def tearDown(self):
        """测试后置清理。
        
        删除测试过程中产生的临时目录及文件。
        """
        shutil.rmtree(self.test_dir)

    def test_initial_scan_empty(self):
        """测试空目录的初始扫描。
        
        验证在没有任何文件时，扫描结果应为空列表。
        """
        added, modified, deleted = self.manager.scan_changes(self.test_dir)
        self.assertEqual(added, [])
        self.assertEqual(modified, [])
        self.assertEqual(deleted, [])

    def test_add_file(self):
        """测试新增文件识别。
        
        验证管理器能准确识别新创建的文件并记录其状态。
        """
        file_path = os.path.join(self.test_dir, "test1.txt")
        with open(file_path, "w") as f:
            f.write("这是测试内容 1")
        
        added, modified, deleted = self.manager.scan_changes(self.test_dir)
        
        abs_path = os.path.abspath(file_path)
        self.assertEqual(added, [abs_path])
        self.assertEqual(modified, [])
        self.assertEqual(deleted, [])
        self.assertIn(abs_path, self.manager.state)

    def test_modify_file(self):
        """测试文件修改识别。
        
        验证当文件内容发生变化时，管理器能通过哈希比对识别出修改。
        """
        # 1. 先添加一个文件
        file_path = os.path.join(self.test_dir, "test1.txt")
        with open(file_path, "w") as f:
            f.write("初始内容")
        self.manager.scan_changes(self.test_dir)
        
        # 2. 修改文件内容
        with open(file_path, "w") as f:
            f.write("已修改的内容")
            
        added, modified, deleted = self.manager.scan_changes(self.test_dir)
        
        abs_path = os.path.abspath(file_path)
        self.assertEqual(added, [])
        self.assertEqual(modified, [abs_path])
        self.assertEqual(deleted, [])

    def test_delete_file(self):
        """测试删除文件识别。
        
        验证当物理文件被删除后，管理器能识别并更新内部状态。
        """
        # 1. 先添加文件
        file_path = os.path.join(self.test_dir, "test1.txt")
        with open(file_path, "w") as f:
            f.write("待删除文件")
        self.manager.scan_changes(self.test_dir)
        
        # 2. 执行物理删除
        os.remove(file_path)
        
        added, modified, deleted = self.manager.scan_changes(self.test_dir)
        
        abs_path = os.path.abspath(file_path)
        self.assertEqual(added, [])
        self.assertEqual(modified, [])
        self.assertEqual(deleted, [abs_path])
        self.assertNotIn(abs_path, self.manager.state)

    def test_state_persistence(self):
        """测试状态持久化。
        
        验证文件状态能否正确写入磁盘，并在重新加载时保持一致。
        """
        file_path = os.path.join(self.test_dir, "test1.txt")
        with open(file_path, "w") as f:
            f.write("持久化内容")
        
        self.manager.scan_changes(self.test_dir)
        self.manager.save_state()
        
        # 创建新的管理器实例指向同一个状态文件
        new_manager = FileStateManager(self.state_file)
        abs_path = os.path.abspath(file_path)
        
        self.assertIn(abs_path, new_manager.state)
        self.assertEqual(new_manager.state[abs_path], self.manager.state[abs_path])

if __name__ == "__main__":
    unittest.main()
