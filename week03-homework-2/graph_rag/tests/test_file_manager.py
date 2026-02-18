import os
import pytest
import tempfile
import shutil
from graph_rag.file_manager import FileStateManager

@pytest.fixture
def temp_data_dir():
    """创建临时数据目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_file_state_manager(temp_data_dir):
    """测试文件状态管理器的基本功能"""
    state_file = os.path.join(temp_data_dir, ".test_state.json")
    manager = FileStateManager(state_file)
    
    # 1. 测试初始扫描（空目录）
    added, modified, deleted = manager.scan_changes(temp_data_dir)
    assert len(added) == 0
    assert len(modified) == 0
    assert len(deleted) == 0
    
    # 2. 测试新增文件
    test_file = os.path.join(temp_data_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("hello world")
    
    added, modified, deleted = manager.scan_changes(temp_data_dir)
    assert len(added) == 1
    assert os.path.abspath(test_file) in added
    
    # 3. 测试修改文件
    manager.save_state() # 保存状态以便下次对比
    with open(test_file, "w") as f:
        f.write("hello world modified")
    
    added, modified, deleted = manager.scan_changes(temp_data_dir)
    assert len(modified) == 1
    assert os.path.abspath(test_file) in modified
    
    # 4. 测试删除文件
    manager.save_state()
    os.remove(test_file)
    
    added, modified, deleted = manager.scan_changes(temp_data_dir)
    assert len(deleted) == 1
    assert os.path.abspath(test_file) in deleted
