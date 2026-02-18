import os
import json
from typing import Dict, List, Tuple
from graph_rag.core.utils import calculate_file_hash
from graph_rag.core.logger import logger

class FileStateManager:
    """
    文件状态管理器
    用于跟踪文件的变化（新增、修改、删除）
    """
    def __init__(self, state_file_path: str = ".file_state.json"):
        """
        初始化文件状态管理器
        
        Args:
            state_file_path: 存储状态的文件路径
        """
        self.state_file_path = state_file_path
        self.state: Dict[str, str] = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        """从文件加载状态"""
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载文件状态失败: {e}")
                return {}
        return {}

    def save_state(self):
        """将当前状态保存到文件"""
        try:
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存文件状态失败: {e}")

    def scan_changes(self, data_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """
        扫描数据目录并与当前状态对比
        
        Args:
            data_dir: 要扫描的数据目录
            
        Returns:
            (新增文件列表, 修改文件列表, 删除文件列表)
        """
        current_files = set()
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    # 忽略隐藏文件或状态文件本身
                    if file.startswith('.') or file == os.path.basename(self.state_file_path):
                        continue
                    
                    full_path = os.path.join(root, file)
                    # 标准化路径
                    norm_path = os.path.abspath(full_path)
                    current_files.add(norm_path)

        stored_files = set(self.state.keys())

        added = []
        modified = []
        deleted = []

        # 检查新增和修改
        for file_path in current_files:
            current_hash = calculate_file_hash(file_path)
            if file_path not in stored_files:
                added.append(file_path)
                # 立即更新状态
                self.state[file_path] = current_hash
            elif self.state.get(file_path) != current_hash:
                modified.append(file_path)
                # 更新状态
                self.state[file_path] = current_hash
        
        # 检查删除
        for file_path in list(stored_files):
            if file_path not in current_files:
                deleted.append(file_path)
                # 从状态中移除
                del self.state[file_path]

        return added, modified, deleted
