import os
import json
from typing import Dict, List, Tuple
from milvus_faq.utils import calculate_file_hash
from milvus_faq.logger import logger

class FileStateManager:
    def __init__(self, state_file_path: str = ".file_state.json"):
        self.state_file_path = state_file_path
        self.state: Dict[str, str] = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load file state: {e}")
                return {}
        return {}

    def save_state(self):
        try:
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save file state: {e}")

    def scan_changes(self, data_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Scans the data directory and compares with current state.
        Returns: (added_files, modified_files, deleted_files)
        """
        current_files = set()
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    # Ignore hidden files or the state file itself
                    if file.startswith('.') or file == os.path.basename(self.state_file_path):
                        continue
                    
                    full_path = os.path.join(root, file)
                    # Normalize path to standard format
                    norm_path = os.path.abspath(full_path)
                    current_files.add(norm_path)

        stored_files = set(self.state.keys())

        added = []
        modified = []
        deleted = []

        # Check for added and modified
        for file_path in current_files:
            current_hash = calculate_file_hash(file_path)
            if file_path not in stored_files:
                added.append(file_path)
                # Update state immediately for added files (will be saved later)
                self.state[file_path] = current_hash
            elif self.state[file_path] != current_hash:
                modified.append(file_path)
                # Update state
                self.state[file_path] = current_hash
        
        # Check for deleted
        for file_path in stored_files:
            if file_path not in current_files:
                deleted.append(file_path)
                # Remove from state
                del self.state[file_path]

        return added, modified, deleted

