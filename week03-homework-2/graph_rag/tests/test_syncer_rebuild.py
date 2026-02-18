import asyncio
import os
from pathlib import Path

import pytest

from graph_rag.services.syncer import DocumentSyncer, BaseSyncProcessor


class FakeProcessor(BaseSyncProcessor):
    """
    用于测试的假同步处理器
    记录被调用情况，验证 DocumentSyncer.rebuild 的行为
    """

    def __init__(self) -> None:
        self.added_files = []
        self.modified_files = []
        self.deleted_files = []
        self.clear_called = 0

    async def process_added(self, file_paths):
        self.added_files.extend(file_paths)

    async def process_modified(self, file_paths):
        self.modified_files.extend(file_paths)

    async def process_deleted(self, file_paths):
        self.deleted_files.extend(file_paths)

    async def clear(self):
        """模拟清空底层存储"""
        self.clear_called += 1


@pytest.mark.asyncio
async def test_documentsyncer_rebuild_resets_state_and_resyncs(tmp_path: Path):
    """
    验证 DocumentSyncer.rebuild 会执行：
    1. 调用底层处理器的 clear
    2. 重置状态文件
    3. 再次扫描目录并将现有文件作为新增文件处理
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 创建一个测试文件
    file_path = data_dir / "test.txt"
    file_path.write_text("hello world", encoding="utf-8")

    state_file = tmp_path / "state.json"

    processor = FakeProcessor()
    syncer = DocumentSyncer(str(data_dir), str(state_file), processor)

    # 模拟已有的脏状态
    syncer.file_manager.state["dummy"] = "old_hash"
    syncer.file_manager.save_state()
    assert os.path.exists(state_file)

    # 执行重建
    await syncer.rebuild()

    # 1. 底层 clear 被调用
    assert processor.clear_called == 1

    # 2. 旧的状态键被清空
    assert "dummy" not in syncer.file_manager.state

    # 3. 现有文件被当作新增文件处理
    added_files = {os.path.abspath(p) for p in processor.added_files}
    assert os.path.abspath(str(file_path)) in added_files

    # 4. 状态文件中应包含当前文件
    #    reload 确认磁盘上状态已更新
    from graph_rag.storage.file_manager import FileStateManager

    fm = FileStateManager(str(state_file))
    assert os.path.abspath(str(file_path)) in fm.state

