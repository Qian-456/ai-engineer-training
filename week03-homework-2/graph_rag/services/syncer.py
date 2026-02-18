from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import os
from graph_rag.storage.file_manager import FileStateManager
from graph_rag.core.logger import logger
import asyncio

class BaseSyncProcessor(ABC):
    """
    同步处理器基类
    定义了如何处理新增、修改和删除的文件
    """
    @abstractmethod
    async def process_added(self, file_paths: List[str]):
        """处理新增文件"""
        pass

    @abstractmethod
    async def process_modified(self, file_paths: List[str]):
        """处理修改的文件"""
        pass

    @abstractmethod
    async def process_deleted(self, file_paths: List[str]):
        """处理删除的文件"""
        pass


from graph_rag.core.interfaces import IDataSyncer

class DocumentSyncer(IDataSyncer):
    """
    通用文档同步器
    负责扫描目录、管理状态并协调处理器执行同步
    实现 IDataSyncer 接口，支持 rebuild 操作
    """
    def __init__(self, data_dir: str, state_file: str, processor: BaseSyncProcessor):
        """
        Args:
            data_dir: 待监控的数据目录
            state_file: 状态存储文件路径
            processor: 具体的同步处理器实例
        """
        self.data_dir = data_dir
        self.file_manager = FileStateManager(state_file)
        self.processor = processor
        
        # 确保目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    async def sync(self):
        """执行一次同步循环"""
        # scan_changes 涉及大量文件 IO，放入 executor
        loop = asyncio.get_running_loop()
        
        added, modified, deleted = await loop.run_in_executor(
            None, 
            self.file_manager.scan_changes, 
            self.data_dir
        )
        
        if not (added or modified or deleted):
            return False

        logger.info(f"[{os.path.basename(self.data_dir)}] 检测到变更: 新增 {len(added)}, 修改 {len(modified)}, 删除 {len(deleted)}")
        
        # 判断是否是 KGManager
        from graph_rag.storage.neo4j_manager import KGManager
        is_kg = isinstance(self.processor, KGManager)
        
        if is_kg:
            # 如果是 KG，只要有变更就报警，不执行具体的 process_
            if added or modified or deleted:
                logger.warning(
                    f"KG: 检测到数据变更 (新增 {len(added)}, 修改 {len(modified)}, 删除 {len(deleted)})。"
                    "由于知识图谱不支持自动增量更新，请手动调用 /kg/rebuild 进行全量重建以保证一致性。"
                )
                return False
        
        try:
            if deleted:
                await self.processor.process_deleted(deleted)
            if added:
                await self.processor.process_added(added)
            if modified:
                await self.processor.process_modified(modified)
                
            # 处理成功后保存状态 (IO)
            await loop.run_in_executor(None, self.file_manager.save_state)
            return True
        except Exception as e:
            logger.error(f"同步目录 {self.data_dir} 失败: {e}")
            return False

    async def rebuild(self):
        """
        执行重建流程：
        1. 调用底层处理器的 clear 方法清空存储（如果实现）
        2. 清空并重置状态文件
        3. 再次执行一次完整同步，将当前目录下的文件视为新增文件
        """
        # 1. Clear Store
        clear_fn = getattr(self.processor, "clear", None)
        if clear_fn is not None:
            try:
                if asyncio.iscoroutinefunction(clear_fn):
                    await clear_fn()
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, clear_fn)
            except Exception as e:
                logger.error(f"重建时清空底层存储失败: {e}")

        # 2. Reset State (Memory & Disk)
        self.file_manager.state.clear()
        try:
            if os.path.exists(self.file_manager.state_file_path):
                os.remove(self.file_manager.state_file_path)
        except Exception as e:
            logger.error(f"删除状态文件失败 {self.file_manager.state_file_path}: {e}")

        # 3. Re-Sync
        # 此时 state 已空，scan_changes 会把所有现有文件当做 added
        
        # 特殊处理：如果是 KGManager，sync() 会因为检测到变更而跳过并报警
        # 所以对于 rebuild，我们需要绕过 sync 的报警检查，直接调用 processor
        
        # 判断是否是 KGManager
        from graph_rag.storage.neo4j_manager import KGManager
        is_kg = isinstance(self.processor, KGManager)
        
        if is_kg:
            # 手动执行 sync 的逻辑，但不检查 is_kg 阻断
            loop = asyncio.get_running_loop()
            
            # 重新扫描所有文件（因为 state 已清空，所有文件都是 added）
            added, _, _ = await loop.run_in_executor(
                None, 
                self.file_manager.scan_changes, 
                self.data_dir
            )
            
            if added:
                logger.info(f"KG Rebuild: 开始处理 {len(added)} 个文件...")
                try:
                    await self.processor.process_added(added)
                    # 处理成功后保存状态
                    await loop.run_in_executor(None, self.file_manager.save_state)
                    logger.info("KG Rebuild: 完成")
                except Exception as e:
                    logger.error(f"KG Rebuild 失败: {e}")
            else:
                logger.info("KG Rebuild: 目录为空，无需处理")
        else:
            # 普通 Syncer，直接复用 sync
            await self.sync()
