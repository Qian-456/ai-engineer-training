from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from graph_rag.core.utils import ReadWriteLock, ReadLockContext, WriteLockContext, split_document
from graph_rag.services.syncer import BaseSyncProcessor
from graph_rag.core.interfaces import IVectorStore
import os
import asyncio

class MilvusManager(BaseSyncProcessor, IVectorStore):
    """
    Milvus 向量数据库管理器
    继承自 BaseSyncProcessor 以支持自动化同步
    实现 IVectorStore 接口
    """
    def __init__(self):
        self.lock = ReadWriteLock()
        self._setup_embedding()
        self._setup_index()

    def _setup_embedding(self):
        """配置 Embedding 模型"""
        try:
            # 使用本地模型或 HuggingFace 模型
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
                cache_folder=settings.milvus.EMBED_CACHE_DIR
            )
            Settings.embed_model = embed_model
            
            # 设置切分器
            self.sentence_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            logger.info("Milvus Embedding 模型初始化成功")
        except Exception as e:
            logger.error(f"Milvus Embedding 模型初始化失败: {e}")
            raise

    def _setup_index(self):
        """连接 Milvus 并加载索引"""
        try:
            # 强制重新初始化 VectorStore，确保连接是最新的
            self.vector_store = MilvusVectorStore(
                uri=settings.milvus.URI,
                token=settings.milvus.TOKEN,
                collection_name=settings.milvus.COLLECTION_NAME,
                dim=settings.milvus.DIMENSION,
                overwrite=False
            )
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
            logger.info(f"成功连接到 Milvus 集合: {settings.milvus.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

    async def process_added(self, file_paths: List[str]):
        """处理新增文件"""
        loop = asyncio.get_running_loop()
        def _sync_process():
            with WriteLockContext(self.lock):
                for file_path in file_paths:
                    self._insert_file(file_path)
                    logger.info(f"Milvus: 已新增文件 {file_path}")
        await loop.run_in_executor(None, _sync_process)

    async def process_modified(self, file_paths: List[str]):
        """处理修改文件（先删后增）"""
        loop = asyncio.get_running_loop()
        def _sync_process():
            with WriteLockContext(self.lock):
                for file_path in file_paths:
                    safe_path = file_path.replace("\\", "/")
                    self.index.delete_ref_doc(safe_path, delete_from_docstore=True)
                    self._insert_file(file_path)
                    logger.info(f"Milvus: 已更新文件 {file_path}")
        await loop.run_in_executor(None, _sync_process)

    async def process_deleted(self, file_paths: List[str]):
        """处理删除文件"""
        loop = asyncio.get_running_loop()
        def _sync_process():
            with WriteLockContext(self.lock):
                for file_path in file_paths:
                    safe_path = file_path.replace("\\", "/")
                    self.index.delete_ref_doc(safe_path, delete_from_docstore=True)
                    logger.info(f"Milvus: 已删除文件 {file_path}")
        await loop.run_in_executor(None, _sync_process)

    def _insert_file(self, file_path: str):
        """插入单个文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 使用统一的切分逻辑，确保 ID 与 ES 一致
            nodes = split_document(file_path, content)
            self.index.insert_nodes(nodes)
        except Exception as e:
            logger.error(f"Milvus 插入文件失败 {file_path}: {e}")


    async def clear(self):
        """实现 IVectorStore.clear (异步)"""
        loop = asyncio.get_running_loop()
        
        def _sync_drop():
            try:
                from pymilvus import utility, connections
                
                connections.connect("default", uri=settings.milvus.URI, token=settings.milvus.TOKEN)
                
                if utility.has_collection(settings.milvus.COLLECTION_NAME):
                    utility.drop_collection(settings.milvus.COLLECTION_NAME)
                    logger.info(f"Milvus 集合 {settings.milvus.COLLECTION_NAME} 已删除")
                
            except Exception as e:
                logger.error(f"Milvus clear 失败: {e}")
        
        await loop.run_in_executor(None, _sync_drop)
        # 重新初始化索引之前，必须确保重新建立连接和上下文
        try:
             # _setup_index 内部会重新创建 MilvusVectorStore，它会处理连接
            self._setup_index()
        except Exception as e:
            logger.error(f"Milvus clear 后重建索引失败: {e}")

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """实现 IVectorStore.search"""
        if not query or not query.strip():
            logger.warning("Milvus search received empty query, returning empty results.")
            return []

        try:
            with ReadLockContext(self.lock):
                # 检查索引是否就绪
                if not hasattr(self, 'index') or self.index is None:
                    logger.warning("Milvus index is not initialized, attempting to re-initialize...")
                    self._setup_index()

                retriever = self.index.as_retriever(similarity_top_k=top_k)
                nodes = retriever.retrieve(query)
                
                results = []
                for node in nodes:
                    results.append({
                        "id": node.node.node_id, # 增加 ID 返回
                        "content": node.node.get_content(), # 注意 node 是 NodeWithScore 类型
                        "score": node.score if node.score is not None else 0.0,
                        "metadata": node.node.metadata
                    })
                return results
        except Exception as e:
            logger.error(f"Milvus 检索失败: {e}", exc_info=True)
            return []
