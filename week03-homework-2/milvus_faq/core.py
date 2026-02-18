from typing import List, Union
from pathlib import Path
import os
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Document, 
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from milvus_faq.config import settings
from milvus_faq.logger import logger, LoggerManager
from milvus_faq.models import QueryResponse, SourceInfo
from milvus_faq.utils import ReadWriteLock, ReadLockContext, WriteLockContext
from milvus_faq.file_manager import FileStateManager

import time
import threading

class RAGManager:
    def __init__(self):
        # Initialize logging
        LoggerManager.setup(
            log_dir=settings.logging.LOG_DIR,
            level=settings.logging.LOG_LEVEL,
            rotation=settings.logging.LOG_ROTATION,
            retention=settings.logging.LOG_RETENTION
        )
        self.lock = ReadWriteLock()
        self.file_manager = FileStateManager(settings.data.STATE_FILE)
        
        self._setup_llm()
        self._setup_embedding()
        self._setup_index()
        
        # Initial synchronization
        self.sync_data_directory()
        
        # Start background watcher
        self._start_watcher()

    def _start_watcher(self):
        """Start a background thread to watch for file changes"""
        def _watch_loop():
            logger.info("Starting background file watcher...")
            while True:
                try:
                    time.sleep(10)  # Check every 10 seconds
                    self.sync_data_directory()
                except Exception as e:
                    logger.error(f"Error in watcher loop: {e}")
        
        self.watcher_thread = threading.Thread(target=_watch_loop, daemon=True)
        self.watcher_thread.start()

    def _setup_llm(self):
        """配置 LLM"""
        try:
            llm = OpenAILike(
                model=settings.llm.MODEL,
                api_key=settings.llm.API_KEY,
                api_base=settings.llm.BASE_URL,
                temperature=settings.llm.TEMPERATURE,
                timeout=settings.llm.TIMEOUT,
                is_chat_model=True
            )
            Settings.llm = llm
            logger.info(f"LLM initialized: {settings.llm.MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _setup_embedding(self):
        """配置 Embedding 模型和切分器"""
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding.MODEL_NAME,
                cache_folder=settings.embedding.CACHE_FOLDER
            )
            Settings.embed_model = embed_model
            logger.info(f"Embedding model initialized: {settings.embedding.MODEL_NAME}")
            
            # 语义切分器
            self.semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=settings.splitter.SEMANTIC_BUFFER_SIZE, 
                breakpoint_percentile_threshold=settings.splitter.SEMANTIC_BREAKPOINT_PERCENTILE, 
                embed_model=embed_model
            )
            # 备用：固定大小切分 + 重叠
            self.sentence_splitter = SentenceSplitter(
                chunk_size=settings.splitter.CHUNK_SIZE, 
                chunk_overlap=settings.splitter.CHUNK_OVERLAP
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Embedding model: {e}")
            raise

    def _setup_index(self):
        """连接 Milvus 并加载/初始化索引"""
        try:
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
            logger.info(f"Connected to Milvus collection: {settings.milvus.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def sync_data_directory(self):
        """同步数据目录中的文件到 Milvus"""
        logger.info(f"Starting data synchronization for {settings.data.DATA_DIR}")
        
        # Ensure data directory exists
        if not os.path.exists(settings.data.DATA_DIR):
            logger.warning(f"Data directory {settings.data.DATA_DIR} does not exist. Creating it.")
            os.makedirs(settings.data.DATA_DIR, exist_ok=True)
            
        added, modified, deleted = self.file_manager.scan_changes(settings.data.DATA_DIR)
        
        if not any([added, modified, deleted]):
            logger.info("No changes detected in data directory.")
            return

        with WriteLockContext(self.lock):
            # Process Deleted Files
            for file_path in deleted:
                try:
                    logger.info(f"Processing deletion for: {file_path}")
                    # Remove from index using file path as ref_doc_id
                    # Normalize path to POSIX style (forward slashes) for Milvus expression compatibility
                    safe_path = file_path.replace("\\", "/")
                    self.index.delete_ref_doc(safe_path, delete_from_docstore=True)
                    logger.info(f"Successfully deleted documents for {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")

            # Process Modified Files (Delete then Insert)
            for file_path in modified:
                try:
                    logger.info(f"Processing modification for: {file_path}")
                    # 1. Delete old (using normalized path)
                    safe_path = file_path.replace("\\", "/")
                    try:
                        self.index.delete_ref_doc(safe_path, delete_from_docstore=True)
                    except:
                        pass # Ignore if not exists
                    
                    # 2. Insert new
                    self._insert_file(file_path)
                    logger.info(f"Successfully updated documents for {file_path}")
                except Exception as e:
                    logger.error(f"Failed to update {file_path}: {e}")

            # Process Added Files
            for file_path in added:
                try:
                    logger.info(f"Processing new file: {file_path}")
                    self._insert_file(file_path)
                    logger.info(f"Successfully inserted documents for {file_path}")
                except Exception as e:
                    logger.error(f"Failed to insert {file_path}: {e}")
            
            # Save state after successful sync
            self.file_manager.save_state()

    def _insert_file(self, file_path: str):
        """Helper to load and insert a single file"""
        # filename_as_id=True ensures doc_id is the absolute file path
        # required for delete_ref_doc to work correctly by path
        print(f"DEBUG: calling SimpleDirectoryReader with {file_path}")
        reader = SimpleDirectoryReader(input_files=[file_path], filename_as_id=True)
        documents = reader.load_data()
        
        # Normalize doc_id to POSIX style for Milvus compatibility
        for doc in documents:
            doc.id_ = doc.id_.replace("\\", "/")

        # Use semantic splitter
        nodes = self.semantic_splitter.get_nodes_from_documents(documents)
        
        if not nodes:
            logger.warning(f"Semantic splitter generated 0 nodes for {file_path}. Falling back to SentenceSplitter.")
            nodes = self.sentence_splitter.get_nodes_from_documents(documents)

        if not nodes:
            logger.warning(f"No nodes generated from file: {file_path}")
            return

        self.index.insert_nodes(nodes)

    def query(self, query_text: str) -> QueryResponse:
        """执行 RAG 查询"""
        with ReadLockContext(self.lock):
            if not self.index:
                logger.warning("Query attempted but index is not initialized.")
                return QueryResponse(answer="Empty Response", sources=[])

            try:
                # Create query engine directly
                query_engine = self.index.as_query_engine(
                    similarity_top_k=settings.rag.SIMILARITY_TOP_K,
                    llm=Settings.llm
                )
                
                logger.info(f"Executing query: '{query_text}'")
                
                # 执行查询
                response = query_engine.query(query_text)
                
                # 调试日志：打印检索到的上下文情况
                if not response.source_nodes:
                    logger.warning("⚠️ No relevant documents found in vector store (source_nodes is empty).")
                else:
                    logger.info(f"✅ Retrieved {len(response.source_nodes)} context nodes.")
                    for i, node in enumerate(response.source_nodes):
                        # 打印前 100 个字符预览
                        content_preview = node.node.get_text()[:100].replace('\n', ' ')
                        logger.info(f"   [Source {i+1} | Score: {node.score:.4f}] {content_preview}...")

                logger.info(f"🤖 LLM Raw Response: {str(response)}")
                
                # 提取来源信息
                sources = []
                if response.source_nodes:
                    for node in response.source_nodes:
                        sources.append(SourceInfo(
                            content=node.node.get_text(),
                            score=node.score
                        ))
                
                return QueryResponse(
                    answer=str(response),
                    sources=sources
                )
                
            except Exception as e:
                logger.error(f"Query failed: {e}")
                raise
