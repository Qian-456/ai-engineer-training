import os
import asyncio
from typing import Optional

from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from graph_rag.core.llm_client import llm_client

# Import Concrete Implementations (Only here!)
from graph_rag.storage.milvus_manager import MilvusManager
from graph_rag.storage.es_manager import ESManager
from graph_rag.storage.neo4j_manager import KGManager

from graph_rag.services.syncer import DocumentSyncer
from graph_rag.services.retriever import HybridRetriever
from graph_rag.services.rebuild_service import RebuildService
from graph_rag.services.generator import LLMGenerator
from graph_rag.services.query_service import QueryService
from graph_rag.services.pipeline import GraphRAGPipeline

class Container:
    """
    依赖注入容器
    负责管理所有服务和组件的生命周期
    """
    _instance = None

    def __init__(self):
        self.milvus_mgr: Optional[MilvusManager] = None
        self.es_mgr: Optional[ESManager] = None
        self.kg_mgr: Optional[KGManager] = None
        
        self.rag_syncer: Optional[DocumentSyncer] = None
        self.es_syncer: Optional[DocumentSyncer] = None
        self.kg_syncer: Optional[DocumentSyncer] = None
        
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[LLMGenerator] = None
        
        self.rebuild_service: Optional[RebuildService] = None
        self.query_service: Optional[QueryService] = None
        self.pipeline: Optional[GraphRAGPipeline] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance

    async def initialize_resources(self):
        """初始化底层资源"""
        logger.info("Initializing resources...")
        
        # 1. Stores
        self.milvus_mgr = MilvusManager()
        self.es_mgr = ESManager()
        
        # Initialize ES async
        await self.es_mgr.initialize()
        
        self.kg_mgr = KGManager(llm_client)
        
        # 2. Syncers
        self.rag_syncer = DocumentSyncer(
            data_dir=settings.data.RAG_DIR,
            state_file=os.path.join(settings.data.DATA_DIR, settings.data.RAG_STATE_FILE),
            processor=self.milvus_mgr
        )
        self.es_syncer = DocumentSyncer(
            data_dir=settings.data.ES_DIR,
            state_file=os.path.join(settings.data.DATA_DIR, settings.data.ES_STATE_FILE),
            processor=self.es_mgr
        )
        self.kg_syncer = DocumentSyncer(
            data_dir=settings.data.KG_DIR,
            state_file=os.path.join(settings.data.DATA_DIR, settings.data.KG_STATE_FILE),
            processor=self.kg_mgr
        )
        
        # 3. Core Services
        self.retriever = HybridRetriever(self.milvus_mgr, self.es_mgr, self.kg_mgr)
        self.generator = LLMGenerator(llm_client)
        
        self.rebuild_service = RebuildService(
            rag_syncer=self.rag_syncer,
            es_syncer=self.es_syncer,
            kg_syncer=self.kg_syncer
        )
        
        self.query_service = QueryService(
            retriever=self.retriever,
            generator=self.generator
        )
        
        # 4. Facade
        # 注意：这里假设 GraphRAGPipeline 的构造函数已经修改为接收这些参数
        self.pipeline = GraphRAGPipeline(
            query_service=self.query_service,
            rebuild_service=self.rebuild_service,
            rag_syncer=self.rag_syncer,
            es_syncer=self.es_syncer,
            kg_syncer=self.kg_syncer
        )
        
        logger.info("Resources initialized.")

    def close_resources(self):
        """释放资源"""
        logger.info("Closing resources...")
        if self.kg_mgr:
            self.kg_mgr.close()
        logger.info("Resources closed.")

    @property
    def graph_rag_pipeline(self) -> GraphRAGPipeline:
        if not self.pipeline:
            raise RuntimeError("Container not initialized. Call initialize_resources() first.")
        return self.pipeline
