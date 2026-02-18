from typing import Dict, Any, List, Optional
import os
import asyncio
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from graph_rag.core.interfaces import IQueryService, IRebuildService, IDataSyncer

class GraphRAGPipeline:
    """
    Graph RAG 管道 (Facade)
    协调数据同步、检索和回答生成
    通过依赖注入接收服务
    """
    def __init__(self,
                 query_service: IQueryService,
                 rebuild_service: IRebuildService,
                 rag_syncer: IDataSyncer,
                 es_syncer: IDataSyncer,
                 kg_syncer: IDataSyncer):
        self.query_service = query_service
        self.rebuild_service = rebuild_service
        self.rag_syncer = rag_syncer
        self.es_syncer = es_syncer
        self.kg_syncer = kg_syncer


    async def sync_data(self):
        """全量同步 (异步)"""
        os.makedirs(settings.data.RAG_DIR, exist_ok=True)
        os.makedirs(settings.data.ES_DIR, exist_ok=True)
        os.makedirs(settings.data.KG_DIR, exist_ok=True)
        
        await self.rag_syncer.sync()
        await self.es_syncer.sync()
        await self.kg_syncer.sync()

    async def rebuild_vector(self):
        await self.rebuild_service.rebuild_vector_store()

    async def rebuild_keyword(self):
        await self.rebuild_service.rebuild_keyword_store()

    async def rebuild_kg(self):
        await self.rebuild_service.rebuild_knowledge_graph()
        
    async def rebuild_all(self):
        await self.rebuild_service.rebuild_all()

    async def run(self, user_query: str) -> Dict[str, Any]:
        """执行混合 RAG 查询"""
        return await self.query_service.query(user_query)
