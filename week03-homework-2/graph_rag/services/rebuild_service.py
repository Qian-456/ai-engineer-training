from typing import List, Dict, Optional
from graph_rag.core.interfaces import IRebuildService, IDataSyncer
from graph_rag.core.logger import logger
from graph_rag.core.config import settings
import asyncio
import os

class RebuildService(IRebuildService):
    """
    重建服务 (RebuildService)
    遵循 SRP: 仅负责协调各个组件的重建流程
    遵循 DIP: 依赖于 IDataSyncer 接口
    """
    def __init__(self, 
                 rag_syncer: IDataSyncer,
                 es_syncer: IDataSyncer,
                 kg_syncer: IDataSyncer):
        self.rag_syncer = rag_syncer
        self.es_syncer = es_syncer
        self.kg_syncer = kg_syncer

    async def rebuild_vector_store(self):
        """重建向量存储 (通过 Syncer)"""
        logger.info("开始重建向量存储...")
        await self.rag_syncer.rebuild()
        logger.info("向量存储重建完成")

    async def rebuild_keyword_store(self):
        """重建关键词存储 (通过 Syncer)"""
        logger.info("开始重建关键词存储...")
        await self.es_syncer.rebuild()
        logger.info("关键词存储重建完成")

    async def rebuild_knowledge_graph(self):
        """重建知识图谱 (通过 Syncer)"""
        logger.info("开始重建知识图谱...")
        await self.kg_syncer.rebuild()
        logger.info("知识图谱重建完成")

    async def rebuild_all(self):
        """重建所有"""
        await asyncio.gather(
            self.rebuild_vector_store(),
            self.rebuild_keyword_store(),
            self.rebuild_knowledge_graph()
        )
