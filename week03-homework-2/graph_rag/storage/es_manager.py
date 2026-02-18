from elasticsearch import AsyncElasticsearch, NotFoundError
from typing import List, Dict, Any, Optional
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from graph_rag.services.syncer import BaseSyncProcessor
from graph_rag.core.utils import split_document
from graph_rag.core.interfaces import IKeywordStore
import os

class ESManager(BaseSyncProcessor, IKeywordStore):
    """
    ElasticSearch 管理器
    负责关键词索引的增删改查
    实现 IKeywordStore 接口
    """
    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None
        self.is_available = False
        self.index_name = settings.es.INDEX_NAME

    async def initialize(self):
        """初始化 ES 连接并检查索引"""
        try:
            basic_auth = None
            if settings.es.USERNAME and settings.es.PASSWORD:
                basic_auth = (settings.es.USERNAME, settings.es.PASSWORD)

            self.client = AsyncElasticsearch(
                hosts=[settings.es.URL],
                basic_auth=basic_auth,
                verify_certs=settings.es.VERIFY_CERTS,
                request_timeout=settings.es.TIMEOUT,  # 使用配置的超时时间
                max_retries=settings.es.MAX_RETRIES,  # 使用配置的重试次数
                retry_on_timeout=True
            )
            
            # 检查连接
            if await self.client.ping():
                self.is_available = True
                logger.info(f"成功连接到 ElasticSearch: {settings.es.URL}")
                await self._create_index_if_not_exists()
                
                # 尝试强制更新 replica 设置，以修复单节点 503 问题
                try:
                    await self.client.indices.put_settings(
                        index=self.index_name, 
                        body={"index": {"number_of_replicas": 0}}
                    )
                except Exception as e:
                    logger.debug(f"更新 ES replica 设置失败 (可忽略): {e}")

            else:
                logger.warning(f"无法连接到 ElasticSearch: {settings.es.URL}，关键词搜索功能将不可用。")
                self.is_available = False
                
        except Exception as e:
            logger.warning(f"ElasticSearch 初始化失败: {e}，系统将降级运行。")
            self.is_available = False

    async def _create_index_if_not_exists(self):
        """创建索引配置（支持中文分词）"""
        if not self.is_available: return

        try:
            exists = await self.client.indices.exists(index=self.index_name)
            if not exists:
                # 定义索引映射，使用 ik_max_word 中文分词
                mapping = {
                    "settings": {
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "default": {
                                    "type": "ik_max_word" 
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                            "file_path": {"type": "keyword"},
                            "chunk_id": {"type": "keyword"}
                        }
                    }
                }
                # 如果没有安装 IK 分词器，回退到 standard
                try:
                    await self.client.indices.create(index=self.index_name, body=mapping)
                except Exception as e:
                    logger.warning(f"创建带 IK 分词的索引失败: {e}，尝试使用默认配置")
                    try:
                        # 回退到标准分词器，避免 'WordListCorpusReader' 相关错误
                        # 该错误通常是因为 IK 分词器未正确安装或版本不兼容，导致 ES 内部报错
                        simple_mapping = {
                            "settings": {"number_of_replicas": 0},
                            "mappings": {
                                "properties": {
                                    "content": {"type": "text", "analyzer": "standard"},
                                    "file_path": {"type": "keyword"},
                                    "chunk_id": {"type": "keyword"}
                                }
                            }
                        }
                        await self.client.indices.create(index=self.index_name, body=simple_mapping)
                    except Exception as e2:
                        logger.error(f"创建默认索引也失败: {e2}")
                    
                logger.info(f"ES 索引 {self.index_name} 创建成功")
        except Exception as e:
            logger.error(f"创建 ES 索引失败: {e}")

    async def close(self):
        """关闭连接"""
        if self.client:
            await self.client.close()

    async def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """索引单个文档"""
        if not self.is_available: return

        try:
            body = {
                "content": content,
                **(metadata or {})
            }
            await self.client.index(index=self.index_name, id=doc_id, document=body)
        except Exception as e:
            logger.error(f"ES 索引文档失败 {doc_id}: {e}")

    async def delete_document(self, doc_id: str):
        """删除单个文档"""
        if not self.is_available: return
        try:
            await self.client.delete(index=self.index_name, id=doc_id)
        except NotFoundError:
            pass
        except Exception as e:
            logger.error(f"ES 删除文档失败 {doc_id}: {e}")

    async def delete_by_file(self, file_path: str):
        """根据文件路径删除相关文档"""
        if not self.is_available: return
        try:
            query = {
                "query": {
                    "term": {
                        "file_path": file_path
                    }
                }
            }

            await self.client.delete_by_query(index=self.index_name, body=query, conflicts="proceed")
            logger.info(f"ES: 已删除文件相关索引 {file_path}")
        except Exception as e:
            logger.error(f"ES 按文件删除失败 {file_path}: {e}")

    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        关键词搜索 (BM25)
        """
        if not self.is_available: return []

        try:
            body = {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": top_k
            }
            logger.debug(f"ES Search Query Body: {body}")
            resp = await self.client.search(index=self.index_name, body=body)
            
            results = []
            for hit in resp["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "content": hit["_source"].get("content", ""),
                    "metadata": {k:v for k,v in hit["_source"].items() if k != "content"}
                })
            return results
        except Exception as e:
            logger.error(f"ES 搜索失败: {e}", exc_info=True)
            return []

    async def search_by_entity(self, entity_name: str) -> List[str]:
        """
        查找包含特定实体的 Chunk ID (用于 Graph Boost)
        使用短语匹配以提高准确率
        """
        if not self.is_available: return []

        try:
            body = {
                "query": {
                    "match_phrase": {
                        "content": entity_name
                    }
                },
                "_source": False, # 不需要内容，只需要 ID
                "size": 100 # 限制数量防止爆炸
            }
            logger.debug(f"ES Search By Entity Query Body: {body}")
            resp = await self.client.search(index=self.index_name, body=body)
            return [hit["_id"] for hit in resp["hits"]["hits"]]
        except Exception as e:
            logger.error(f"ES 实体反查失败: {e}")
            return []

    async def clear(self):
        """清空索引"""
        if not self.is_available: return
        try:
            # conflicts="proceed" 忽略版本冲突 (例如文档被并发删除)
            await self.client.delete_by_query(
                index=self.index_name, 
                body={"query": {"match_all": {}}}, 
                conflicts="proceed"
            )
            logger.info("ES: 索引已清空")
        except Exception as e:
            logger.error(f"ES 清空索引失败: {e}")

    # --- BaseSyncProcessor 实现 ---
    
    async def process_added(self, file_paths: List[str]):
        """处理新增文件"""
        if not self.is_available: return
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用统一的切分逻辑
                nodes = split_document(file_path, content)
                
                for node in nodes:
                    # 使用确定性的 ID
                    await self.index_document(
                        doc_id=node.node_id,
                        content=node.text,
                        metadata={
                            **node.metadata,
                            "file_path": file_path,
                            "chunk_id": node.node_id
                        }
                    )
                logger.info(f"ES: 已索引文件 {file_path} ({len(nodes)} chunks)")
            except Exception as e:
                logger.error(f"ES 处理新增文件失败 {file_path}: {e}")

    async def process_modified(self, file_paths: List[str]):
        """处理修改文件"""
        if not self.is_available: return
        
        for file_path in file_paths:
            # 先删
            await self.delete_by_file(file_path)
        
        await self.process_added(file_paths)

    async def process_deleted(self, file_paths: List[str]):
        """处理删除文件"""
        if not self.is_available: return
        
        for file_path in file_paths:
            await self.delete_by_file(file_path)
