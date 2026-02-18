from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class IVectorStore(ABC):
    """向量存储接口"""
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """语义检索"""
        pass
    
    @abstractmethod
    async def clear(self):
        """清空数据"""
        pass

class IKeywordStore(ABC):
    """关键词存储接口"""
    @abstractmethod
    async def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """关键词检索"""
        pass

    @abstractmethod
    async def search_by_entity(self, entity_name: str) -> List[str]:
        """实体反查"""
        pass

    @abstractmethod
    async def clear(self):
        """清空索引"""
        pass

class IGraphStore(ABC):
    """知识图谱存储接口"""
    @abstractmethod
    def graph_reasoning(self, entities: List[str]) -> Dict[str, Any]:
        """图谱推理"""
        pass

    @abstractmethod
    async def clear(self):
        """清空数据库"""
        pass

class IRanker(ABC):
    """排序/评分接口"""
    @abstractmethod
    def rank(self, 
             vec_res: List[Dict[str, Any]], 
             kw_res: List[Dict[str, Any]], 
             kg_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        对多路召回结果进行评分、融合与过滤
        Returns:
            Dict: {
                "joint_results": List[Dict],
                "debug_info": Dict
            }
        """
        pass

class IDataSyncer(ABC):
    """数据同步器接口 (用于 Rebuild)"""
    @abstractmethod
    async def sync(self):
        """执行同步"""
        pass

    @abstractmethod
    async def rebuild(self):
        """执行重建"""
        pass

class IGenerator(ABC):
    """文本生成接口"""
    @abstractmethod
    async def generate(self, context: str, query: str) -> str:
        """根据上下文和问题生成回答"""
        pass

class IQueryService(ABC):
    """查询服务接口"""
    @abstractmethod
    async def query(self, question: str) -> Dict[str, Any]:
        """执行问答流程"""
        pass

class IRebuildService(ABC):
    """重建服务接口"""
    @abstractmethod
    async def rebuild_all(self):
        """重建所有数据"""
        pass
    
    @abstractmethod
    async def rebuild_vector_store(self):
        """重建向量库"""
        pass
        
    @abstractmethod
    async def rebuild_keyword_store(self):
        """重建关键词库"""
        pass
        
    @abstractmethod
    async def rebuild_knowledge_graph(self):
        """重建知识图谱"""
        pass
