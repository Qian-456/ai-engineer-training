from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
import os

class BaseLoggingSettings(BaseSettings):
    """通用日志配置基类"""
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_DIR: str = "logs"
    LOG_RETENTION: str = "30 days"
    LOG_ROTATION: str = "10 MB"

class BaseLLMSettings(BaseSettings):
    """通用 LLM 配置基类"""
    API_KEY: Optional[str] = Field(None, description="LLM API Key")
    BASE_URL: Optional[str] = Field(None, description="LLM Base URL")
    MODEL: str = "deepseek-chat"
    TEMPERATURE: float = 0.7
    TIMEOUT: int = 60

    @field_validator("BASE_URL")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v:
            v = v.rstrip("/")
            if not v.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
        return v

class MilvusSettings(BaseSettings):
    """Milvus 配置"""
    URI: str = Field("http://localhost:19530", description="Milvus URI")
    TOKEN: Optional[str] = Field(None, description="Milvus Token")
    # Change collection name to force new creation with correct dimension
    COLLECTION_NAME: str = "faq_collection"
    # BAAI/bge-small-zh-v1.5 has a dimension of 512
    DIMENSION: int = 512 

class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置"""
    MODEL_NAME: str = Field("BAAI/bge-small-zh-v1.5", description="HuggingFace Embedding 模型名称")
    CACHE_FOLDER: Optional[str] = Field(None, description="模型缓存目录")

class SplitterSettings(BaseSettings):
    """文档切分配置"""
    CHUNK_SIZE: int = Field(512, description="切片大小")
    CHUNK_OVERLAP: int = Field(50, description="切片重叠")
    SEMANTIC_BUFFER_SIZE: int = Field(1, description="语义切分缓冲区大小")
    SEMANTIC_BREAKPOINT_PERCENTILE: int = Field(95, description="语义切分断点百分位")

class RAGSettings(BaseSettings):
    """RAG 相关配置"""
    SIMILARITY_TOP_K: int = Field(3, description="检索结果数量")
    SIMILARITY_THRESHOLD: float = Field(0.7, description="相似度阈值")

class DataSettings(BaseSettings):
    """数据管理配置"""
    DATA_DIR: str = Field("milvus_faq/data", description="自动加载的数据目录")
    STATE_FILE: str = Field(".file_state.json", description="文件状态记录文件")

class AppConfig(BaseSettings):
    """
    应用配置
    """
    PROJECT_NAME: str = "Milvus FAQ System"
    
    logging: BaseLoggingSettings = BaseLoggingSettings()
    llm: BaseLLMSettings = BaseLLMSettings()
    milvus: MilvusSettings = MilvusSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    splitter: SplitterSettings = SplitterSettings()
    rag: RAGSettings = RAGSettings()
    data: DataSettings = DataSettings()

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding='utf-8',
        extra="ignore",
        env_nested_delimiter='__',
        _env_file_sentinel=object()
    )

settings = AppConfig()
