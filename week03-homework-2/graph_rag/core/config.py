from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
import os

class LoggingSettings(BaseSettings):
    """
    日志配置类
    """
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_DIR: str = "logs"
    LOG_RETENTION: str = "30 days"
    LOG_ROTATION: str = "10 MB"

class LLMSettings(BaseSettings):
    """
    LLM 配置类
    """
    API_KEY: str = Field(..., description="LLM API Key")
    BASE_URL: str = Field(..., description="LLM Base URL")
    MODEL: str = Field("deepseek-chat", description="LLM 模型名称")
    TEMPERATURE: float = 0.1
    TEMPERATURE_TEXT: float = Field(1.0, description="普通文本生成时的默认温度")
    TEMPERATURE_JSON: float = Field(0.0, description="JSON 结构化提取时的默认温度")
    TIMEOUT: int = 60
    MAX_RETRIES: int = 2

    @field_validator("BASE_URL")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v:
            v = v.rstrip("/")
            if not v.startswith(("http://", "https://")):
                raise ValueError("URL 必须以 http:// 或 https:// 开头")
        return v

class MilvusSettings(BaseSettings):
    """
    Milvus 配置类
    """
    URI: str = Field("http://localhost:19530", description="Milvus URI")
    TOKEN: Optional[str] = Field(None, description="Milvus Token")
    COLLECTION_NAME: str = "graph_rag_collection"
    DIMENSION: int = 512
    INDEX_TYPE: str = "IVF_FLAT"
    METRIC_TYPE: str = "L2"
    PARAMS: dict = {"nlist": 128}
    EMBED_CACHE_DIR: str = "./graph_rag/embed_cache"

class ElasticsearchSettings(BaseSettings):
    """
    ElasticSearch 配置类
    """
    URL: str = Field("http://localhost:9200", description="ES URL")
    USERNAME: Optional[str] = Field(None, description="ES 用户名")
    PASSWORD: Optional[str] = Field(None, description="ES 密码")
    INDEX_NAME: str = "graph_rag_index"
    VERIFY_CERTS: bool = False
    TIMEOUT: int = Field(30, description="ES 连接/请求超时时间(秒)")
    MAX_RETRIES: int = Field(3, description="ES 重试次数")

class Neo4jSettings(BaseSettings):
    """
    Neo4j 配置类
    """
    URI: str = Field(..., description="Neo4j URI")
    USERNAME: str = Field(..., description="Neo4j 用户名")
    PASSWORD: str = Field(..., description="Neo4j 密码")

class RAGSettings(BaseSettings):
    """
    RAG 混合检索配置类
    """
    VECTOR_WEIGHT: float = 0.5
    KEYWORD_WEIGHT: float = 0.3
    GRAPH_WEIGHT: float = 0.2
    MIN_SCORE_THRESHOLD: float = 0.2
    TOP_K: int = 10

class DataSettings(BaseSettings):
    """
    数据管理配置类
    """
    DATA_DIR: str = Field("data", description="根数据目录")
    RAG_DIR: str = Field("data/rag_documents", description="RAG 原始文档目录")
    ES_DIR: str = Field("data/es_documents", description="ES 原始文档目录")
    KG_DIR: str = Field("data/kg_documents", description="知识图谱原始文档目录")
    RAG_STATE_FILE: str = Field(".rag_state.json", description="RAG 状态文件")
    ES_STATE_FILE: str = Field(".es_state.json", description="ES 状态文件")
    KG_STATE_FILE: str = Field(".kg_state.json", description="KG 状态文件")

class AppConfig(BaseSettings):
    """
    总应用配置类
    """
    PROJECT_NAME: str = "Graph RAG System"
    
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    llm: LLMSettings
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    es: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    neo4j: Neo4jSettings
    rag: RAGSettings = Field(default_factory=RAGSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    model_config = SettingsConfigDict(
        env_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env")),
        env_file_encoding='utf-8',
        extra="ignore",
        env_nested_delimiter='__',
    )

settings = AppConfig()
