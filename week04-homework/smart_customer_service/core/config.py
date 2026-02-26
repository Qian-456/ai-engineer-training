import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    DEEPSEEK_API_KEY: str
    MODEL: str = "deepseek-chat"
    BASE_URL: str = "https://api.deepseek.com"

class RedisSettings(BaseSettings):
    HOST: str = "localhost"
    PORT: int = 6379
    DB: int = 0
    PASSWORD: str | None = None

class LoggingSettings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "7 days"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Customer Service"
    VERSION: str = "0.1.0"
    
    llm: LLMSettings = LLMSettings()
    redis: RedisSettings = RedisSettings()
    logging: LoggingSettings = LoggingSettings()

    # Priority: current dir .env > parent dir .env
    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="ignore" 
    )

settings = Settings()
