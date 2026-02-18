import uvicorn
from milvus_faq.config import settings
from milvus_faq.logger import logger

def main():
    """
    启动 FastAPI 应用
    """
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    uvicorn.run("milvus_faq.api:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
