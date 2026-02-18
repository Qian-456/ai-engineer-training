from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import os
from milvus_faq.models import QueryResponse
from milvus_faq.core import RAGManager
from milvus_faq.logger import logger, LoggerManager
from milvus_faq.config import settings

# Global instance
rag_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    LoggerManager.setup(
        log_dir=settings.logging.LOG_DIR,
        level=settings.logging.LOG_LEVEL,
        rotation=settings.logging.LOG_ROTATION,
        retention=settings.logging.LOG_RETENTION
    )
    
    global rag_manager
    try:
        rag_manager = RAGManager()
        logger.info("RAG Manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Manager: {e}")
        # We might want to exit here or run in degraded mode
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

@app.get("/query", response_model=QueryResponse)
async def query_index(query: str = Query(..., description="用户查询文本")):
    """
    RAG 查询接口
    
    Args:
        query: 用户输入的自然语言问题
    """
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        response = rag_manager.query(query)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
