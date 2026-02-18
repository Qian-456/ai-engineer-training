from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
from graph_rag.services.pipeline import GraphRAGPipeline
from graph_rag.core.container import Container
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
import os
from contextlib import asynccontextmanager

# 全局 Pipeline 实例
pipeline = None
container = Container.get_instance()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理
    """
    global pipeline
    logger.info("正在初始化 Graph RAG Pipeline...")
    
    # 使用容器初始化资源
    await container.initialize_resources()
    pipeline = container.graph_rag_pipeline
    
    # 启动后台同步任务（使用与 FastAPI 相同的事件循环）
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    async def background_sync_worker():
        logger.info("启动后台同步任务，间隔: 10s")
        while not stop_event.is_set():
            try:
                await pipeline.sync_data()
                logger.debug("后台同步完成")
            except Exception as e:
                logger.error(f"后台同步失败: {e}")

            try:
                # 等待 10 秒或直到收到停止信号
                await asyncio.wait_for(stop_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                continue

    sync_task = loop.create_task(background_sync_worker())

    try:
        yield
    finally:
        # 清理资源
        logger.info("正在关闭服务并清理资源...")
        stop_event.set()
        try:
            await sync_task
        except Exception as e:
            logger.error(f"等待后台同步任务结束时出错: {e}")
        container.close_resources()

app = FastAPI(
    title="Graph RAG System API",
    description="提供混合 RAG 查询接口，自动同步本地文档",
    version="1.0.0",
    lifespan=lifespan
)

# --- 请求与响应模型 ---

class QueryRequest(BaseModel):
    text: str = Field(..., description="用户查询文本")

class QueryResponse(BaseModel):
    answer: str
    retrieval_results: List[Dict[str, Any]] = Field(..., description="检索结果")
    reasoning_log: List[str] = Field(default=[], description="推理路径日志")

# --- API 接口 ---

@app.post("/query", response_model=QueryResponse, summary="混合 RAG 查询")
async def query_rag(request: QueryRequest):
    """
    混合 RAG 查询：结合向量检索和知识图谱检索生成答案。
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="系统正在初始化，请稍后再试")
    try:
        # 调用简化后的查询方法
        result = await pipeline.run(request.text)
        
        if not result["answer"]:
            raise HTTPException(status_code=400, detail="查询结果中缺少答案字段")
        if not result["retrieval_results"]:
            logger.warning("查询结果中缺少检索结果字段")
            
        return result
    except Exception as e:
        import traceback
        error_msg = f"API 查询失败: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        with open("error.log", "w", encoding="utf-8") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild/vector", summary="重建向量存储")
async def rebuild_vector(background_tasks: BackgroundTasks):
    """重建向量存储 (Milvus)"""
    if pipeline is None: raise HTTPException(status_code=503, detail="系统初始化中")
    try:
        background_tasks.add_task(pipeline.rebuild_vector)
        return {"message": "向量存储重建任务已启动"}
    except Exception as e:
        logger.error(f"启动向量重建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild/keyword", summary="重建关键词存储")
async def rebuild_keyword(background_tasks: BackgroundTasks):
    """重建关键词存储 (ElasticSearch)"""
    if pipeline is None: raise HTTPException(status_code=503, detail="系统初始化中")
    try:
        background_tasks.add_task(pipeline.rebuild_keyword)
        return {"message": "关键词存储重建任务已启动"}
    except Exception as e:
        logger.error(f"启动关键词重建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild/kg", summary="重建知识图谱")
async def rebuild_kg(background_tasks: BackgroundTasks):
    """重建知识图谱 (Neo4j)"""
    if pipeline is None: raise HTTPException(status_code=503, detail="系统初始化中")
    try:
        background_tasks.add_task(pipeline.rebuild_kg)
        return {"message": "知识图谱重建任务已启动"}
    except Exception as e:
        logger.error(f"启动 KG 重建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild/all", summary="全量重建所有存储")
async def rebuild_all(background_tasks: BackgroundTasks):
    """重建所有存储"""
    if pipeline is None: raise HTTPException(status_code=503, detail="系统初始化中")
    try:
        background_tasks.add_task(pipeline.rebuild_all)
        return {"message": "全量重建任务已启动"}
    except Exception as e:
        logger.error(f"启动全量重建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
