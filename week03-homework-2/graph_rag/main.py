import argparse
import uvicorn
import sys
import os

# 将项目根目录添加到 pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag.core.logger import logger
from graph_rag.api.api_server import app

def main():
    """
    主入口：启动 FastAPI 服务器
    """
    parser = argparse.ArgumentParser(description="Graph RAG 系统服务入口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    
    args = parser.parse_args()

    logger.info(f"正在启动 Graph RAG API 服务 @ {args.host}:{args.port}")
    # 使用导入字符串以便支持 reload
    uvicorn.run("graph_rag.api.api_server:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
