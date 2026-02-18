import sys
import os
from loguru import logger
from typing import Optional

class LoggerManager:
    """
    通用日志管理器 (基于 loguru)
    """
    
    _initialized = False
    
    @classmethod
    def setup(cls, log_dir: str = "logs", level: str = "INFO", rotation: str = "10 MB", retention: str = "30 days"):
        """
        初始化日志系统
        
        Args:
            log_dir: 日志存储目录
            level: 日志级别
            rotation: 日志滚动策略
            retention: 日志保留策略
        """
        if cls._initialized:
            return

        # 1. 移除默认的 handler
        logger.remove()
        
        # 2. 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 3. 控制台输出 (带颜色)
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=level,
            colorize=True
        )
        
        # 4. 业务日志 (包含 INFO 及以上)
        logger.add(
            os.path.join(log_dir, "app.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8",
            enqueue=True  # 异步写入，线程安全
        )
        
        # 5. 错误日志 (仅包含 ERROR 及以上)
        logger.add(
            os.path.join(log_dir, "error.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation=rotation,
            retention="90 days",
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        cls._initialized = True
        logger.info(f"日志系统已启动 | 级别: {level} | 目录: {log_dir}")

# 初始化日志
from graph_rag.core.config import settings
LoggerManager.setup(
    log_dir=settings.logging.LOG_DIR,
    level=settings.logging.LOG_LEVEL,
    rotation=settings.logging.LOG_ROTATION,
    retention=settings.logging.LOG_RETENTION
)

__all__ = ["logger"]
