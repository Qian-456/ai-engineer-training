import json
import redis.asyncio as redis
from typing import List, Dict, Any, Optional
from smart_customer_service.core.config import settings

from smart_customer_service.core.logger import logger

class RedisMemory:
    """
    基于 Redis 的对话记忆管理 (结构化存储)。
    """
    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis.HOST,
            port=settings.redis.PORT,
            db=settings.redis.DB,
            password=settings.redis.PASSWORD,
            decode_responses=True
        )
        # 保持最新的 5 轮对话
        self.max_turns = 5
        self.store_turns = 2 * self.max_turns
        

    async def get_history_messages(self, session_id: str) -> List[str]:
        """
        获取用于 Context 的历史消息列表 。
        """
        key = f"memory:{session_id}"
        # Redis 返回的是 bytes 列表，但在 __init__ 中 decode_responses=True，所以是 str 列表
        history = await self.client.lrange(key, 0, self.max_turns - 1)
        logger.debug(f"[Memory Read] Session: {session_id}, Retrieved {len(history)} turns.")
        return history

    async def save_history_messages(self, session_id: str, turn_data: Dict[str, Any]):
        """
        添加新的对话轮次 (结构化数据)。
        
        turn_data 示例:
        {
            "initial_input": "查一下 ORD-123",
            "intent": "query_order",
            "order_num": "ORD-123",
            "final_response": "订单状态...",
            "retry_count": 0,
            "retry_inputs": ""
        }
        """
        key = f"memory:{session_id}"
        turn_json = json.dumps(turn_data, ensure_ascii=False)
        await self.client.lpush(key, turn_json)
        
        logger.debug(f"[Memory Save] Session: {session_id}, Turn Saved. Intent: {turn_data.get('intent')}")
        
        # 保持列表长度不超过 store_turns (10)
        # 如果超过，修剪到 max_turns (5)
        length = await self.client.llen(key)
        if length > self.store_turns:
            await self.client.ltrim(key, 0, self.max_turns - 1)
            logger.debug(f"[Memory Trim] Session: {session_id}, Trimmed to {self.max_turns} turns.")
            
    async def clear_history(self, session_id: str):
        """清除历史"""
        key = f"memory:{session_id}"
        await self.client.delete(key)
