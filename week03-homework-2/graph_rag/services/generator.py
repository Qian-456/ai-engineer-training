from graph_rag.core.interfaces import IGenerator
import asyncio
from typing import Any

class LLMGenerator(IGenerator):
    """
    基于 LLM 的文本生成器
    """
    def __init__(self, client: Any):
        self.client = client

    async def generate(self, context: str, query: str) -> str:
        prompt = (
            f"基于以下参考信息回答问题。如果参考信息不足，请结合自身知识但需注明。\n\n"
            f"参考信息：\n{context}\n\n"
            f"用户问题：{query}\n"
            f"回答："
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # 假设 client.chat 是同步的，使用 to_thread 避免阻塞
        return await asyncio.to_thread(self.client.chat, messages)
