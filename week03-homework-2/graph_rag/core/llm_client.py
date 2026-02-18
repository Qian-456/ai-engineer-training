import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from graph_rag.core.config import settings
from graph_rag.core.logger import logger
from typing import List, Dict, Any, Optional

class LLMClient:
    """
    LLM 客户端，支持重试机制
    """
    def __init__(self):
        api_key = settings.llm.API_KEY or "dummy_key"
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=settings.llm.BASE_URL
        )
        self.model = settings.llm.MODEL
        self.default_temp_text = settings.llm.TEMPERATURE_TEXT
        self.default_temp_json = settings.llm.TEMPERATURE_JSON
        self.max_retries = settings.llm.MAX_RETRIES

    @retry(
        stop=stop_after_attempt(3), # 初始尝试 + 2次重试 = 3次
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError)),
        before_sleep=lambda retry_state: logger.warning(f"LLM 调用失败，正在进行第 {retry_state.attempt_number} 次重试...")
    )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        发送对话请求
        
        Args:
            messages: 对话消息列表
            **kwargs: 其他 OpenAI 参数
            
        Returns:
            LLM 生成的文本内容
        """
        try:
            # 根据 response_format 自动选择默认 temperature
            is_json = kwargs.get("response_format", {}).get("type") == "json_object"
            default_temp = self.default_temp_json if is_json else self.default_temp_text
            
            # 如果 kwargs 中有传入 temperature 则覆盖默认值
            temperature = kwargs.pop("temperature", default_temp)
            
            # Debug log
            logger.debug(f"LLM Call: temperature={temperature}, kwargs={kwargs.keys()}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                timeout=settings.llm.TIMEOUT,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 调用发生错误: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示 (如果需要使用 LLM 的 embedding)
        注：本项目主要使用本地 HuggingFace 模型，此处保留作为备选
        """
        # 这里可以根据需要实现
        pass

llm_client = LLMClient()
