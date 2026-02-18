from pydantic import BaseModel, Field
from typing import List, Optional


class SourceInfo(BaseModel):
    """来源信息模型"""
    content: str = Field(..., description="来源内容片段")
    score: Optional[float] = Field(None, description="相似度得分")

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="生成的回答")
    sources: List[SourceInfo] = Field(default_factory=list, description="参考来源列表")
