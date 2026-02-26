import operator
from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict, Annotated

class OrderInfo(TypedDict):
    """订单信息定义"""
    order_num: str
    order_date: str
    product_name: str
    quantity: int
    price: float
    total_price: float

class GraphState(TypedDict):
    """图状态定义 """
    messages: Annotated[list[str], operator.add] 
    order_info: Optional[OrderInfo]
    intent: Optional[str]
    retry_count: int
    slots: Dict[str, Any]
    history: Optional[str]
