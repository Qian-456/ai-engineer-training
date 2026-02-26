from typing import Dict, Any

# 模拟数据库
MOCK_DB: Dict[str, Any] = {
    "ORD-001": {
        "order_num": "ORD-001", 
        "order_date": "2023-09-20", 
        "product_name": "电脑", 
        "quantity": 1, 
        "price": 8000.0, 
        "total_price": 8000.0
    },
    "ORD-002": {
        "order_num": "ORD-002", 
        "order_date": "2026-02-01", 
        "product_name": "耳机", 
        "quantity": 1, 
        "price": 300.0, 
        "total_price": 300.0
    }
}
