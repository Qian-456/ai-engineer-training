from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Entity:
    """
    实体数据类
    """
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relationship:
    """
    关系数据类
    """
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)

# 定义节点类型
NODE_TYPES = {
    "COMPANY": "公司或企业实体，如'A公司'、'D科技'",
    "GROUP": "集团实体，如'A集团'",
    "PERSON": "个人实体，如'张三'、'李四'",
    "INVESTOR": "投资机构或个人投资者，如'B资本'",
    "FUND": "基金实体，如'C基金'"
}

# 定义关系类型
RELATIONSHIP_TYPES = {
    "INVESTS_IN": "投资持股关系，通常包含 properties: {'share_percentage': '60%'}",
    "CHAIRMAN": "担任董事长关系",
    "FOUNDED_BY": "由某人或某机构创立的关系",
    "MANAGES": "管理关系，如基金管理人",
    "CONTROLS": "控股或母子公司关系"
}

# 定义常用的查询模式
PATTERNS = {
    "FIND_CONTROL_CHAIN": "MATCH (a)-[:INVESTS_IN*]->(b) WHERE a.name = $name RETURN b",
    "FIND_ULTIMATE_BENEFICIARY": "MATCH (a:PERSON)-[:INVESTS_IN*]->(b:COMPANY) WHERE b.name = $name RETURN a"
}

# 提取提示词模板
EXTRACTION_PROMPT = """
你是一个专业的金融数据分析师。请从提供的文本或图像描述中提取企业股权关系信息。
请输出为 JSON 格式，包含 nodes 和 relationships 两个列表。

约束条件：
1. 节点类型 (type) 必须是以下之一：{node_types}
2. 关系类型 (type) 必须是以下之一：{rel_types}
3. 每个节点必须有 'name' 属性。
4. 每个关系必须有 'source', 'target', 'type' 属性，以及可选的 'properties'。

待分析内容：
{content}

JSON 输出：
"""

def get_extraction_prompt(content: str) -> str:
    """生成实体关系提取的提示词"""
    return EXTRACTION_PROMPT.format(
        node_types=list(NODE_TYPES.keys()),
        rel_types=list(RELATIONSHIP_TYPES.keys()),
        content=content
    )

def validate_extraction(data: Dict[str, Any]) -> bool:
    """
    验证提取的数据是否符合 Schema
    
    Args:
        data: 包含 nodes 和 relationships 的字典
        
    Returns:
        bool: 是否有效
    """
    if not isinstance(data, dict):
        return False
        
    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])
    
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        return False
        
    valid_node_types = set(NODE_TYPES.keys())
    valid_rel_types = set(RELATIONSHIP_TYPES.keys())
    
    # 验证节点
    for node in nodes:
        if not isinstance(node, dict):
            return False
        if "name" not in node or not node["name"]:
            return False
        if "type" not in node or node["type"] not in valid_node_types:
            # 允许默认类型或做宽容处理？这里严格校验
            return False
            
    # 验证关系
    for rel in relationships:
        if not isinstance(rel, dict):
            return False
        if "source" not in rel or not rel["source"]:
            return False
        if "target" not in rel or not rel["target"]:
            return False
        if "type" not in rel or rel["type"] not in valid_rel_types:
            return False
            
    return True
