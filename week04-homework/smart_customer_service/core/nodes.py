import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from smart_customer_service.core.models import GraphState, OrderInfo
from smart_customer_service.core.config import settings
from smart_customer_service.core.data import MOCK_DB
from smart_customer_service.core.logger import logger

# Initialize LLM
llm = ChatDeepSeek(
    api_key=settings.llm.DEEPSEEK_API_KEY,
    model=settings.llm.MODEL,
    temperature=0.1
)

# --- Helper Functions ---

def extract_last_message(messages: List[Any]) -> str:
    """提取最后一条消息的文本内容"""
    if not messages:
        return ""
    last_msg = messages[-1]
    if isinstance(last_msg, str):
        return last_msg
    if hasattr(last_msg, "content"):
        return last_msg.content
    return str(last_msg)

def parse_json_response(response: Any) -> Dict[str, Any]:
    """解析LLM返回的JSON字符串"""
    try:
        content = response.content.strip()
        # 处理可能的 markdown 代码块
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        result = json.loads(content)
        return result
    except Exception as e:
        logger.error(f"JSON Parse Error: {e}, Content: {response.content}")
        return {"intent": "unknown", "slots": {}, "complete": False, "retry_count": 0}

# --- Nodes ---

def intent_analyze_node(state: GraphState) -> Command[Any]:
    """意图识别 + 槽位填充检查"""
    messages = state['messages']
    last_message = extract_last_message(messages)
    
    slots = state.get("slots", {})
    intent = state.get("intent", "unknown")
    retry_count = state.get("retry_count", 0)
    history = state.get("history", "")
    
    logger.debug(f"[Intent Analyze Node Start] Retry Count: {retry_count}, Intent: {intent}, Slots: {slots}")
    
    prompt = ChatPromptTemplate.from_template("""
    你是智能客服，分析用户输入完成订单服务流程。

    历史对话记录：
    {history}

    支持意图及所需槽位：
    - query_order: 需要 "order_num" (如 ORD-001)
    - request_refund: 需要 "order_num" + "refund_reason" (如 "ORD-001", "因为商品质量问题")
    - unknown: 无槽位需求

    规则：
    1. 识别意图，更新/提取槽位。请注意：
       - 不要覆盖已有的有效槽位，除非用户提供了新的信息。
       - 当前意图为 "{intent}"。如果用户输入仅为对缺失槽位的补充（如只输入了订单号），请保持原意图不变。
       - 只有当用户明确表达了改变意图（如"改查订单"、"我要退款"）时，才修改意图。
    2. 检查意图是否为"unknown"：
       - 是：返回{{"intent": "unknown", "slots": {{}}, "complete": false, "retry_count": 0}}
       - 否：继续下一步
    3. 检查槽位是否完整：
       - 缺少：返回{{"intent": "query_order/request_refund", "slots": {{"order_num": "...", "refund_reason": "..."}}, "complete": false, "retry_count": 0}}
       - 完整：返回{{"intent": "query_order/request_refund", "slots": {{"order_num": "...", "refund_reason": "..."}}, "complete": true, "retry_count": 0}}
    4. 仅JSON输出，无其他内容

    当前意图：{intent}
    当前槽位：{slots}
    最新用户输入：{input}
    
    输出：
    {{
        "intent": "query_order/request_refund/unknown",
        "slots": {{"order_num": "...", "refund_reason": "..."}},
        "complete": true/false
    }}
    示例：
    用户:"查订单123" -> {{"intent":"query_order","slots":{{"order_num":"123"}},"complete":true}}
    用户:"退款" -> {{"intent":"request_refund","slots":{{}}, "complete":false}}
    """)
    
    try:
        # 使用 LCEL 链式调用
        chain = prompt | llm
        response = chain.invoke({
            "input": last_message,
            "slots": json.dumps(slots),
            "intent": intent,
            "history": history
        })
        
        # 记录 LLM 原始响应，方便 Debug
        logger.debug(f"[LLM Raw Response] {response.content}")
        
        result = parse_json_response(response)  
        
        intent = result.get("intent", "unknown")
        new_slots = result.get("slots", {})
        complete = result.get("complete", False)
        
        final_slots = {**slots, **new_slots}
        # 清理空值
        final_slots = {k: v for k, v in final_slots.items() if v}

        # 检查是否真的 complete
        required_slots = []
        if intent == "query_order":
            required_slots = ["order_num"]
        elif intent == "request_refund":
            required_slots = ["order_num", "refund_reason"]
            
        missing_slots = [key for key in required_slots if key not in final_slots]
        
        if missing_slots:
            complete = False
        else:
            if intent != "unknown":
                complete = True

        if intent == "unknown":
            # 如果意图不明确，直接跳转到 fallback 节点
            return Command(goto="fallback", update={"intent": "unknown", "slots": {}, "retry_count": 0})
                
        if not complete:
            # 超过3次重试（即总共尝试4次后），放弃并跳转到fallback
            if retry_count >= 3:
                 return Command(goto="fallback", update={"intent": intent, "slots": final_slots, "retry_count": 3})

            # 跳转到槽位填充节点
            return Command(
                goto="slot_filling_node", 
                update={
                    "intent": intent, 
                    "slots": final_slots
                }
            )
        
        # 成功且完整
        msg = f"好的，正在为您处理{intent}..."
        logger.info(f"Node: intent_analyze_node, Response: {msg}")
        
        # 更新状态中的 order_num 等字段以便后续节点使用
        updates = {
            "intent": intent,
            "slots": final_slots,
            "retry_count": retry_count,
            "messages": [msg]
        }
        if "order_num" in final_slots:
            updates["order_num"] = final_slots["order_num"]
        if "refund_reason" in final_slots:
            updates["refund_reason"] = final_slots["refund_reason"]
            
        # 路由到业务节点
        next_node = "query_order" if intent == "query_order" else "check_refund"
        return Command(goto=next_node, update=updates)

    except Exception as e:
        logger.error(f"Intent Analyze Node Error: {e}")
        # 指数退避重试
        if retry_count >= 3:
            return Command(goto="fallback", update={"retry_count": 3})
            
        wait_time = 2 ** retry_count
        logger.info(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1})")
        time.sleep(wait_time)
        
        return Command(
            goto="intent_analyze_node", 
            update={"retry_count": retry_count + 1}
        )

def slot_filling_node(state: GraphState) -> Command[Any]:
    """槽位填充引导节点：生成引导语并等待用户输入"""
    intent = state.get("intent")
    slots = state.get("slots", {})
    retry_count = state.get("retry_count", 0)
    
    # 重新计算缺失槽位 (逻辑与 intent_analyze_node 类似，但这里只负责生成消息)
    required_slots = []
    if intent == "query_order":
        required_slots = ["order_num"]
    elif intent == "request_refund":
        required_slots = ["order_num", "refund_reason"]
        
    missing_slots = [key for key in required_slots if key not in slots]
    
    # 生成引导语
    if intent == "query_order":
        guide_msg = "请提供您要查询的订单号。"
    elif intent == "request_refund":
        if "order_num" in missing_slots and "refund_reason" in missing_slots:
            guide_msg = "请提供您要申请退款的订单号及退款理由。"
        elif "order_num" in missing_slots:
            guide_msg = "请提供您要申请退款的订单号。"
        else:
            guide_msg = "请提供您的退款理由。"
    else:
        if missing_slots:
            guide_msg = f"缺少信息：{', '.join(missing_slots)}。请补充。"
        else:
            guide_msg = "请提供更多信息以完成操作。"
    
    logger.info(f"Node: slot_filling_node, Response: {guide_msg} (Interrupt)")
    
    # 中断，等待用户输入
    # 这里的 guide_msg 会被 main.py 捕获并显示给用户
    user_input = interrupt(guide_msg)
    
    # 恢复后，将用户输入作为新的 HumanMessage 加入历史
    # 并重新跳转回意图分析节点
    return Command(
        update={
            "messages": [user_input],
            "retry_count": retry_count + 1 # 增加重试计数，防止无限循环
        },
        goto="intent_analyze_node"
    )

def query_order_node(state: GraphState) -> Dict[str, Any]:
    """查询订单详情"""
    order_num = state.get("order_num")
    # 也可以从 slots 取
    if not order_num and state.get("slots"):
        order_num = state["slots"].get("order_num")
        
    info = MOCK_DB.get(order_num)
    
    if info:
        order_info = OrderInfo(**info)
        msg = f"为您查询到订单 {order_num}：{info['product_name']}，金额 {info['total_price']}元，下单日期 {info['order_date']}。"
        logger.info(f"Node: query_order_node, Response: {msg}")
        return {"order_info": order_info, "messages": [msg]}
    else:
        msg = f"抱歉，未找到订单号为 {order_num} 的订单。"
        logger.info(f"Node: query_order_node, Response: {msg}")
        return {"order_info": None, "messages": [msg]}

def check_refund_eligibility_node(state: GraphState) -> Command[Any]:
    """检查退款资格"""
    order_num = state["slots"].get("order_num")

    if order_num in MOCK_DB:
        order_price = MOCK_DB[order_num]["total_price"]
        if order_price <= 500:
            msg = f"订单 {order_num} 符合退款条件。正在为您办理退款..."
            logger.info(f"Node: check_refund_eligibility_node, Response: {msg}")
            # 符合条件，继续到确认退款
            return Command(
                update={"messages": [msg]},
                goto="confirm_refund"
            )
        else:
            # 结束流程
            msg = f"抱歉，订单 {order_num} 金额超过500元，不符合退款条件。"
            logger.info(f"Node: check_refund_eligibility_node, Response: {msg}")
            return Command(
                update={"messages": [msg]},
                goto=END
            )
    else:
        msg = f"抱歉，订单 {order_num} 不存在。"
        logger.info(f"Node: check_refund_eligibility_node, Response: {msg}")
        return Command(
            update={"messages": [msg]},
            goto=END
        )

def confirm_refund_node(state: GraphState) -> Dict[str, Any]:
    """确认退款并执行"""
    order_num = state.get("order_num")
    if not order_num and state.get("slots"):
        order_num = state["slots"].get("order_num")
    msg = f"退款申请已提交，订单 {order_num} 的款项将原路返回。"
    logger.info(f"Node: confirm_refund_node, Response: {msg}")
    return {"messages": [msg]}

def fallback_node(state: GraphState) -> Dict[str, Any]:
    """无法识别意图或多次重试失败时的回复"""
    intent = state.get("intent")
    
    if intent and intent != "unknown":
        intent_name = "查询订单" if intent == "query_order" else "申请退款"
        msg = f"抱歉，虽然我明白您想要{intent_name}，但尝试多次后仍无法获取完整信息。请重新描述您的需求或练习人工客服。"
    else:
        msg = "抱歉，我没听懂您的意思。请问您是要查询订单还是申请退款？"
        
    logger.info(f"Node: fallback_node, Response: {msg}")
    return {"messages": [msg]}

# --- Graph Construction ---

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("intent_analyze_node", intent_analyze_node)
workflow.add_node("slot_filling_node", slot_filling_node)
workflow.add_node("query_order", query_order_node)
workflow.add_node("check_refund", check_refund_eligibility_node)
workflow.add_node("confirm_refund", confirm_refund_node)
workflow.add_node("fallback", fallback_node)

# Set entry point
workflow.set_entry_point("intent_analyze_node")


workflow.add_edge("query_order", END)

workflow.add_edge("confirm_refund", END)
workflow.add_edge("fallback", END)


from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    png_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
