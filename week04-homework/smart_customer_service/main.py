import sys
import json
import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import functional_serializers

# Import config and logger from local core modules
from smart_customer_service.core.config import settings
from smart_customer_service.core.logger import LoggerManager, logger
from smart_customer_service.core.nodes import app
from langgraph.types import Command
from smart_customer_service.core.memory import RedisMemory

async def main():
    # Initialize logging system first
    LoggerManager.setup(
        log_dir=settings.logging.LOG_DIR,
        level=settings.logging.LOG_LEVEL,
        rotation=settings.logging.LOG_ROTATION,
        retention=settings.logging.LOG_RETENTION
    )
    
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    
    # Initialize Memory
    try:
        memory = RedisMemory()
        logger.info("Redis memory initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis memory: {e}")
        print("⚠️ 无法连接 Redis，请检查配置。程序将以无记忆模式运行或退出。")
        sys.exit(1)

    print("=" * 50)
    print("🤖 智能客服助手已启动")
    print("📅 支持：查询订单 (如 '查订单 ORD-001')、申请退款")
    print("❌ 退出请输入：quit, q, 退出, 拜拜, exit")
    print("=" * 50)
    
    # Use to_thread for input to avoid blocking, but for simple prompts, blocking is often okay if no background tasks running *before* input.
    # But here we want to potentially run save_history in background while waiting for input.
    session_id = await asyncio.to_thread(input, "\n请输入您的 Session ID (用于记忆): ")
    session_id = session_id.strip()
    if not session_id:
        session_id = "default_user"
    print(f"✅ Session ID: {session_id}")
    
    # Initialize history cache
    history_messages_json = None
    
    # Configuration for LangGraph checkpointer
    config = {"configurable": {"thread_id": session_id}}

    while True:
        try:
            # 1. 从 Redis 获取历史 Context (放在循环开头以便在任何时刻使用)
            history = ""
            if history_messages_json is None:
                history_messages_json = await memory.get_history_messages(session_id)
            
            for i in range(len(history_messages_json)):
                history += f'\n----{i+1}/{len(history_messages_json)}轮记忆----\n'  + history_messages_json[-i-1]

            # 检查是否有待处理的中断
            current_state = await app.aget_state(config)
            is_interrupted = False
            interrupt_value = None
            
            if current_state.tasks:
                # 检查是否有中断
                if hasattr(current_state, "tasks") and current_state.tasks:
                     # This is tricky in async. Let's rely on tasks being present and next being empty?
                     # actually, tasks[0].interrupts is the way
                     task = current_state.tasks[0]
                     if task.interrupts:
                         is_interrupted = True
                         interrupt_value = task.interrupts[0].value
            
            if is_interrupted:
                # 打印引导语 (interrupt_value)
                print(f"🤖 助手: {interrupt_value}")
                
                # 获取用户补充输入
                human_input = await asyncio.to_thread(input, "\n👤 您 (补充信息): ")
                human_input = human_input.strip()
                
                if human_input.lower() in ["quit", "q", "退出", "拜拜", "exit"]:
                    print("\n👋 谢谢使用，再见！")
                    break

                if human_input == "/history":
                    print(history)
                    continue
                
                if human_input == "/clear":
                    await memory.clear_history(session_id)
                    history_messages_json = []
                    print("✅ 历史记忆已清除")
                    continue
                
                # 恢复执行
                print("🤖 助手: 正在处理...", end="", flush=True)
                result = await app.ainvoke(Command(resume=human_input), config=config)
                # 移除 "正在处理..."
                print("\r" + " " * 20 + "\r", end="", flush=True)
                
            else:
                # 正常新一轮对话
                human_input = await asyncio.to_thread(input, "\n👤 您: ")
                human_input = human_input.strip()
                
                exit_commands = ["quit", "q", "退出", "拜拜", "exit"]
                if human_input.lower() in exit_commands:
                    logger.info("User requested exit")
                    print("\n👋 谢谢使用，再见！")
                    break
                
                if not human_input:
                    continue

                if human_input == "/history":
                    print(history)
                    continue
                
                if human_input == "/clear":
                    await memory.clear_history(session_id)
                    history_messages_json = []
                    print("✅ 历史记忆已清除")
                    continue

                current_messages = []
                
                # 添加当前用户输入
                current_messages.append(human_input)
                
                initial_state = {
                    "messages": current_messages,
                    "retry_count": 0, 
                    "order_num": None,
                    "intent": None,
                    "slots": {},
                    "history": history
                }
                
                print("🤖 助手: 正在处理...", end="", flush=True)
                
                # 2. 运行图
                result = await app.ainvoke(initial_state, config=config)
                # 移除 "正在处理..."
                print("\r" + " " * 20 + "\r", end="", flush=True)
            

            current_snapshot = await app.aget_state(config)
            if current_snapshot.tasks and current_snapshot.tasks[0].interrupts:
               
                continue

            if not result.get("messages", []):
                continue
            
            # 打印最终回复
            current_messages = result.get("messages", [])
            display_messages = []
            
            # 这里的 messages 是累积的，我们需要提取本轮生成的 AI 消息
            # 简单起见，我们倒序查找，直到遇到 HumanMessage 或开头
            # 但实际上 app.ainvoke 返回的 state 包含所有历史消息
            # 我们只需要显示本次交互产生的 AI 回复
            
            # 找到最后一条 HumanMessage 的索引
            last_human_idx = -1
            for i in range(len(current_messages) - 1, -1, -1):
                msg = current_messages[i]
                if isinstance(msg, HumanMessage) or (isinstance(msg, str) and not msg.startswith("正在") and not msg.startswith("退款") and not msg.startswith("抱歉") and not msg.startswith("订单")):
                     # 这里的判断有点脆弱，最好是依靠类型
                     # 由于我们在 nodes.py 中返回的是 {"messages": [msg]}，LangGraph 会将其追加
                     # 我们可以假设最后几条是 AI 的回复
                     pass
            
            # 更稳健的方法：我们只关心最后生成的几条
            # 如果是拒绝退款，通常只有一条 "抱歉..."
            # 如果是成功，可能有 "正在办理..." 和 "退款申请已提交..."
            
            # 取最后 5 条消息进行分析
            recent_msgs = current_messages[-5:]
            ai_responses = []
            
            for msg in recent_msgs:
                content = msg.content if hasattr(msg, "content") else str(msg)
                # 过滤掉用户输入（假设用户输入不包含特定关键词，或者通过上下文判断）
                # 在我们的流程中，HumanMessage 是用户输入，AIMessage 是助手回复
                # 但 nodes.py 中返回的是字符串列表，LangGraph 可能会将其转换为 AIMessage 或保持字符串
                # 让我们假设它是字符串或 AIMessage
                
                # 简单过滤：只显示包含特定关键词的 AI 回复
                if any(k in content for k in ["抱歉", "退款申请已提交", "正在为您办理", "为您查询到"]):
                    ai_responses.append(content)
            
            # 如果有拒绝消息，只显示拒绝
            rejection_msg = next((m for m in ai_responses if "不符合退款条件" in m), None)
            if rejection_msg:
                print(f"🤖 助手: {rejection_msg}")
            else:
                # 否则显示最后一条
                if ai_responses:
                    print(f"🤖 助手: {ai_responses[-1]}")

            

            # 3. 保存结构化记忆 (仅在完整对话结束时保存)
            # 构造 Turn Data
            # 注意：如果是 resume 后的结果，initial_input 可能不是最新的 human_input
            # 这里简化处理，记录最后一次意图和结果
            
            # 如果是 resume 模式，human_input 可能是补充信息
            # 我们可能需要记录完整的交互过程，或者只记录最终结果
            final_response_text = result.get("messages", [])[-1]
            turn_data = {
                "initial_input": human_input, # 这里的 input 可能是补充信息，作为一轮记录可能不完美，但在简单场景下可接受
                "intent": result.get("intent"),
                "slots": result.get("slots", {}),
                "final_response": final_response_text,
                "retry_count": result.get("retry_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            latest_history = json.dumps(turn_data, ensure_ascii=False)
            
            # Use create_task for non-blocking save
            asyncio.create_task(memory.save_history_messages(session_id, turn_data))
            
            # Update local cache for next iteration: Add newest to front, keep 5
            if history_messages_json is None:
                history_messages_json = []
            history_messages_json.insert(0, latest_history)
            if len(history_messages_json) > 5:
                history_messages_json = history_messages_json[:5]
            
            logger.debug(f"Turn complete. Intent: {turn_data['intent']}, Slots: {turn_data['slots']}")
            
        except KeyboardInterrupt:
            logger.warning("Program interrupted by user")
            print("\n\n⚠️ 程序被用户中断，谢谢使用！")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"程序发生错误：{e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch interrupt during asyncio.run
        print("\n\n⚠️ 程序被用户中断，谢谢使用！")
        sys.exit(0)
