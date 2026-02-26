import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY in .env file (or environment variables)")
        pass

    # Initialize LLM
    try:
        llm = ChatDeepSeek(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=0.7
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # Define Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手。当前系统时间是：{current_time}。请根据当前时间回答用户问题，特别是涉及相对时间（如昨天、明天、上周三）的推断。"),
        ("user", "{input}")
    ])

    # Build Chain
    chain = prompt | llm | StrOutputParser()

    print("🤖 时间感知助手已启动 (输入 'quit' 退出)")
    print(f"🕒 当前系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}")
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 再见！")
                break

            # Get current time with weekday
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
            
            print("🤖 助手: ", end="", flush=True)
            
            # Invoke Chain
            try:
                for chunk in chain.stream({"current_time": current_time, "input": user_input}):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n⚠️ 调用 LLM 失败: {e}")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n⚠️ 发生错误: {e}")

if __name__ == "__main__":
    main()
