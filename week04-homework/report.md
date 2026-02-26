# 第四周作业：智能客服系统

本项目包含两个阶段的作业实现：基础对话系统与多轮对话智能客服。

## 项目结构

- `time_aware_bot/`: 阶段一 - 基础对话系统 (Prompt → LLM → OutputParser)
- `smart_customer_service/`: 阶段二 - 多轮对话与工具调用 (LangGraph)
- `tests/`: 测试用例

## 环境准备

1.  **进入项目目录**
    ```bash
    cd week04-homework
    ```

2.  **创建虚拟环境 (推荐使用 Conda)**
    ```bash
    conda create -n week04 python=3.12
    conda activate week04
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量**
    复制 `.env.example` 为 `.env` 并填入你的 API Key：
    ```bash
    cp .env.example .env
    ```
    在 `.env` 中配置：
    ```ini
    DEEPSEEK_API_KEY=your_api_key_here
    # Redis 配置 (阶段二需要)
    REDIS_HOST=localhost
    REDIS_PORT=6379
    ```

## 作业思路与实现

### 阶段一：基础对话系统搭建 (Time Aware Bot)
该阶段目标是构建一个能够感知当前时间的对话机器人。

- **核心实现**: 使用 `LangChain` 的 `Chain` 结构：`PromptTemplate | ChatModel | OutputParser`。
- **Prompt 设计**: 在 System Prompt 中动态注入当前的系统时间（精确到星期几）。
- **代码位置**: `time_aware_bot/main.py`
- **关键逻辑**:
  ```python
  prompt = ChatPromptTemplate.from_messages([
      ("system", "你是一个智能助手。当前系统时间是：{current_time}。请根据当前时间回答用户问题..."),
      ("user", "{input}")
  ])
  ```

### 阶段二：多轮对话与工具调用 (Smart Customer Service)
该阶段构建了一个基于图（Graph）的智能客服系统，支持状态管理和中断恢复。

- **架构**: 使用 `LangGraph` 构建状态图。
- **核心节点**:
    - `intent_analyze_node`: 识别用户意图（查订单/退款）并提取槽位。如果信息缺失，会主动跳转到 `slot_filling_node`。
    - `slot_filling_node`: 生成追问话术并触发 `interrupt` 中断，等待用户补充信息。
    - `query_order_node` / `confirm_refund_node`: 具体的业务逻辑节点。
- **状态管理**: 使用 `Redis` 持久化会话历史，支持跨会话记忆。
- **中断恢复**: 利用 `LangGraph` 的 `checkpointer` 机制，在用户补充信息后恢复执行流程。
- **代码位置**: `smart_customer_service/core/nodes.py` (图定义), `smart_customer_service/main.py` (运行入口)

---

## 运行指南

### 前置准备

1.  **进入项目目录**
    ```bash
    cd week04-homework
    ```

2.  **创建虚拟环境 (推荐使用 Conda)**
    ```bash
    conda create -n week04 python=3.12
    conda activate week04
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量**
    本项目支持在根目录或各子目录中配置 `.env`。建议在根目录配置一份即可。
    ```bash
    cp .env.example .env
    ```
    在 `.env` 中填入：
    ```ini
    DEEPSEEK_API_KEY=your_api_key_here
    REDIS_HOST=localhost
    REDIS_PORT=6379
    ```

### 运行阶段一：基础对话系统

```bash
# 运行方式 
python -m time_aware_bot.main
```

### 运行阶段二：智能客服系统

**注意**: 必须先启动 Redis 服务。

```bash
# 确保在根目录
cd week04-homework (如果不在的话)
# 运行方式 
python -m smart_customer_service.main
```

**测试演示**：
```
✅ Session ID: Jason
 
 👤 您: 我要查订单ORD-002 
 🤖 助手: 为您查询到订单 ORD-002：耳机，金额 300.0元，下单日期 2026-02-01。 
 
 👤 您: 我要查订单 
 🤖 助手: 请提供您要查询的订单号。 
 
 👤 您 (补充信息): ORD-001 
 🤖 助手: 为您查询到订单 ORD-001：电脑，金额 8000.0元，下单日期 2023-09-20。 
 
 👤 您: 我要退款     
 🤖 助手: 请提供您要申请退款的订单号及退款理由。 
 
 👤 您 (补充信息): ORD-002太贵了 
 🤖 助手: 退款申请已提交，订单 ORD-002 的款项将原路返回。 
 
 👤 您: 你好         
 🤖 助手: 抱歉，我没听懂您的意思。请问您是要查询订单还是申请退款？ 
 
 👤 您: 我要退款     
 🤖 助手: 请提供您要申请退款的订单号及退款理由。 
 
 👤 您 (补充信息): 我要退款 
 🤖 助手: 请提供您要申请退款的订单号及退款理由。 
 
 👤 您 (补充信息): 我要退款 
 🤖 助手: 请提供您要申请退款的订单号及退款理由。 
 
 👤 您 (补充信息): 我要退款 
 🤖 助手: 抱歉，虽然我明白您想要申请退款，但尝试多次后仍无法获取完整信息。请重新描述您的需求或练习人工客服。 
```

