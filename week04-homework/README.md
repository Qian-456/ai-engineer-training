# 第四周作业

## 任务
构建一个小型多轮对话智能客服，支持工具调用以及模型与插件的热更新。

## 作业思路指导
### 阶段一：基础对话系统搭建
使用 LangChain 构建基础 Chain：Prompt → LLM → OutputParser
用户说“我昨天下的单”，系统能结合当前时间推断“昨天”的具体日期

### 阶段二：多轮对话与工具调用
实现“订单查询”“退款申请”等多轮交互流程，支持工具自动调用。
使用 LangGraph 构建以下流程：
- 用户说“查订单” → 追问“请提供订单号”
- 收到订单号后 → 调用 query_order(order_id) 工具
- 返回订单状态与物流信息

### 阶段三：热更新与生产部署
实现模型与插件的热更新，完成系统部署与监控。
1. 模型热更新
2. 插件热重载
3. 暴露健康检查接口 /health
4. 编写自动化测试脚本
- 测试“发票开具”插件的功能正确性
- 验证热更新后旧会话不受影响

## 如何提交作业
请fork本仓库，然后在以下目录分别完成编码作业：
- [week04-homework/smart_customer_service](./smart_customer_service)

其中:
- main.py是作业的入口


完成作业后，请在【极客时间】上提交你的fork仓库链接，精确到本周的目录，例如：
```
https://github.com/your-username/ai-engineer-training/tree/main/week04-homework
```
