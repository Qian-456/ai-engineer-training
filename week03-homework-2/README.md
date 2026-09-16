# 第三周作业 Part 2

## 作业一：构建一个基于 Milvus 的 FAQ 检索系统

### 输入输出定义
- **输入：** 用户自然语言问题（如“如何退货？”）
- **输出：** 最相关的 FAQ 条目及其答案

### 扩展项
- 支持热更新知识库（ 自动 re-index）
- 提供 RESTful API 接口（FastAPI 封装）

### 工程化要求
- 使用 LlamaIndex 构建索引
- 部署 Milvus 作为向量库
- 实现文档切片优化（语义切分 + 重叠）

## 作业二：构建一个融合文档检索、图谱推理的多跳问答系统

### 场景设定
- **用户问：** “A 公司的最大股东是谁？”

### 系统流程
1. 检索 A 公司相关信息（RAG）
2. 图谱中查找控股关系（KG）
3. 生成最终回答（LLM）

### 技术难点
- 如何将 RAG 与图谱推理融合？
- 如何设计联合评分机制？
- 如何防止错误传播？（如图谱中错误关系导致错误回答）

### 工程化要求
- 使用 Neo4j 构建企业股权图谱
- 使用 LlamaIndex 实现文档检索
- 实现多跳查询逻辑（Cypher + LLM 协同）
- 构建可解释性输出（展示推理路径）

## 如何提交作业
请fork本仓库，然后在以下目录分别完成编码作业：
- [week03-homework-2/milvus_faq](milvus_faq)
- [week03-homework-2/graph_rag](graph_rag)

其中:
- main.py是作业的入口
- report.md是作业的报告


完成作业后，请在【极客时间】上提交你的fork仓库链接，精确到本周的目录，例如：
```
https://github.com/your-username/ai-engineer-training/tree/main/week03-homework-2
```

## 公开数据检索评测

见 [GraphRAG 评测与人工验收](graph_rag/evaluation/README.md)：固定 HotpotQA 数据、调参/测试划分、共享融合评分、图消融及逐题结果。
