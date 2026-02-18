# Graph RAG 系统实验报告

## 1. 作业概述：构建一个融合文档检索、图谱推理的多跳问答系统

本实验旨在构建一个能够处理复杂多跳查询的智能问答系统。通过融合 RAG（检索增强生成）与 KG（知识图谱）推理，解决传统 RAG 在处理结构化关系时的不足。

### 1.1 场景设定
- **用户提问**：例如“A 集团的最大股东是干什么的？”这类涉及实体关系和多跳推理的问题。
- **目标**：准确识别实体，利用图谱推理出潜在关系，结合文档证据生成准确回答。

### 1.2 系统流程
1.  **检索 A 公司相关信息（RAG）**：从非结构化文本中检索语义相关的片段。
2.  **图谱中查找控股关系（KG）**：在企业股权图谱中执行多跳查询，寻找股东、子公司等关系。
3.  **生成最终回答（LLM）**：融合检索到的文本和图谱推理结果，由大模型生成最终答案。

### 1.3 技术难点
-   **如何将 RAG 与图谱推理融合？**：文本与图谱数据模态不同，需要设计有效的融合机制。
-   **如何设计联合评分机制？**：需要平衡向量相似度、关键词匹配度和图谱相关性。
-   **如何防止错误传播？**：图谱中的错误关系或错误的实体提取可能导致回答偏差，需要抑制噪声。

### 1.4 工程化要求
-   使用 **Neo4j** 构建企业股权图谱。
-   使用 **LlamaIndex** 实现文档检索（向量存储使用 Milvus）。
-   实现 **多跳查询逻辑**（Cypher + LLM 协同）。
-   构建 **可解释性输出**（展示推理路径，让用户了解系统的思考过程）。

---

## 2. 系统实现细节

### 2.1 查询理解与实体提取
-   **实体提取**：使用 LLM 从用户 Query 中提取关键实体（如“A集团”），并识别实体类型（如 COMPANY）。
-   **兜底策略**：若 LLM 提取失败，自动降级为基于规则（正则表达式）的关键词匹配，确保系统鲁棒性。

### 2.2 混合检索 (Hybrid Retrieval)
系统并发执行三路召回，互补优劣：
1.  **向量检索 (Vector Search)**：使用 Milvus 存储文档 Embedding，召回语义相关的非结构化文本。
2.  **关键词检索 (Keyword Search)**：使用 Elasticsearch (BM25) 进行精确匹配，弥补向量检索在专有名词（如特定公司名）上的不足。
3.  **图谱推理 (Graph Reasoning)**：
    -   利用 Neo4j 执行多跳 Cypher 查询。
    -   **1-Hop**：查找直接关联实体（如直接持股），置信度设为 1.0。
    -   **2-Hop**：查找间接关联实体（如间接控股），置信度衰减为 0.8，以捕捉深层关系。

### 2.3 结果融合与生成 (Graph Boosting)
为解决“RAG 与 KG 融合”及“联合评分”的难点，本系统采用了 **Graph Boosting** 机制：
-   **图谱增益**：系统不直接将图谱三元组作为唯一上下文，而是利用图谱推理出的“高置信度实体”，去**反向增强**包含这些实体的文本文档的排序分数。
-   **联合评分公式**：
    ```python
    FinalScore = (VectorScore * w1) + (KeywordScore * w2) + GraphBoost
    ```
    其中 `GraphBoost` 取决于图谱推理的置信度。
-   **加权排序 (WeightedRanker)**：对三路结果进行归一化和加权融合，并应用动态阈值过滤低质量结果。

### 2.4 防止错误传播
-   **置信度衰减**：随着推理跳数增加，置信度逐级递减，避免长路径引入噪声。
-   **双重验证与过滤**：
    1.  **绝对阈值**：设定最小分数阈值，直接过滤掉低质量的检索结果。
    2.  **动态阈值**：检索结果的分数必须大于最大分数的 50%（`max_score * 0.5`），否则被视为长尾噪声进行截断。这有效防止了即使图谱推理出实体，但其在文本中相关性极低的情况。

### 2.5 可解释性输出
系统在 API 响应中增加了 `reasoning_log` 字段，完整展示：
1.  接收到的查询。
2.  三路检索的召回数量。
3.  KG 推理出的实体及置信度。
4.  每个文档的评分详情（向量分、关键词分、图谱增益）。
5.  最终生成的上下文长度。

---

## 3. 运行指南

按照以下步骤运行本作业：

### 3.1 启动基础服务
首先启动 Docker 容器，运行 Milvus, Neo4j, Elasticsearch 等中间件：
```bash
docker-compose -f docker-compose.yml up -d
```
*注意：确保 Docker 容器全部正常启动（尤其是 Neo4j 和 Milvus）。*

### 3.2 环境准备与依赖安装

建议使用 Conda 创建独立的 Python 环境：

```bash
conda create -n graph_rag python=3.12
conda activate graph_rag
```

然后安装项目所需的 Python 依赖：
```bash
pip install -r graph_rag/requirements.txt
```

### 3.3 启动 API 服务
在项目根目录下运行主程序：
```bash
python -m graph_rag.main
```
服务默认监听在 `http://0.0.0.0:8000`。

### 3.4 数据初始化 (Rebuild)
服务启动后，需要调用 API 初始化数据（构建索引和图谱）。建议调用全量重建接口：

**PowerShell 示例:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/rebuild/all" -Method Post
```

**或者使用 curl:**
```bash
curl -X POST "http://localhost:8000/rebuild/all"
```
*注意：此过程是异步的，请关注后台日志等待重建完成。*

### 3.5 执行查询
待数据重建完成后，调用查询接口询问 A 集团的董事长：

**PowerShell 示例:**
```powershell
$body = @{ text = "A集团的董事长是谁？" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -Body $body -ContentType "application/json"
```

**或者使用 curl:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "A集团的董事长是谁？"}'
```

### 3.6 API 文档
服务启动后，可以访问 `http://localhost:8000/docs` 查看 Swagger UI 界面，在网页上直接进行 API 调用和测试。

查看返回结果中的 `answer` 字段获取答案，并查看 `reasoning_log` 字段了解系统的推理路径。
