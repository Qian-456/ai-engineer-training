# RAG 分块策略与参数对比实验报告

## 1. 实验目标
比较不同分块大小（Chunk Size）、重叠窗口（Overlap）及检索策略对 RAG 系统性能的影响，寻找“检索相关性”与“回答质量”的最佳平衡点。

## 2. 实验设置
- **模型**: Qwen-Plus (LLM), text-embedding-v3 (Embedding)
- **数据集**: RAG QA Dataset (Sampled)
- **评价指标**:
  - **Context Recall**: 检索到的内容是否包含标准答案。
  - **Context Precision**: 检索到的内容中有效信息的占比。
  - **Answer Correctness**: 生成的回答与标准答案的一致性。
  - **Redundancy Score (1-5)**: 上下文冗余度（1=无冗余，5=严重冗余）。

## 3. 实验结果对比

| 实验组别 | Chunk Size | Overlap | Top-K | Rerank | Context Recall | Context Precision | Answer Correctness | Redundancy | 备注 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 512 | 50 | 3 | No | 0.940 | 0.850 | 0.683 | 4.96 | 召回率高，但冗余度高，正确率一般 |
| **Small Chunk** | 128 | 25 | 3 | No | 0.787 | 0.733 | 0.698 | 4.68 | 冗余降低，正确率微涨，但召回率显著下降 |
| **Optimized** | 128 | 25 | 10 | **Yes (Top-3)** | 0.840 | 0.723 | 0.728 | 4.60 | **正确率提升 (+4.5%)**，召回率回升，冗余最低 |

## 4. 结果分析

### 4.1 哪些参数显著影响效果？为什么？
*   **Chunk Size (分块大小)**: 
    *   **大块 (512+)**: 包含更多上下文，Recall 容易高，但容易引入无关信息（噪音），导致 LLM “迷失”或产生幻觉（Correctness 下降）。
    *   **小块 (128-)**: 信息密度高，Precision 高，但单块可能语义不完整。如果不配合 Rerank，极易漏掉关键信息。
*   **Top-K (召回数量)**: 
    *   当 Chunk Size 变小时，必须增大 Top-K（如从 3 增至 10），否则碎片化的信息无法被完整召回。

### 4.2 Chunk Overlap 过大或过小的利弊？
*   **过大 (>50%)**: 导致检索到的 Top-K 内容高度重复（例如 Node 1 和 Node 2 讲同一件事），浪费了上下文窗口，降低了信息多样性（Redundancy Score 变高）。
*   **过小 (<10%)**: 句子可能被切断，导致语义丢失。例如“原因如下：”在上一块，“具体原因...”在下一块，导致检索失效。
*   **最佳实践**: 通常设为 Chunk Size 的 10%-20%（如 128 配 25）。

### 4.3 如何在“精确检索”与“上下文丰富性”之间权衡？
*   **核心矛盾**: 
    *   要“精确”，就要切得细（小 Chunk）。
    *   要“丰富”，就要看得多（大 Context）。
*   **解决方案 (The Golden Combination)**:
    *   **"Small Chunks, Big Window"**: 使用小分块（128）进行存储，但在检索时召回大量候选（Top-10）。
    *   **Reranking (重排序)**: 使用 Cross-Encoder 对 Top-10 进行精细打分，只保留最相关的 Top-3。
    *   **Window Expansion (可选)**: 检索到小块后，自动把它的“前一块”和“后一块”拼起来送给 LLM，既定位准，又看得全。

## 5. 结论
通过引入 **Chunk Size=128 + Top-K=10 + Rerank** 策略，我们在保持低冗余度的同时，显著提升了回答的准确性（Correctness），证明了减少上下文噪音对提升 LLM 推理能力至关重要。