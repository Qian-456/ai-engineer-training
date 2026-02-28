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
| **SentenceWindow (w=3)** | - | - | 10 | **Yes (Top-3)** | 0.970 | 0.930 | 0.599 | 4.60 | **Precision/Recall 极高**，但 Correctness 仅为 0.599。原因：窗口过大 (w=3) 导致上下文过长，引入过多噪音，LLM 生成受干扰。 |
| **SentenceWindow (w=1)** | - | - | 10 | **Yes (Top-3)** | 0.940 | 0.867 | 0.638 | 4.68 | 缩小窗口后，Correctness 回升至 0.638，说明减少噪音有助于提升回答质量，但仍需进一步优化 Rerank 策略。 |
| **SentenceWindow (w=1 + Filter + Norm)** | - | - | 10 | **Yes + Threshold** | 0.898 | 0.878 | 0.693 | 4.14 | **Answer Correctness 提升至 0.693**，冗余度降至 4.14。改进点：1. 双重阈值过滤 (Abs>0.4, Rel>0.5) 排除低分噪音；2. 中文标点转英文解决切分失效；3. 先段落后句子切分策略。 |

## 4. 结果分析

### 4.1 哪些参数显著影响效果？为什么？
*   **Chunk Size (分块大小)**: 
    *   **大块 (512+)**: 包含更多上下文，Recall 容易高，但容易引入无关信息（噪音），导致 LLM “迷失”或产生幻觉（Correctness 下降）。
    *   **小块 (128-)**: 信息密度高，Precision 高，但单块可能语义不完整。如果不配合 Rerank，极易漏掉关键信息。
*   **Top-K (召回数量)**: 
    *   当 Chunk Size 变小时，必须增大 Top-K（如从 3 增至 10），否则碎片化的信息无法被完整召回。
*   **Sentence Window Strategy (深度优化)**:
    *   **w=3**: 召回了大量相关内容（Recall 0.97），但由于上下文包含过多冗余信息，LLM 难以抓取重点，导致回答准确性（Correctness 0.599）反而不如 Optimized 组。
    *   **w=1**: 缩小窗口后，回答准确性显著提升（0.599 -> 0.638），再次印证了**“降低上下文噪音”**的重要性。
    *   **w=1 + Filter + Norm**: 通过引入**双重阈值过滤**（Abs>0.4, Rel>0.5）有效剔除了 Rerank 后依然存在的低分凑数结果；同时解决了**中文标点导致切分失效**的问题（通过标点标准化+先段落后句子切分），使得冗余度显著降低（4.68 -> 4.14），最终 Answer Correctness 提升至 0.693，接近 Optimized 组的最佳水平。

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
    *   **Threshold Filtering (新发现)**: 在 Sentence Window 实验中发现，Reranker 虽然能把最佳结果排在第一（Score > 0.9），但 Top-3 中的后两名往往是“凑数”的（Score < 0.01）。因此，建议增加**绝对阈值 (Score > 0.4)** 和 **相对阈值 (Score > Max_Score * 0.5)** 过滤，宁缺毋滥，进一步降低噪音。
    *   **Text Normalization (特定问题)**: 对于中文文档，标点符号（如 `\n` vs `\n\n`）对切分器影响巨大。**标准化预处理**（如统一标点、先分段后分句）是确保 Sentence Window 策略生效的关键前提。

## 5. 结论
通过引入 **Chunk Size=128 + Top-K=10 + Rerank** 策略，我们在保持低冗余度的同时，显著提升了回答的准确性（Correctness）。同时，**Sentence Window** 策略虽然能提供极高的召回率，但需警惕窗口过大带来的噪音干扰，且必须配合严格的 **Rerank 阈值过滤** 以及**正确的文本切分预处理**才能发挥最大效用。实验证明，解决分句器对中文支持不佳的问题后，Sentence Window 策略的性能得到了显著修复和提升。