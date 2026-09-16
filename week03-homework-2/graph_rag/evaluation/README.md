# GraphRAG 公开数据检索评测与人工验收

本实验调的是检索融合参数，没有训练或微调模型。线上 `WeightedRanker` 与评测共用 `services/fusion.py`，但离线语料、向量模型、BM25 和图构建方式独立。线上默认参数保持原值。

## 数据与许可证

- [HotpotQA 官方项目](https://hotpotqa.github.io/)提供多跳问题及支持事实，许可为 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。引用：Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018。
- 使用 [HotpotQA 发布者的 Hugging Face 数据集](https://huggingface.co/datasets/hotpotqa/hotpot_qa)，固定 revision、文件 SHA256 见 `prepare.py`。原官方下载在本环境出现 EOF，改用该镜像；没有更换数据集。
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 使用模型页标注的 Apache-2.0 许可；固定 revision 见 `prepare.py`。模型可能接触过公开语料，未验证训练数据与本评测的重叠。
- 原始问题、正文、模型、逐题 HTML 仅保存在忽略目录 `artifacts/`。仓库 evidence 只发布题目 ID、数值指标、划分、配置和运行环境，不重新发布数据集正文。派生数据外发时仍须遵守原许可及署名要求。

## 复现（Python 3.12；从 week03-homework-2 执行）

```powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r graph_rag/evaluation/requirements.txt
.\.venv-eval\Scripts\python.exe -m unittest graph_rag.evaluation.test_evaluation -v
.\.venv-eval\Scripts\python.exe -m graph_rag.evaluation.prepare
.\.venv-eval\Scripts\python.exe -m graph_rag.evaluation.benchmark --selection-file graph_rag/evaluation/evidence/selection.json
```

首次 prepare 需要网络下载；benchmark 使用本地模型和数据，无需 LLM API Key、Milvus、Elasticsearch 或 Neo4j。GPU 可用时自动使用 GPU，否则 CPU；速度与浮点排序可能因环境略有差异。若要从头重做调参，移除 `--selection-file`；这仅复现既定网格，不能把反复查看本测试集后追加的试验当作独立测试。

命令最后打印新 `index.html` 路径，双击即可查看。每次运行创建新目录，不覆盖历史。`summary.json` 保存环境、源码哈希、git revision 和 dirty 标志；`manifest.json` 保存题目 ID 和语料哈希；`selection.json` 保存全部调参结果与冻结时间。

## 固定协议

1. 对 validation 的问题 ID 排序，用 seed=20260916 打乱；前 120 题调参，后 240 题测试。两组问题 ID 不重叠，共享同一检索语料，文档允许重叠。
2. 汇总这 360 题的全部 distractor contexts，按标题合并段落，得到 3,562 篇候选。每个问题搜索全部候选，**不是每题仅选 10 篇，也不是官方 full-wiki 评测**。语料范围由所选题目上下文限定，不能外推开放域规模。
3. Dense 使用归一化 MiniLM 向量、精确余弦，最长 256 tokens；BM25 使用英文小写字母数字分词，k1=1.5、b=0.75。两路各取前 40，统一评估前 5。
4. 图从正文中的唯一标题提及构建有向边，问题中的标题作为种子，种子分 1、一跳邻居分 0.8。图不读答案/支持标签，只对已经召回的候选加分。它不是生产 Neo4j 的关系抽取效果评测。
5. 混合检索比较 12 个配置（向量权重 .25/.5/.75；加权融合相对阈值 0/.5；RRF k=20/60）；图增强比较 36 个配置（额外图权重 0/.1/.2）。按调参集 Recall、全部支持文档命中率、MRR、较小图权重选参。
6. 冻结参数后计算测试指标。首轮完成后补同参数图消融及配对 bootstrap，重用已冻结参数，不再选参。保留首次冻结时间与 replay 来源哈希。

Recall@5：每题命中的支持文档数 / 该题支持文档总数，再对题目平均。MRR@5：前 5 中首次相关文档排名倒数。全部支持命中率：前 5 覆盖全部支持文档的题目比例。空标签不计作满分；本数据没有自然无答案测试集，不宣称验证拒答能力。

## 已执行结果

2026-09-16，Python 3.12.3，CUDA；完整环境见 evidence/summary.json。

| 方法 | Recall@5 | MRR@5 | 全部支持命中率 |
|---|---:|---:|---:|
| Dense | 77.29% | 88.81% | 57.50% |
| BM25 | 72.71% | 87.01% | 47.92% |
| 原加权公式（无图） | 78.75% | 91.18% | 58.75% |
| 调参混合检索 | 79.79% | 91.75% | 60.42% |
| 调参图增强 | 83.54% | 93.33% | 68.33% |
| 图增强所选参数，仅关图增益 | 76.67% | 89.33% | 54.58% |

图增强相对 Dense +6.25 个百分点，48 题改善、19 题退化、173 题不变。相对同参数关图 +6.875 个百分点，34 题改善、1 题退化、205 题不变；2,000 次配对 bootstrap 的 95% 区间为 [+4.79, +9.38] 个百分点。区间仅描述本次 240 题的抽样不确定性，不包含调参、模型、数据分布变化的不确定性。

这些是检索指标，不是答案准确率、中文企业问答效果或生产规模吞吐。没有调用答案生成器，没有把英文所选权重写入线上默认配置。

## 手动验收 / 面试官可核验

1. 跑上述命令，打开 HTML，确认 120/240/3562 的口径、六组指标和方法限制可见。
2. 展开任意题，核对 relevant_docs 与各方法 top_docs，手算 Recall@5。对照 evidence/per_query_metrics.json 中的题目 ID 和分数。
3. 查看 selection.json，图增强选择向量 .25、关键词 .75、图 .2；消融只把图设为 0。调参混合检索为向量 .75、关键词 .25，不能把两者差距全部归因于图。
4. 在逐题结果找图增强相对 Dense 退化的题目，说明标题匹配、候选召回、256-token 截断的限制；保留失败，不能只展示成功案例。
5. 阅读 `services/fusion.py` 和 `test_no_graph_only_empty_document`：给一个未召回 ID 很高的图分，不会生成空正文结果。

HR 可以核验任务、演示报告、真实 PR；技术面试官可以核验公式、分组、同参数对照、源码和失败题。个人贡献边界及待填写复盘见 `../DECISIONS.md`。
