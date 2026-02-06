import os
import json
import asyncio
import time
import warnings
import numpy as np
from openai import project
import pandas as pd
from tqdm.asyncio import tqdm
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

warnings.filterwarnings("ignore", category=DeprecationWarning)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

try:
    from ragas import evaluate
    from ragas.metrics import context_recall, context_precision, answer_correctness
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"Warning: Ragas or LangChain import failed: {e}")
    print("Ragas metrics will be skipped.")

try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

try:
    from llama_index.core.postprocessor import LLMRerank
    LLM_RERANK_AVAILABLE = True
except ImportError:
    LLM_RERANK_AVAILABLE = False
    print("Warning: LLMRerank not found.")


# 自定义冗余度评估 Prompt
REDUNDANCY_PROMPT = """
你是RAG系统上下文质量评估专家。
请对比以下两段内容，评估【检索到的上下文】相对于【标准上下文】的冗余程度。

评分标准：
1：几乎全是必要证据，基本无赘余
3：有部分背景/重复句，但不影响定位证据
5：大量无关信息/多主题混杂，明显污染上下文

【标准上下文 (Ground Truth)】:
{ground_truth}

[Retrieved Context (Retrieved)]:
{retrieved}

Please output only a single digit score (1, 3, 5):
"""

async def evaluate_redundancy(llm, retrieved_text, ground_truth_text):
    """使用 LLM 评估上下文冗余度"""
    prompt = REDUNDANCY_PROMPT.format(
        ground_truth=ground_truth_text[:2000], # 截断以防过长
        retrieved=retrieved_text[:2000]
    )
    try:
        response = await llm.acomplete(prompt)
        score_str = response.text.strip()
        # 提取数字
        import re
        match = re.search(r'\b(1|3|5)\b', score_str)
        if match:
            return int(match.group(1))
        return 3 # 默认中等
    except Exception as e:
        print(f"Error in redundancy eval: {e}")
        return 3

async def process_one_query(query_engine, item):
    """处理单个 Query 的检索和生成"""
    query = item.get("query") or item.get("question")
    ground_truth_answer = item.get("answer")
    ground_truth_context = item.get("context")
    
    if not query:
        return None

    # 增加手动重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 异步查询
            response = await query_engine.aquery(query)
            
            retrieved_nodes = response.source_nodes
            retrieved_info = [{"text": node.text, "score": node.score} for node in retrieved_nodes]
            retrieved_texts = [node.text for node in retrieved_nodes]
            retrieved_scores = [node.score for node in retrieved_nodes]
            generated_answer = response.response

            return {
                "question": query,
                "answer": generated_answer,
                "contexts": retrieved_texts,
                "retrieved_nodes": retrieved_info, # 新增结构化字段
                "retrieved_scores": retrieved_scores,
                "ground_truth": ground_truth_answer, # Ragas 需要
                "ground_truth_context": ground_truth_context # 自定义评估需要
            }
        except Exception as e:
            print(f"Error processing query '{query}' (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # 指数退避策略
                wait_time = (attempt + 1) * 2
                await asyncio.sleep(wait_time)
            else:
                print(f"Failed to process query '{query}' after {max_retries} attempts.")
                return None

def init_settings():
    """初始化 LlamaIndex 设置"""
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True
    )

    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=2,
        embed_input_length=8192
    )

def build_or_load_index(doc_dir="./docs", chunk_size=512, chunk_overlap=50, force_rebuild=False):
    """
    构建或加载索引
    :param doc_dir: 文档目录
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠大小
    :param force_rebuild: 是否强制重建索引
    """
    # 根据参数生成唯一的存储目录，避免不同实验混淆
    # 获取 doc_dir 的父目录作为项目根目录
    project_root = os.path.dirname(doc_dir)
    storage_base = os.path.join(project_root, "storage")
    persist_dir = os.path.join(storage_base, f"storage_chunk_{chunk_size}_{chunk_overlap}")
    
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"Loading index from {persist_dir}...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            return load_index_from_storage(storage_context)
        except Exception as e:
            print(f"Failed to load index from {persist_dir}: {e}. Rebuilding...")
    
    # 重建索引逻辑
    print(f"Building index with Chunk Size: {chunk_size}, Overlap: {chunk_overlap}...")
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(f"Document directory '{doc_dir}' not found.")
        
    documents = SimpleDirectoryReader(doc_dir).load_data()
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # 增加重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Building index (Attempt {attempt + 1}/{max_retries})...")
            # 确保目录存在
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
                
            index = VectorStoreIndex.from_documents(documents, transformations=[sentence_splitter])
            # 保存索引到磁盘
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"Index created and saved to {persist_dir}")
            return index
        except Exception as e:
            print(f"Error building index: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Exiting.")
                raise e

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

async def main():
    global RAGAS_AVAILABLE, RERANKER_AVAILABLE, LLM_RERANK_AVAILABLE
    DEBUG_MODE = False

    # 修改路径逻辑：现在所有资源都在当前文件夹下
    project_dir = os.path.dirname(os.path.abspath(__file__))
    doc_dir = os.path.join(project_dir, "docs")
    dataset_dir = os.path.join(project_dir, "rag_dataset")
    
    # 1. 初始化 Settings
    init_settings()

    # 实验参数设置
    CHUNK_SIZE = 128
    CHUNK_OVERLAP = 25
    FORCE_REBUILD = False # 如果需要强制重新切分并构建索引，设为 True

    # 2. 构建或加载索引
    try:
        # 传递参数，自动管理不同参数的存储路径
        index = build_or_load_index(
            doc_dir=doc_dir,
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP, 
            force_rebuild=FORCE_REBUILD
        )
    except Exception as e:
        print(f"Failed to initialize index: {e}")
        return

    # 适当增加 top_k 以便召回更多上下文进行评估
    # EXPERIMENT: 启用 Rerank 策略
    # 策略: 先召回 Top-10 (high recall)，然后 Rerank 到 Top-3 (high precision)
    USE_RERANK = True 
    INITIAL_TOP_K = 10
    FINAL_TOP_N = 3

    node_postprocessors = []
    if USE_RERANK:
        # 1. 尝试使用 BGE Reranker
        if RERANKER_AVAILABLE:
            print(f"Reranking Enabled (BGE): Top-{INITIAL_TOP_K} -> Top-{FINAL_TOP_N}")
            try:
                reranker = SentenceTransformerRerank(
                    model="BAAI/bge-reranker-base",
                    top_n=FINAL_TOP_N
                )
                node_postprocessors.append(reranker)
            except Exception as e:
                print(f"Failed to load SentenceTransformerRerank: {e}")
                RERANKER_AVAILABLE = False # 标记失败，尝试 fallback

        # 2. 如果 BGE 失败或不可用，尝试使用 LLM Rerank
        if not RERANKER_AVAILABLE:
            if LLM_RERANK_AVAILABLE:
                print(f"Reranking Enabled (LLM Fallback): Top-{INITIAL_TOP_K} -> Top-{FINAL_TOP_N}")
                try:
                    reranker = LLMRerank(
                        top_n=FINAL_TOP_N, 
                        llm=Settings.llm
                    )
                    node_postprocessors.append(reranker)
                except Exception as e:
                     print(f"Failed to load LLMRerank: {e}")
                     INITIAL_TOP_K = FINAL_TOP_N # 彻底失败，回退到无 Rerank
            else:
                 print("Warning: No Reranker available. Skipping rerank step.")
                 INITIAL_TOP_K = FINAL_TOP_N
    else:
        print(f"Reranking Disabled. Using direct Top-{FINAL_TOP_N} retrieval.")
        INITIAL_TOP_K = FINAL_TOP_N
        node_postprocessors = []

    query_engine = index.as_query_engine(
        similarity_top_k=INITIAL_TOP_K,
        node_postprocessors=node_postprocessors
    )

    # 3. 加载评测数据集
    dataset_path = os.path.join(dataset_dir, "rag_qa_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    if DEBUG_MODE:
        qa_data = qa_data[:2] # 仅使用前2个样本
        print("DEBUG MODE: ON (Processing only 2 samples)")
    
    print(f"Loaded {len(qa_data)} QA pairs.")

    # 4. 异步并发执行 RAG 查询
    print("Running Async RAG queries...")
    
    # 使用 Semaphore 限制并发数，防止 ConnectionResetError
    # 降低并发数以提高稳定性，特别是在开启 Rerank 后
    sem = asyncio.Semaphore(2)

    async def safe_process(item):
        async with sem:
            # 添加随机延时以进一步错开请求
            await asyncio.sleep(np.random.uniform(0.5, 1.5))
            return await process_one_query(query_engine, item)

    tasks = [safe_process(item) for item in qa_data]
    results = await tqdm.gather(*tasks)
    results = [r for r in results if r is not None]

    # DEBUG: 打印每个 Query 检索到的 Node 分数
    if DEBUG_MODE:
        print("\n=== DEBUG: Retrieved Node Scores ===")
        for res in results:
            print(f"Q: {res['question']}")
            if "retrieved_nodes" in res:
                 for i, node in enumerate(res["retrieved_nodes"]):
                     print(f"  Node {i+1} (Score: {node.get('score', 0.0):.4f}): {node.get('text', '')}")
            else:
                print(f"Scores: {res.get('retrieved_scores', [])}")
            print("-" * 30)

    # 5. 计算自定义冗余度指标 (Redundancy Score)
    print("Evaluating Redundancy (Custom Metric)...")
    redundancy_scores = []
    redundancy_tasks = []
    
    for res in results:
        retrieved_concat = "\n\n".join(res["contexts"])
        gt_context = res["ground_truth_context"]
        redundancy_tasks.append(evaluate_redundancy(Settings.llm, retrieved_concat, gt_context))
    
    redundancy_scores = await tqdm.gather(*redundancy_tasks)
    avg_redundancy = np.mean(redundancy_scores)
    
    # 将冗余度分数添加到结果中以便查看
    for i, res in enumerate(results):
        res["redundancy_score"] = redundancy_scores[i]

    # 6. 计算 Ragas 指标 (改为分批次 evaluate)
    ragas_metrics_result = {}
    if RAGAS_AVAILABLE:
        print("Evaluating with Ragas (Context Recall/Precision, Answer Correctness) - Batch Mode...")
        
        # 配置 Ragas 使用的 LLM/Embeddings
        ragas_llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            max_retries=3,
            request_timeout=60
        )
        ragas_embeddings = OpenAIEmbeddings(
            model="text-embedding-v3",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            check_embedding_ctx_length=False,
            max_retries=3,
            request_timeout=60
        )
        
        metrics = [context_recall, context_precision, answer_correctness]
        
        # 分批处理以避免并发过高导致的连接错误
        BATCH_SIZE = 5
        all_ragas_scores = []
        
        # 将 results 分割为 batches
        batches = [results[i:i + BATCH_SIZE] for i in range(0, len(results), BATCH_SIZE)]
        
        print(f"Total batches: {len(batches)}")
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}...")
            
            # 构造当前批次的数据集
            batch_data = {
                "question": [r["question"] for r in batch],
                "answer": [r["answer"] for r in batch],
                "contexts": [r["contexts"] for r in batch],
                "ground_truth": [r["ground_truth"] for r in batch]
            }
            dataset = Dataset.from_dict(batch_data)
            
            try:
                # 使用标准的 evaluate，它会自动处理并发 (通常基于 ThreadPool)
                # 由于这是一个同步循环中调用，它会等待当前批次完成再进行下一批
                # 这样可以有效控制并发量
                batch_results = evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    raise_exceptions=False # 防止单个错误中断整个批次
                )
                
                # 将结果转换为 list of dicts 并添加到总列表
                # to_pandas().to_dict('records') 是一种方便的转换方式
                batch_scores = batch_results.to_pandas().to_dict('records')
                all_ragas_scores.extend(batch_scores)
                
            except Exception as e:
                print(f"Error evaluating batch {i+1}: {e}")
                # 为该批次填充空值以保持索引对齐
                empty_scores = [{m.name: np.nan for m in metrics} for _ in batch]
                all_ragas_scores.extend(empty_scores)
            
            # 批次间短暂暂停
            time.sleep(1)

        # 将 Ragas 分数合并回 results
        # 注意：all_ragas_scores 的顺序应该与 results 一致
        if len(all_ragas_scores) == len(results):
            for idx, scores in enumerate(all_ragas_scores):
                results[idx].update(scores)
        else:
            print(f"Warning: Ragas scores count ({len(all_ragas_scores)}) mismatch with results count ({len(results)})")

        # 构建 DataFrame 以便后续处理
        ragas_df = pd.DataFrame(all_ragas_scores)
        # 确保列存在
        for m in metrics:
            if m.name not in ragas_df.columns:
                ragas_df[m.name] = 0.0

    # 7. 保存结果到本地
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 使用 SentenceSplitter 默认的类名作为文件夹标识，如果未来支持多种splitter，可参数化
    splitter_name = "sentence_splitter"
    output_dir = os.path.join(project_dir, "output", splitter_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"rag_eval_results_{timestamp}.json")
    
    # 构造要保存的数据结构
    final_output = {
        "timestamp": timestamp,
        "splitter": splitter_name,
        "metrics": {
            "custom_redundancy_score": avg_redundancy,
        },
        "details": results # 包含每个QA的生成结果、redundancy score等
    }
    
    if RAGAS_AVAILABLE:
        # 处理 NaN 值：用列均值填充
        for col in ["context_recall", "context_precision", "answer_correctness"]:
            if col in ragas_df.columns:
                ragas_df[col] = pd.to_numeric(ragas_df[col], errors='coerce')
                mean_val = ragas_df[col].mean()
                if pd.isna(mean_val): 
                    mean_val = 0.0
                ragas_df[col] = ragas_df[col].fillna(mean_val)

        # 更新总体指标为处理 NaN 后的均值
        final_output["metrics"].update({
            "ragas_context_recall": float(ragas_df["context_recall"].mean()),
            "ragas_context_precision": float(ragas_df["context_precision"].mean()),
            "ragas_answer_correctness": float(ragas_df["answer_correctness"].mean())
        })
        
        f1_scores = []
        for idx, row in ragas_df.iterrows():
            recall = row.get("context_recall", 0.0)
            precision = row.get("context_precision", 0.0)
            if (recall + precision) > 0:
                f1 = 2 * (recall * precision) / (recall + precision)
            else:
                f1 = 0.0
            f1_scores.append(f1)
            
            # 更新 results (final_output['details'])
            results[idx]["ragas_context_recall"] = recall
            results[idx]["ragas_context_precision"] = precision
            results[idx]["ragas_answer_correctness"] = row.get("answer_correctness", 0.0)
            results[idx]["ragas_f1"] = f1
            
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        final_output["metrics"]["ragas_context_f1_avg"] = float(avg_f1)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results to {output_file}: {e}")
        # 尝试保存到备用位置
        backup_file = os.path.join(output_dir, f"rag_eval_backup_{timestamp}.json")
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"Results saved to backup file: {backup_file}")
        except Exception as e2:
             print(f"CRITICAL: Failed to save backup results: {e2}")

    # 8. 汇总与输出
    print("\n" + "="*50)
    print("Evaluation Report")
    print("="*50)
    
    if RAGAS_AVAILABLE and ragas_metrics_result:
        # EvaluationResult 使用索引访问
        # 确保取出的是标量值，如果 Ragas 返回的是列表（单条数据时可能），取平均或第一个值
        # 通常 EvaluationResult[...] 返回的是平均值，但在某些版本或单样本下行为可能不同
        # 最安全的做法是强制转换为 float
        try:
            c_recall = float(ragas_metrics_result["context_recall"])
            c_prec = float(ragas_metrics_result["context_precision"])
            ans_corr = float(ragas_metrics_result["answer_correctness"])
        except (TypeError, ValueError):
            # 降级处理：如果是列表，取平均
            c_recall = np.mean(ragas_metrics_result["context_recall"])
            c_prec = np.mean(ragas_metrics_result["context_precision"])
            ans_corr = np.mean(ragas_metrics_result["answer_correctness"])

        # 使用上面计算好的平均 F1 (如果存在)
        context_f1_avg = final_output["metrics"].get("ragas_context_f1_avg", 0.0)
            
        print(f"Ragas Context Recall:    {c_recall:.4f}")
        print(f"Ragas Context Precision: {c_prec:.4f}")
        print(f"Ragas Context F1 (Avg):  {context_f1_avg:.4f}")
        print(f"Ragas Answer Correctness:{ans_corr:.4f}")
    else:
        print("Ragas metrics skipped.")

    print(f"Custom Redundancy Score: {avg_redundancy:.4f} (1=Best, 5=Worst)")
    print("-" * 50)
    
    # 打印部分详细结果
    for i, res in enumerate(results[:3]):
        print(f"\n[Sample {i+1}]")
        print(f"Q: {res['question']}")
        print(f"Generated Answer: {res['answer'][:100]}...")
        print(f"Redundancy: {res['redundancy_score']}")

if __name__ == "__main__":
    asyncio.run(main())