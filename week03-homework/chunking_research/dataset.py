import json
import os
import random
from collections import defaultdict
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True
    )

Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )

# 修改路径逻辑
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir, "docs")

if not os.path.exists(docs_dir):
    print(f"Error: Docs directory not found at {docs_dir}")
else:
    documents = SimpleDirectoryReader(docs_dir).load_data()

    parser = SentenceSplitter(
        chunk_size=150,
        chunk_overlap=20,
        paragraph_separator="\n\n"
    )
    nodes = parser.get_nodes_from_documents(documents)

    # ==========================================
    # 新增逻辑：限制生成的 QA 数量并均分到每个文件
    # ==========================================
    TOTAL_QA_LIMIT = 50  # 设置总共要生成的 QA 对数量限制

    nodes_by_file = defaultdict(list)
    for node in nodes:
        # 使用文件名作为分组键，如果没有则使用 'unknown'
        file_key = node.metadata.get("file_name") or node.metadata.get("file_path") or "unknown"
        nodes_by_file[file_key].append(node)

    selected_nodes = []
    files = list(nodes_by_file.keys())
    num_files = len(files)

    if num_files > 0:
        base_count = TOTAL_QA_LIMIT // num_files
        remainder = TOTAL_QA_LIMIT % num_files
        
        print(f"Distributing {TOTAL_QA_LIMIT} QA pairs across {num_files} files...")
        
        for i, file_key in enumerate(files):
            file_nodes = nodes_by_file[file_key]
            # 将余数分配给前几个文件
            count_for_file = base_count + (1 if i < remainder else 0)
            
            if count_for_file > 0:
                # 如果节点数足够，则随机采样；否则全选
                if len(file_nodes) > count_for_file:
                    selected = random.sample(file_nodes, count_for_file)
                else:
                    selected = file_nodes
                selected_nodes.extend(selected)
                print(f"  File '{file_key}': Selected {len(selected)}/{len(file_nodes)} nodes")
            else:
                print(f"  File '{file_key}': Skipped (quota 0)")

    print(f"Total nodes selected for processing: {len(selected_nodes)}")
    # ==========================================

    QA_PROMPT = """
    你是RAG系统评测数据构造器。

    给定以下文档片段：

    ----------------
    {context}
    ----------------

    请生成 1 个高质量的问答对，要求：

    1. 问题必须仅依赖此段文本可回答，且具有一定的深度
    2. 不允许引入外部知识
    3. 答案必须是原文事实的改写，必须准确、完整
    4. 输出必须是合法的 JSON 格式

    输出 JSON:

    [
    {{
        "question": "...",
        "answer": "...",
    }}
    ]
    """
    qa_pairs = []
    count = 0
    failed_details = []
    total_nodes = len(selected_nodes)
    count_node = 0

    print(f"Total nodes to process: {total_nodes}")
    for node in selected_nodes:
        count_node += 1
        print(f"Processing node {count_node}/{total_nodes} (ID: {node.node_id})")
        
        prompt = QA_PROMPT.format(context=node.text)
        response_text = "" # 初始化 response_text 以便异常处理时访问
        try:
            response = Settings.llm.complete(prompt)
            response_text = response.text.strip()
            
            # 处理 Markdown 代码块
            content = response_text
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            data = json.loads(content)
            
            for item in data:
                # 兼容字段名
                q = item.get("question") or item.get("query")
                a = item.get("answer")
                
                if q and a:
                    qa_pairs.append({
                        "query": q,
                        "answer": a,
                        "context": node.text
                    })
                    print(f"  [Success] Generated pair {len(qa_pairs)}/{len(selected_nodes)}")
                else:
                    raise ValueError("Missing 'query' or 'answer' field")

        except Exception as e:
            count += 1
            print(f"No.{count} Error parsing JSON for node_id: {node.node_id}")
            
            failed_details.append({
                "node_id": node.node_id,
                "error": str(e),
                "node_text": node.text,
                "llm_response": response_text
            })



    # ========= Save Results =========

    output_dir = os.path.join(current_dir, "rag_dataset")
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ 主 QA 数据集（JSON）
    qa_path = os.path.join(output_dir, "rag_qa_dataset.json")

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved QA dataset to: {os.path.abspath(qa_path)}")



    # 2️⃣ JSONL 格式（推荐给 RAGAs / 批处理）

    jsonl_path = os.path.join(output_dir, "rag_qa_dataset.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved QA dataset (jsonl): {jsonl_path}")


    # 3️⃣ 失败 Node 记录（详细版）

    failed_path = os.path.join(output_dir, "failed_nodes_details.json")

    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_details, f, ensure_ascii=False, indent=2)

    print(f"Saved failed node details: {failed_path}")


    # 4️⃣ 统计信息（实验日志）

    stats = {
        "total_documents": len(documents),
        "total_original_nodes": len(nodes),
        "total_selected_nodes": len(selected_nodes),
        "success_qa_pairs": len(qa_pairs),
        "failed_nodes": len(failed_details),
        "avg_qa_per_node": round(len(qa_pairs) / max(len(selected_nodes), 1), 2)
    }

    stats_path = os.path.join(output_dir, "stats.json")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Generation Stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))