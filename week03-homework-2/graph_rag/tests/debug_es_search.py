import asyncio
from elasticsearch import AsyncElasticsearch
from graph_rag.core.config import settings

async def main():
    print(f"Connecting to ES at {settings.es.URL}...")
    client = AsyncElasticsearch(
        hosts=[settings.es.URL],
        verify_certs=settings.es.VERIFY_CERTS,
        request_timeout=30
    )
    
    test_index = "debug_search_test"
    
    try:
        # 1. 删除旧索引
        if await client.indices.exists(index=test_index):
            await client.indices.delete(index=test_index)
            print(f"Deleted old index: {test_index}")

        # 2. 创建索引 (强制 replicas=0)
        await client.indices.create(
            index=test_index,
            body={
                "settings": {
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {"type": "standard"} # 简化环境依赖，用 standard
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "title": {"type": "text"}
                    }
                }
            }
        )
        print(f"Created index: {test_index}")

        # 3. 插入数据
        docs = [
            {"title": "Doc A", "content": "A集团是一家大型科技公司"},
            {"title": "Doc B", "content": "张三是A集团的董事长"},
            {"title": "Doc C", "content": "B资本投资了A集团"}
        ]
        
        for i, doc in enumerate(docs):
            await client.index(index=test_index, id=str(i), document=doc)
        
        # 强制刷新，确保立即可搜
        await client.indices.refresh(index=test_index)
        print(f"Indexed {len(docs)} documents")

        # 4. 测试 match 查询 (全文检索)
        print("\n--- Testing match (Query: 'A集团') ---")
        resp = await client.search(
            index=test_index,
            body={"query": {"match": {"content": "A集团"}}}
        )
        hits = resp['hits']['hits']
        print(f"Found {len(hits)} hits:")
        for hit in hits:
            print(f" - {hit['_source']['content']} (score: {hit['_score']})")

        # 5. 测试 match_phrase 查询 (短语匹配)
        print("\n--- Testing match_phrase (Query: '张三是') ---")
        resp = await client.search(
            index=test_index,
            body={"query": {"match_phrase": {"content": "张三是"}}}
        )
        hits = resp['hits']['hits']
        print(f"Found {len(hits)} hits:")
        for hit in hits:
            print(f" - {hit['_source']['content']} (score: {hit['_score']})")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
