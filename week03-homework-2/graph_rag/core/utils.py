import hashlib
import threading
from typing import List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
import os
import nltk

# 预加载 NLTK 资源，防止多线程懒加载冲突
try:
    # 尝试加载 punkt 分词器
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    # 尝试加载 stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    # 强制加载一次 LazyCorpusLoader
    from nltk.corpus import stopwords
    try:
        _ = stopwords.words('english')
    except Exception:
        pass
except Exception:
    # 忽略加载错误，避免阻断程序启动
    pass

def calculate_file_hash(file_path: str) -> str:
    """计算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class ReadWriteLock:
    """简单的读写锁实现"""
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()

class ReadLockContext:
    def __init__(self, rw_lock: ReadWriteLock):
        self.rw_lock = rw_lock

    def __enter__(self):
        self.rw_lock.acquire_read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_read()
        return False

class WriteLockContext:
    def __init__(self, rw_lock: ReadWriteLock):
        self.rw_lock = rw_lock

    def __enter__(self):
        self.rw_lock.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_write()
        return False


def get_chunk_id(content: str, file_path: str, index: int) -> str:
    """
    生成确定性的 Chunk ID
    使用 md5(file_name + "_" + str(index))
    注意：使用文件名而不是绝对路径，以保证在不同目录下的一致性
    """
    file_name = os.path.basename(file_path)
    unique_str = f"{file_name}_{index}"
    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

def split_document(file_path: str, content: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[TextNode]:
    """
    统一的文档切分逻辑
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 创建 LlamaIndex Document 对象
    doc = Document(text=content, metadata={"file_path": file_path})
    
    # 切分
    nodes = splitter.get_nodes_from_documents([doc])
    
    # 重新分配确定性 ID
    for i, node in enumerate(nodes):
        node.id_ = get_chunk_id(content, file_path, i)
        # 确保 metadata 中有 file_path
        node.metadata["file_path"] = file_path
        
    return nodes
