"""基础工具类单元测试模块。

测试读写锁（ReadWriteLock）的并发控制逻辑以及文件哈希计算。
"""

import sys
import os

# 为了支持 IDE 直接点击运行按钮，将项目根目录加入 sys.path
# 确保在 milvus_faq 包外部运行也能正确识别包路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import threading
import time
import tempfile
from milvus_faq.utils import ReadWriteLock, ReadLockContext, WriteLockContext, calculate_file_hash


class TestUtils(unittest.TestCase):
    """工具类测试集合。"""

    def test_calculate_file_hash(self):
        """测试文件 MD5 哈希计算。
        
        验证计算结果是否与标准哈希值匹配，并处理不存在文件的情况。
        """
        # 创建临时文件并写入内容
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("hello world")
            filepath = f.name
        
        try:
            # "hello world" 的标准 MD5 
            expected_hash = "5eb63bbbe01eeed093cb22bb8f5acdc3"
            self.assertEqual(calculate_file_hash(filepath), expected_hash)
            
            # 测试异常路径：文件不存在
            self.assertEqual(calculate_file_hash("non_existent_file.txt"), "")
        finally:
            # delete=False 时必须手动删除文件
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_read_write_lock_multiple_readers(self):
        """测试读写锁：允许多个读取者并行。
        
        5 个并发读取线程应该在极短时间内（接近单个线程耗时）完成任务。
        """
        lock = ReadWriteLock()
        counter = 0
        
        def reader():
            nonlocal counter
            with ReadLockContext(lock):
                temp = counter
                time.sleep(0.1)
                # 读锁期间 counter 不应发生变动
                self.assertEqual(counter, temp)
        
        # 批量启动线程
        threads = [threading.Thread(target=reader) for _ in range(5)]
        start_time = time.time()
        
        # 并发启动
        for t in threads: t.start()
        # 等待归队
        for t in threads: t.join()
        
        end_time = time.time()
        
        # 如果是串行的，耗时应 > 0.5s；并行的耗时应显著小于 0.4s
        self.assertLess(end_time - start_time, 0.4)

    def test_read_write_lock_write_blocks_read(self):
        """测试读写锁：写入操作阻塞读取操作。
        
        确保在写入者完成修改前，读取者无法进入临界区。
        """
        lock = ReadWriteLock()
        resource = []
        
        def writer():
            with WriteLockContext(lock):
                time.sleep(0.2)
                resource.append(1)
                
        def reader():
            # 延迟启动读取，确保写入者先拿到锁
            time.sleep(0.05)
            with ReadLockContext(lock):
                # 此时应已能看到写入者的修改结果
                self.assertEqual(len(resource), 1)

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)
        
        t_writer.start()
        t_reader.start()
        
        t_writer.join()
        t_reader.join()

    def test_read_write_lock_writer_mutual_exclusion(self):
        """测试读写锁：多个写入者互斥。
        
        确保同一时间只有一个写入者能进入临界区。
        通过验证 3 个各耗时 0.1s 的写入任务总耗时大于 0.3s 来确认其串行执行。
        """
        lock = ReadWriteLock()
        counter = 0
        
        def writer():
            nonlocal counter
            with WriteLockContext(lock):
                temp = counter
                time.sleep(0.1)
                counter = temp + 1
        
        threads = [threading.Thread(target=writer) for _ in range(3)]
        start_time = time.time()
        for t in threads: t.start()
        for t in threads: t.join()
        end_time = time.time()
        
        # 验证结果正确性
        self.assertEqual(counter, 3)
        # 验证串行执行：总耗时应显著大于单个任务耗时的总和
        self.assertGreater(end_time - start_time, 0.3)

if __name__ == "__main__":
    unittest.main()
