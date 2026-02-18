import threading
import hashlib
import os

class ReadWriteLock:
    """
    A simple Reader-Writer lock.
    Allows multiple readers or one writer.
    """
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
    def __init__(self, rw_lock):
        self.rw_lock = rw_lock
    def __enter__(self):
        self.rw_lock.acquire_read()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_read()

class WriteLockContext:
    def __init__(self, rw_lock):
        self.rw_lock = rw_lock
    def __enter__(self):
        self.rw_lock.acquire_write()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_write()

def calculate_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return ""
