"""下载固定版本公开数据和模型；原始材料只保存在忽略目录。"""
import hashlib
from pathlib import Path
import shutil
import urllib.request

ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
DATA_REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_SHA256 = 'c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6'
DATA_URL = f'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{DATA_REVISION}/distractor/validation-00000-of-00001.parquet'


def prepare():
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / 'hotpot-validation.parquet'
    if not path.exists():
        partial = path.with_suffix('.partial')
        with urllib.request.urlopen(DATA_URL, timeout=90) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != DATA_SHA256:
            raise ValueError('Dataset checksum mismatch')
        partial.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError('Dataset checksum mismatch')
    print('dataset_verified', path, flush=True)
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION,
        local_dir=str(ARTIFACTS / 'model'),
        allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*', 'README.md'])
    print('model_ready', model_path, flush=True)


if __name__ == '__main__':
    prepare()
