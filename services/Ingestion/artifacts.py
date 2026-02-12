import json
import os
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_chunks_jsonl(chunks: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def write_docstore_jsonl(docstore: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in docstore:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_embeddings_bin(embeddings: list[list[float]], path: str):
    arr = np.asarray(embeddings, dtype=np.float32)
    arr.tofile(path)  # simple binary float32 dump
