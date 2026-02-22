import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _load_docstore_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_embeddings_bin(path: Path, dim: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % dim != 0:
        raise ValueError(f"{path} size {data.size} not divisible by dim={dim}")
    return data.reshape(-1, dim)


class UserCorpusStore:
    """
    Per-user append-only corpus:
      - vectors.f32: float32 matrix rows (append-only)
      - ids.txt: chunk_id per row
      - docstore.jsonl: metadata per row (same order)
      - manifest.json: doc_id -> {active_version, active_ids, last_checksum, last_version, last_updated}
      - deleted_ids.txt: tombstones to ignore at query time
    """
    def __init__(self, user_dir: Path, dim: int, metric: str = "cosine"):
        self.user_dir = user_dir
        self.dim = dim
        self.metric = metric

        self.corpus_dir = user_dir / "_corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

        self.vectors_path = self.corpus_dir / "vectors.f32"
        self.ids_path = self.corpus_dir / "ids.txt"
        self.docstore_path = self.corpus_dir / "docstore.jsonl"
        self.meta_path = self.corpus_dir / "meta.json"
        self.manifest_path = self.corpus_dir / "manifest.json"
        self.deleted_path = self.corpus_dir / "deleted_ids.txt"

        # init files
        self.vectors_path.touch(exist_ok=True)
        self.ids_path.touch(exist_ok=True)
        self.docstore_path.touch(exist_ok=True)
        if not self.meta_path.exists():
            _write_json(self.meta_path, {"dim": dim, "metric": metric, "count": 0})
        if not self.manifest_path.exists():
            _write_json(self.manifest_path, {"docs": {}})
        self.deleted_path.touch(exist_ok=True)

    def meta(self) -> dict:
        return _read_json(self.meta_path, {"dim": self.dim, "metric": self.metric, "count": 0})

    def set_count(self, count: int) -> None:
        m = self.meta()
        m["count"] = count
        _write_json(self.meta_path, m)

    def count(self) -> int:
        return int(self.meta().get("count", 0))

    def manifest(self) -> dict:
        return _read_json(self.manifest_path, {"docs": {}})

    def save_manifest(self, manifest: dict) -> None:
        _write_json(self.manifest_path, manifest)

    def load_deleted(self) -> set[str]:
        txt = self.deleted_path.read_text(encoding="utf-8").strip()
        return set(txt.splitlines()) if txt else set()

    def add_tombstones(self, chunk_ids: Iterable[str]) -> None:
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return
        with self.deleted_path.open("a", encoding="utf-8") as f:
            for cid in chunk_ids:
                f.write(cid + "\n")

    def append_doc_leaf(self, doc_leaf: Path) -> tuple[int, int, list[str]]:
        """
        Append one document leaf folder (v<version>) into the corpus store.

        Returns: (start_row, n_rows, appended_chunk_ids)
        """
        docstore = _load_docstore_jsonl(doc_leaf / "docstore.jsonl")
        vectors = _load_embeddings_bin(doc_leaf / "embeddings.bin", dim=self.dim)

        ids = [row["chunk_id"] for row in docstore]
        if vectors.shape[0] != len(ids):
            raise ValueError("docstore rows and embeddings rows mismatch")

        start = self.count()
        n = vectors.shape[0]

        # append vectors
        with self.vectors_path.open("ab") as f:
            vectors.astype(np.float32).tofile(f)

        # append ids
        with self.ids_path.open("a", encoding="utf-8") as f:
            for cid in ids:
                f.write(cid + "\n")

        # append docstore (annotate with corpus_row)
        with self.docstore_path.open("a", encoding="utf-8") as f:
            for i, row in enumerate(docstore):
                row = dict(row)
                row["corpus_row"] = start + i
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.set_count(start + n)
        return start, n, ids

    def load_vectors(self) -> np.ndarray:
        data = np.fromfile(self.vectors_path, dtype=np.float32)
        if data.size % self.dim != 0:
            raise ValueError("corpus vectors file size not divisible by dim")
        return data.reshape(-1, self.dim)

    def load_ids(self) -> list[str]:
        return self.ids_path.read_text(encoding="utf-8").splitlines()