import json
import re
from pathlib import Path
from typing import Any, Optional, Iterable

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())

class UserBM25:
    """
    Per-user BM25 index persisted in artifacts/user_<id>/_corpus/_bm25/
    """
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.bm25_dir = corpus_dir / "_bm25"
        self.bm25_dir.mkdir(parents=True, exist_ok=True)

        self.tokens_path = self.bm25_dir / "tokens.jsonl"
        self.meta_path = self.bm25_dir / "bm25_meta.json"

        self._bm25: Optional[BM25Okapi] = None
        self._tokens: Optional[list[list[str]]] = None

    def _load_docstore_rows(self) -> list[dict[str, Any]]:
        rows = []
        p = self.corpus_dir / "docstore.jsonl"
        if not p.exists():
            return rows
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def build_or_update(self) -> None:
        """
        For simplicity: rebuild tokens file if counts mismatch.
        For production: make this incremental in updater.py.
        """
        rows = self._load_docstore_rows()
        # Expect optional "text" field; if absent, BM25 will be weak.
        tokens = [tokenize(r.get("text", "")) for r in rows]

        with self.tokens_path.open("w", encoding="utf-8") as f:
            for t in tokens:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        self.meta_path.write_text(json.dumps({"count": len(tokens)}, indent=2), encoding="utf-8")
        self._bm25 = BM25Okapi(tokens)
        self._tokens = tokens

    def load(self) -> None:
        if not self.tokens_path.exists():
            self.build_or_update()
            return

        toks: list[list[str]] = []
        with self.tokens_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    toks.append(json.loads(line))

        self._tokens = toks
        self._bm25 = BM25Okapi(toks)

    def search(self, query: str, top_k: int, allowed_rows: Optional[set[int]] = None) -> list[tuple[int, float]]:
        if self._bm25 is None:
            self.load()

        assert self._bm25 is not None
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        # restrict to allowed rows if provided (doc filter)
        if allowed_rows is not None:
            candidates = [(i, float(scores[i])) for i in allowed_rows]
        else:
            candidates = [(i, float(s)) for i, s in enumerate(scores)]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]