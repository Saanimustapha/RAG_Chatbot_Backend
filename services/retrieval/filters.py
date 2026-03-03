import json
from pathlib import Path
from typing import Optional

def load_manifest(corpus_dir: Path) -> dict:
    p = corpus_dir / "manifest.json"
    if not p.exists():
        return {"docs": {}}
    return json.loads(p.read_text(encoding="utf-8"))

def allowed_rows_for_docs(corpus_dir: Path, doc_ids: list[str]) -> set[int]:
    man = load_manifest(corpus_dir)
    docs = man.get("docs", {})
    rows: set[int] = set()

    for doc_id in doc_ids:
        info = docs.get(str(doc_id))
        if not info:
            continue
        start = int(info["corpus_start"])
        n = int(info["corpus_count"])
        rows.update(range(start, start + n))

    return rows