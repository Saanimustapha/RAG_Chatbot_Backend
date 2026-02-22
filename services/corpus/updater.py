import json
from datetime import datetime
from pathlib import Path
import numpy as np

from RAG_Chatbot_Backend.services.corpus.corpus_store import UserCorpusStore, _load_docstore_jsonl
from RAG_Chatbot_Backend.services.hnsw.hnsw_persist import load_hnsw, save_hnsw
from RAG_Chatbot_Backend.services.hnsw.hnsw_index import HNSWIndex, HNSWParams


def _extract_doc_signature(doc_leaf: Path) -> tuple[str | None, int | None, str | None]:
    """
    Reads docstore.jsonl and returns (checksum, doc_version, document_id)
    (Your chunks.jsonl contains checksum; docstore.jsonl might not—so we read from chunks.jsonl if needed.)
    """
    # Prefer chunks.jsonl because you write checksum there
    chunks_path = doc_leaf / "chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            first = json.loads(next(f))
            return first.get("checksum"), first.get("doc_version"), first.get("document_id")

    # fallback docstore.jsonl
    rows = _load_docstore_jsonl(doc_leaf / "docstore.jsonl")
    if not rows:
        return None, None, None
    r0 = rows[0]
    return r0.get("checksum"), r0.get("doc_version"), r0.get("document_id")


def update_user_corpus_and_hnsw(
    *,
    user_dir: Path,
    doc_leaf: Path,
    embed_dim: int,
    hnsw_params: HNSWParams,
    rebuild_threshold_ratio: float = 0.30,
) -> None:
    """
    Incrementally updates the per-user corpus + HNSW index after a doc ingestion.

    Strategy:
      - Track current doc version/checksum in manifest.json
      - If doc changed: tombstone old chunk_ids from manifest, then append new rows
      - Append-only vectors + metadata
      - Incremental HNSW insertion for appended vectors
      - If too many docs changed since last rebuild, rebuild HNSW (optional)
    """
    store = UserCorpusStore(user_dir=user_dir, dim=embed_dim, metric=hnsw_params.metric)
    manifest = store.manifest()
    docs = manifest.get("docs", {})

    checksum, doc_version, document_id = _extract_doc_signature(doc_leaf)
    if not document_id:
        # folder name is vX under doc_<id>, so doc id can be inferred:
        # .../doc_<docid>/v<version>
        document_id = doc_leaf.parent.name.replace("doc_", "")

    doc_id = str(document_id)
    key = doc_id

    prev = docs.get(key)

    # --- detect change ---
    changed = False
    if not prev:
        changed = True
    else:
        if prev.get("checksum") != checksum or prev.get("doc_version") != doc_version:
            changed = True

    # If unchanged, nothing to do
    if not changed:
        return

    # --- tombstone old active ids (if any) ---
    if prev and prev.get("active_chunk_ids"):
        store.add_tombstones(prev["active_chunk_ids"])

    # --- append new leaf into corpus store ---
    start_row_before = store.count()
    start, n, new_chunk_ids = store.append_doc_leaf(doc_leaf)

    # update manifest entry for this doc
    docs[key] = {
        "checksum": checksum,
        "doc_version": doc_version,
        "active_chunk_ids": new_chunk_ids,  # used for tombstones next time
        "last_update": datetime.utcnow().isoformat(),
        "corpus_start": start,
        "corpus_count": n,
    }
    manifest["docs"] = docs
    store.save_manifest(manifest)

    # --- load or build HNSW ---
    hnsw_dir = store.corpus_dir / "_hnsw"
    hnsw_meta = hnsw_dir / "hnsw_meta.json"

    vectors_all = store.load_vectors()

    if not hnsw_meta.exists():
        # build from scratch
        idx = HNSWIndex(params=hnsw_params)
        idx.build(vectors_all)
        save_hnsw(idx, hnsw_dir)
        return

    # load and attach vectors
    idx = load_hnsw(hnsw_dir)
    idx.params = hnsw_params  # ensure consistent params
    idx.vectors = idx._prepare_vectors(vectors_all)
    idx.N, idx.dim = idx.vectors.shape

    # incremental insert only the appended part
    appended = vectors_all[start_row_before:]
    if appended.shape[0] > 0:
        idx.add_vectors(appended)
        save_hnsw(idx, hnsw_dir)