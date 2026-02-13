from typing import Any

def format_citation(md: dict[str, Any]) -> str:
    """
    Builds a human-friendly citation label from Pinecone metadata.
    Priority:
    - PDF: "<filename> p. X" or "p. X–Y"
    - HTML: "<filename or title> > <section>"
    - Else: "<filename or title>"
    """
    filename = (md.get("filename") or "").strip()
    title = (md.get("title") or "").strip()
    source_type = (md.get("source_type") or "").strip().lower()
    section = (md.get("section") or "").strip()

    page_start = md.get("page_start")
    page_end = md.get("page_end")

    base = filename or title or "source"

    # PDF citation
    if source_type == "pdf":
        if isinstance(page_start, int) and isinstance(page_end, int):
            if page_start == page_end:
                return f"{base} p. {page_start}"
            return f"{base} p. {page_start}–{page_end}"
        if isinstance(page_start, int):
            return f"{base} p. {page_start}"
        return base

    # HTML citation
    if source_type == "html":
        if section:
            return f"{base} > {section}"
        return base

    # DOCX/TXT/etc.
    if section:
        return f"{base} > {section}"
    return base
