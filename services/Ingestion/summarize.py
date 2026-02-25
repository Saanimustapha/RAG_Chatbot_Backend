from __future__ import annotations
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def page_summary(text: str, max_sentences: int = 3) -> str:
    """
    Very simple extractive summary:
    take the first few sentences that look informative.
    Works decently as a baseline and costs nothing.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 30]
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences])