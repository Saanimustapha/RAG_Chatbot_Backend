from __future__ import annotations
from collections import Counter
import re

# Remove weird control chars (incl NUL) + normalize whitespace
_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

def _normalize(text: str) -> str:
    text = _CTRL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()

def build_repeated_line_filter(pages: list[dict], min_freq_ratio: float = 0.35) -> set[str]:
    """
    Finds lines that repeat across many pages (likely headers/footers).
    Returns a set of lines to remove.
    """
    lines = []
    for p in pages:
        txt = p.get("text") or ""
        for ln in txt.splitlines():
            ln = _normalize(ln)
            if not ln:
                continue
            # Keep only short-ish lines; headers/footers tend to be short
            if len(ln) <= 80:
                lines.append(ln)

    if not lines:
        return set()

    counts = Counter(lines)
    num_pages = max(1, len(pages))
    threshold = max(2, int(num_pages * min_freq_ratio))

    repeated = {ln for ln, c in counts.items() if c >= threshold}
    # Common junk tokens
    repeated |= {"TLFeBOOK"}
    return repeated

def clean_pdf_page(text: str, repeated_lines: set[str]) -> str:
    text = _normalize(text)

    out_lines = []
    for ln in text.splitlines():
        ln2 = _normalize(ln)
        if not ln2:
            continue
        if ln2 in repeated_lines:
            continue
        # Drop obvious book artifacts
        if "TLFeBOOK" in ln2:
            continue
        out_lines.append(ln)

    cleaned = "\n".join(out_lines)
    cleaned = _normalize(cleaned)
    return cleaned