from __future__ import annotations
import re

_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-,:;()]{6,}$")  # ALL CAPS-ish headings

def split_into_blocks(page_text: str) -> list[dict]:
    """
    Splits a cleaned PDF page into blocks based on blank lines and headings.
    Returns list of {"section": str|None, "text": str}
    """
    lines = [ln.strip() for ln in page_text.splitlines()]
    blocks: list[list[str]] = []
    cur: list[str] = []

    for ln in lines:
        if not ln:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln)
    if cur:
        blocks.append(cur)

    section = None
    out = []
    for b in blocks:
        if not b:
            continue
        # if first line looks like a heading, treat it as section
        first = b[0]
        if _HEADING_RE.match(first) and len(first) <= 80:
            section = first.title()  # nicer formatting
            # remaining text after heading
            rest = "\n".join(b[1:]).strip()
            if rest:
                out.append({"section": section, "text": rest})
        else:
            out.append({"section": section, "text": "\n".join(b).strip()})
    return [x for x in out if x["text"]]