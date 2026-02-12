from bs4 import BeautifulSoup
from services.extractor import extract_pdf, extract_docx, extract_txt  # adjust import to match your project
from cleaning import normalize_text

def load_pdf_bytes(file_bytes: bytes) -> list[dict]:
    # extract_pdf returns (full_text, pages)
    _full, pages = extract_pdf(file_bytes)
    out = []
    for p in pages:
        out.append({"page": p["page"], "text": normalize_text(p["text"])})
    return out

def load_docx_bytes(file_bytes: bytes) -> str:
    return normalize_text(extract_docx(file_bytes))

def load_text_bytes(file_bytes: bytes) -> str:
    return normalize_text(extract_txt(file_bytes))

def load_html_bytes(file_bytes: bytes) -> list[dict]:
    html = file_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # remove non-content tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # walk through and capture text under headings
    blocks = []
    current_section = "root"
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        name = el.name.lower()
        text = normalize_text(el.get_text(" ", strip=True))
        if not text:
            continue
        if name in ("h1", "h2", "h3"):
            current_section = text[:200]
        else:
            blocks.append({"section": current_section, "text": text})

    # fallback if empty
    if not blocks:
        full = normalize_text(soup.get_text("\n", strip=True))
        if full:
            blocks = [{"section": "root", "text": full}]
    return blocks
