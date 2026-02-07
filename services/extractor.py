from io import BytesIO
from pypdf import PdfReader
from docx import Document as DocxDocument

def extract_pdf(file_bytes: bytes) -> tuple[str, list[dict]]:
    """
    Returns full text + per-page metadata list for citations.
    """
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    full = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
        full.append(text)
    return "\n".join(full), pages

def extract_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts)

def extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")
