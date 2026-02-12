import re
import unicodedata

_whitespace_re = re.compile(r"[ \t]+")
_multi_newline_re = re.compile(r"\n{3,}")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # fix common PDF hyphen line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # collapse excessive whitespace
    text = _whitespace_re.sub(" ", text)

    # collapse excessive newlines
    text = _multi_newline_re.sub("\n\n", text)

    return text.strip()
