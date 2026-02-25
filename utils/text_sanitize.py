import re

# Remove NUL + other non-printing control chars except \n and \t
_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def sanitize_text(s: str | None) -> str:
    if not s:
        return ""
    # remove NUL and other problematic control chars
    s = _CTRL.sub("", s)
    return s