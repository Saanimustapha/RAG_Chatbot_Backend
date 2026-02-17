from typing import Any

def clean_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """
    Pinecone metadata values must be:
      - string, number, boolean
      - list[str]
    Nulls are NOT allowed. Remove keys with None.
    """
    cleaned: dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            cleaned[k] = v
    return cleaned
