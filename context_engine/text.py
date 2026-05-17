import re


def clean_text(text: str) -> str:
    """Normalize document-level text before chunking."""
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"[^\w\s\.\,\-\%\(\)]", " ", text)
    text = re.sub(r"\n\s*\.\s*\n", "\n", text)
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_chunk(text: str) -> str:
    """Normalize text at the chunk level."""
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"Print to PDF", "", text)
    text = re.sub(r"^\s*\.\s*", "", text)
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"and drying, green coffee beans.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
