from markdowncleaner import CleanerOptions, MarkdownCleaner
import re

def clean_rag_markdown(text: str) -> str:
    """Clean PDF-converted Markdown without removing meaningful sections."""

    options = CleanerOptions()
    options.fix_encoding_mojibake = True
    options.normalize_quotation_symbols = True
    options.contract_empty_lines = True
    options.crimp_linebreaks = True
    options.remove_duplicate_headlines = True

    options.remove_short_lines = False
    options.remove_sections = False
    options.remove_references_heuristically = False
    options.remove_footnotes_in_text = False

    cleaner = MarkdownCleaner(options=options)
    return cleaner.clean_markdown_string(text)


def clean_repeated_special_chars(text: str) -> str:
    """Normalize repeated punctuation noise commonly introduced by PDFs."""

    text = re.sub(r"(\\_){2,}", "______", text)
    text = re.sub(r"_{3,}", "______", text)
    text = re.sub(r"([ \t])\1{2,}", r"\1", text)

    replacements = {
        r"!{2,}": "!",
        r"@{2,}": "@",
        r"\${2,}": "$",
        r"%{2,}": "%",
        r"\^{2,}": "^",
        r"&{2,}": "&",
        r"={2,}": "=",
        r"\+{2,}": "+",
        r";{2,}": ";",
        r":{2,}": ":",
        r",{2,}": ",",
        r"\?{2,}": "?",
        r"/{2,}": "/",
        r"[\\]{2,}": "\\\\",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text