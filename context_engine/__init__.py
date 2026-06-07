"""Context engine entry point plus composable RAG building blocks."""

from .chunkers.schema import Doc
from .loaders.schema import ParsedDoc, ExtractionResult
from .engine import ContextEngine
from .loaders.pdf_markdown import PdfMarkdownLoader
from .loaders.auto import AutoLoader

__all__ = [
    "AutoLoader",
    "ContextEngine",
    "Doc",
    "ExtractionResult",
    "ParsedDoc",
    "PdfMarkdownLoader",
    ]
