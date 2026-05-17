"""Context engine entry point plus composable RAG building blocks."""

from .config import ContextEngineConfig
from .documents import Doc, ParsedDoc
from .engine import ContextEngine
from .loaders import AutoLoader, ExtractionResult, PdfMarkdownLoader

__all__ = [
    "AutoLoader",
    "ContextEngine",
    "ContextEngineConfig",
    "Doc",
    "ExtractionResult",
    "ParsedDoc",
    "PdfMarkdownLoader",
]
