"""Composable RAG building blocks.

Compose a pipeline by hand or via Ingestor.from_config(RAGConfig(...)).
"""

from .config import RAGConfig
from .loaders import AutoLoader, ExtractionResult, PdfMarkdownLoader
from .rag import Ingestor
from .types import Doc, ParsedDoc

__all__ = [
    "AutoLoader",
    "Doc",
    "ExtractionResult",
    "Ingestor",
    "ParsedDoc",
    "PdfMarkdownLoader",
    "RAGConfig",
]
