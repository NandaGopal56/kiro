"""Context engine entry point plus composable RAG building blocks."""

__all__ = [
    "AutoLoader",
    "ContextEngine",
    "Doc",
    "ExtractionResult",
    "ParsedDoc",
    "PdfMarkdownLoader",
    ]


def __getattr__(name: str):
    if name == "AutoLoader":
        from .loaders.auto import AutoLoader

        return AutoLoader
    if name == "ContextEngine":
        from .engine import ContextEngine

        return ContextEngine
    if name == "Doc":
        from .chunkers.schema import Doc

        return Doc
    if name == "ExtractionResult":
        from .loaders.schema import ExtractionResult

        return ExtractionResult
    if name == "ParsedDoc":
        from .loaders.schema import ParsedDoc

        return ParsedDoc
    if name == "PdfMarkdownLoader":
        from .loaders.pdf_markdown import PdfMarkdownLoader

        return PdfMarkdownLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
