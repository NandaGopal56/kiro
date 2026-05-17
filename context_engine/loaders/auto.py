from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc
from .base import Loader
from .pdf_markdown import PdfMarkdownLoader


class AutoLoader(Loader):
    """Choose the right loader from the source file extension."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        clean: bool = True,
        loaders_by_suffix: dict[str, Loader] | None = None,
    ) -> None:
        self.loaders_by_suffix = loaders_by_suffix or {
            ".pdf": PdfMarkdownLoader(output_dir=output_dir, clean=clean),
        }

    def load(self, source: str) -> ParsedDoc:
        return self._loader_for(source).load(source)

    def _loader_for(self, source: str | Path) -> Loader:
        suffix = Path(source).suffix.lower()
        try:
            return self.loaders_by_suffix[suffix]
        except KeyError as exc:
            available = sorted(self.loaders_by_suffix)
            raise ValueError(
                f"No loader registered for {suffix!r}. Available suffixes: {available}"
            ) from exc
