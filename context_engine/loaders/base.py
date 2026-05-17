from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import ParsedDoc


@dataclass(slots=True)
class ExtractionResult:
    """File extraction output, kept with loaders to avoid a second abstraction."""

    source_path: Path
    text_path: Path
    text: str
    artifacts_dir: Path | None = None
    metadata: dict[str, Any] | None = None


class Loader(ABC):
    """Base for source-specific loaders. Implementations turn a source into a ParsedDoc."""

    @abstractmethod
    def load(self, source: str) -> ParsedDoc:
        ...
