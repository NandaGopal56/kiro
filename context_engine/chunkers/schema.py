from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Doc:
    """A chunk-shaped document used by chunkers, stores, and retrievers."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}

