from __future__ import annotations

from typing import Any, Protocol


class Summarizer(Protocol):
    """Multimodal summarizer used by loaders to describe tables and images."""

    def summarize_table(self, html: str, context: str) -> dict[str, Any]: ...

    def summarize_image(self, block: dict[str, Any], context: str) -> dict[str, Any]: ...
