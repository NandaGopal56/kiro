from __future__ import annotations

from typing import Protocol


class Service(Protocol):
    """Minimal async lifecycle contract for runtime modules."""

    name: str

    async def start(self, **kwargs) -> None:
        ...

    async def stop(self) -> None:
        ...
