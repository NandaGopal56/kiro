from __future__ import annotations

from collections.abc import Iterable

from .types import Service


class Orchestrator:
    """Starts services in order and stops them in reverse order."""

    def __init__(self, services: Iterable[Service] | None = None) -> None:
        self.services = list(services or [])
        self._started: list[Service] = []

    def add(self, service: Service) -> None:
        self.services.append(service)

    async def start(self) -> None:
        try:
            for service in self.services:
                await service.start()
                self._started.append(service)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        while self._started:
            service = self._started.pop()
            await service.stop()
