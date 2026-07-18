from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from vision.common.types import FrameResult


class VisionClient(ABC):
    """Common entry point contract shared by every vision module.

    Each subsystem (detection, tracking, vlm) exposes a concrete client
    implementing this interface. The client is used both by the module's
    own ``__main__`` for standalone CLI runs and by the orchestrating
    pipeline, so there is a single code path per module.
    """

    name: str = "vision-client"

    @abstractmethod
    def run(self, frame, result: Optional[FrameResult] = None) -> FrameResult:
        """Process a frame (optionally extending an existing FrameResult)."""
        ...
