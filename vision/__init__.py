"""Vision subsystem: detection, tracking, recognition, and VLM adapters.

Each submodule (detection, tracking, vlm) exposes a client implementing the
common ``VisionClient`` interface plus its own ``__main__`` for standalone CLI
execution. ``VisionPipeline`` is the single orchestrator that composes those
clients.
"""

from vision.common.client import VisionClient
from vision.common.env import init_env
from vision.pipeline import VisionPipeline

init_env()

__all__ = ["VisionClient", "VisionPipeline"]
