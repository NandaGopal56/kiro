"""Vision subsystem: detection, tracking, and VLM adapters.

Each submodule (detection, tracking, vlm) is independent and exposes its own
client (e.g. ``DetectionClient``) plus a ``__main__`` for standalone CLI use.
The modules are joined *only* by ``VisionPipeline`` (the single orchestrator),
which is the one place that knows about all of them.
"""

from dotenv import find_dotenv, load_dotenv

from vision.pipeline import VisionPipeline

load_dotenv(find_dotenv())

__all__ = ["VisionPipeline"]
