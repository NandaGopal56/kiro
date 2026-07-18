from __future__ import annotations

from typing import Optional

from dotenv import find_dotenv, load_dotenv

from shared.logging import get_logger
from vision.common.types import FrameResult
from vision.vlm.base import VLMResponse, VisionLanguageModel
from vision.vlm.factory import create_vlm

logger = get_logger("vision.vlm.client", log_file="vision_vlm.log")

load_dotenv(find_dotenv())


class VLMClient:
    """Runs a vision-language model on a frame given a prompt (extends FrameResult).

    Standalone entry point used by ``vlm/__main__.py`` (CLI) and composed by
    ``VisionPipeline`` (for captions / on-demand QA).
    """

    def __init__(self, backend: str = "openai", **kwargs):
        self.vlm: VisionLanguageModel = create_vlm(backend, **kwargs)

    def analyze(self, frame, prompt: str, history=None) -> VLMResponse:
        resp = self.vlm.analyze(frame, prompt, history)
        logger.debug("VLM %s responded (%d chars)", self.vlm.name, len(resp.text))
        return resp

    def run(self, frame, result: Optional[FrameResult] = None) -> FrameResult:
        # VLM is prompt-driven; `run` performs a default scene caption.
        if result is None:
            result = FrameResult(frame=frame)
        resp = self.analyze(frame, "Describe what is happening in this scene.")
        result.caption = resp.text
        return result
