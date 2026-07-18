from __future__ import annotations

from typing import Optional

from vision.common.client import VisionClient
from vision.common.env import init_env
from vision.common.logging import get_logger
from vision.common.types import FrameResult
from vision.vlm.base import VLMResponse, VisionLanguageModel
from vision.vlm.factory import create_vlm

logger = get_logger("vision.vlm.client", log_file="vision_vlm.log")

init_env()


class VLMClient(VisionClient):
    """Runs a vision-language model on a frame given a prompt (extends FrameResult).

    Single entry point used by ``vlm/__main__.py`` (CLI) and by the
    orchestrating pipeline (for captions / on-demand QA).
    """

    name = "vlm"

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
