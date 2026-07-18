from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Optional

import cv2
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from vision.common.env import init_env
from vision.common.logging import get_logger
from vision.vlm.base import VLMResponse, VisionLanguageModel

logger = get_logger("vision.vlm.openai", log_file="vision_vlm_openai.log")

init_env()


def _frame_to_data_url(frame, encode: str = ".jpg") -> str:
    ok, buf = cv2.imencode(encode, frame)
    if not ok:
        raise ValueError("Failed to encode frame to image buffer.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class OpenAIVisionLM(VisionLanguageModel):
    """LangChain-backed OpenAI vision model (e.g. gpt-4o).

    Takes a camera frame + prompt and returns a natural-language analysis.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        self._model = model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is not set (load it via init_env / .env).")
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=resolved_key,
        )

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def analyze(self, frame, prompt: str, history: Optional[list[str]] = None) -> VLMResponse:
        data_url = _frame_to_data_url(frame)

        content: list[object] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

        if history:
            prior = "\n".join(f"- {h}" for h in history)
            content.insert(0, {"type": "text", "text": f"Prior context:\n{prior}"})

        message = HumanMessage(content=content)
        result = self._llm.invoke([message])

        text = result.content if isinstance(result.content, str) else str(result.content)
        logger.debug("VLM %s responded (%d chars)", self.name, len(text))

        return VLMResponse(text=text, model=self.name, raw=result)
