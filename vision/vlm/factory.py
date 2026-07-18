from __future__ import annotations

from vision.vlm.base import VisionLanguageModel
from vision.vlm.openai import OpenAIVisionLM


def create_vlm(name: str, **kwargs) -> VisionLanguageModel:
    name = name.lower()

    if name in {"openai", "gpt-4o", "gpt4o", "openai-vision"}:
        return OpenAIVisionLM(**kwargs)

    raise ValueError(f"Unsupported VLM backend: {name}")
