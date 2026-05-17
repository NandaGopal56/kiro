from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class _TableSummary(BaseModel):
    useful_table: bool
    description: str


class _ImageSummary(BaseModel):
    useful_image: bool
    description: str


class OpenAISummarizer:
    """Summarize non-text PDF elements into retrievable text."""

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 400) -> None:
        llm = ChatOpenAI(model=model, max_tokens=max_tokens)
        self._image_llm = llm.with_structured_output(_ImageSummary)
        self._table_llm = llm.with_structured_output(_TableSummary)

    def summarize_table(self, html: str, context: str) -> dict[str, Any]:
        messages = [
            SystemMessage(
                content=(
                    "You analyze tables extracted from documents. Produce dense, "
                    "information-rich descriptions suitable for semantic search."
                )
            ),
            HumanMessage(
                content=f"""
Context around the table:
{context}

Table HTML:
{html}

Determine whether the table contains meaningful structured information.
If useful, write 3 to 5 concise sentences covering topic, entities, metrics,
row/column relationships, trends, and units. If not useful, mark it not useful.
"""
            ),
        ]
        return self._table_llm.invoke(messages).model_dump()

    def summarize_image(self, block: dict[str, Any], context: str) -> dict[str, Any]:
        image_path = block.get("image_path")
        if not image_path:
            return {"useful_image": False, "description": "No image path provided."}

        with Path(image_path).open("rb") as file:
            image_b64 = base64.b64encode(file.read()).decode("utf-8")

        messages = [
            SystemMessage(
                content=(
                    "You analyze images extracted from documents. Determine if each "
                    "image is useful and produce a dense semantic description."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"""
Context around the image:
{context}

Determine whether this image contains useful document information.
If useful, write 3 to 5 concise sentences covering subject, labels, visible
text, diagrams, relationships, numbers, and units. If not useful, mark it not useful.
""",
                    },
                ]
            ),
        ]
        return self._image_llm.invoke(messages).model_dump()
