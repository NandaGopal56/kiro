from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

if TYPE_CHECKING:
    from agents.client import AgentGateway
    from stt.engine import STTEngine


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"


@dataclass
class InputEvent:
    """Unified event that can be consumed by any downstream module."""

    text: str
    modality: Modality = Modality.TEXT
    timestamp: float = field(default_factory=time.monotonic)
    language: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.modality.value}] {self.text!r}"


class SpeechToAgentAdapter:
    """Bridge STT output into agent invocations without coupling modules."""

    def __init__(
        self,
        stt_engine: "STTEngine",
        agent_gateway: "AgentGateway",
        agent_name: str = "supervisor",
        thread_id: str = "1",
        language: Optional[str] = None,
    ) -> None:
        self.stt_engine = stt_engine
        self.agent_gateway = agent_gateway
        self.agent_name = agent_name
        self.thread_id = thread_id
        self.language = language

    async def listen_and_respond(self) -> AsyncIterator[tuple[InputEvent, str]]:
        async for item in self.stt_engine.stream():
            if isinstance(item, InputEvent):
                event = item
            else:
                event = InputEvent(
                    text=str(item),
                    modality=Modality.AUDIO,
                    language=self.language,
                    metadata={"source": "stt"},
                )

            if event.language is None and self.language is not None:
                event.language = self.language
            if not event.metadata.get("source"):
                event.metadata["source"] = "stt"

            response = await self.agent_gateway.invoke(
                agent_name=self.agent_name,
                task=event.text,
                thread_id=self.thread_id,
                context={"input_event": event},
            )
            yield event, response

    async def close(self) -> None:
        if hasattr(self.stt_engine, "close"):
            await self.stt_engine.close()
