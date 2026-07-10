from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

if TYPE_CHECKING:
    from agents.client import AgentGateway
    from kiro.audio_player import AudioPlayer
    from stt.engine import STTEngine
    from tts.engine import TTSEngine


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


class TextInputSource:
    """Yield text prompts from a predefined list of lines or from stdin."""

    def __init__(self, text_lines: Optional[list[str]] = None, language: Optional[str] = None) -> None:
        self.text_lines = text_lines
        self.language = language

    async def stream(self) -> AsyncIterator[InputEvent]:
        if self.text_lines is not None:
            for text in self.text_lines:
                if not text:
                    continue
                yield InputEvent(
                    text=text,
                    modality=Modality.TEXT,
                    language=self.language,
                    metadata={"source": "text"},
                )
                await asyncio.sleep(0)
            return

        while True:
            try:
                prompt = "input> "
                text = await asyncio.to_thread(input, prompt)
            except EOFError:
                break

            stripped = text.strip()
            if not stripped:
                continue

            yield InputEvent(
                text=stripped,
                modality=Modality.TEXT,
                language=self.language,
                metadata={"source": "text"},
            )


class SpeechToAgentAdapter:
    """Bridge STT or text input into agent invocations without coupling modules."""

    def __init__(
        self,
        stt_engine: Optional["STTEngine"],
        agent_gateway: "AgentGateway",
        agent_name: str = "supervisor",
        thread_id: str = "1",
        language: Optional[str] = None,
        input_source: Optional[AsyncIterator[InputEvent] | TextInputSource] = None,
        tts_engine: Optional["TTSEngine"] = None,
        audio_player: Optional["AudioPlayer"] = None,
    ) -> None:
        self.stt_engine = stt_engine
        self.agent_gateway = agent_gateway
        self.agent_name = agent_name
        self.thread_id = thread_id
        self.language = language
        self.input_source = input_source
        self.tts_engine = tts_engine
        self.audio_player = audio_player

    async def listen_and_respond(self) -> AsyncIterator[tuple[InputEvent, str]]:
        stream_source = self.input_source
        if stream_source is None:
            if self.stt_engine is None:
                raise ValueError("Either an STT engine or input_source must be provided")
            stream_source = self.stt_engine.stream()

        if isinstance(stream_source, TextInputSource):
            async for item in stream_source.stream():
                event = item
                await self._handle_event(event)
                response = await self._invoke_agent(event)
                print(f'response: {response}')
                await self._speak_response(response)
                yield event, response
        else:
            async for item in stream_source:
                if isinstance(item, InputEvent):
                    event = item
                else:
                    event = InputEvent(
                        text=str(item),
                        modality=Modality.AUDIO,
                        language=self.language,
                        metadata={"source": "stt"},
                    )

                await self._handle_event(event)
                response = await self._invoke_agent(event)
                await self._speak_response(response)
                yield event, response

    async def _handle_event(self, event: InputEvent) -> None:
        if event.language is None and self.language is not None:
            event.language = self.language
        if not event.metadata.get("source"):
            event.metadata["source"] = "stt"

    async def _invoke_agent(self, event: InputEvent) -> str:
        return await self.agent_gateway.invoke(
            agent_name=self.agent_name,
            task=event.text,
            thread_id=self.thread_id,
            context={"input_event": event},
        )

    async def _speak_response(self, response: str) -> None:
        if not response or self.tts_engine is None:
            return

        try:
            audio_b64 = await self.tts_engine.speak(response)
            if audio_b64 and self.audio_player is not None:
                await self.audio_player.play_b64(audio_b64)
        except Exception as exc:
            print(f"Warning: failed to synthesize/play agent response audio: {exc}")

    async def close(self) -> None:
        if self.stt_engine is not None and hasattr(self.stt_engine, "close"):
            await self.stt_engine.close()
        if self.tts_engine is not None and hasattr(self.tts_engine, "close"):
            await self.tts_engine.close()
