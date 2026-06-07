"""
transcription_providers.py
---------------------------

This module defines the interface and implementations for streaming
speech-to-text (STT) providers.

How it works
------------
Each provider exposes a single long-lived streaming session that you open
once and reuse across many utterances, rather than creating a new WebSocket
connection for every turn. This avoids the connection-setup latency that
you would otherwise pay at the start of each utterance.

The session has two methods you interact with:

  send_audio(chunk)
      Push a raw PCM byte-string into the provider. Call this continuously
      as audio arrives from the microphone.

  events()
      An async generator that yields TranscriptEvent objects as the
      provider sends them back. Each event carries the transcript text
      and two boolean flags that signal sentence boundaries and end-of-
      utterance, respectively.

TranscriptEvent fields
----------------------
  text         The transcript string. May be empty for VAD-only events
               (like START_SPEECH or END_SPEECH).

  is_final     True when the provider considers the current segment a
               complete sentence. You can use this to display a polished
               final line to the user.

  is_endpoint  True when the provider's VAD decides the speaker has fully
               stopped talking. VoiceProcessor treats this as the EOU
               (end-of-utterance) signal and stops waiting for more speech.

Adding a new provider
---------------------
1. Subclass BaseTranscriptionSession and implement send_audio(), events(),
   and close().
2. Subclass BaseTranscriptionProvider and implement streaming_session() as
   an async context manager that yields your session.
3. The rest of the system needs no changes — it only depends on the base
   types and TranscriptEvent.
"""

import asyncio
import base64
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from sarvamai import AsyncSarvamAI
from sarvamai.types.events_data import EventsData
from sarvamai.types.speech_to_text_transcription_data import SpeechToTextTranscriptionData

from stt_audio_processor.utils.config import AUDIO_CONFIG


# ---------------------------------------------------------------------------
# Shared event type — every provider produces these
# ---------------------------------------------------------------------------

@dataclass
class TranscriptEvent:
    text: str
    is_final: bool = False
    is_endpoint: bool = False


# ---------------------------------------------------------------------------
# Base contracts
# ---------------------------------------------------------------------------

class BaseTranscriptionSession(ABC):

    @abstractmethod
    async def send_audio(self, chunk: bytes): ...

    @abstractmethod
    def events(self) -> AsyncIterator[TranscriptEvent]: ...

    @abstractmethod
    async def close(self): ...


class BaseTranscriptionProvider(ABC):

    @asynccontextmanager
    @abstractmethod
    async def streaming_session(self) -> AsyncIterator[BaseTranscriptionSession]:
        """
        Open a streaming STT session.

        This is an async context manager. The session is kept alive for as
        long as the block runs, so you can send audio and read transcripts
        across multiple utterances without reconnecting.

        Example usage:

            async with provider.streaming_session() as session:
                await session.send_audio(pcm_bytes)
                async for event in session.events():
                    print(event.text)
        """
        ...


# ---------------------------------------------------------------------------
# Sarvam provider
# ---------------------------------------------------------------------------

class SarvamTranscriptionSession(BaseTranscriptionSession):
    """
    Wraps a single Sarvam WebSocket connection and exposes it as a
    reusable session that survives across multiple utterances.

    Internally, a background reader task (_read_loop) drains WebSocket
    messages into an in-memory queue as fast as they arrive. This keeps
    the sender and receiver completely decoupled — you can push audio
    and read transcripts at different rates without either side blocking
    the other.

    Transcript events are printed to the console immediately as they
    arrive in _read_loop, which is why you see output even before the
    full utterance is assembled. This is intentional: it gives you
    real-time feedback without waiting for the final aggregated result.

    Sarvam sends three kinds of messages:

    - START_SPEECH  The VAD detected the beginning of speech. We log it
                    but do not emit an event, since VoiceProcessor uses
                    its own RMS-based detection for the send side.

    - END_SPEECH    The VAD detected the end of speech on the server side.
                    We emit an event with is_endpoint=True so the caller
                    can treat it as an additional EOU signal.

    - Transcript    A partial or final transcription of what was spoken.
                    We emit it immediately so the console output feels
                    responsive even on longer recordings.
    """

    def __init__(self, ws):
        self._ws = ws
        self._queue: asyncio.Queue[Optional[TranscriptEvent]] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None

    def start_reader(self):
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        """
        Runs as a background task for the lifetime of the session.
        Parses every incoming WebSocket message and immediately puts the
        resulting event into the queue. Printing happens here, so it is
        as close to real-time as possible.
        """
        try:
            async for message in self._ws:
                event = self._parse(message)
                if event is not None:
                    if event.text:
                        # Print immediately so long recordings feel responsive.
                        print(f"\r[Transcript] {event.text}", end="", flush=True)
                        if event.is_final:
                            print()  # move to next line on final segment
                    await self._queue.put(event)
        except Exception as e:
            print(f"\n[Sarvam] Reader error: {e}")
        finally:
            # A None sentinel tells any waiting events() call to stop iterating.
            await self._queue.put(None)

    async def send_audio(self, chunk: bytes):
        b64 = base64.b64encode(chunk).decode("utf-8")
        await self._ws.transcribe(
            audio=b64,
            encoding=AUDIO_CONFIG.encoding,
            sample_rate=AUDIO_CONFIG.sample_rate,
        )

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        """
        Yield TranscriptEvents as they arrive from the WebSocket.
        Stops when the background reader signals that the stream is done.
        """
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def close(self):
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _parse(message) -> Optional[TranscriptEvent]:
        data = message.data

        if isinstance(data, EventsData):
            event_type = str(getattr(data, "event", "") or "").upper()
            if "END_SPEECH" in event_type:
                print("\n[Sarvam] END_SPEECH received")
                return TranscriptEvent(text="", is_final=True, is_endpoint=True)
            if "START_SPEECH" in event_type:
                print("[Sarvam] START_SPEECH received")
            return None

        if isinstance(data, SpeechToTextTranscriptionData):
            text = (data.transcript or "").strip()
            if not text:
                return None
            return TranscriptEvent(text=text, is_final=False, is_endpoint=False)

        return None


class SarvamTranscriptionProvider(BaseTranscriptionProvider):
    """
    Manages a single long-lived Sarvam WebSocket session.

    The session is opened once when you enter the context manager and is
    reused for every utterance until the context exits. This eliminates
    the per-utterance reconnection cost and means there is no gap between
    utterances during which speech could be missed.
    """

    def __init__(self):
        self.client = AsyncSarvamAI(
            api_subscription_key=os.getenv("SARVAM_API_KEY")
        )

    @asynccontextmanager
    async def streaming_session(self) -> AsyncIterator[SarvamTranscriptionSession]:
        print("[Sarvam] Opening WebSocket session...")
        async with self.client.speech_to_text_streaming.connect(
            model="saaras:v3",
            mode="transcribe",
            language_code=AUDIO_CONFIG.language_code,
            high_vad_sensitivity=True,
            vad_signals=True,
        ) as ws:
            session = SarvamTranscriptionSession(ws)
            session.start_reader()
            try:
                yield session
            finally:
                await session.close()
                print("[Sarvam] WebSocket session closed.")