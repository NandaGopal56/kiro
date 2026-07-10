"""
Sarvam AI speech-to-text provider.

Uses the synchronous Sarvam client inside a dedicated thread instead of
AsyncSarvamAI, whose background ping task causes event-loop race
conditions. Mic capture and the transcribe-send loop also run in that
thread; transcripts cross into asyncio via an asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import base64
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from sarvamai import SarvamAI
from sarvamai.types.speech_to_text_transcription_data import (
    SpeechToTextTranscriptionData,
)

from stt.base import STTProvider
from kiro.microphone import KiroMicrophone, SAMPLE_RATE, BLOCKSIZE

MODEL = "saaras:v3"


class SarvamSTT(STTProvider):
    """Streams microphone audio to Sarvam and yields transcripts."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "en-IN",
        model: str = MODEL,
        microphone: Optional[KiroMicrophone] = None,
    ) -> None:
        self.api_key = api_key or os.environ["SARVAM_API_KEY"]
        self.language_code = language_code
        self.model = model

        self._loop = asyncio.get_event_loop()
        self._out_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._microphone = microphone or KiroMicrophone(sample_rate=SAMPLE_RATE, blocksize=BLOCKSIZE)
        self._executor.submit(self._run_sync)

    def _run_sync(self) -> None:
        """Runs entirely in a worker thread: mic capture + Sarvam sync ws."""
        client = SarvamAI(api_subscription_key=self.api_key)
        self._microphone.start()

        with client.speech_to_text_streaming.connect(
            model=self.model,
            language_code=self.language_code,
            mode="translate",
            high_vad_sensitivity=True,
            flush_signal=True,
        ) as ws:
            sender = threading.Thread(
                target=self._sender_loop, args=(ws,), daemon=True
            )
            sender.start()

            try:
                for msg in ws:
                    if self._stop_event.is_set():
                        break
                    if getattr(msg, "type", None) != "data":
                        continue
                    if not isinstance(msg.data, SpeechToTextTranscriptionData):
                        continue
                    transcript = (msg.data.transcript or "").strip()
                    if transcript:
                        self._loop.call_soon_threadsafe(
                            self._out_queue.put_nowait, transcript
                        )
            finally:
                self._microphone.stop()

    def _sender_loop(self, ws) -> None:
        """Pulls captured audio chunks off the microphone queue and forwards them."""
        while not self._stop_event.is_set():
            chunk = self._microphone.get_audio_frame(timeout=0.5)
            if not chunk:
                continue
            ws.transcribe(
                audio=base64.b64encode(chunk).decode(),
                encoding="audio/wav",
                sample_rate=SAMPLE_RATE,
            )

    async def stream(self) -> AsyncIterator[str]:
        while not self._stop_event.is_set() or not self._out_queue.empty():
            transcript = await self._out_queue.get()
            yield transcript

    async def pause(self) -> None:
        self._microphone.pause()

    async def resume(self) -> None:
        self._microphone.resume()

    def attach_microphone(self, microphone: KiroMicrophone) -> None:
        self._microphone = microphone

    async def close(self) -> None:
        self._stop_event.set()
        self._microphone.stop()
        self._executor.shutdown(wait=False)


async def _main() -> None:
    provider = SarvamSTT()
    print("Listening... (Ctrl+C to stop)\n")
    try:
        async for transcript in provider.stream():
            print(f"\r{transcript}", end="", flush=True)
    finally:
        await provider.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nStopped.")