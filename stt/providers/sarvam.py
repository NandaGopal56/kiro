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
from dotenv import load_dotenv

load_dotenv('/Users/nnandagopal/Desktop/personal_projects/RAG/.env')

import sounddevice as sd
from sarvamai import SarvamAI
from sarvamai.types.speech_to_text_transcription_data import (
    SpeechToTextTranscriptionData,
)

from stt.base import STTProvider

SAMPLE_RATE = 16000
FRAME_MS = 40
BLOCKSIZE = SAMPLE_RATE * FRAME_MS // 1000
MODEL = "saaras:v3"


class SarvamSTT(STTProvider):
    """Streams microphone audio to Sarvam and yields transcripts."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "en-IN",
        model: str = MODEL,
    ) -> None:
        self.api_key = api_key or os.environ["SARVAM_API_KEY"]
        self.language_code = language_code
        self.model = model

        self._loop = asyncio.get_event_loop()
        self._out_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._run_sync)

    def _run_sync(self) -> None:
        """Runs entirely in a worker thread: mic capture + Sarvam sync ws."""
        client = SarvamAI(api_subscription_key=self.api_key)
        audio_queue: "queue.Queue[bytes]" = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_queue.put(bytes(indata))

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCKSIZE,
            callback=audio_callback,
        )

        with client.speech_to_text_streaming.connect(
            model=self.model,
            language_code=self.language_code,
            mode="translate",
            high_vad_sensitivity=True,
            flush_signal=True,
        ) as ws:
            stream.start()
            sender = threading.Thread(
                target=self._sender_loop, args=(ws, audio_queue), daemon=True
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
                stream.stop()
                stream.close()

    def _sender_loop(self, ws, audio_queue: "queue.Queue[bytes]") -> None:
        """Pulls captured audio chunks off the queue and forwards them."""
        while not self._stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
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

    async def close(self) -> None:
        self._stop_event.set()
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