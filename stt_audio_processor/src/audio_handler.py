"""
audio_handler.py
-----------------

Captures microphone audio in 100ms PCM chunks and puts them into an async
queue so the rest of the system can consume them at its own pace.

There is no voice-activity detection at this layer. Silence vs. speech is
decided by the RMS energy checks in VoiceProcessor, and by Sarvam's own VAD
signals on the server side.

Public controls
---------------
mute() / unmute()
    Stop or resume pushing chunks into the queue. Useful when the system is
    playing back TTS so it doesn't accidentally hear itself.

recalibrate()
    Measures ambient room noise for one second and prints the RMS value. Use
    that number to tune SILENCE_RMS_THRESHOLD in your config if the end-of-
    utterance timer fires too early in a noisy environment.

drain()
    Removes and returns every chunk currently sitting in the queue. Call this
    just before you need a clean slate — for example, between utterances —
    so that stale audio from a previous turn is not fed into a new session.
"""

import asyncio
from typing import Optional

import pyaudio
import numpy as np

from stt_audio_processor.utils.config import AUDIO_CONFIG

CHUNK_MS     = 100
CHUNK_FRAMES = AUDIO_CONFIG.sample_rate * CHUNK_MS // 1000


class AudioHandler:

    def __init__(self):
        self._pa       = None
        self._stream   = None
        self._loop     = None
        self.muted     = False
        self.data_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._pa   = pyaudio.PyAudio()
        await self._open_stream()

    async def stop(self):
        self._close_stream()
        if self._pa:
            self._pa.terminate()
        print("[Audio] Stopped")

    # ------------------------------------------------------------------
    # Reading audio
    # ------------------------------------------------------------------

    async def get_chunk(self) -> Optional[bytes]:
        """Return the next 100ms chunk, or None if the queue is empty right now."""
        try:
            return self.data_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def drain(self) -> list[bytes]:
        """
        Remove and return all chunks currently waiting in the queue.

        Use this between utterances to capture audio that arrived while the
        previous WebSocket session was closing, so it can be replayed into
        the next session without dropping any speech.
        """
        chunks = []
        while True:
            try:
                chunks.append(self.data_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return chunks

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def mute(self):
        self.muted = True
        print("[Audio] Muted")

    def unmute(self):
        self.muted = False
        print("[Audio] Unmuted")

    async def recalibrate(self):
        """
        Opens a temporary microphone stream for one second, measures the
        average RMS energy of the room noise, and prints the result.

        Run this once in a quiet room and use the printed value as your
        SILENCE_RMS_THRESHOLD baseline. Add a comfortable margin on top
        so ordinary background noise does not prevent the EOU timer from
        counting down.
        """
        print("[Audio] Measuring ambient noise for 1 second...")
        frames = []
        temp_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_CONFIG.sample_rate,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )
        for _ in range(int(AUDIO_CONFIG.sample_rate / CHUNK_FRAMES)):
            data = temp_stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        temp_stream.stop_stream()
        temp_stream.close()

        noise_rms = int(np.sqrt(np.mean(np.concatenate(frames).astype(np.float32) ** 2)))
        print(f"[Audio] Ambient RMS: {noise_rms}  —  set SILENCE_RMS_THRESHOLD above this value")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _open_stream(self):
        print(f"[Audio] Opening microphone — {AUDIO_CONFIG.sample_rate} Hz, {CHUNK_MS}ms chunks")
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_CONFIG.sample_rate,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        print("[Audio] Stream started")

    def _close_stream(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def _callback(self, in_data, frame_count, time_info, status):
        """
        Called by PyAudio from a C-level thread every CHUNK_MS milliseconds.
        We hand the raw bytes off to the async event loop immediately using
        call_soon_threadsafe so no audio is ever dropped.
        """
        if not self.muted and in_data:
            self._loop.call_soon_threadsafe(self.data_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)


audio_handler = AudioHandler()