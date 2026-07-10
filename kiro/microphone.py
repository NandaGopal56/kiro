"""Kiro-owned microphone capture used by the STT adapter pipeline."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_MS = 40
BLOCKSIZE = SAMPLE_RATE * FRAME_MS // 1000


class KiroMicrophone:
    """Capture microphone audio frames and optionally mute them during playback."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 1,
        dtype: str = "int16",
        blocksize: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize or BLOCKSIZE
        self._audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        self._stop_event.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started = True

    def _run(self) -> None:
        def audio_callback(indata, frames, time_info, status):
            if self._stop_event.is_set():
                return
            self.queue_audio_frame(bytes(indata))

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=audio_callback,
        )

        with stream:
            stream.start()
            try:
                while not self._stop_event.is_set():
                    sd.sleep(100)
            finally:
                stream.stop()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def queue_audio_frame(self, frame: bytes) -> None:
        payload = b"" if self._paused.is_set() else frame
        self._audio_queue.put(payload)

    def get_audio_frame(self, timeout: float = 0.5) -> bytes:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._started = False
