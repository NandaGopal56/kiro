import threading
import queue
from typing import Any

from .logger import logger

# Shared queue
audio_queue: "queue.Queue[Any]" = queue.Queue()

# Flag to stop playback
stop_playback = threading.Event()


class TTSPlayer:
    """Explicitly started audio playback worker."""

    def __init__(self, audio_controller=None):
        self.audio_controller = audio_controller
        self.audio_queue: "queue.Queue[Any]" = queue.Queue()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def enqueue(self, segment: Any) -> None:
        self.audio_queue.put(segment)

    def _worker(self) -> None:
        logger.info("Starting playback worker")

        while not self.stop_event.is_set():
            logger.debug("Waiting for audio to play")
            try:
                segment = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.audio_controller is not None:
                self.audio_controller.mute()

            self.play(segment)

            if self.audio_controller is not None:
                self.audio_controller.unmute()
            self.audio_queue.task_done()

        logger.info("Playback worker stopped")

    def play(self, segment: Any) -> None:
        import simpleaudio as sa

        play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )
        play_obj.wait_done()


def playback_worker():
    """Continuously play audio from queue."""

    logger.info('Starting playback worker')

    while not stop_playback.is_set():
        logger.debug('Waiting for audio to play')
        try:
            segment = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        
        TTSPlayer().play(segment)
        logger.info('Played TTS audio')
        audio_queue.task_done()

    logger.info('Playback worker stopped')
