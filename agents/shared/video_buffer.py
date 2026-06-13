# agents/shared/video_buffer.py
#
# A simple in-memory buffer that holds the most recent video frame.
# The personal agent reads from this when the user asks about what it sees.
# Ported directly from the original video_topic_buffer.py.

import threading
from typing import Optional


class VideoBuffer:
    """Thread-safe holder for the latest camera frame."""

    def __init__(self):
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()

    def update(self, frame: bytes) -> None:
        """Store the latest frame. Called by your video capture pipeline."""
        with self._lock:
            self._frame = frame

    def latest(self) -> Optional[bytes]:
        """Return the most recent frame, or None if none has arrived yet."""
        with self._lock:
            return self._frame

    def clear(self) -> None:
        with self._lock:
            self._frame = None


# Module-level singleton — import this everywhere you need it
video_buffer = VideoBuffer()