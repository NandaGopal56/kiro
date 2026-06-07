import time
from collections import deque
from typing import Dict, Any, Deque, Tuple, List, Optional


class VideoTopicBuffer:
    """
    Maintains a rolling time window of frames for a single video topic.
    """

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.buffer: Deque[Tuple[float, Dict[str, Any]]] = deque()

    def on_frame(self, topic: str, payload: Dict[str, Any]):
        now = time.time()
        self.buffer.append((now, payload))

        cutoff = now - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.buffer:
            return None
        return self.buffer[-1][1]

    def clip(self, seconds: int) -> List[Dict[str, Any]]:
        now = time.time()
        cutoff = now - seconds
        return [p for ts, p in self.buffer if ts >= cutoff]


video_buffer = VideoTopicBuffer(window_seconds=60)