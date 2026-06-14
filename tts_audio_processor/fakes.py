from __future__ import annotations


class FakeTTSPlayer:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.items = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def enqueue(self, segment) -> None:
        self.items.append(segment)


def fake_synthesizer(text: str):
    return {"audio": text}
