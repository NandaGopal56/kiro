import asyncio
import base64
import io
import wave

from kiro.audio_player import AudioPlayer


class DummyMicrophone:
    def __init__(self):
        self.paused = False

    async def pause(self) -> None:
        self.paused = True

    async def resume(self) -> None:
        self.paused = False


def _make_wav_b64() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_play_b64_uses_wave_stream(monkeypatch):
    calls = []

    class DummyStream:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def start(self):
            calls.append(("start", self.kwargs))

        def write(self, data):
            calls.append(("write", len(data)))

        def stop(self):
            calls.append(("stop",))

    monkeypatch.setattr("kiro.audio_player.sd.RawOutputStream", DummyStream)

    microphone = DummyMicrophone()
    player = AudioPlayer(microphone=microphone)

    asyncio.run(player.play_b64(_make_wav_b64()))

    assert microphone.paused is False
    assert calls[0][0] == "start"
    assert calls[1][0] == "write"
    assert calls[2][0] == "stop"
