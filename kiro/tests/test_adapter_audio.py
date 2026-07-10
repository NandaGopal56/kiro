import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kiro.adapter import SpeechToAgentAdapter, TextInputSource


class DummyGateway:
    def __init__(self, response: str):
        self.response = response

    async def invoke(self, **kwargs):
        return self.response


class DummyTTS:
    def __init__(self):
        self.calls: list[str] = []

    async def speak(self, text: str) -> str:
        self.calls.append(text)
        return "abc123"


class DummyAudioPlayer:
    def __init__(self):
        self.played: list[str] = []

    async def play_b64(self, audio_b64: str) -> None:
        self.played.append(audio_b64)


def test_response_is_synthesized_and_played():
    gateway = DummyGateway("hello from agent")
    tts = DummyTTS()
    player = DummyAudioPlayer()

    adapter = SpeechToAgentAdapter(
        stt_engine=None,
        agent_gateway=gateway,
        input_source=TextInputSource(text_lines=["hi"]),
        tts_engine=tts,
        audio_player=player,
    )

    events = []

    async def run() -> None:
        async for event, response in adapter.listen_and_respond():
            events.append((event.text, response))

    asyncio.run(run())

    assert events == [("hi", "hello from agent")]
    assert tts.calls == ["hello from agent"]
    assert player.played == ["abc123"]
