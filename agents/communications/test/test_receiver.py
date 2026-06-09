import asyncio

import pytest

from agents.communications import receiver


@pytest.mark.asyncio
async def test_on_human_message_schedules_conversation(monkeypatch):
    calls = []
    completed = asyncio.Event()

    async def fake_invoke_conversation(message: str, thread_id: int):
        calls.append((message, thread_id))
        yield "chunk"
        completed.set()

    monkeypatch.setattr(receiver, "invoke_conversation", fake_invoke_conversation)

    receiver.on_human_message("voice/commands", "hello")

    await asyncio.wait_for(completed.wait(), timeout=1)
    assert calls == [("hello", 1)]


@pytest.mark.asyncio
async def test_on_human_message_accepts_canonical_payload_and_publishes_response(monkeypatch):
    completed = asyncio.Event()
    published = []

    class ResponseBus:
        async def publish(self, topic, payload):
            published.append((topic, payload))
            completed.set()

    async def fake_invoke_conversation(message: str, thread_id: int):
        assert message == "hello"
        yield "hi"

    monkeypatch.setattr(receiver, "invoke_conversation", fake_invoke_conversation)

    receiver.on_human_message(
        "voice/commands",
        {"text": "hello", "source": "test"},
        response_bus=ResponseBus(),
    )

    await asyncio.wait_for(completed.wait(), timeout=1)
    assert published == [("voice/commands/llm_response", {"text": "hi"})]
