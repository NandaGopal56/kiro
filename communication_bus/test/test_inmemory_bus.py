import asyncio

import pytest

from communication_bus import create_bus


@pytest.mark.asyncio
async def test_sync_callback_receives_message():
    bus = create_bus()
    received = asyncio.Event()
    payloads = []

    def callback(topic, payload):
        payloads.append((topic, payload))
        received.set()

    await bus.start()
    bus.subscribe("topic", callback)
    await bus.publish("topic", {"value": 1})
    await asyncio.wait_for(received.wait(), timeout=1)
    await bus.stop()

    assert payloads == [("topic", {"value": 1})]


@pytest.mark.asyncio
async def test_async_callback_receives_message():
    bus = create_bus()
    received = asyncio.Event()

    async def callback(topic, payload):
        assert topic == "topic"
        assert payload == "hello"
        received.set()

    await bus.start()
    bus.subscribe("topic", callback)
    await bus.publish("topic", "hello")
    await asyncio.wait_for(received.wait(), timeout=1)
    await bus.stop()


@pytest.mark.asyncio
async def test_unsubscribe_removes_callback():
    bus = create_bus()
    calls = []

    def callback(topic, payload):
        calls.append(payload)

    await bus.start()
    bus.subscribe("topic", callback)
    bus.unsubscribe("topic", callback)
    await bus.publish("topic", "ignored")
    await asyncio.sleep(0.05)
    await bus.stop()

    assert calls == []


@pytest.mark.asyncio
async def test_connect_disconnect_are_idempotent():
    bus = create_bus()

    await bus.connect()
    await bus.connect()
    assert bus.running is True

    await bus.disconnect()
    await bus.disconnect()
    assert bus.running is False
