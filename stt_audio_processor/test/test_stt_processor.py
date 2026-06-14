import asyncio

import pytest

from communication_bus import create_bus
from stt_audio_processor import create_service
from stt_audio_processor.fakes import FakeVoiceProcessor


def test_import_package_does_not_create_live_assistant():
    import stt_audio_processor

    assert hasattr(stt_audio_processor, "create_service")


@pytest.mark.asyncio
async def test_stt_service_publishes_fake_utterance():
    bus = create_bus()
    received = asyncio.Event()
    payloads = []

    def on_command(topic, payload):
        payloads.append(payload)
        received.set()

    bus.subscribe("voice/commands", on_command)
    service = create_service(
        bus=bus,
        assistant=FakeVoiceProcessor(utterances=["turn on lights"]),
    )
    service.assistant.on_utterance = service._publish_utterance

    await service.start()
    await asyncio.wait_for(received.wait(), timeout=1)
    await service.stop()

    assert payloads == [{"text": "turn on lights", "source": "stt"}]
