import asyncio

import pytest

from communication_bus import create_bus
from tts_audio_processor import create_service
from tts_audio_processor.fakes import FakeTTSPlayer, fake_synthesizer
from tts_audio_processor.text_reader import extract_response_text


def test_extract_response_text_accepts_canonical_and_legacy_payloads():
    assert extract_response_text({"text": "hello"}) == "hello"
    assert extract_response_text({"llm_response": "legacy"}) == "legacy"
    assert extract_response_text("raw") == "raw"


@pytest.mark.asyncio
async def test_tts_service_handles_bus_message_with_fakes():
    bus = create_bus()
    player = FakeTTSPlayer()
    service = create_service(
        bus=bus,
        synthesizer=fake_synthesizer,
        player=player,
    )

    await service.start()
    await bus.publish("voice/commands/llm_response", {"text": "hello"})
    await asyncio.sleep(0.05)
    await service.stop()

    assert player.started is True
    assert player.stopped is True
    assert player.items == [{"audio": "hello"}]
