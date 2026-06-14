from __future__ import annotations

import argparse
import asyncio

from communication_bus import create_bus

from .fakes import FakeVoiceProcessor
from .stt_processor import STTAudioProcessorService, create_service


async def _run_mock() -> None:
    bus = create_bus()
    received = asyncio.Event()

    def on_command(topic, payload):
        print(f"{topic}: {payload}")
        received.set()

    bus.subscribe("voice/commands", on_command)
    service = STTAudioProcessorService(
        bus=bus,
        assistant_factory=lambda emit: FakeVoiceProcessor(on_utterance=emit),
    )
    await service.start()
    await asyncio.wait_for(received.wait(), timeout=1)
    await service.stop()
    await bus.stop()


async def _run_live() -> None:
    service = create_service()
    await service.start()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await service.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STT audio processor.")
    parser.add_argument("--mock", action="store_true", help="Run fake STT once.")
    parser.add_argument("--live", action="store_true", help="Run live microphone STT.")
    args = parser.parse_args()

    if args.mock:
        asyncio.run(_run_mock())
    elif args.live:
        asyncio.run(_run_live())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
