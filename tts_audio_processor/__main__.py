from __future__ import annotations

import argparse
import asyncio

from communication_bus import create_bus

from .fakes import FakeTTSPlayer, fake_synthesizer
from .tts_processor import create_service


async def _run_text(text: str, mock: bool) -> None:
    bus = create_bus()
    if mock:
        player = FakeTTSPlayer()
        service = create_service(bus=bus, synthesizer=fake_synthesizer, player=player)
    else:
        service = create_service(bus=bus)

    await service.start()
    await bus.publish("voice/commands/llm_response", {"text": text})
    await asyncio.sleep(0.1)
    await service.stop()
    await bus.stop()
    if mock:
        print(player.items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TTS audio processor.")
    parser.add_argument("--text", help="Text to synthesize.")
    parser.add_argument("--mock", action="store_true", help="Use fake synthesizer/player.")
    parser.add_argument("--live", action="store_true", help="Use live TTS provider.")
    args = parser.parse_args()

    if args.text:
        asyncio.run(_run_text(args.text, mock=args.mock or not args.live))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
