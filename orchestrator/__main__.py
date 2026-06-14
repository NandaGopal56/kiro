from __future__ import annotations

import argparse
import asyncio

from communication_bus import create_bus

from .runner import Orchestrator


def _selected_services(args: argparse.Namespace):
    profile = args.profile
    with_ui = args.with_ui or profile in {"ui", "all"}
    with_stt = args.with_stt or profile in {"voice", "all"}
    with_tts = args.with_tts or profile in {"voice", "all"}

    shared_bus = create_bus()

    from agents import create_service as create_agent_service

    services = [shared_bus, create_agent_service(bus=shared_bus)]

    if with_ui:
        from live_interaction import create_service as create_ui_service

        services.append(
            create_ui_service(bus=shared_bus, host=args.host, port=args.port)
        )

    if with_stt:
        from stt_audio_processor import create_service as create_stt_service

        services.append(create_stt_service(bus=shared_bus))

    if with_tts:
        from tts_audio_processor import create_service as create_tts_service

        services.append(create_tts_service(bus=shared_bus))

    return services


async def _run(args: argparse.Namespace) -> None:
    orchestrator = Orchestrator(_selected_services(args))
    await orchestrator.start()
    print("Orchestrator running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await orchestrator.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run assistant services.")
    parser.add_argument(
        "--profile",
        choices=["core", "ui", "voice", "all"],
        default="core",
        help="Service profile to run. Defaults to safe core.",
    )
    parser.add_argument("--with-ui", action="store_true", help="Enable UI service.")
    parser.add_argument("--with-stt", action="store_true", help="Enable live STT.")
    parser.add_argument("--with-tts", action="store_true", help="Enable live TTS.")
    parser.add_argument("--host", default="localhost", help="UI host.")
    parser.add_argument("--port", type=int, default=8000, help="UI port.")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
