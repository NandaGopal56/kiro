"""
CLI entry point for the STT module.

Usage:
    python -m stt                          # sarvam, en-IN (defaults)
    python -m stt --provider sarvam --language hi-IN
"""

from __future__ import annotations

import argparse
import asyncio

from stt.engine import STTEngine


def _build_provider(name: str, language: str):
    if name == "sarvam":
        from stt.providers.sarvam import SarvamSTT

        return SarvamSTT(language_code=language)

    if name == "elevenlabs":
        raise NotImplementedError("ElevenLabs STT provider is not implemented yet.")

    raise ValueError(f"Unknown provider: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="stt", description="Run the STT engine from the CLI.")
    parser.add_argument(
        "--provider",
        default="sarvam",
        choices=["sarvam", "elevenlabs"],
        help="STT provider to use (default: sarvam)",
    )
    parser.add_argument(
        "--language",
        default="en-IN",
        help="Language code passed to the provider (default: en-IN)",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    provider = _build_provider(args.provider, args.language)
    engine = STTEngine(provider, language=args.language)

    print(f"[{args.provider}] Listening... (Ctrl+C to stop)\n")
    try:
        async for event in engine.stream():
            print(event)
    finally:
        await engine.close()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()