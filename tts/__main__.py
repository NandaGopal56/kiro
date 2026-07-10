"""CLI entry point for the TTS module.

Usage:
    python -m tts --text "Hello world"
    python -m tts --provider sarvam
"""

from __future__ import annotations

import argparse
import asyncio

from tts.engine import TTSEngine


def _build_provider(name: str, voice: str | None, output_path: str | None):
    if name == "sarvam":
        from tts.providers.sarvam import SarvamTTS

        return SarvamTTS(
            speaker=voice or "shubh",
            output_path=output_path,
        )

    raise ValueError(f"Unknown provider: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tts",
        description="Run the TTS engine from the CLI.",
    )

    parser.add_argument(
        "--provider",
        default="sarvam",
        choices=["sarvam"],
        help="TTS provider to use",
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize",
    )

    parser.add_argument(
        "--voice",
        default="shubh",
        help="Speaker/voice name",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output audio file",
    )

    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    provider = _build_provider(args.provider, args.voice, args.output)
    engine = TTSEngine(provider)

    print(f"[{args.provider}] Speaking: {args.text}")

    try:
        await engine.speak(args.text)

        if getattr(provider, "last_output_path", None):
            print(f"Saved to: {provider.last_output_path}")

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