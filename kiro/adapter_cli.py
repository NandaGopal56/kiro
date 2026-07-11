#adapter_cli.py
from __future__ import annotations

import argparse
import asyncio
import datetime as dt

from agents.client import gateway
from kiro.adapter import SpeechToAgentAdapter, TextInputSource
from kiro.audio_player import AudioPlayer
from kiro.microphone import KiroMicrophone
from stt.engine import STTEngine
from stt.providers.sarvam import SarvamSTT
from tts.engine import TTSEngine
from tts.providers.sarvam import SarvamTTS


async def run_streaming(
    agent_name: str = "supervisor",
    thread_id: str = "1",
    language: str = "en-IN",
    input_mode: str = "audio",
) -> None:
    tts_provider = SarvamTTS(language_code=language)
    tts_engine = TTSEngine(tts_provider)

    if input_mode == "text":
        adapter = SpeechToAgentAdapter(
            stt_engine=None,
            agent_gateway=gateway,
            agent_name=agent_name,
            thread_id=thread_id,
            language=language,
            input_source=TextInputSource(language=language),
            tts_engine=tts_engine,
            audio_player=None,
        )
        print(f"Text input mode ({language}). Type a message and press Enter. Ctrl+C to stop")
    else:
        microphone = KiroMicrophone()
        provider = SarvamSTT(language_code=language, microphone=microphone)
        stt_engine = STTEngine(provider, language=language)
        # Wire the mic itself (not the STT engine) into the player so
        # playback can pause/resume capture directly.
        audio_player = AudioPlayer(microphone=microphone)
        adapter = SpeechToAgentAdapter(
            stt_engine=stt_engine,
            agent_gateway=gateway,
            agent_name=agent_name,
            thread_id=thread_id,
            language=language,
            tts_engine=tts_engine,
            audio_player=audio_player,
        )
        print(f"Listening for speech ({language})... Ctrl+C to stop")

    try:
        async for event, response in adapter.listen_and_respond():
            input_timestamp = dt.datetime.now().strftime("%H:%M:%S")
            print(f"[{input_timestamp}] input: {event.text}")

            output_timestamp = dt.datetime.now().strftime("%H:%M:%S")
            print(f"[{output_timestamp}] output: {response}")
    finally:
        await adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge STT to agents using the kiro adapter")
    parser.add_argument("--agent", default="supervisor", choices=["supervisor", "personal", "deep_research"])
    parser.add_argument("--thread-id", default="1")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--input-mode", default="audio", choices=["audio", "text"], help="Choose input modality")
    args = parser.parse_args()

    try:
        asyncio.run(
            run_streaming(
                agent_name=args.agent,
                thread_id=args.thread_id,
                language=args.language,
                input_mode=args.input_mode,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()