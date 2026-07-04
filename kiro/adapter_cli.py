from __future__ import annotations

import argparse
import asyncio
import datetime as dt

from agents.client import gateway
from stt.engine import STTEngine
from stt.providers.sarvam import SarvamSTT
from kiro.adapter import SpeechToAgentAdapter


async def run_streaming(agent_name: str = "supervisor", thread_id: str = "1", language: str = "en-IN") -> None:
    provider = SarvamSTT(language_code=language)
    stt_engine = STTEngine(provider, language=language)
    adapter = SpeechToAgentAdapter(
        stt_engine=stt_engine,
        agent_gateway=gateway,
        agent_name=agent_name,
        thread_id=thread_id,
        language=language,
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
    args = parser.parse_args()

    try:
        asyncio.run(run_streaming(agent_name=args.agent, thread_id=args.thread_id, language=args.language))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
