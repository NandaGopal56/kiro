from __future__ import annotations

import argparse
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

from agents.client import gateway
from agents.shared.storage import create_thread, init_db

load_dotenv()


def _split_response_words(response_text: str) -> AsyncGenerator[str, None]:
    async def _generator():
        words = response_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0)

    return _generator()


async def invoke_conversation(
    message: str,
    thread_id: str = "1",
    agent_name: str = "supervisor",
) -> AsyncGenerator[str, None]:
    response_text = await gateway.invoke(
        agent_name=agent_name,
        task=message,
        thread_id=thread_id,
    )
    if response_text:
        async for word in _split_response_words(response_text):
            yield word


async def cli_chat(
    agent_name: str = "supervisor",
    thread_id: str = "1",
) -> None:
    await init_db()
    gateway.save_graphs()
    if thread_id == "1":
        thread_id = str(await create_thread("CLI session"))

    agents = gateway.registered_agents()
    print("Live Chat")
    print(f"Mode: {agent_name}")
    print("Agents: " + ", ".join(agents.keys()))
    print(f"Thread ID: {thread_id}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print("Assistant: ", end="", flush=True)

        async for word in invoke_conversation(user_input, thread_id, agent_name):
            print(word, end="", flush=True)

        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agents package.")
    parser.add_argument(
        "--agent",
        choices=["supervisor", "personal", "deep_research"],
        default="supervisor",
        help="Which agent to invoke.",
    )
    parser.add_argument(
        "--message",
        default="hi",
        help="Send one message and print the response.",
    )
    parser.add_argument("--thread-id", default="1")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat.")
    args = parser.parse_args()

    if args.chat:
        asyncio.run(cli_chat(agent_name=args.agent, thread_id=args.thread_id))
    else:
        async def _run_message() -> None:
            async for token in invoke_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                print(token, end="", flush=True)

        asyncio.run(_run_message())


if __name__ == "__main__":
    main()
