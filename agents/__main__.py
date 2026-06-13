from __future__ import annotations

import argparse
import asyncio


async def _run_message(message: str, thread_id: int) -> None:
    from .bot import invoke_conversation

    async for token in invoke_conversation(message, thread_id=thread_id):
        print(token, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agents package.")
    parser.add_argument("--message", default='hi', help="Send one message and print the response.")
    parser.add_argument("--thread-id", type=int, default=1)
    parser.add_argument("--chat", action="store_true", help="Start interactive chat.")
    args = parser.parse_args()

    if args.message:
        asyncio.run(_run_message(args.message, args.thread_id))
    elif args.chat:
        from .bot import cli_chat

        asyncio.run(cli_chat())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
