from __future__ import annotations

import argparse
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

from agents.client import gateway
from agents.shared.storage import create_thread, init_db
from agents.shared.logging import (
    get_agent_logger,
    log_invoke_start,
)
logger = get_agent_logger("cli", "__main__")

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
    log_invoke_start(logger, agent_name, thread_id=thread_id, mode="cli", task_preview=message)
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
    
    # Initialize the database and create a new thread for the CLI session if needed
    # await init_db()

    # save_graphs() is commented out to avoid saving graphs everytime, as it may not be necessary always.
    # gateway.save_graphs()

    logger.info("Starting CLI chat: agent=%s thread=%s", agent_name, thread_id)
    if thread_id == "1":
        thread_id = str(await create_thread("CLI session"))

    agents = gateway.registered_agents()
    logger.info("Live Chat")
    logger.info("Mode: %s", agent_name)
    logger.info("Agents: %s", ", ".join(agents.keys()))
    logger.info("Thread ID: %s", thread_id)
    logger.info("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("CLI chat interrupted by user")
            logger.info("Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            logger.info("CLI chat exit requested")
            logger.info("Goodbye.")
            break

        logger.debug("User input received: %s", user_input)
        response_accum = ""
        async for word in invoke_conversation(user_input, thread_id, agent_name):
            response_accum += word

        print(f'{agent_name} Agent: {response_accum}', end='\n\n')
        logger.info("Assistant response (agent=%s thread=%s): %s", agent_name, thread_id, response_accum)
        logger.debug("Finished streaming response for input")


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
            logger.info("Running single-message invocation: agent=%s message=%s", args.agent, args.message)
            response_accum = ""
            async for token in invoke_conversation(
                args.message,
                thread_id=args.thread_id,
                agent_name=args.agent,
            ):
                # accumulate tokens and log at the end for non-interactive mode
                response_accum += token

            logger.info("Invocation result (agent=%s thread=%s): %s", args.agent, args.thread_id, response_accum)

        asyncio.run(_run_message())


if __name__ == "__main__":
    main()
