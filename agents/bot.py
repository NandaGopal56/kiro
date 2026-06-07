
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from .graph import build_workflow

load_dotenv()
workflow = build_workflow()


class TokenStreamProcessor:
    def __init__(self, max_buffer_size: int = 50):
        self.buffer = ""
        self.max_buffer_size = max_buffer_size

    async def process_chunk(self, chunk) -> AsyncGenerator[str, None]:
        if chunk[1].get("langgraph_node") != "call_model":
            return

        for msg in chunk:
            token = msg[0] if isinstance(msg, tuple) else msg
            if hasattr(token, "content") and token.content:
                self.buffer += token.content

                if (
                    any(p in self.buffer for p in [".", "!", "?", ","])
                    or len(self.buffer) >= self.max_buffer_size
                ):
                    yield self.buffer
                    self.buffer = ""

    async def finalize(self) -> AsyncGenerator[str, None]:
        if self.buffer.strip():
            yield self.buffer
            self.buffer = ""


# ---------------- CORE INVOCATION ---------------- #

async def invoke_conversation(
    message: str,
    thread_id: int = 1,
) -> AsyncGenerator[str, None]:
    """
    Reusable streaming invocation.
    Works for CLI, API, WebSocket, TTS, etc.
    """
    processor = TokenStreamProcessor()

    async for mode, data in workflow.astream(
        {"messages": [HumanMessage(content=message)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode=["messages"],
    ):
        if mode == "messages":
            async for chunk in processor.process_chunk(data):
                yield chunk

    async for chunk in processor.finalize():
        yield chunk


# ---------------- CLI ONLY ---------------- #

async def cli_chat():
    print("Live Chat (type 'exit' to quit)\n")
    thread_id = 1

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        print("Assistant: ", end="", flush=True)

        async for token in invoke_conversation(user_input, thread_id):
            print(token, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    asyncio.run(cli_chat())