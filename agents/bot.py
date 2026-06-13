# bot.py
#
# Conversation entry point — routes all messages through the Supervisor.
# Self-contained: no dependency on main.py.

import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from agents.deep_research.graph import DeepResearchAgent
from agents.personal.graph import PersonalAgent
from agents.supervisor.grpah import Supervisor

from agents.shared.storage import init_db, create_thread

load_dotenv()


# ---------------------------------------------------------------------------
# Supervisor singleton — built once at first call, reused forever.
# Add new agents here as you build them.
# ---------------------------------------------------------------------------

_supervisor = None


def get_supervisor():
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor(agents={
            "personal":      PersonalAgent(),
            "deep_research": DeepResearchAgent(),
        })
    return _supervisor


# ---------------------------------------------------------------------------
# _extract_response
#
# supervisor.stream() yields {"node": str, "update": dict}.
# These are the nodes that produce user-facing text:
#   "delegate"  — personal agent's full response
#   "ask_user"  — supervisor asking for clarification
#   "finish"    — deep research final answer
# ---------------------------------------------------------------------------

_RESPONSE_NODES = {"delegate", "ask_user", "finish"}


from agents.personal.graph import build_personal_graph
from agents.deep_research.graph import build_research_graph
from agents.supervisor.grpah import build_supervisor_graph
from pathlib import Path

def save_graphs():
    _artifacts = Path(__file__).parent / "artifacts"
    _artifacts.mkdir(parents=True, exist_ok=True)

    personal  = build_personal_graph()
    research  = build_research_graph()
    supervisor = build_supervisor_graph()

    personal.get_graph().draw_mermaid_png(output_file_path=str(_artifacts / "personal_agent.png"))
    research.get_graph().draw_mermaid_png(output_file_path=str(_artifacts / "deep_research_agent.png"))
    supervisor.get_graph().draw_mermaid_png(output_file_path=str(_artifacts / "supervisor.png"))

    # Combined: supervisor only — xray can't reach into agent.run() calls
    # To get a true combined view, subgraphs must be added as graph nodes.
    # For now this is the same as supervisor.png — remove if not needed.
    supervisor.get_graph(xray=True).draw_mermaid_png(output_file_path=str(_artifacts / "full_system_xray.png"))

    print(f"Graphs saved to {_artifacts}")

def _extract_response(node_name: str, update: dict) -> str:
    if node_name not in _RESPONSE_NODES:
        return ""

    if "response" in update and update["response"]:
        return update["response"]

    messages = update.get("messages", [])
    if not isinstance(messages, list):
        messages = [messages]
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)

    return ""


# ---------------------------------------------------------------------------
# invoke_conversation — reusable async generator, same signature as original
# ---------------------------------------------------------------------------

async def invoke_conversation(
    message: str,
    thread_id: str = "1",
) -> AsyncGenerator[str, None]:
    """
    Send a message through the supervisor and stream the response word by word.

    Args:
        message:   The user's message.
        thread_id: Numeric string matching a threads.id row in the DB.

    Yields:
        Words of the response one at a time.
    """
    supervisor    = get_supervisor()
    response_text = ""

    async for chunk in supervisor.stream(message, thread_id=thread_id):
        node_name = chunk.get("node", "")
        update    = chunk.get("update", {})
        text      = _extract_response(node_name, update)
        if text:
            response_text = text
            break

    if response_text:
        words = response_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# cli_chat — for local testing
# ---------------------------------------------------------------------------

async def cli_chat():
    # Initialise DB and create a thread for this CLI session
    await init_db()
    save_graphs()
    thread_id = str(await create_thread("CLI session"))

    supervisor = get_supervisor()
    print("Live Chat")
    print(f"Agents: {', '.join(supervisor.registered_agents().keys())}")
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

        async for word in invoke_conversation(user_input, thread_id):
            print(word, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    asyncio.run(cli_chat())