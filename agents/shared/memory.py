# agents/shared/memory.py
#
# Thin async wrappers around the SQLite storage layer.
# Every agent calls only these four functions — the storage backend
# is swappable here without touching any agent code.
#
# thread_id throughout the agents is a str (LangGraph convention).
# The SQLite storage uses int. We cast at this boundary so neither
# side needs to know about the other's type.

from typing import Any, Dict, List, Optional

from agents.shared.storage import (
    add_message     as _add_message,
    add_tool_call   as _add_tool_call,
    update_tool_result as _update_tool_result,
    get_full_thread as _get_full_thread,
)


async def save_message(
    thread_id: str,
    role: str,        # "user" or "assistant"
    content: str = "",
) -> int:
    """Save a message and return its integer DB row ID."""
    return await _add_message(int(thread_id), role, content)


async def save_tool_call(
    message_id: int,
    call_id: str,
    tool_input: Dict[str, Any],
) -> None:
    """Attach a tool call to an assistant message."""
    await _add_tool_call(
        message_id=message_id,
        tool_input_json=tool_input,
        call_id=call_id,
    )


async def save_tool_result(
    message_id: int,
    call_id: str,
    output: str,
) -> None:
    """Store the result of a tool call."""
    await _update_tool_result(
        message_id=message_id,
        call_id=call_id,
        tool_output=output,
    )


async def load_thread(thread_id: str) -> List[Dict[str, Any]]:
    """
    Load the full message history for a thread.
    Returns a list of dicts — each with: role, content, tool_calls.
    """
    return await _get_full_thread(int(thread_id))