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


async def save_message_idempotent(
    thread_id: str,
    role: str,
    content: str = "",
) -> int:
    """Save a message only if it is not already the last message in the thread history."""
    db_history = await load_thread(thread_id)
    if db_history:
        last_msg = db_history[-1]
        if last_msg.get("role") == role and last_msg.get("content") == content:
            return last_msg.get("message_id")
    return await save_message(thread_id, role, content)


async def rebuild_messages_from_db(thread_id: str) -> list:
    """Load and reconstruct message history from database into LangChain message objects."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    import json

    db_history = await load_thread(thread_id)
    loaded_messages = []
    for msg in db_history:
        role = msg.get("role")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "user":
            loaded_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tcs = []
            for tc in tool_calls:
                tc_dict = tc.get("tool_input_json")
                if tc_dict:
                    tcs.append({
                        "name": tc_dict.get("name"),
                        "args": tc_dict.get("args"),
                        "id": tc.get("call_id"),
                        "type": "tool_call"
                    })
            loaded_messages.append(AIMessage(content=content, tool_calls=tcs))

            for tc in tool_calls:
                call_id = tc.get("call_id")
                tool_output = tc.get("tool_output")
                if tool_output is not None:
                    if isinstance(tool_output, str):
                        try:
                            tool_output = json.loads(tool_output)
                        except json.JSONDecodeError:
                            pass

                    if not isinstance(tool_output, str):
                        tool_output_str = json.dumps(tool_output)
                    else:
                        tool_output_str = tool_output
                    loaded_messages.append(ToolMessage(content=tool_output_str, tool_call_id=call_id))
    return loaded_messages