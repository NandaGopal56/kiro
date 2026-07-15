# agents/shared/memory.py
#
# Thin async wrappers around the SQLite storage layer.
# Every agent calls only these four functions — the storage backend
# is swappable here without touching any agent code.

from typing import Any, Dict, List, Optional

from agents.shared.logging import log_save
from agents.shared.storage import (
    add_message     as _add_message,
    add_tool_call   as _add_tool_call,
    update_tool_result as _update_tool_result,
    get_full_thread as _get_full_thread,
)


def _preview(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [len={len(text)}]"


async def save_tool_call(
    message_id: int,
    call_id: str,
    tool_input: Dict[str, Any],
) -> None:
    """Attach a tool call to an assistant message."""
    tool_name = tool_input.get("name", "unknown")
    log_save(
        "tool_call",
        "insert",
        message_id=message_id,
        call_id=call_id,
        tool=tool_name,
    )
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
    log_save(
        "tool_result",
        "update",
        message_id=message_id,
        call_id=call_id,
        output_len=len(output) if output else 0,
        output_preview=_preview(output) if output else "",
    )
    await _update_tool_result(
        message_id=message_id,
        call_id=call_id,
        tool_output=output,
    )


async def load_thread(thread_id: str) -> List[Dict[str, Any]]:
    """Load the full message history for a thread."""
    return await _get_full_thread(int(thread_id))


async def save_message_idempotent(
    thread_id: str,
    role: str,
    content: str = "",
) -> int:
    """Save a message only if it is not already the last message in the thread history."""
    db_history = await _get_full_thread(int(thread_id))
    if db_history:
        last_msg = db_history[-1]
        if last_msg.get("role") == role and last_msg.get("content") == content:
            message_id = last_msg.get("message_id")
            log_save(
                "message",
                "skip_duplicate",
                thread_id=thread_id,
                role=role,
                message_id=message_id,
                content_len=len(content),
            )
            return message_id

    message_id = await _add_message(int(thread_id), role, content)
    log_save(
        "message",
        "insert",
        thread_id=thread_id,
        role=role,
        message_id=message_id,
        content_len=len(content),
        content_preview=_preview(content),
    )
    return message_id


async def rebuild_messages_from_db(thread_id: str) -> list:
    """Load and reconstruct message history from database into LangChain message objects."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    import json

    db_history = await _get_full_thread(int(thread_id))
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

    log_save(
        "thread",
        "rebuild_messages",
        thread_id=thread_id,
        loaded_count=len(loaded_messages),
    )
    return loaded_messages
