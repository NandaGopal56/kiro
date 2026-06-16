from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from agents.shared.memory import (
    load_thread,
    save_message,
    save_tool_call,
    save_tool_result,
    save_message_idempotent,
    rebuild_messages_from_db,
)
from agents.shared.models import get_classifier_llm, get_llm
from agents.shared.tools import personal_tools
from agents.shared.video_buffer import video_buffer

from .prompts import STEP_CLASSIFIER_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from .state import PersonalState

# Built-in LangGraph tool executor
_tool_node = ToolNode(tools=personal_tools)

# Main LLM with tools bound — created once at import time
_llm_with_tools = get_llm().bind_tools(personal_tools)

# How many past user turns to keep in the prompt window (beyond the summary)
MAX_HISTORY_TURNS = 6


# ---------------------------------------------------------------------------
# Node 1 — load_history
#
# Only saves the incoming message to external storage.
# Does NOT reload history from storage — LangGraph's MemorySaver checkpointer
# already carries the full message history across turns automatically.
#
# When you switch to a persistent checkpointer (AsyncSqliteSaver, RedisSaver),
# this node can be removed entirely — the checkpointer handles it.
# ---------------------------------------------------------------------------

async def load_history(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "")
    messages  = list(state.get("messages", []))

    last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    user_msg_id   = None
    if last_user_msg:
        text        = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)
        user_msg_id = await save_message_idempotent(thread_id, "user", text)

    # Rebuild history from DB and replace the messages list in the state
    loaded_messages = await rebuild_messages_from_db(thread_id)
    all_messages = [RemoveMessage(id=m.id) for m in messages if m.id] + loaded_messages

    return {
        "thread_id":               thread_id,
        "current_user_message_id": user_msg_id,
        "messages":                all_messages,
    }


# ---------------------------------------------------------------------------
# Node 2 — decide_steps
# Asks a cheap classifier which context steps (if any) to run before answering.
# Returns a list like ["web_search"] or [] for plain conversation.
# ---------------------------------------------------------------------------

async def decide_steps(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    messages      = list(state.get("messages", []))
    last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        return {"steps_needed": []}

    user_text = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)

    response = await get_classifier_llm().ainvoke([
        SystemMessage(content=STEP_CLASSIFIER_PROMPT),
        HumanMessage(content=user_text),
    ])

    try:
        steps = json.loads(response.content)
        steps = steps if isinstance(steps, list) else []
    except (json.JSONDecodeError, ValueError):
        steps = []

    return {"steps_needed": steps}


# ---------------------------------------------------------------------------
# Router — pick_context_steps
# Called after decide_steps. Returns the list of parallel node names to run,
# or ["call_llm"] when no context gathering is needed.
# ---------------------------------------------------------------------------

def pick_context_steps(state: PersonalState) -> List[str]:
    step_map = {
        "video_capture":   "grab_video_frame",
        "web_search":      "fetch_web_context",
        "document_search": "fetch_doc_context",
    }
    steps_needed = state.get("steps_needed", [])
    selected     = [step_map[s] for s in steps_needed if s in step_map]
    return selected if selected else ["call_llm"]


# ---------------------------------------------------------------------------
# Node 3a — grab_video_frame
# ---------------------------------------------------------------------------

async def grab_video_frame(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    frame = video_buffer.latest()
    if frame is None:
        return {"messages": [HumanMessage(content=[{"type": "text", "text": "No camera frame available right now."}])]}

    b64      = base64.b64encode(frame).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    return {"messages": [HumanMessage(content=[
        {"type": "text",      "text": "Here is the current camera view:"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ])]}


# ---------------------------------------------------------------------------
# Node 3b — fetch_web_context
# Hook for proactive web retrieval before the LLM call.
# ---------------------------------------------------------------------------

async def fetch_web_context(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    # TODO: run web_search(last user message) and inject results as a SystemMessage
    return {}


# ---------------------------------------------------------------------------
# Node 3c — fetch_doc_context
# Hook for proactive RAG retrieval before the LLM call.
# ---------------------------------------------------------------------------

async def fetch_doc_context(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    # TODO: run document_search(last user message) and inject results as a SystemMessage
    return {}


# ---------------------------------------------------------------------------
# Node 3d — join_context
# Waits for all parallel context branches before calling the LLM.
# ---------------------------------------------------------------------------

def join_context(state: PersonalState) -> PersonalState:
    return state


# ---------------------------------------------------------------------------
# Node 4 — call_llm
# Builds prompt from full checkpointer history, calls LLM, saves response.
# ---------------------------------------------------------------------------

async def call_llm(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "")
    messages  = list(state.get("messages", []))
    summary   = state.get("summary", "")

    # Build prompt: system message, optional summary, then recent history
    prompt: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
    prompt.extend(_trim_to_recent_turns(messages, MAX_HISTORY_TURNS))

    response         = await _llm_with_tools.ainvoke(prompt, config=config)
    content          = response.content if isinstance(response.content, str) else ""
    assistant_msg_id = await save_message(thread_id, "assistant", content)

    for tc in getattr(response, "tool_calls", []) or []:
        await save_tool_call(message_id=assistant_msg_id, call_id=tc.get("id", ""), tool_input=tc)

    return {
        "messages":                     [response],
        "current_assistant_message_id": assistant_msg_id,
    }


# ---------------------------------------------------------------------------
# Router — what_next
# Called after call_llm. Decides: run tools, compress history, or finish.
# ---------------------------------------------------------------------------

def what_next(state: PersonalState) -> str:
    messages = list(state.get("messages", []))
    if not messages:
        return "done"
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "run_tools"
    if len(messages) > 12:
        return "compress_history"
    return "done"


# ---------------------------------------------------------------------------
# Node 5 — run_tools
# Executes tool calls the LLM made, saves results, loops back to call_llm.
# ---------------------------------------------------------------------------

async def run_tools(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    messages   = list(state.get("messages", []))
    last_msg   = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) if last_msg else None
    if not tool_calls:
        return {}

    result        = await _tool_node.ainvoke({"messages": messages}, config)
    tool_messages = result.get("messages", [])

    assistant_msg_id = state.get("current_assistant_message_id")
    for tm in tool_messages:
        if assistant_msg_id:
            await save_tool_result(message_id=assistant_msg_id, call_id=tm.tool_call_id, output=tm.content)

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Node 6 — compress_history
# Produces a rolling summary to keep prompts short on long conversations.
# ---------------------------------------------------------------------------

async def compress_history(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    existing_summary = state.get("summary", "")
    messages         = list(state.get("messages", []))

    msg_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
        f"{m.content if isinstance(m.content, str) else str(m.content)}"
        for m in messages if isinstance(m, (HumanMessage, AIMessage))
    )

    prompt   = SUMMARY_PROMPT.format(existing_summary=existing_summary or "None yet.", new_messages=msg_text)
    response = await get_llm().ainvoke([HumanMessage(content=prompt)])
    return {"summary": response.content}


# ---------------------------------------------------------------------------
# Internal helper — _trim_to_recent_turns
# Keeps only the last N human turns so the context window stays manageable.
# ---------------------------------------------------------------------------

def _trim_to_recent_turns(messages: Sequence[Any], max_turns: int) -> List[Any]:
    if not messages:
        return []

    last_human_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
        None,
    )
    if last_human_idx is None:
        return list(messages)

    kept, turns_seen = [], 0
    for msg in reversed(messages[:last_human_idx]):
        kept.append(msg)
        if isinstance(msg, HumanMessage):
            turns_seen += 1
            if turns_seen >= max_turns:
                break

    kept.reverse()
    kept.extend(messages[last_human_idx:])
    return kept