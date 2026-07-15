from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from agents.shared.memory import (
    save_tool_call,
    save_tool_result,
    save_message_idempotent
)
from agents.shared.models import get_classifier_llm, get_llm
from agents.shared.tools import personal_tools
from agents.shared.video_buffer import video_buffer
from agents.shared.logging import (
    get_agent_logger,
    log_branch,
    log_node_enter,
    log_node_exit,
    log_route,
)

from .prompts import STEP_CLASSIFIER_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from .state import PersonalState

logger = get_agent_logger("personal", "nodes")

_tool_node = ToolNode(tools=personal_tools)
_llm_with_tools = get_llm().bind_tools(personal_tools)
MAX_HISTORY_TURNS = 6


def _thread_id(config: RunnableConfig, state: PersonalState) -> str:
    return config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")


async def decide_steps(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    log_node_enter(logger, "decide_steps", thread_id=tid, state=state)

    messages      = list(state.get("messages", []))
    last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_user_msg:
        log_node_exit(logger, "decide_steps", thread_id=tid, steps_needed=[])
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

    log_node_exit(logger, "decide_steps", thread_id=tid, steps_needed=steps)
    return {"steps_needed": steps}


def pick_context_steps(state: PersonalState) -> List[str]:
    tid = state.get("thread_id", "")
    log_node_enter(logger, "pick_context_steps", thread_id=tid, state=state)

    step_map = {
        "video_capture":   "grab_video_frame",
        "web_search":      "fetch_web_context",
        "document_search": "fetch_doc_context",
    }
    steps_needed = state.get("steps_needed", [])
    selected     = [step_map[s] for s in steps_needed if s in step_map]
    chosen = selected if selected else ["call_llm"]

    log_route(
        logger,
        "pick_context_steps",
        ",".join(chosen),
        reason="context steps selected by classifier" if selected else "no context needed",
        steps_needed=steps_needed,
        selected=chosen,
    )
    return chosen


async def grab_video_frame(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    log_node_enter(logger, "grab_video_frame", thread_id=tid, state=state)

    frame = video_buffer.latest()
    if frame is None:
        log_branch(logger, "grab_video_frame", "no_frame_available")
        log_node_exit(logger, "grab_video_frame", thread_id=tid, frame_available=False)
        return {"messages": [HumanMessage(content=[{"type": "text", "text": "No camera frame available right now."}])]}

    b64      = base64.b64encode(frame).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    log_node_exit(logger, "grab_video_frame", thread_id=tid, frame_available=True)
    return {"messages": [HumanMessage(content=[
        {"type": "text",      "text": "Here is the current camera view:"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ])]}


async def fetch_web_context(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    log_node_enter(logger, "fetch_web_context", thread_id=tid, state=state)
    log_node_exit(logger, "fetch_web_context", thread_id=tid, status="stub")
    return {}


async def fetch_doc_context(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    log_node_enter(logger, "fetch_doc_context", thread_id=tid, state=state)
    log_node_exit(logger, "fetch_doc_context", thread_id=tid, status="stub")
    return {}


def join_context(state: PersonalState) -> PersonalState:
    tid = state.get("thread_id", "")
    log_node_enter(logger, "join_context", thread_id=tid, state=state)
    log_node_exit(logger, "join_context", thread_id=tid)
    return state


async def call_llm(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = _thread_id(config, state)
    messages  = list(state.get("messages", []))
    summary   = state.get("summary", "")
    log_node_enter(
        logger,
        "call_llm",
        thread_id=thread_id,
        state=state,
        summary_len=len(summary) if summary else 0,
        message_count=len(messages),
    )

    prompt: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    if summary:
        prompt.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
    prompt.extend(_trim_to_recent_turns(messages, MAX_HISTORY_TURNS))

    response         = await _llm_with_tools.ainvoke(prompt, config=config)
    content          = response.content if isinstance(response.content, str) else ""
    assistant_msg_id = await save_message_idempotent(thread_id, "assistant", content)

    tool_calls = getattr(response, "tool_calls", []) or []
    log_branch(
        logger,
        "call_llm",
        "response_ready",
        assistant_msg_id=assistant_msg_id,
        tool_call_count=len(tool_calls),
    )

    for tc in tool_calls:
        await save_tool_call(message_id=assistant_msg_id, call_id=tc.get("id", ""), tool_input=tc)

    log_node_exit(
        logger,
        "call_llm",
        thread_id=thread_id,
        response_len=len(content),
        tool_call_count=len(tool_calls),
    )
    return {
        "messages":                     [response],
        "current_assistant_message_id": assistant_msg_id,
    }


def what_next(state: PersonalState) -> str:
    tid = state.get("thread_id", "")
    messages = list(state.get("messages", []))
    if not messages:
        target = "done"
        log_route(logger, "what_next", target, reason="no messages", thread_id=tid)
        return target

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        target = "run_tools"
        log_route(
            logger,
            "what_next",
            target,
            reason="assistant issued tool calls",
            thread_id=tid,
            tool_call_count=len(last_msg.tool_calls),
        )
        return target

    if len(messages) > 12:
        target = "compress_history"
        log_route(
            logger,
            "what_next",
            target,
            reason="message history exceeds threshold",
            thread_id=tid,
            message_count=len(messages),
        )
        return target

    log_route(logger, "what_next", "done", reason="conversation turn complete", thread_id=tid)
    return "done"


async def run_tools(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    log_node_enter(logger, "run_tools", thread_id=tid, state=state)

    messages   = list(state.get("messages", []))
    last_msg   = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) if last_msg else None
    if not tool_calls:
        log_node_exit(logger, "run_tools", thread_id=tid, tool_messages=0)
        return {}

    result        = await _tool_node.ainvoke({"messages": messages}, config)
    tool_messages = result.get("messages", [])

    assistant_msg_id = state.get("current_assistant_message_id")
    for tm in tool_messages:
        if assistant_msg_id:
            await save_tool_result(message_id=assistant_msg_id, call_id=tm.tool_call_id, output=tm.content)

    log_node_exit(logger, "run_tools", thread_id=tid, tool_messages=len(tool_messages))
    return {"messages": tool_messages}


async def compress_history(state: PersonalState, config: RunnableConfig) -> Dict[str, Any]:
    tid = _thread_id(config, state)
    log_node_enter(logger, "compress_history", thread_id=tid, state=state)

    existing_summary = state.get("summary", "")
    messages         = list(state.get("messages", []))

    msg_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
        f"{m.content if isinstance(m.content, str) else str(m.content)}"
        for m in messages if isinstance(m, (HumanMessage, AIMessage))
    )

    prompt   = SUMMARY_PROMPT.format(existing_summary=existing_summary or "None yet.", new_messages=msg_text)
    response = await get_llm().ainvoke([HumanMessage(content=prompt)])
    summary_len = len(response.content) if response and getattr(response, "content", None) else 0
    log_node_exit(logger, "compress_history", thread_id=tid, summary_len=summary_len)
    return {"summary": response.content}


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
