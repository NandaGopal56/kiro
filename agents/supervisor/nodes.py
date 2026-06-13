# agents/supervisor/nodes.py
#
# The supervisor's three nodes:
#
#   route_request   → decides which agent should handle the user's message
#   delegate        → hands the task to the chosen agent and waits for the answer
#   ask_user        → asks for clarification when the router is unsure
#
# The registered agents are injected at startup via `set_agent_registry()`.
# Nodes themselves are plain async functions — easy to read and test.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import RunnableConfig

from agents.base import BaseAgent
from agents.shared.models import get_classifier_llm

from .prompts import (
    CLARIFICATION_PREFIX,
    FALLBACK_RESPONSE,
    ROUTER_PROMPT,
)
from .state import SupervisorState

# ---------------------------------------------------------------------------
# Agent registry — injected at startup, not hardcoded here
# ---------------------------------------------------------------------------

_agents: Dict[str, BaseAgent] = {}


def set_agent_registry(agents: Dict[str, BaseAgent]) -> None:
    """
    Called once at startup with a dict of {agent_id: agent_instance}.
    Nodes read from this dict — no imports of individual agents.
    """
    _agents.clear()
    _agents.update(agents)


def _agent_list_for_prompt() -> str:
    """
    Format agent descriptions for injection into the routing prompt.
    Example output:
      - personal: Handles everyday chat... (long tasks: no)
      - deep_research: Handles complex research... (long tasks: yes)
    """
    lines = []
    for agent in _agents.values():
        info = agent.info
        long = "yes" if info.can_handle_long_tasks else "no"
        lines.append(f"- {info.agent_id}: {info.description} (long tasks: {long})")
    return "\n".join(lines) if lines else "- personal: General purpose assistant"


# ---------------------------------------------------------------------------
# Node 1 — route_request
# Asks the classifier LLM which agent should handle this input.
# ---------------------------------------------------------------------------

async def route_request(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Read the user's message and decide which agent handles it.
    Writes chosen_agent, routing_reason, and needs_clarification to state.
    """
    user_input = state.get("user_input", "")

    prompt = ROUTER_PROMPT.format(agent_list=_agent_list_for_prompt())

    llm = get_classifier_llm()
    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_input),
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        # If parsing fails, default to personal agent
        data = {
            "chosen_agent":          "personal",
            "reason":                "Could not parse routing decision — defaulting to personal.",
            "needs_clarification":   False,
            "clarification_question": "",
        }

    chosen = data.get("chosen_agent", "personal")

    # Guard: if the model hallucinated an agent ID, fall back to personal
    if chosen and chosen not in _agents:
        chosen = "personal"
        data["reason"] += f" (unknown agent '{chosen}' requested — fell back to personal)"

    return {
        "chosen_agent":          chosen,
        "routing_reason":        data.get("reason", ""),
        "needs_clarification":   data.get("needs_clarification", False),
        "clarification_question": data.get("clarification_question", ""),
    }


# ---------------------------------------------------------------------------
# Node 2 — delegate
# Hands the task to the chosen agent and stores the response.
# ---------------------------------------------------------------------------

async def delegate(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invoke the chosen agent with the user's task.
    Waits for the agent to finish and stores its response.
    """
    agent_id   = state.get("chosen_agent", "personal")
    user_input = state.get("user_input", "")
    thread_id  = state.get("thread_id", "default")

    agent = _agents.get(agent_id) or _agents.get("personal")

    if agent is None:
        return {
            "response": FALLBACK_RESPONSE,
            "messages": [AIMessage(content=FALLBACK_RESPONSE)],
        }

    try:
        answer = await agent.run(
            task=user_input,
            thread_id=thread_id,
            config=RunnableConfig(configurable={"thread_id": thread_id}),
        )
    except Exception as exc:
        answer = f"The {agent_id} agent encountered an error: {exc}"

    return {
        "response": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Node 3 — ask_user
# Returns a clarification question when the router is unsure.
# ---------------------------------------------------------------------------

async def ask_user(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    When the router cannot confidently pick an agent, ask the user to clarify.
    The conversation continues after the user replies.
    """
    question = state.get("clarification_question", "Could you give me more details?")
    message  = CLARIFICATION_PREFIX + question

    return {
        "response": message,
        "messages": [AIMessage(content=message)],
    }


# ---------------------------------------------------------------------------
# Router — what_to_do
# Called after route_request to pick the next node.
# ---------------------------------------------------------------------------

def what_to_do(state: SupervisorState) -> str:
    """
    After routing:
    - If we need more information → ask the user
    - Otherwise → delegate to the chosen agent
    """
    if state.get("needs_clarification", False):
        return "ask_user"
    return "delegate"