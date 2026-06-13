from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import RunnableConfig

from agents.base import BaseAgent
from agents.shared.models import get_classifier_llm

from .prompts import CLARIFICATION_PREFIX, ROUTER_PROMPT
from .state import SupervisorState


def make_route_request(agents: Dict[str, BaseAgent]):
    """
    Build a router node bound to the provided agent registry.
    This avoids hidden module-global state.
    """

    def _agent_list_for_prompt() -> str:
        lines = []
        for agent in agents.values():
            info = agent.info
            long = "yes" if info.can_handle_long_tasks else "no"
            lines.append(f"- {info.agent_id}: {info.description} (long tasks: {long})")
        return "\n".join(lines) if lines else "- personal: General purpose assistant"

    async def route_request(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
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
            data = {
                "chosen_agent": "personal",
                "reason": "Could not parse routing decision - defaulting to personal.",
                "needs_clarification": False,
                "clarification_question": "",
            }

        chosen = data.get("chosen_agent", "personal")

        if chosen and chosen not in agents:
            unknown = chosen
            chosen = "personal"
            data["reason"] += f" (unknown agent '{unknown}' requested - fell back to personal)"

        return {
            "chosen_agent": chosen,
            "routing_reason": data.get("reason", ""),
            "needs_clarification": data.get("needs_clarification", False),
            "clarification_question": data.get("clarification_question", ""),
        }

    return route_request


async def ask_user(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    question = state.get("clarification_question", "Could you give me more details?")
    message = CLARIFICATION_PREFIX + question

    return {
        "response": message,
        "messages": [AIMessage(content=message)],
    }


def what_to_do(state: SupervisorState) -> str:
    if state.get("needs_clarification", False):
        return "ask_user"
    return state.get("chosen_agent", "personal")
