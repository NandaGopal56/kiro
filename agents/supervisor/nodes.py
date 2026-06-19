from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.types import RunnableConfig

from agents.shared.memory import save_message_idempotent, rebuild_messages_from_db
from agents.shared.checkpointer import load_previous_state, merge_with_new_messages

from agents.base import BaseAgent
from agents.shared.models import get_classifier_llm

from .prompts import CLARIFICATION_PREFIX, ROUTER_PROMPT
from .state import SupervisorState


def make_route_request(agents: Dict[str, BaseAgent], parent_graph=None):
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
        thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")

        # Save user message to database
        await save_message_idempotent(thread_id, "user", user_input)

        # Load history to provide routing context
        loaded_messages = await rebuild_messages_from_db(thread_id)

        prompt = ROUTER_PROMPT.format(agent_list=_agent_list_for_prompt())

        llm = get_classifier_llm()
        llm_messages = [SystemMessage(content=prompt)]
        llm_messages.extend(loaded_messages[-10:])  # Give the classifier context of the last few turns

        response = await llm.ainvoke(llm_messages)

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

        messages = list(state.get("messages", []))
        all_messages = [RemoveMessage(id=m.id) for m in messages if m.id] + loaded_messages

        # Attempt to load previous checkpoint from both the agent's own DB
        # (from standalone runs) and from the supervisor's DB (previous
        # supervised runs). Prefer supervisor state when present.
        try:
            previous_agent_state = await load_previous_state(agents[chosen].graph, thread_id, chosen)
        except Exception:
            previous_agent_state = None

        try:
            previous_supervisor_state = None
            if parent_graph is not None:
                previous_supervisor_state = await load_previous_state(parent_graph, thread_id, "supervisor")
        except Exception:
            previous_supervisor_state = None

        print(f"DEBUG: route_request previous_agent_state_present={bool(previous_agent_state)} previous_supervisor_state_present={bool(previous_supervisor_state)}")

        # Merge precedence: supervisor state (most recent) overrides agent DB.
        # Start with agent DB state, then overlay supervisor state, then add
        # the new routing messages.
        merged_base = merge_with_new_messages(previous_agent_state, {})
        merged_with_supervisor = merge_with_new_messages(merged_base, previous_supervisor_state or {})
        new_state_values = {"messages": all_messages, "thread_id": thread_id}
        merged_agent_state = merge_with_new_messages(merged_with_supervisor, new_state_values)

        print(f"DEBUG: route_request merged_messages_count={len(merged_agent_state.get('messages', [])) if merged_agent_state else 0}")

        result = {
            "chosen_agent": chosen,
            "routing_reason": data.get("reason", ""),
            "needs_clarification": data.get("needs_clarification", False),
            "clarification_question": data.get("clarification_question", ""),
            # Always pass messages for supervisor transparency
            "messages": all_messages,
        }

        # Inject merged agent state fields so the downstream agent graph will
        # receive the restored checkpoint values when invoked as a node.
        if merged_agent_state:
            for k, v in merged_agent_state.items():
                # Avoid overwriting supervisor routing fields
                if k in ("chosen_agent", "routing_reason", "needs_clarification", "clarification_question"):
                    continue
                result[k] = v

        return result

    return route_request


async def ask_user(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
    question = state.get("clarification_question", "Could you give me more details?")
    message = CLARIFICATION_PREFIX + question

    await save_message_idempotent(thread_id, "assistant", message)

    return {
        "response": message,
        "messages": [AIMessage(content=message)],
    }


def what_to_do(state: SupervisorState) -> str:
    if state.get("needs_clarification", False):
        return "ask_user"
    return state.get("chosen_agent", "personal")
