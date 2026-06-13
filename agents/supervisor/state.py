# agents/supervisor/state.py
#
# The supervisor's state.
# It only needs to know: what came in, which agent was chosen, and what came back.

from __future__ import annotations

from typing import Annotated, Optional, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class SupervisorState(TypedDict, total=False):
    # The conversation — carries the user message and the final response
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Which thread this request belongs to (passed through to the sub-agent)
    thread_id: str

    # The user's original message text
    user_input: str

    # The agent the router decided should handle this (e.g. "personal", "deep_research")
    chosen_agent: str

    # One-sentence reason the router chose this agent (useful for debugging)
    routing_reason: str

    # The final answer returned by the sub-agent
    response: str

    # Set to True if the router could not confidently pick an agent
    needs_clarification: bool

    # Clarification question to ask the user (when needs_clarification is True)
    clarification_question: Optional[str]