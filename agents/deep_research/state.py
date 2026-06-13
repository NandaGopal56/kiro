from __future__ import annotations

from typing import Annotated, List, Optional, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    # The conversation (carries the original question and all back-and-forth)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Thread this research session belongs to
    thread_id: str

    # The original raw question from the user — never changed
    original_question: str

    # The refined, confirmed goal — set after user confirms it
    # This is what the planner and executor actually work from
    goal: str

    # Clarifying questions the agent asked the user
    clarifying_questions: List[str]

    # User's answers to those questions (free text)
    user_clarification: str

    # Whether the user has confirmed the goal and we can start researching
    # False → still in clarify/confirm loop
    # True  → proceed to planning and execution
    goal_confirmed: bool

    # Ordered list of sub-questions / steps produced by the planner
    plan: List[str]

    # "Done when" criterion produced by the planner
    done_when: str

    # Which step in the plan we are currently executing (0-based index)
    current_step: int

    # Running log of everything found so far, appended after each step
    findings: str

    # What the reflection node says to focus on next
    next_focus: str

    # Set to True by the reflection node when the goal is fully answered
    is_done: bool

    # How many execute→reflect cycles have run (guards against infinite loops)
    iteration: int

    # The final synthesised answer, written by the finish node
    final_answer: str