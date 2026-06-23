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

    # The refined goal — updated as clarify_goal loops with the user
    # This is what the planner and executor actually work from
    goal: str

    # Clarifying questions the agent asked the user (most recent round)
    clarifying_questions: List[str]

    # User's answers to those questions (free text), consumed by clarify_goal
    user_clarification: str

    # Whether clarify_goal judges the goal sufficiently clear to plan from.
    # False → still looping in clarify_goal (ends run, user replies next turn)
    # True  → proceed to create_plan in the same invocation
    goal_ready: bool

    # Ordered list of sub-questions / steps produced by the planner
    plan: List[str]

    # "Done when" criterion produced by the planner
    done_when: str

    # Whether the user has confirmed the drafted plan via interrupt().
    # False → check_plan_confirmation interrupt loop (revise via create_plan)
    # True  → proceed to execute_step
    plan_confirmed: bool

    # User's revision feedback on the plan, consumed by create_plan to
    # revise the existing plan in place (not regenerate from scratch)
    plan_revision_notes: str

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