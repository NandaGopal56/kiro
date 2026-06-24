from __future__ import annotations

from typing import Annotated, List, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    """State carried through the deep research graph.

    Notes:
        - `messages` holds the conversation history merged by LangGraph.
        - `goal` is the latest refined research goal draft.
        - `goal_confirmed` means the user has explicitly approved the goal.
        - `plan_confirmed` means the user has explicitly approved the plan.
        - `goal_revision_notes` and `plan_revision_notes` are used when the
          user provides additional context or correction that requires another
          refine/revise loop.
    """

    # Conversation / thread
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    original_question: str

    # Goal refinement / confirmation
    goal: str
    clarifying_questions: List[str]
    user_clarification: str
    goal_ready: bool
    goal_confirmed: bool
    goal_revision_notes: str

    # Plan creation / confirmation
    plan: List[str]
    done_when: str
    plan_confirmed: bool
    plan_revision_notes: str

    # Execution progress
    current_step: int
    findings: str
    next_focus: str
    is_done: bool
    iteration: int

    # Final output
    final_answer: str