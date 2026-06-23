# agents/deep_research/planner.py
#
# Plain async functions — not LangGraph nodes — so they can be called from
# a node, tested in isolation, or reused elsewhere easily.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.models import get_llm

from .prompts import PLANNER_PROMPT, PLANNER_REVISE_PROMPT


@dataclass
class ResearchPlan:
    """The output of the planner."""
    goal: str
    steps: List[str] = field(default_factory=list)
    done_when: str = ""

    def step_list(self) -> str:
        """Return steps as a numbered string for prompts / display."""
        return "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.steps))


async def make_plan(goal: str) -> ResearchPlan:
    """
    Ask the LLM to break the research goal into ordered sub-questions.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    llm = get_llm(strong=True)

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Research goal: {goal}"),
    ])

    data = _parse_plan_json(response.content)
    if data is None:
        return ResearchPlan(
            goal=goal,
            steps=[
                f"Research: {goal}",
                "Synthesise all findings into a final answer.",
            ],
            done_when="The goal is fully answered with supporting evidence.",
        )

    return ResearchPlan(
        goal=data.get("goal", goal),
        steps=data.get("steps", [goal]),
        done_when=data.get("done_when", ""),
    )


async def revise_plan(
    goal: str,
    existing_steps: List[str],
    done_when: str,
    revision_notes: str,
) -> ResearchPlan:
    """
    Edit the EXISTING plan in place based on the user's revision feedback.
    Only changes steps the feedback calls for; keeps the rest word-for-word.
    Falls back to the existing plan unchanged if the LLM returns bad JSON,
    so a parsing hiccup never silently drops the user's plan.
    """
    llm = get_llm(strong=True)

    existing_steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(existing_steps))

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_REVISE_PROMPT.format(
            goal=goal,
            existing_steps=existing_steps_text,
            done_when=done_when,
            revision_notes=revision_notes,
        )),
        HumanMessage(content="Revise the plan now, returning the full revised step list."),
    ])

    data = _parse_plan_json(response.content)
    if data is None:
        # Keep existing plan intact rather than losing it on a bad parse.
        return ResearchPlan(goal=goal, steps=existing_steps, done_when=done_when)

    return ResearchPlan(
        goal=data.get("goal", goal),
        steps=data.get("steps", existing_steps),
        done_when=data.get("done_when", done_when),
    )


def _parse_plan_json(raw: str):
    """Parse JSON from LLM output, stripping markdown fences if present.
    Returns None on failure so callers apply their own fallback."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError, ValueError):
        return None