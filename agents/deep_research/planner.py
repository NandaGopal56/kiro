# agents/deep_research/planner.py
#
# The planner takes a research goal and returns an ordered list of steps.
# It is a plain async function — not a LangGraph node — so it can be
# called from a node, tested in isolation, or reused elsewhere easily.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.shared.models import get_llm

from .prompts import PLANNER_PROMPT


@dataclass
class ResearchPlan:
    """The output of the planner."""
    goal: str
    steps: List[str] = field(default_factory=list)
    done_when: str = ""

    def step_list(self) -> str:
        """Return steps as a numbered string for prompts."""
        return "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.steps))


async def make_plan(goal: str) -> ResearchPlan:
    """
    Ask the LLM to break the research goal into ordered sub-questions.
    Returns a ResearchPlan with a list of steps and a done-when criterion.

    Falls back gracefully if the LLM returns malformed JSON.
    """
    llm = get_llm(strong=True)  # use the stronger model for planning

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Research goal: {goal}"),
    ])

    try:
        raw = response.content.strip()
        # Strip markdown code fences if the model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError, ValueError):
        # Fallback: treat the whole goal as a single step
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