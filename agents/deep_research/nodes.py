from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from agents.shared.models import get_llm
from agents.shared.tools import research_tools

from .planner import ResearchPlan, make_plan
from .prompts import (
    CLARIFICATION_MESSAGE,
    CLARIFIER_PROMPT,
    CONFIRMATION_MESSAGE,
    EXECUTOR_PROMPT,
    FINISHER_PROMPT,
    GOAL_UPDATER_PROMPT,
    REFLECTOR_PROMPT,
)
from .state import ResearchState

MAX_ITERATIONS = 10

_tool_node       = ToolNode(tools=research_tools)
_llm_with_tools  = get_llm(strong=True).bind_tools(research_tools)


# ---------------------------------------------------------------------------
# Node 1 — clarify_goal
# Sharpens the question and asks any needed clarifying questions.
# The graph PAUSES after this node (interrupt_after) waiting for user reply.
# ---------------------------------------------------------------------------

async def clarify_goal(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Two jobs:
    1. First call: refine the raw question + ask clarifying questions.
    2. Subsequent calls (after user replied): incorporate their feedback.

    The graph pauses after this node so the user can reply.
    When the user replies, the supervisor feeds their message back in and
    the graph resumes at check_confirmation.
    """
    original     = state.get("original_question", "")
    clarification = state.get("user_clarification", "")
    prev_goal    = state.get("goal", "")

    if not original:
        messages = list(state.get("messages", []))
        from langchain_core.messages import HumanMessage as HM
        for msg in reversed(messages):
            if isinstance(msg, HM):
                original = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

    llm = get_llm(strong=True)

    # ── First pass: refine and ask questions ──────────────────────────────
    if not prev_goal:
        response = await llm.ainvoke([
            SystemMessage(content=CLARIFIER_PROMPT),
            HumanMessage(content=original),
        ])
        data = _parse_json(response.content, {
            "refined_goal": f"Research goal: {original}",
            "questions":    [],
            "reason":       "",
        })

        refined_goal = data.get("refined_goal", f"Research goal: {original}")
        questions    = data.get("questions", [])

        # Build the message shown to the user
        if questions:
            q_block = "**I have a few questions to sharpen the research:**\n" + \
                      "\n".join(f"- {q}" for q in questions)
            message = CLARIFICATION_MESSAGE.format(
                original_question=original,
                refined_goal=refined_goal,
                questions_block=q_block,
            )
        else:
            message = CONFIRMATION_MESSAGE.format(
                original_question=original,
                refined_goal=refined_goal,
            )

        return {
            "goal":                 refined_goal,
            "clarifying_questions": questions,
            "goal_confirmed":       False,
            "messages":             [AIMessage(content=message)],
        }

    # ── Subsequent pass: user gave feedback, update the goal ─────────────
    if clarification:
        response = await llm.ainvoke([
            SystemMessage(content=GOAL_UPDATER_PROMPT.format(
                original_question=original,
                refined_goal=prev_goal,
                user_clarification=clarification,
            )),
            HumanMessage(content="Update the goal based on my feedback."),
        ])
        data = _parse_json(response.content, {"updated_goal": prev_goal})
        updated_goal = data.get("updated_goal", prev_goal)

        message = CONFIRMATION_MESSAGE.format(
            original_question=original,
            refined_goal=updated_goal,
        )
        return {
            "goal":             updated_goal,
            "goal_confirmed":   False,
            "user_clarification": "",   # clear so next pass starts fresh
            "messages":         [AIMessage(content=message)],
        }

    # ── Nothing changed — just re-show the confirmation prompt ───────────
    message = CONFIRMATION_MESSAGE.format(
        original_question=original,
        refined_goal=prev_goal,
    )
    return {
        "goal_confirmed": False,
        "messages":       [AIMessage(content=message)],
    }


# ---------------------------------------------------------------------------
# Node 2 — check_confirmation
# Reads the user's reply and decides: confirmed or needs more changes?
# This node runs AFTER the graph resumes from the interrupt.
# ---------------------------------------------------------------------------

async def check_confirmation(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Read the latest user message.
    - If it's a confirmation ("yes", "confirmed", "ok", "looks good", etc.) → set goal_confirmed = True
    - Otherwise → treat it as clarification feedback, store it, loop back to clarify_goal
    """
    from langchain_core.messages import HumanMessage as HM
    messages = list(state.get("messages", []))

    # Find the most recent human message (the user's reply)
    user_reply = ""
    for msg in reversed(messages):
        if isinstance(msg, HM):
            user_reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Simple confirmation check — catches common affirmative replies
    confirm_words = {"yes", "confirmed", "confirm", "ok", "okay", "looks good",
                     "good", "correct", "proceed", "start", "go", "yep", "sure",
                     "perfect", "great", "fine", "approved", "approve"}
    is_confirmed = any(w in user_reply.lower() for w in confirm_words)

    if is_confirmed:
        return {"goal_confirmed": True, "user_clarification": ""}
    else:
        # Treat their reply as clarification/corrections
        return {"goal_confirmed": False, "user_clarification": user_reply}


# Router after check_confirmation
def confirmation_router(state: ResearchState) -> str:
    """Confirmed → plan. Not confirmed → loop back to clarify."""
    return "create_plan" if state.get("goal_confirmed", False) else "clarify_goal"


# ---------------------------------------------------------------------------
# Node 3 — create_plan
# Breaks the confirmed goal into an ordered list of steps.
# ---------------------------------------------------------------------------

async def create_plan(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Called once after the goal is confirmed.
    Produces the research plan the executor will follow step by step.
    """
    goal = state.get("goal", "")

    if state.get("plan"):   # already planned (e.g. resumed session)
        return {}

    plan: ResearchPlan = await make_plan(goal)

    plan_text = (
        f"📋 **Research plan** ({len(plan.steps)} steps):\n"
        + plan.step_list()
        + f"\n\n✅ Done when: {plan.done_when}\n\n_Starting research now..._"
    )

    return {
        "plan":         plan.steps,
        "done_when":    plan.done_when,
        "current_step": 0,
        "findings":     "",
        "is_done":      False,
        "iteration":    0,
        "next_focus":   "",
        "messages":     [AIMessage(content=plan_text)],
    }


# ---------------------------------------------------------------------------
# Node 4 — execute_step
# Runs one step from the plan using the LLM + search tools.
# ---------------------------------------------------------------------------

async def execute_step(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the current plan step with search tools, append findings."""
    goal        = state.get("goal", "")
    plan        = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    findings    = state.get("findings", "")
    next_focus  = state.get("next_focus", "")
    iteration   = state.get("iteration", 0)

    if current_idx >= len(plan):
        return {"is_done": True}

    current_step_text = plan[current_idx]
    completed_steps   = "\n".join(f"  ✓ {plan[i]}" for i in range(current_idx)) or "  (none yet)"

    prompt = EXECUTOR_PROMPT.format(
        goal=goal,
        completed_steps=completed_steps,
        findings=findings[-3000:] if findings else "(none yet)",
        current_step=current_step_text,
        next_focus=next_focus or "none",
    )

    step_findings = await _run_with_tools([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Execute step {current_idx + 1}: {current_step_text}"),
    ], config)

    step_header  = f"\n\n--- Step {current_idx + 1}: {current_step_text} ---\n"
    new_findings = findings + step_header + step_findings

    return {
        "findings":     new_findings,
        "current_step": current_idx + 1,
        "iteration":    iteration + 1,
        "messages":     [AIMessage(content=f"✅ Step {current_idx + 1} done: {current_step_text}")],
    }


# ---------------------------------------------------------------------------
# Node 5 — reflect
# Evaluates progress and decides: continue or finish?
# ---------------------------------------------------------------------------

async def reflect(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Check if the goal is answered. If not, identify the next focus."""
    goal         = state.get("goal", "")
    plan         = state.get("plan", [])
    findings     = state.get("findings", "")
    iteration    = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    done_when    = state.get("done_when", f"All {len(plan)} steps are complete.")

    prompt = REFLECTOR_PROMPT.format(
        goal=goal,
        done_when=done_when,
        plan="\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)),
        steps_completed=current_step,
        total_steps=len(plan),
        iteration=iteration,
        max_iterations=MAX_ITERATIONS,
        findings=findings[-4000:] if findings else "(none yet)",
    )

    llm = get_llm(strong=True)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    data = _parse_json(response.content, {"is_done": False, "reason": "", "next_focus": ""})

    is_done    = bool(data.get("is_done", False))
    next_focus = data.get("next_focus", "")
    reason     = data.get("reason", "")
    status_msg = "✔ Research complete." if is_done else f"🔄 Continuing — {reason}"

    return {
        "is_done":    is_done,
        "next_focus": next_focus,
        "messages":   [AIMessage(content=status_msg)],
    }


# Router after reflect
def should_continue(state: ResearchState) -> str:
    """Done or hit limit → finish. Otherwise → next step."""
    if state.get("is_done", False):
        return "finish"
    iteration    = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    plan         = state.get("plan", [])
    if iteration >= MAX_ITERATIONS or current_step >= len(plan):
        return "finish"
    return "execute_step"


# ---------------------------------------------------------------------------
# Node 6 — finish
# Synthesises all findings into the final answer.
# ---------------------------------------------------------------------------

async def finish(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """Produce the final, well-structured research answer."""
    goal     = state.get("goal", "")
    findings = state.get("findings", "")

    llm = get_llm(strong=True)
    response = await llm.ainvoke([
        SystemMessage(content=FINISHER_PROMPT.format(goal=goal, findings=findings)),
        HumanMessage(content="Write the final research answer now."),
    ])

    return {
        "final_answer": response.content,
        "messages":     [AIMessage(content=response.content)],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_with_tools(messages: List[Any], config: RunnableConfig) -> str:
    """Mini tool-use loop for one research step. Returns final text."""
    current_messages = list(messages)
    for _ in range(5):
        response = await _llm_with_tools.ainvoke(current_messages, config=config)
        current_messages.append(response)
        if not getattr(response, "tool_calls", None):
            return response.content if isinstance(response.content, str) else str(response.content)
        tool_result = await _tool_node.ainvoke({"messages": current_messages}, config)
        current_messages.extend(tool_result.get("messages", []))
    last = current_messages[-1]
    return getattr(last, "content", "") or ""


def _parse_json(raw: str, fallback: Dict) -> Dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError, IndexError):
        return fallback
