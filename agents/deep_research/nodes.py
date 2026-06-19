from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from agents.shared.memory import save_message_idempotent, rebuild_messages_from_db
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

_tool_node      = ToolNode(tools=research_tools)
_llm_with_tools = get_llm(strong=True).bind_tools(research_tools)


# ---------------------------------------------------------------------------
# Node — clarify_goal
# Sharpens the question and asks any needed clarifying questions, or
# incorporates the user's clarification into an updated goal.
# This node always ends the graph run (see graph.py: clarify_goal -> END) —
# the user's next message is what triggers the follow-up turn.
# ---------------------------------------------------------------------------
async def clarify_goal(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    original      = state.get("original_question", "")
    clarification = state.get("user_clarification", "")
    prev_goal     = state.get("goal", "")

    # Fallback: scan messages only if original_question wasn't persisted
    if not original:
        for msg in reversed(list(state.get("messages", []))):
            if isinstance(msg, HumanMessage):
                original = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

    print(f"DEBUG: clarify_goal original={original!r} prev_goal={prev_goal!r} clarification={clarification!r}")

    llm = get_llm(strong=True)
    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")

    # ── First pass: no goal yet ───────────────────────────────────────────
    if not prev_goal:
        response = await llm.ainvoke([
            SystemMessage(content=CLARIFIER_PROMPT),
            HumanMessage(content=original),
        ])
        data = _parse_json(response.content, {
            "refined_goal": f"Research goal: {original}",
            "questions":    [],
        })

        refined_goal = data.get("refined_goal", f"Research goal: {original}")
        questions    = data.get("questions", [])

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

        await save_message_idempotent(thread_id, "assistant", message)

        return {
            "original_question":    original,
            "goal":                 refined_goal,
            "clarifying_questions": questions,
            "goal_confirmed":       False,
            "messages":             [AIMessage(content=message)],
        }

    # ── Subsequent pass: user gave clarification/corrections ─────────────
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
        await save_message_idempotent(thread_id, "assistant", message)

        return {
            "original_question":  original,
            "goal":               updated_goal,
            "goal_confirmed":     False,
            "user_clarification": "",
            "messages":           [AIMessage(content=message)],
        }

    # ── Fallback: goal exists, no clarification text extracted ───────────
    # (e.g. check_confirmation couldn't tell yes/no and didn't extract
    # anything useful) — re-show the same confirmation prompt.
    message = CONFIRMATION_MESSAGE.format(
        original_question=original,
        refined_goal=prev_goal,
    )
    await save_message_idempotent(thread_id, "assistant", message)

    return {
        "original_question": original,
        "goal":              prev_goal,
        "goal_confirmed":    False,
        "messages":          [AIMessage(content=message)],
    }


# ---------------------------------------------------------------------------
# Node — check_confirmation
# Interprets the user's reply to a pending clarification: confirmed, or
# more corrections needed? Only reached when state["goal"] is already set
# and not yet confirmed (see graph.py: _entry_router), so the latest human
# message in this invocation's state IS the reply to interpret.
# ---------------------------------------------------------------------------
async def check_confirmation(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")

    # The merged state's messages already end with this turn's HumanMessage
    # (see DeepResearchAgent._run), so just take the latest one directly —
    # no need to round-trip through the DB to find it.
    user_reply = ""
    for msg in reversed(list(state.get("messages", []))):
        if isinstance(msg, HumanMessage):
            user_reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    print(f"DEBUG: check_confirmation thread_id={thread_id} goal={state.get('goal')!r} user_reply={user_reply!r}")

    llm = get_llm(strong=True)
    llm_prompt = (
        f"Determine whether the following user reply confirms the research goal.\n"
        f"Research goal: {state.get('goal', '')}\n"
        f"Assistant asked the user to confirm the goal.\n"
        f"User reply: {user_reply}\n\n"
        "Return a JSON object with keys: \"is_confirmed\" (true/false),"
        " and \"clarification\" (string; any extra instructions from the user, empty if none)."
        " Respond with JSON only."
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=llm_prompt),
            HumanMessage(content="Please respond with JSON only."),
        ])
        data = _parse_json(response.content, {"is_confirmed": False, "clarification": ""})
    except Exception:
        data = {"is_confirmed": False, "clarification": ""}

    is_confirmed            = bool(data.get("is_confirmed", False))
    clarification_from_user = data.get("clarification", "")
    print(f"DEBUG: check_confirmation is_confirmed={is_confirmed} clarification={clarification_from_user!r}")

    if is_confirmed:
        return {"goal_confirmed": True, "user_clarification": ""}

    # Treat the reply as clarification/corrections. Prefer what the LLM
    # extracted; fall back to the raw reply if it extracted nothing.
    return {
        "goal_confirmed":     False,
        "user_clarification": clarification_from_user or user_reply,
    }


# Router after check_confirmation
def confirmation_router(state: ResearchState) -> str:
    """Confirmed → plan. Not confirmed → loop back to clarify (which ends
    the run after re-asking)."""
    return "create_plan" if state.get("goal_confirmed", False) else "clarify_goal"


# ---------------------------------------------------------------------------
# Node — create_plan
# Breaks the confirmed goal into an ordered list of steps.
# ---------------------------------------------------------------------------
async def create_plan(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    goal = state.get("goal", "")

    if state.get("plan"):   # already planned (e.g. resumed session)
        return {}

    plan: ResearchPlan = await make_plan(goal)

    plan_text = (
        f"📋 **Research plan** ({len(plan.steps)} steps):\n"
        + plan.step_list()
        + f"\n\n✅ Done when: {plan.done_when}\n\n_Starting research now..._"
    )

    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
    await save_message_idempotent(thread_id, "assistant", plan_text)

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
# Node — execute_step
# Runs one step from the plan using the LLM + search tools.
# ---------------------------------------------------------------------------
async def execute_step(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    goal        = state.get("goal", "")
    plan        = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    findings    = state.get("findings", "")
    next_focus  = state.get("next_focus", "")
    iteration   = state.get("iteration", 0)

    if current_idx >= len(plan):
        return {"is_done": True}

    current_step_text = plan[current_idx]
    completed_steps    = "\n".join(f"  ✓ {plan[i]}" for i in range(current_idx)) or "  (none yet)"

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
# Node — reflect
# Evaluates progress and decides: continue or finish?
# ---------------------------------------------------------------------------
async def reflect(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
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
# Node — finish
# Synthesises all findings into the final answer.
# ---------------------------------------------------------------------------
async def finish(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    goal     = state.get("goal", "")
    findings = state.get("findings", "")

    llm = get_llm(strong=True)
    response = await llm.ainvoke([
        SystemMessage(content=FINISHER_PROMPT.format(goal=goal, findings=findings)),
        HumanMessage(content="Write the final research answer now."),
    ])

    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
    content = response.content if isinstance(response.content, str) else str(response.content)
    await save_message_idempotent(thread_id, "assistant", content)

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