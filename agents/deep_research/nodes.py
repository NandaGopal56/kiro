from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig, interrupt

from agents.shared.memory import (
    save_message_idempotent,
    save_tool_call,
    save_tool_result,
)
from agents.shared.models import get_llm
from agents.shared.tools import research_tools

from .planner import ResearchPlan, make_plan, revise_plan
from .prompts import (
    CLARIFICATION_MESSAGE,
    CLARIFIER_PROMPT,
    EXECUTOR_PROMPT,
    FINISHER_PROMPT,
    GOAL_UPDATER_PROMPT,
    PLAN_CONFIRMATION_MESSAGE,
    REFLECTOR_PROMPT,
)
from .state import ResearchState

# ---------------------------------------------------------------------------
# Debug logging — toggle DEBUG_MODE to enable/disable state logging.
# When enabled, every node logs its full input state to
# .logs/deep_research.log before doing any work.
# ---------------------------------------------------------------------------
DEBUG_MODE = True

_log = logging.getLogger("deep_research")
if not _log.handlers:
    import os
    os.makedirs(".logs", exist_ok=True)
    _fh = logging.FileHandler(".logs/deep_research.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_fh)
    _log.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)


def _log_state(node_name: str, state: Dict[str, Any]) -> None:
    """Log the full state at node entry. No-op when DEBUG_MODE is False."""
    if not DEBUG_MODE:
        return
    # Truncate long text fields so the log stays readable
    loggable = {}
    for k, v in state.items():
        if isinstance(v, str) and len(v) > 500:
            loggable[k] = v[:500] + f"... [truncated, total={len(v)}]"
        elif k == "messages":
            loggable[k] = f"[{len(v)} messages]"
        else:
            loggable[k] = v
    _log.debug("NODE ENTRY: %s\nSTATE: %s", node_name, json.dumps(loggable, default=str, indent=2))


MAX_ITERATIONS = 10

_tool_node      = ToolNode(tools=research_tools)
_llm_with_tools = get_llm(strong=True).bind_tools(research_tools)


# ---------------------------------------------------------------------------
# Node — clarify_goal
#
# Loops (via END → next user message → _entry_router → back here) until
# its OWN judgment says the goal is clear enough to plan from, at which
# point it sets goal_ready=True and the graph edge sends the run straight
# on to create_plan IN THE SAME INVOCATION — no user round-trip for goal
# confirmation.
# ---------------------------------------------------------------------------
async def clarify_goal(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    _log_state("clarify_goal", state)

    original      = state.get("original_question", "")
    prev_goal     = state.get("goal", "")

    # Fallback: pull original question from messages if not persisted
    if not original:
        for msg in reversed(list(state.get("messages", []))):
            if isinstance(msg, HumanMessage):
                original = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

    # On re-entry after a clarifying question, the latest HumanMessage IS
    # the user's reply — extract it as the clarification.
    user_reply = ""
    if prev_goal:
        for msg in reversed(list(state.get("messages", []))):
            if isinstance(msg, HumanMessage):
                user_reply = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

    _log.debug("clarify_goal: original=%r prev_goal=%r user_reply=%r", original, prev_goal, user_reply)

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
            "goal_ready":   False,
        })

        refined_goal = data.get("refined_goal", f"Research goal: {original}")
        questions    = data.get("questions", [])
        goal_ready   = bool(data.get("goal_ready", not questions))

        _log.debug("clarify_goal first-pass: refined_goal=%r questions=%r goal_ready=%r",
                   refined_goal, questions, goal_ready)

        if not goal_ready and questions:
            q_block = "**I have a few questions to sharpen the research:**\n" + \
                      "\n".join(f"- {q}" for q in questions)
            message = CLARIFICATION_MESSAGE.format(
                original_question=original,
                refined_goal=refined_goal,
                questions_block=q_block,
            )
            await save_message_idempotent(thread_id, "assistant", message)
            return {
                "original_question":    original,
                "goal":                 refined_goal,
                "clarifying_questions": questions,
                "goal_ready":           False,
                "messages":             [AIMessage(content=message)],
            }

        # Goal already clear — no need to ask the user anything
        return {
            "original_question":    original,
            "goal":                 refined_goal,
            "clarifying_questions": [],
            "goal_ready":           True,
        }

    # ── Subsequent pass: incorporate user's reply ─────────────────────────
    response = await llm.ainvoke([
        SystemMessage(content=GOAL_UPDATER_PROMPT.format(
            original_question=original,
            refined_goal=prev_goal,
            user_clarification=user_reply,
        )),
        HumanMessage(content="Update the goal based on my feedback, and judge if it's now ready to plan from."),
    ])
    data = _parse_json(response.content, {
        "updated_goal": prev_goal,
        "questions":    [],
        "goal_ready":   True,
    })
    updated_goal = data.get("updated_goal", prev_goal)
    questions    = data.get("questions", [])
    goal_ready   = bool(data.get("goal_ready", not questions))

    _log.debug("clarify_goal subsequent-pass: updated_goal=%r questions=%r goal_ready=%r",
               updated_goal, questions, goal_ready)

    if not goal_ready and questions:
        q_block = "**A couple more things to sharpen the research:**\n" + \
                  "\n".join(f"- {q}" for q in questions)
        message = CLARIFICATION_MESSAGE.format(
            original_question=original,
            refined_goal=updated_goal,
            questions_block=q_block,
        )
        await save_message_idempotent(thread_id, "assistant", message)
        return {
            "original_question":    original,
            "goal":                 updated_goal,
            "clarifying_questions": questions,
            "goal_ready":           False,
            "user_clarification":   "",
            "messages":             [AIMessage(content=message)],
        }

    return {
        "original_question":  original,
        "goal":               updated_goal,
        "clarifying_questions": [],
        "goal_ready":         True,
        "user_clarification": "",
    }


# ---------------------------------------------------------------------------
# Node — create_plan
#
# Drafts a plan from the ready goal, OR — if re-entered after a rejected
# plan confirmation with revision notes — edits the EXISTING plan in place
# using that feedback (not regenerating from scratch, so unaffected steps
# keep their original wording and order).
# ---------------------------------------------------------------------------
async def create_plan(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    _log_state("create_plan", state)

    goal           = state.get("goal", "")
    existing_plan  = state.get("plan", [])
    revision_notes = state.get("plan_revision_notes", "")

    _log.debug("create_plan: goal=%r existing_plan_len=%d revision_notes=%r",
               goal, len(existing_plan), revision_notes)

    if existing_plan and revision_notes:
        plan: ResearchPlan = await revise_plan(
            goal=goal,
            existing_steps=existing_plan,
            done_when=state.get("done_when", ""),
            revision_notes=revision_notes,
        )
    else:
        plan: ResearchPlan = await make_plan(goal)

    plan_text = (
        f"📋 **Research plan** ({len(plan.steps)} steps):\n"
        + plan.step_list()
        + f"\n\n✅ Done when: {plan.done_when}"
    )

    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
    await save_message_idempotent(thread_id, "assistant", plan_text)

    _log.debug("create_plan: produced %d steps", len(plan.steps))

    return {
        "plan":                plan.steps,
        "done_when":           plan.done_when,
        "current_step":        0,
        "findings":            "",
        "is_done":             False,
        "iteration":           0,
        "next_focus":          "",
        "plan_confirmed":      False,
        "plan_revision_notes": "",
        "messages":            [AIMessage(content=plan_text)],
    }


# ---------------------------------------------------------------------------
# Node — check_plan_confirmation
#
# Mid-graph interrupt(): pauses execution here, shows the plan to the user,
# and waits for their reply. On resume (via Command(resume=user_reply) in
# Supervisor._run), the node reruns from the top but interrupt() returns
# the user's reply instead of pausing. The node then interprets it as
# confirm or revise and sets plan_confirmed accordingly.
#
# Docs confirm: resume via graph.invoke(Command(resume=value), same_config)
# — no new state dict, same thread_id. That's exactly what
# Supervisor._pending_interrupt + _run does.
# ---------------------------------------------------------------------------
async def check_plan_confirmation(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    _log_state("check_plan_confirmation", state)

    goal = state.get("goal", "")
    plan = state.get("plan", [])

    message = PLAN_CONFIRMATION_MESSAGE.format(
        goal=goal,
        plan_steps="\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan)),
    )

    _log.debug("check_plan_confirmation: issuing interrupt for plan confirmation goal=%r steps=%d",
               goal, len(plan))

    # Pauses here on first pass. On resume, returns the user's reply string.
    user_reply = interrupt({
        "type":    "plan_confirmation",
        "goal":    goal,
        "plan":    plan,
        "message": message,
    })
    user_reply_text = user_reply if isinstance(user_reply, str) else str(user_reply)

    _log.debug("check_plan_confirmation: resumed with user_reply=%r", user_reply_text)

    llm = get_llm(strong=True)
    llm_prompt = (
        f"Determine whether the user reply confirms the research plan as-is.\n"
        f"Research goal: {goal}\n"
        f"Plan:\n" + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan)) + "\n\n"
        f"User reply: {user_reply_text}\n\n"
        "Return JSON only: {{\"is_confirmed\": true/false, "
        "\"revision_notes\": \"user's requested changes, or empty string\"}}"
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=llm_prompt),
            HumanMessage(content="Respond with JSON only."),
        ])
        data = _parse_json(response.content, {"is_confirmed": False, "revision_notes": ""})
    except Exception:
        data = {"is_confirmed": False, "revision_notes": ""}

    is_confirmed   = bool(data.get("is_confirmed", False))
    revision_notes = data.get("revision_notes", "")

    _log.debug("check_plan_confirmation: is_confirmed=%r revision_notes=%r",
               is_confirmed, revision_notes)

    if is_confirmed:
        return {"plan_confirmed": True, "plan_revision_notes": ""}

    return {
        "plan_confirmed":      False,
        "plan_revision_notes": revision_notes or user_reply_text,
    }


# Router after check_plan_confirmation
def confirmation_router(state: ResearchState) -> str:
    """Confirmed → execute. Not confirmed → back to create_plan to revise."""
    return "execute_step" if state.get("plan_confirmed", False) else "create_plan"


# ---------------------------------------------------------------------------
# Node — execute_step
#
# Runs one step from the confirmed plan using the LLM + search tools.
# No interrupt — goal and plan are fixed by this point, so execution is
# fully autonomous.
# ---------------------------------------------------------------------------
async def execute_step(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    _log_state("execute_step", state)

    goal        = state.get("goal", "")
    plan        = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    findings    = state.get("findings", "")
    next_focus  = state.get("next_focus", "")
    iteration   = state.get("iteration", 0)

    _log.debug("execute_step: current_idx=%d iteration=%d total_steps=%d next_focus=%r",
               current_idx, iteration, len(plan), next_focus)

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

    _log.debug("execute_step: step %d done, findings_total_len=%d", current_idx + 1, len(new_findings))

    return {
        "findings":     new_findings,
        "current_step": current_idx + 1,
        "iteration":    iteration + 1,
        "messages":     [AIMessage(content=f"✅ Step {current_idx + 1} done: {current_step_text}")],
    }


# ---------------------------------------------------------------------------
# Node — reflect
# ---------------------------------------------------------------------------
async def reflect(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    _log_state("reflect", state)

    goal         = state.get("goal", "")
    plan         = state.get("plan", [])
    findings     = state.get("findings", "")
    iteration    = state.get("iteration", 0)
    current_step = state.get("current_step", 0)
    done_when    = state.get("done_when", f"All {len(plan)} steps are complete.")

    _log.debug("reflect: iteration=%d current_step=%d total_steps=%d", iteration, current_step, len(plan))

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

    _log.debug("reflect: is_done=%r next_focus=%r reason=%r", is_done, next_focus, reason)

    return {
        "is_done":    is_done,
        "next_focus": next_focus,
        "messages":   [AIMessage(content=status_msg)],
    }


# Router after reflect
def should_continue(state: ResearchState) -> str:
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
# ---------------------------------------------------------------------------
async def finish(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    _log_state("finish", state)

    goal     = state.get("goal", "")
    findings = state.get("findings", "")

    _log.debug("finish: goal=%r findings_len=%d", goal, len(findings))

    llm = get_llm(strong=True)
    response = await llm.ainvoke([
        SystemMessage(content=FINISHER_PROMPT.format(goal=goal, findings=findings)),
        HumanMessage(content="Write the final research answer now."),
    ])

    thread_id = config.get("configurable", {}).get("thread_id", "") or state.get("thread_id", "")
    content = response.content if isinstance(response.content, str) else str(response.content)
    await save_message_idempotent(thread_id, "assistant", content)

    _log.debug("finish: final_answer_len=%d", len(content))

    return {
        "final_answer": content,
        "messages":     [AIMessage(content=content)],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
async def _run_with_tools(messages: List[Any], config: RunnableConfig) -> str:
    """Mini tool-use loop for one research step. Returns final text."""
    current_messages = list(messages)
    thread_id = (
        config.get("configurable", {}).get("thread_id", "")
        or config.get("metadata", {}).get("thread_id", "")
    )

    for _ in range(5):
        response = await _llm_with_tools.ainvoke(current_messages, config=config)

        content = response.content if isinstance(response.content, str) else str(response.content)
        assistant_msg_id = None
        if thread_id:
            assistant_msg_id = await save_message_idempotent(thread_id, "assistant", content)
            for tc in getattr(response, "tool_calls", []) or []:
                await save_tool_call(message_id=assistant_msg_id, call_id=tc.get("id", ""), tool_input=tc)

        current_messages.append(response)

        if not getattr(response, "tool_calls", None):
            return content

        tool_result = await _tool_node.ainvoke({"messages": current_messages}, config)
        tool_messages = tool_result.get("messages", [])
        for tm in tool_messages:
            if assistant_msg_id:
                await save_tool_result(message_id=assistant_msg_id, call_id=tm.tool_call_id, output=tm.content)
        current_messages.extend(tool_messages)

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