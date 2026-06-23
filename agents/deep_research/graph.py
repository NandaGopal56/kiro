from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import (
    check_plan_confirmation,
    clarify_goal,
    confirmation_router,
    create_plan,
    execute_step,
    finish,
    reflect,
    should_continue,
)
from .state import ResearchState


def _entry_router(state: ResearchState) -> str:
    """
    Decides where a fresh invocation enters the graph.

    Two cases handled here:
      - plan_confirmed=True: research is mid-execution (e.g. resumed after
        restart). Jump straight to execute_step.
      - Anything else (new session, or re-entering during the clarify loop):
        go to clarify_goal, which handles both "no goal yet" and
        "user replied to a clarifying question" internally.

    Note: the plan-confirmation interrupt (check_plan_confirmation) is
    NOT handled here. When that interrupt is pending, Supervisor._run
    detects it via _pending_interrupt() and passes Command(resume=task)
    directly to ainvoke/astream — bypassing START and _entry_router
    entirely. LangGraph's own resume machinery threads the value down
    into the waiting interrupt() call.
    """
    if state.get("plan_confirmed", False):
        return "execute_step"
    return "clarify_goal"


def build_research_graph(checkpointer=None):
    g = StateGraph(ResearchState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("clarify_goal",            clarify_goal)
    g.add_node("create_plan",             create_plan)
    g.add_node("check_plan_confirmation", check_plan_confirmation)
    g.add_node("execute_step",            execute_step)
    g.add_node("reflect",                 reflect)
    g.add_node("finish",                  finish)

    # -- Edges ----------------------------------------------------------------
    g.add_conditional_edges(
        START,
        _entry_router,
        path_map={
            "clarify_goal": "clarify_goal",
            "execute_step": "execute_step",
        },
    )

    # clarify_goal either:
    #   a) ends the run (goal_ready=False) — user replies next turn, which
    #      re-enters at START → _entry_router → clarify_goal again, or
    #   b) falls straight through to create_plan (goal_ready=True) in the
    #      same invocation — no user round-trip for goal confirmation.
    g.add_conditional_edges(
        "clarify_goal",
        lambda state: "create_plan" if state.get("goal_ready", False) else "__end__",
        path_map={
            "create_plan": "create_plan",
            "__end__":     END,
        },
    )

    # create_plan always feeds the plan into the confirmation interrupt.
    g.add_edge("create_plan", "check_plan_confirmation")

    # check_plan_confirmation pauses with interrupt() — shows the plan and
    # waits for the user to confirm or give revision feedback.
    # On resume: confirmed → execute_step; not confirmed → create_plan
    # revises the existing plan, then the loop repeats.
    g.add_conditional_edges(
        "check_plan_confirmation",
        confirmation_router,
        path_map={
            "execute_step": "execute_step",
            "create_plan":  "create_plan",
        },
    )

    g.add_edge("execute_step", "reflect")

    g.add_conditional_edges(
        "reflect",
        should_continue,
        path_map={
            "execute_step": "execute_step",
            "finish":       "finish",
        },
    )

    g.add_edge("finish", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# DeepResearchAgent
# ---------------------------------------------------------------------------

class DeepResearchAgent(BaseAgent):
    """
    Multi-step research agent.

    Flow:
      1. clarify_goal loops autonomously (LLM judges readiness) until the
         goal is clear enough to plan from — no separate "confirm goal?" step.
      2. create_plan drafts a plan, then check_plan_confirmation interrupts
         to show it to the user. User confirms or gives revision notes;
         create_plan revises in place and shows again. Repeats until confirmed.
      3. execute_step → reflect → (loop) → finish runs fully autonomously —
         no further interrupts since goal and plan are now fixed.

    In STANDALONE mode (direct use), this agent's own SQLite checkpointer
    is used. In SUPERVISOR mode, get_compiled_graph(checkpointer=True) is
    called instead, so the subgraph inherits the supervisor's checkpointer
    (same DB, namespaced by checkpoint_ns). invoke()/stream() are only
    used in standalone mode.
    """

    def __init__(self):
        self._graph = None

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="deep_research",
            name="Deep Research Agent",
            description=(
                "Handles complex questions needing multi-step research, "
                "synthesis across many sources, or thorough investigation. "
                "Clarifies the goal, then confirms the plan before starting. "
                "Best for: research reports, in-depth analysis, "
                "fact-checking across many sources."
            ),
            can_handle_long_tasks=True,
        )

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_research_graph(get_checkpointer("deep_research"))
        return self._graph

    def get_compiled_graph(self, checkpointer=None):
        """Return a compiled subgraph for supervisor use.
        checkpointer=True → inherit supervisor's checkpointer (recommended).
        checkpointer=None → no persistence (testing only).
        """
        return build_research_graph(checkpointer=checkpointer)

    async def _is_interrupted(self, cfg: RunnableConfig) -> bool:
        """Check if this thread is paused at an interrupt (standalone mode)."""
        try:
            snapshot = await self.graph.aget_state(cfg)
            return bool(snapshot.next)
        except Exception:
            return False

    async def _run(self, task: str, thread_id: str, config: Optional[RunnableConfig] = None):
        """
        Shared setup for standalone invoke()/stream().

        If the thread has a pending interrupt, resume it via
        Command(resume=task) — same config, no new state dict.
        Otherwise build/merge state and enter normally.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        if await self._is_interrupted(cfg):
            return Command(resume=task), cfg

        previous_state = await load_previous_state(self.graph, thread_id, "deep_research")

        if previous_state is None:
            state = ResearchState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                original_question=task,
                goal_ready=False,
                plan_confirmed=False,
            )
        else:
            state = merge_with_new_messages(previous_state, {
                "messages":   [HumanMessage(content=task)],
                "thread_id":  thread_id,
            })

        return state, cfg

    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        run_input, cfg = await self._run(task, thread_id, config)
        result = await self.graph.ainvoke(run_input, config=cfg)

        pending = result.get("__interrupt__")
        if pending:
            payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
            return payload.get("message", str(payload)) if isinstance(payload, dict) else str(payload)

        from langchain_core.messages import AIMessage
        for msg in reversed(list(result.get("messages", []))):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)

        return result.get("final_answer", "")

    async def stream(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        run_input, cfg = await self._run(task, thread_id, config)
        async for update in self.graph.astream(run_input, config=cfg, stream_mode="updates"):
            for node_name, node_update in update.items():
                yield {"node": node_name, "update": node_update}