from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import (
    check_confirmation,
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
    Decide where a fresh invocation enters the graph.

    - No goal yet (or goal not confirmed and never set): this is either the
      very first question, or the user is replying to a clarifying question.
      Either way, run clarify_goal — it already knows how to tell those
      cases apart (no prev_goal vs. prev_goal + clarification).
    - Goal is set but not confirmed: the previous turn asked the user to
      confirm/clarify, and this invocation IS that reply — go straight to
      check_confirmation to interpret it.
    - Goal already confirmed (e.g. a resumed/replayed session): skip ahead
      to plan creation.
    """
    if state.get("goal_confirmed", False):
        return "create_plan"
    if state.get("goal"):
        return "check_confirmation"
    return "clarify_goal"


def build_research_graph(checkpointer=None):
    g = StateGraph(ResearchState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("clarify_goal",        clarify_goal)
    g.add_node("check_confirmation",  check_confirmation)
    g.add_node("create_plan",         create_plan)
    g.add_node("execute_step",        execute_step)
    g.add_node("reflect",             reflect)
    g.add_node("finish",              finish)

    # -- Edges ----------------------------------------------------------------
    # Route at entry instead of always starting at clarify_goal. Each
    # invocation does exactly one logical step then returns — no interrupt
    # needed, since the caller (DeepResearchAgent) controls turn-taking by
    # invoking once per user message.
    g.add_conditional_edges(
        START,
        _entry_router,
        path_map={
            "clarify_goal":       "clarify_goal",
            "check_confirmation": "check_confirmation",
            "create_plan":        "create_plan",
        },
    )

    # clarify_goal always ends the run here — it either asked a question
    # or re-showed the confirmation prompt. The user's next message is what
    # drives the next invocation, which _entry_router will send to
    # check_confirmation.
    g.add_edge("clarify_goal", END)

    # check_confirmation → create_plan  OR  loop back to clarify_goal
    g.add_conditional_edges(
        "check_confirmation",
        confirmation_router,
        path_map={
            "create_plan":  "create_plan",
            "clarify_goal": "clarify_goal",
        },
    )

    g.add_edge("create_plan",  "execute_step")
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

    # No interrupt: each call to invoke()/stream() is one full pass through
    # the graph from the appropriate entry point, driven by goal/goal_confirmed
    # in the (checkpointed) state. The checkpointer still persists state
    # between calls so the graph can pick up where it left off.
    graph = g.compile(checkpointer=checkpointer)
    return graph


# ---------------------------------------------------------------------------
# DeepResearchAgent
# ---------------------------------------------------------------------------

class DeepResearchAgent(BaseAgent):
    """
    Multi-step research agent with goal clarification before starting.

    Conversation flow:
      1. Agent asks clarifying questions + shows refined goal
      2. User confirms ("yes") or gives corrections
      3. Agent builds a plan, executes it step-by-step, reflects, finishes

    Each call to invoke()/stream() handles exactly one user message and
    returns. The graph's entry router uses the persisted state
    (goal / goal_confirmed) to decide whether this message is the original
    question, a reply to a clarifying question, or arrives after the goal
    is already confirmed.
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
                "Clarifies the goal with the user before starting. "
                "Best for: research reports, in-depth analysis, "
                "fact-checking across many sources."
            ),
            can_handle_long_tasks=True,
        )

    @property
    def graph(self):
        if self._graph is None:
            # Standalone agent: compile with its own persistent checkpointer
            self._graph = build_research_graph(get_checkpointer("deep_research"))
        return self._graph

    def get_compiled_graph(self, checkpointer=None):
        """Return a compiled subgraph. When `checkpointer` is None the
        compiled subgraph will inherit the parent's checkpointer when added
        as a node to a parent graph (recommended for supervisor use).
        """
        return build_research_graph(checkpointer=checkpointer)

    async def _run(self, task: str, thread_id: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Shared setup for invoke()/stream(): persist the user message,
        load + merge prior state, and return (state, cfg) ready to run."""
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        previous_state = await load_previous_state(self.graph, thread_id, "deep_research")
        print(f"DEBUG: deep_research._run previous_state for thread={thread_id} -> {'present' if previous_state else 'none'}")

        if previous_state is None:
            state = ResearchState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                original_question=task,
                goal_confirmed=False,
            )
        else:
            new_state_values = {
                "messages": [HumanMessage(content=task)],
                "thread_id": thread_id,
            }
            state = merge_with_new_messages(previous_state, new_state_values)
            print(f"DEBUG: deep_research._run merged_state_keys={list(state.keys())} merged_messages={len(state.get('messages', []))}")

        return state, cfg

    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Handle one user message for this research thread (question, a reply
        to a clarifying question, or anything in between) and return the
        agent's response text.
        """
        state, cfg = await self._run(task, thread_id, config)
        result = await self.graph.ainvoke(state, config=cfg)

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
        """Stream all node updates for one user message."""
        state, cfg = await self._run(task, thread_id, config)

        async for update in self.graph.astream(state, config=cfg, stream_mode="updates"):
            for node_name, node_update in update.items():
                yield {"node": node_name, "update": node_update}