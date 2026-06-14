from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agents.base import AgentInfo, BaseAgent

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


def build_research_graph():
    g = StateGraph(ResearchState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("clarify_goal",        clarify_goal)
    g.add_node("check_confirmation",  check_confirmation)
    g.add_node("create_plan",         create_plan)
    g.add_node("execute_step",        execute_step)
    g.add_node("reflect",             reflect)
    g.add_node("finish",              finish)

    # -- Edges ----------------------------------------------------------------
    g.add_edge(START, "clarify_goal")

    # clarify_goal → check_confirmation (after graph resumes from interrupt)
    g.add_edge("clarify_goal", "check_confirmation")

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

    # Pause after clarify_goal so the user can reply
    # The supervisor feeds the user's reply back in and calls .ainvoke(None) to resume
    graph = g.compile(checkpointer=MemorySaver(), interrupt_after=["clarify_goal"])
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
            self._graph = build_research_graph()
        return self._graph

    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Start or continue a research session.

        First call:   task = the user's research question
                      → graph runs clarify_goal, pauses, returns clarification message
        Second call+: task = the user's reply (confirmation or corrections)
                      → graph resumes, processes reply, either loops or proceeds to research
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        # Check if a session already exists for this thread
        existing = await self.graph.aget_state(cfg)
        session_exists = existing and existing.values

        if not session_exists:
            # First call — start a new session
            initial_state = ResearchState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                original_question=task,
                goal_confirmed=False,
            )
            result = await self.graph.ainvoke(initial_state, config=cfg)
        else:
            # Subsequent call — inject the user's reply and resume
            await self.graph.aupdate_state(
                cfg,
                {"messages": [HumanMessage(content=task)]},
            )
            result = await self.graph.ainvoke(None, config=cfg)

        # Return the last AI message as the response
        messages = list(result.get("messages", []))
        from langchain_core.messages import AIMessage
        for msg in reversed(messages):
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
        """
        Stream all node updates, including clarification and research steps.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        existing = await self.graph.aget_state(cfg)
        session_exists = existing and existing.values

        if not session_exists:
            initial_state = ResearchState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                original_question=task,
                goal_confirmed=False,
            )
            async for update in self.graph.astream(
                initial_state, config=cfg, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    yield {"node": node_name, "update": node_update}
        else:
            await self.graph.aupdate_state(
                cfg,
                {"messages": [HumanMessage(content=task)]},
            )
            async for update in self.graph.astream(
                None, config=cfg, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    yield {"node": node_name, "update": node_update}
