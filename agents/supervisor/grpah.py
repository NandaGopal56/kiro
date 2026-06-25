from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional
from shared.logging import get_logger, log_state

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import ask_user, make_route_request, what_to_do
from .state import SupervisorState

DEBUG_MODE = True
logger = get_logger("agents.supervisor", log_file="supervisor.log")


def build_supervisor_graph(agents: Dict[str, BaseAgent]):
    """Build and compile the supervisor's LangGraph."""
    logger.info("Building supervisor graph for agents: %s", list(agents.keys()))
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", make_route_request(agents))
    g.add_node("ask_user",      ask_user)
    logger.debug("Added core nodes: route_request, ask_user")

    for agent_id, agent in agents.items():
        try:
            subgraph = agent.get_compiled_graph(checkpointer=True)
            logger.debug("Compiled subgraph for agent=%s (standalone compiled)", agent_id)
        except Exception:
            subgraph = agent.graph
            logger.debug("Using existing compiled graph for agent=%s", agent_id)
        g.add_node(agent_id, subgraph)

    # -- Edges ----------------------------------------------------------------
    g.add_edge(START, "route_request")

    g.add_conditional_edges(
        "route_request",
        what_to_do,
        path_map={"ask_user": "ask_user", **{agent_id: agent_id for agent_id in agents}},
    )

    g.add_edge("ask_user", END)
    for agent_id in agents:
        g.add_edge(agent_id, END)

    checkpointer = get_checkpointer("supervisor")
    compiled = g.compile(checkpointer=checkpointer)
    logger.info("Supervisor graph compiled with checkpointer for 'supervisor'")
    return compiled


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor(BaseAgent):
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self._graph = build_supervisor_graph(agents)

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="supervisor",
            name="Supervisor",
            description="Routes requests across registered agents.",
            can_handle_long_tasks=True,
        )

    @property
    def graph(self):
        return self._graph

    async def _pending_interrupt(self, thread_id: str, cfg: RunnableConfig) -> bool:
        """
        Check whether this thread is paused at an interrupt ANYWHERE in the
        graph, including inside a subgraph node (e.g. deep_research's
        check_plan_confirmation).

        subgraphs=True is mandatory: without it, aget_state only reads the
        supervisor's own top-level state and never sees an interrupt() raised
        inside a child graph's checkpoint_ns.
        """
        if not thread_id:
            return False
        try:
            snapshot = await self._graph.aget_state(cfg, subgraphs=True)
            pending = bool(snapshot.next)
            logger.debug("_pending_interrupt thread=%s pending=%s next=%s", thread_id, pending, snapshot.next)
            return pending
        except Exception as e:
            logger.debug("_pending_interrupt thread=%s error=%s", thread_id, e)
            return False

    async def _run(self, task: str, thread_id: str, config: Optional[RunnableConfig]):
        """
        Shared setup for invoke()/stream().

        If the thread is paused at an interrupt anywhere in the graph
        (including inside a subgraph), resume it with Command(resume=task)
        using the SAME config — no new state dict. This is the correct
        LangGraph resume pattern: LangGraph's own machinery delivers `task`
        to the waiting interrupt() call.

        Only when nothing is pending do we fall through to the normal
        route_request path, which rebuilds state and classifies the intent.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})
        logger.info("_run start: thread=%s task_preview=%s", thread_id, (task[:200] + "...") if len(task) > 200 else task)

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)
        logger.debug("Saved user message idempotent for thread=%s", thread_id)

        if await self._pending_interrupt(thread_id, cfg):
            logger.debug("_run thread=%s resuming pending interrupt reply=%r", thread_id, task)
            return Command(resume=task), cfg

        previous_state = await load_previous_state(self.graph, thread_id, "supervisor")
        logger.debug("Loaded previous_state present=%s for thread=%s", bool(previous_state), thread_id)
        log_state(logger, "supervisor.previous_state", previous_state)

        if previous_state is None:
            state = SupervisorState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                user_input=task,
            )
            logger.debug("Initialized new SupervisorState for thread=%s", thread_id)
        else:
            state = merge_with_new_messages(previous_state, {
                "messages":   [HumanMessage(content=task)],
                "thread_id":  thread_id,
                "user_input": task,
            })
            logger.debug("Merged previous state with new message for thread=%s", thread_id)
            log_state(logger, "supervisor.merged_state", state)

        return state, cfg

    async def _extract_response(self, result: Dict[str, Any], thread_id: str) -> str:
        from agents.shared.memory import load_thread
        db_history = await load_thread(thread_id)
        for msg in reversed(db_history):
            if msg.get("role") == "assistant":
                return msg.get("content") or ""
        messages = list(result.get("messages", []))
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return result.get("response", "")

    async def invoke(
        self,
        task: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        run_input, cfg = await self._run(task, thread_id, config)
        result = await self._graph.ainvoke(run_input, config=cfg)

        # If this invocation caused a NEW interrupt (e.g. deep_research just
        # drafted a plan and paused), return the interrupt's message directly.
        # No assistant message is in the DB yet at this point.
        pending = result.get("__interrupt__")
        if pending:
            payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
            logger.debug("invoke thread=%s new interrupt payload=%r", thread_id, payload)
            return payload.get("message", str(payload)) if isinstance(payload, dict) else str(payload)

        return await self._extract_response(result, thread_id)

    async def stream(
        self,
        task: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        run_input, cfg = await self._run(task, thread_id, config)
        async for update in self._graph.astream(run_input, config=cfg, stream_mode="updates"):
            for node_name, node_update in update.items():
                yield {"node": node_name, "update": node_update}