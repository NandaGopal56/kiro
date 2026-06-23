from __future__ import annotations

import logging
import json
from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import ask_user, make_route_request, what_to_do
from .state import SupervisorState

# ---------------------------------------------------------------------------
# Debug logging — toggle DEBUG_MODE to enable/disable state logging.
# When enabled, every node logs its full input state to
# .logs/supervisor.log before doing any work.
# ---------------------------------------------------------------------------
DEBUG_MODE = True

_log = logging.getLogger("supervisor")
if not _log.handlers:
    import os
    os.makedirs(".logs", exist_ok=True)
    _fh = logging.FileHandler(".logs/supervisor.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_fh)
    _log.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)


def _log_state(node_name: str, state: Dict[str, Any]) -> None:
    """Log the full state at node entry. No-op when DEBUG_MODE is False."""
    if not DEBUG_MODE:
        return
    loggable = {}
    for k, v in state.items():
        if isinstance(v, str) and len(v) > 500:
            loggable[k] = v[:500] + f"... [truncated, total={len(v)}]"
        elif k == "messages":
            loggable[k] = f"[{len(v) if hasattr(v, '__len__') else '?'} messages]"
        else:
            loggable[k] = v
    _log.debug("NODE ENTRY: %s\nSTATE: %s", node_name, json.dumps(loggable, default=str, indent=2))


def build_supervisor_graph(agents: Dict[str, BaseAgent]):
    """Build and compile the supervisor's LangGraph."""
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", make_route_request(agents))
    g.add_node("ask_user",      ask_user)

    for agent_id, agent in agents.items():
        try:
            subgraph = agent.get_compiled_graph(checkpointer=True)
        except Exception:
            subgraph = agent.graph
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
    return g.compile(checkpointer=checkpointer)


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
            _log.debug("_pending_interrupt thread=%s pending=%s next=%s",
                       thread_id, pending, snapshot.next)
            return pending
        except Exception as e:
            _log.debug("_pending_interrupt thread=%s error=%s", thread_id, e)
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

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        if await self._pending_interrupt(thread_id, cfg):
            _log.debug("_run thread=%s resuming pending interrupt reply=%r", thread_id, task)
            return Command(resume=task), cfg

        previous_state = await load_previous_state(self.graph, thread_id, "supervisor")

        if previous_state is None:
            state = SupervisorState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                user_input=task,
            )
        else:
            state = merge_with_new_messages(previous_state, {
                "messages":   [HumanMessage(content=task)],
                "thread_id":  thread_id,
                "user_input": task,
            })

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
            _log.debug("invoke thread=%s new interrupt payload=%r", thread_id, payload)
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