from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import ask_user, make_route_request, what_to_do
from .state import SupervisorState

def build_supervisor_graph(agents: Dict[str, BaseAgent]):
    """Build and compile the supervisor's LangGraph."""
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", make_route_request(agents))  # remove parent_graph=g
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

    # Use persistent SQLite checkpointer for the supervisor
    checkpointer = get_checkpointer("supervisor")

    # Agents that have an interactive clarify/confirm loop need the supervisor
    # to pause after they run so the user can reply before the next invocation.
    interactive_agents = [
        agent_id for agent_id, agent in agents.items()
        if getattr(agent, "requires_clarification", False)
    ]

    graph = g.compile(
        checkpointer=checkpointer,
        # interrupt_after=interactive_agents if interactive_agents else None,
    )
    return graph

# ---------------------------------------------------------------------------
# Supervisor — the public-facing class main.py instantiates
# ---------------------------------------------------------------------------

class Supervisor(BaseAgent):
    """
    Top-level entry point.

    Create one at startup:
        supervisor = Supervisor(agents={
            "personal":      PersonalAgent(),
            "deep_research": DeepResearchAgent(),
        })

    Then call it per request:
        answer = await supervisor.run("...", thread_id="user-123")
    """

    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self._graph = build_supervisor_graph(agents)

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="supervisor",
            name="Supervisor",
            description=(
                "Routes requests across registered agents and can act as a "
                "single entry point for multi-agent execution."
            ),
            can_handle_long_tasks=True,
        )

    @property
    def graph(self):
        return self._graph

    async def _extract_response(self, result: Dict[str, Any], thread_id: str) -> str:
        from agents.shared.memory import load_thread
        db_history = await load_thread(thread_id)
        for msg in reversed(db_history):
            if msg.get("role") == "assistant":
                return msg.get("content") or ""
        # Fallback to the old method if database is empty or doesn't have assistant messages
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
        """
        Handle a user message end-to-end with persistent state restoration.
        
        Loads previous checkpoint for the thread_id, merges with new message,
        then routes the request to the appropriate agent.
        
        Returns the final response as a plain string.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        # Load previous checkpoint for this thread_id
        previous_state = await load_previous_state(self.graph, thread_id, "supervisor")

        if previous_state is None:
            # First call — start a new session
            initial_state = SupervisorState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                user_input=task,
            )
            result = await self._graph.ainvoke(initial_state, config=cfg)
        else:
            # Subsequent call — merge new message with previous checkpoint state
            new_state_values = {
                "messages": [HumanMessage(content=task)],
                "thread_id": thread_id,
                "user_input": task,
            }
            merged_state = merge_with_new_messages(previous_state, new_state_values)
            result = await self._graph.ainvoke(merged_state, config=cfg)

        return await self._extract_response(result, thread_id)

    async def stream(
        self,
        task: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the supervisor's progress node by node with persistent state restoration.
        
        Loads previous checkpoint for the thread_id, merges with new message,
        then streams all routing and agent updates.
        
        Each yielded dict has {"node": str, "update": dict}.
        Useful for showing the user "Routing your request..." then
        live updates from the chosen agent.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        # Load previous checkpoint for this thread_id
        previous_state = await load_previous_state(self.graph, thread_id, "supervisor")

        if previous_state is None:
            # First call — start a new session
            initial_state = SupervisorState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
                user_input=task,
            )
            async for update in self._graph.astream(
                initial_state, config=cfg, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    yield {"node": node_name, "update": node_update}
        else:
            # Subsequent call — merge new message with previous checkpoint state
            new_state_values = {
                "messages": [HumanMessage(content=task)],
                "thread_id": thread_id,
                "user_input": task,
            }
            merged_state = merge_with_new_messages(previous_state, new_state_values)
            async for update in self._graph.astream(
                merged_state, config=cfg, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    yield {"node": node_name, "update": node_update}