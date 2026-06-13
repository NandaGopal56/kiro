# agents/supervisor/graph.py
#
# The supervisor graph — the single entry point for all user requests.
#
# Flow:
#   START
#     → route_request    (which agent should handle this?)
#         → personal | deep_research  (subgraph nodes)
#         → ask_user     (ask for clarification if needed)
#     → END
#
# Usage:
#   supervisor = Supervisor(agents={"personal": PersonalAgent(), ...})
#   answer = await supervisor.run("What is the capital of France?", thread_id="t1")

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agents.base import AgentInfo, BaseAgent

from .nodes import ask_user, make_route_request, what_to_do
from .state import SupervisorState


def build_supervisor_graph(agents: Dict[str, BaseAgent]):
    """Build and compile the supervisor's LangGraph."""
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", make_route_request(agents))
    g.add_node("ask_user",      ask_user)

    for agent_id, agent in agents.items():
        g.add_node(agent_id, agent.graph)

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

    graph = g.compile(checkpointer=MemorySaver())
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

    def _extract_response(self, result: Dict[str, Any]) -> str:
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
        Handle a user message end-to-end.
        Returns the final response as a plain string.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        initial_state = SupervisorState(
            messages=[HumanMessage(content=task)],
            thread_id=thread_id,
            user_input=task,
        )

        result = await self._graph.ainvoke(initial_state, config=cfg)
        return self._extract_response(result)

    async def stream(
        self,
        task: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the supervisor's progress node by node.
        Each yielded dict has {"node": str, "update": dict}.

        Useful for showing the user "Routing your request..." then
        live updates from the chosen agent.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

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

    def registered_agents(self) -> Dict[str, str]:
        """Return {agent_id: agent_name} — useful for health checks / admin UIs."""
        return {aid: a.info.name for aid, a in self.agents.items()}
