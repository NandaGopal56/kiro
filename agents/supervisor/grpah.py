# agents/supervisor/graph.py
#
# The supervisor graph — the single entry point for all user requests.
#
# Flow:
#   START
#     → route_request    (which agent should handle this?)
#         → delegate     (hand off to the chosen agent, return answer)
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

from agents.base import BaseAgent

from .nodes import (
    ask_user,
    delegate,
    route_request,
    set_agent_registry,
    what_to_do,
)
from .state import SupervisorState


def build_supervisor_graph():
    """Build and compile the supervisor's LangGraph."""
    g = StateGraph(SupervisorState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("route_request", route_request)
    g.add_node("delegate",      delegate)
    g.add_node("ask_user",      ask_user)

    # -- Edges ----------------------------------------------------------------
    g.add_edge(START, "route_request")

    g.add_conditional_edges(
        "route_request",
        what_to_do,
        path_map={
            "delegate": "delegate",
            "ask_user": "ask_user",
        },
    )

    g.add_edge("delegate", END)
    g.add_edge("ask_user", END)

    graph = g.compile(checkpointer=MemorySaver())
    return graph


# ---------------------------------------------------------------------------
# Supervisor — the public-facing class main.py instantiates
# ---------------------------------------------------------------------------

class Supervisor:
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
        # Push the agent registry into the nodes module
        set_agent_registry(agents)
        self.agents = agents
        self._graph = build_supervisor_graph()

    async def run(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> str:
        """
        Handle a user message end-to-end.
        Returns the final response as a plain string.
        """
        config = RunnableConfig(configurable={"thread_id": thread_id})

        initial_state = SupervisorState(
            messages=[HumanMessage(content=user_input)],
            thread_id=thread_id,
            user_input=user_input,
        )

        result = await self._graph.ainvoke(initial_state, config=config)
        return result.get("response", "")

    async def stream(
        self,
        user_input: str,
        thread_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the supervisor's progress node by node.
        Each yielded dict has {"node": str, "update": dict}.

        Useful for showing the user "Routing your request..." then
        live updates from the chosen agent.
        """
        config = RunnableConfig(configurable={"thread_id": thread_id})

        initial_state = SupervisorState(
            messages=[HumanMessage(content=user_input)],
            thread_id=thread_id,
            user_input=user_input,
        )

        async for update in self._graph.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_name, node_update in update.items():
                yield {"node": node_name, "update": node_update}

    def registered_agents(self) -> Dict[str, str]:
        """Return {agent_id: agent_name} — useful for health checks / admin UIs."""
        return {aid: a.info.name for aid, a in self.agents.items()}