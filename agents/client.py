from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from langgraph.types import RunnableConfig

from agents.base import BaseAgent
from agents.deep_research.graph import DeepResearchAgent
from agents.personal.graph import PersonalAgent
from agents.supervisor.grpah import Supervisor


class AgentGateway:
    """
    Thin client-side gateway for selecting and invoking agents.
    Keeps one shared entrypoint for direct agent use and supervisor use.
    """

    def __init__(self) -> None:
        self._agents: Optional[Dict[str, BaseAgent]] = None
        self._supervisor: Optional[Supervisor] = None

    def get_individual_agents(self) -> Dict[str, BaseAgent]:
        if self._agents is None:
            self._agents = {
                "personal": PersonalAgent(),
                "deep_research": DeepResearchAgent(),
            }
        return self._agents

    def get_supervisor(self) -> Supervisor:
        if self._supervisor is None:
            self._supervisor = Supervisor(agents=self.get_individual_agents())
        return self._supervisor

    def get_agent(self, agent_name: str) -> BaseAgent:
        if agent_name == "supervisor":
            return self.get_supervisor()

        agents = self.get_individual_agents()
        if agent_name not in agents:
            raise KeyError(f"Unknown agent '{agent_name}'")
        return agents[agent_name]

    async def invoke(
        self,
        agent_name: str,
        task: str,
        thread_id: str = "1",
        context: Optional[Dict[str, object]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        return await self.get_agent(agent_name).invoke(
            task=task,
            thread_id=thread_id,
            context=context,
            config=config,
        )

    async def stream(
        self,
        agent_name: str,
        task: str,
        thread_id: str = "1",
        context: Optional[Dict[str, object]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, object]]:
        async for update in self.get_agent(agent_name).stream(
            task=task,
            thread_id=thread_id,
            context=context,
            config=config,
        ):
            yield update

    def save_graphs(self) -> None:
        artifacts = Path(__file__).parent / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        personal = self.get_individual_agents()["personal"]
        research = self.get_individual_agents()["deep_research"]
        supervisor = self.get_supervisor()

        personal.graph.get_graph().draw_mermaid_png(
            output_file_path=str(artifacts / "personal_agent.png")
        )
        research.graph.get_graph().draw_mermaid_png(
            output_file_path=str(artifacts / "deep_research_agent.png")
        )
        supervisor.graph.get_graph().draw_mermaid_png(
            output_file_path=str(artifacts / "supervisor_shell.png")
        )
        supervisor.graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path=str(artifacts / "supervisor_full_system.png")
        )

        print(f"Graphs saved to {artifacts}")

    def registered_agents(self) -> Dict[str, str]:
        agents = self.get_individual_agents()
        return {
            **{aid: agent.info.name for aid, agent in agents.items()},
            "supervisor": self.get_supervisor().info.name,
        }


gateway = AgentGateway()
