from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from langgraph.types import RunnableConfig

from agents.base import BaseAgent
from agents.deep_research.graph import DeepResearchAgent
from agents.personal.graph import PersonalAgent
from agents.supervisor.grpah import Supervisor
from shared.logging import get_logger, log_state


logger = get_logger("agents.client", log_file="supervisor.log")


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
            logger.info("Initializing individual agents")
            self._agents = {
                "personal": PersonalAgent(),
                "deep_research": DeepResearchAgent(),
            }
            logger.debug("Initialized agents: %s", list(self._agents.keys()))
        return self._agents

    def get_supervisor(self) -> Supervisor:
        if self._supervisor is None:
            logger.info("Creating Supervisor with agents")
            self._supervisor = Supervisor(agents=self.get_individual_agents())
            logger.debug("Supervisor created: %s", type(self._supervisor))
        return self._supervisor

    def get_agent(self, agent_name: str) -> BaseAgent:
        logger.debug("Selecting agent '%s'", agent_name)
        if agent_name == "supervisor":
            return self.get_supervisor()

        agents = self.get_individual_agents()
        if agent_name not in agents:
            logger.error("Unknown agent requested: %s", agent_name)
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
        agent_logger = get_logger(f"agent.{agent_name}", log_file=f"{agent_name}.log")
        agent_logger.info("Invoke request: agent=%s thread=%s task_summary=%s", agent_name, thread_id, (task[:200] + "...") if len(task) > 200 else task)
        log_state(agent_logger, "invoke.context", context)
        try:
            result = await self.get_agent(agent_name).invoke(
                task=task,
                thread_id=thread_id,
                context=context,
                config=config,
            )
            agent_logger.info("Invoke completed: agent=%s thread=%s response_length=%d", agent_name, thread_id, len(result) if result else 0)
            return result
        except Exception as e:
            agent_logger.exception("Exception while invoking agent %s: %s", agent_name, e)
            raise

    async def stream(
        self,
        agent_name: str,
        task: str,
        thread_id: str = "1",
        context: Optional[Dict[str, object]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, object]]:
        agent_logger = get_logger(f"agent.{agent_name}", log_file=f"{agent_name}.log")
        agent_logger.info("Stream requested: agent=%s thread=%s", agent_name, thread_id)
        log_state(agent_logger, "stream.context", context)
        async for update in self.get_agent(agent_name).stream(
            task=task,
            thread_id=thread_id,
            context=context,
            config=config,
        ):
            agent_logger.debug("Stream update: %s", update)
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

        logger.info("Graphs saved to %s", artifacts)

    def registered_agents(self) -> Dict[str, str]:
        agents = self.get_individual_agents()
        return {
            **{aid: agent.info.name for aid, agent in agents.items()},
            "supervisor": self.get_supervisor().info.name,
        }


gateway = AgentGateway()
