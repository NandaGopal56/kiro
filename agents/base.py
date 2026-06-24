from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, Optional

from langgraph.types import RunnableConfig
from shared.logging import get_logger, log_state


logger = get_logger("agents.base", log_file="agents_base.log")


class AgentInfo:
    """
    Simple description of an agent.
    The supervisor reads this to know which agent handles what.
    """
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        can_handle_long_tasks: bool = False,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description          # shown to the routing LLM
        self.can_handle_long_tasks = can_handle_long_tasks


class BaseAgent(abc.ABC):
    """
    Every agent subclasses this and implements three methods.
    The supervisor only ever calls .info, .invoke(), and .stream().
    """

    @property
    @abc.abstractmethod
    def info(self) -> AgentInfo:
        """Return static info about this agent. Called at startup."""
        ...

    @abc.abstractmethod
    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Run the agent and return the final response as a string.
        Use this for standard request/response interactions.
        """
        ...


def log_agent_call(agent: "BaseAgent", task: str, thread_id: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Utility helper to log agent invocation start state.

    Callers (or agent implementations) can call this before heavy operations.
    """
    try:
        name = getattr(agent, "__class__", type(agent)).__name__
    except Exception:
        name = str(agent)
    a_logger = get_logger(f"agent.{name.lower()}", log_file=f"{name.lower()}.log")
    a_logger.info("Starting agent: %s thread=%s", name, thread_id)
    log_state(a_logger, "agent.start_state", {"task": task, "context": context})

    @abc.abstractmethod
    async def stream(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream intermediate state updates.
        Use this for long-running tasks or real-time UI feedback.
        Yield dicts with at minimum {"node": str, "update": dict}.
        """
        ...
