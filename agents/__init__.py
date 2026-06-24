"""Agents package."""

__all__ = ["AgentProcessor", "create_service", "invoke_conversation"]


def __getattr__(name: str):
    from shared.logging import get_logger

    logger = get_logger("agents.init", log_file="agents_init.log")
    logger.debug("__getattr__ requested: %s", name)

    if name in {"AgentProcessor", "create_service"}:
        from .legacy.agent_processor import AgentProcessor, create_service

        return {"AgentProcessor": AgentProcessor, "create_service": create_service}[name]

    if name == "invoke_conversation":
        from .bot import invoke_conversation

        logger.debug("Providing invoke_conversation from .bot")
        return invoke_conversation

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
