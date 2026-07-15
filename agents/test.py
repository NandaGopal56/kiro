from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from agents.client import gateway
from agents.shared.logging import get_agent_logger
from shared.logging import log_state

logger = get_agent_logger("test", "smoke")


@dataclass(frozen=True)
class SmokeCase:
    agent: str
    message: str
    thread_id: str


CASES = (
    SmokeCase("personal", "Say hello in one sentence.", "101"),
    SmokeCase("deep_research", "Research the tradeoffs between batteries and hydrogen for transport.", "102"),
    SmokeCase("supervisor", "Say hello in one sentence.", "103"),
)


async def run_case(case: SmokeCase) -> None:
    logger.info("Running smoke case: %s", case)
    result = await gateway.invoke(case.agent, case.message, thread_id=case.thread_id)
    log_state(logger, "smoke_case.response", {"agent": case.agent, "response_preview": (result[:400] + "...") if len(result) > 400 else result})
    assert result.strip(), f"{case.agent} returned an empty response"
    logger.info("Smoke case passed for %s (thread=%s)", case.agent, case.thread_id)
    logger.info("Smoke case result: %s", (result[:1000] + "...") if len(result) > 1000 else result)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke tests for the agents.")
    parser.add_argument(
        "--agent",
        choices=["personal", "deep_research", "supervisor", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.agent == "all":
        selected = CASES
    else:
        selected = [case for case in CASES if case.agent == args.agent]

    for case in selected:
        await run_case(case)

    logger.info("All smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
