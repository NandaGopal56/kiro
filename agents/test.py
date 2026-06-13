from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from agents.client import gateway


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
    print(f"\n== {case.agent} ==")
    result = await gateway.invoke(case.agent, case.message, thread_id=case.thread_id)
    assert result.strip(), f"{case.agent} returned an empty response"
    print(result)


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

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
