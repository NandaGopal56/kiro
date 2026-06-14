from __future__ import annotations

import argparse
import asyncio

from .inmemory_bus import create_bus


async def _smoke() -> None:
    bus = create_bus()
    received = asyncio.Event()

    def on_message(topic, payload):
        print(f"{topic}: {payload}")
        received.set()

    await bus.start()
    bus.subscribe("smoke", on_message)
    await bus.publish("smoke", {"ok": True})
    await asyncio.wait_for(received.wait(), timeout=1)
    await bus.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication bus utilities.")
    parser.add_argument("--smoke", action="store_true", help="Run pub/sub smoke test.")
    args = parser.parse_args()
    if args.smoke:
        asyncio.run(_smoke())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
