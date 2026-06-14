from __future__ import annotations

import argparse
import asyncio


async def _run(host: str, port: int) -> None:
    from .ui_service import create_service

    service = create_service(host=host, port=port)
    await service.start()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await service.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live interaction UI.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(_run(args.host, args.port))


if __name__ == "__main__":
    main()
