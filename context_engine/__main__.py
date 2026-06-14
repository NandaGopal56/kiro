from __future__ import annotations

import argparse
import asyncio


async def _ingest(source: str) -> None:
    from .engine import ContextEngine

    engine = ContextEngine()
    ids = await engine.ingest(source)
    print("\n".join(ids))


async def _retrieve(query: str, k: int) -> None:
    from .engine import ContextEngine

    engine = ContextEngine()
    docs = await engine.retrieve(query, k=k)
    for doc in docs:
        print(doc.as_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Context engine utilities.")
    parser.add_argument("--ingest", help="Source path to ingest.")
    parser.add_argument("--retrieve", help="Query to retrieve.")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    if args.ingest:
        asyncio.run(_ingest(args.ingest))
    elif args.retrieve:
        asyncio.run(_retrieve(args.retrieve, args.k))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
