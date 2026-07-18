from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def init_env() -> None:
    """Load .env from the project root (or nearest parent) exactly once.

    Uses ``find_dotenv`` so the path is never hardcoded and the same call
    works whether the code is run as a CLI (``python -m vision...``) or
    imported into an application.
    """
    load_dotenv(find_dotenv())
