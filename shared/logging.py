from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / ".logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    if not name.endswith(".log"):
        name = name + ".log"
    return name


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Return a module-level logger writing to a file under .logs.

    - ``name`` is the logger name (e.g. ``agents.client``).
    - ``log_file`` if provided overrides the log filename (e.g. ``personal.log``).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        return logger

    filename = _sanitize_filename(log_file) if log_file else _sanitize_filename(name)
    path = LOGS_DIR / filename

    fh = logging.handlers.RotatingFileHandler(
        filename=str(path), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(level)

    fmt = (
        "%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d %(funcName)s() - %(message)s"
    )
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.log_file_path = str(path)  # type: ignore[attr-defined]
    return logger


def log_state(logger: logging.Logger, label: str, state: Any) -> None:
    """Log a structured representation of ``state`` at DEBUG level."""
    try:
        payload = json.dumps(state, default=str, ensure_ascii=False)
    except Exception:
        payload = repr(state)

    logger.debug("STATE %s: %s", label, payload)
