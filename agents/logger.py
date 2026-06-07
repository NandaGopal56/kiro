# logger.py
import logging
import os

os.makedirs("logs", exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("agents_bot")
logger.setLevel(getattr(logging, LOG_LEVEL))

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    # Example output:
    # 2025-09-16 22:45:12 - agents_bot - INFO - main.py:23 - Starting the agent...

    file_handler = logging.FileHandler("logs/agents.log")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.propagate = False
