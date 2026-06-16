"""
Persistent SQLite checkpointer utilities for multi-turn agent conversations.

Provides:
  - get_checkpointer: Creates agent-specific AsyncSqliteSaver instances
  - load_previous_state: Restores prior state from checkpoint for a thread
  - merge_with_new_messages: Merges restored state with new incoming messages
"""

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# Shared event loop for bootstrapping AsyncSqliteSaver from sync get_checkpointer()
_SAVER_LOOP: Optional[asyncio.AbstractEventLoop] = None
_SAVER_THREAD: Optional[threading.Thread] = None


def _get_saver_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for checkpoint saver initialization."""
    global _SAVER_LOOP, _SAVER_THREAD
    
    if _SAVER_LOOP is None:
        _SAVER_LOOP = asyncio.new_event_loop()
        _SAVER_THREAD = threading.Thread(target=_SAVER_LOOP.run_forever, daemon=True)
        _SAVER_THREAD.start()
    
    return _SAVER_LOOP


def get_checkpointer(agent_id: str) -> AsyncSqliteSaver:
    """
    Create an AsyncSqliteSaver for a specific agent.

    Each agent gets its own SQLite database file in `.storages/<agent_id>.db`
    at the project root.

    Args:
        agent_id: Unique agent identifier (e.g., "deep_research", "personal", "supervisor")

    Returns:
        AsyncSqliteSaver: Ready to use async checkpoint saver
    """
    storages_dir = Path(__file__).parent.parent.parent / ".storages"
    storages_dir.mkdir(exist_ok=True, parents=True)

    db_path = storages_dir / f"{agent_id}.db"
    
    # Create aiosqlite connection and AsyncSqliteSaver in background loop
    async def _create_saver() -> AsyncSqliteSaver:
        conn = await aiosqlite.connect(str(db_path))
        return AsyncSqliteSaver(conn)
    
    loop = _get_saver_loop()
    future = asyncio.run_coroutine_threadsafe(_create_saver(), loop)
    return future.result()


async def load_previous_state(
    graph,
    thread_id: str,
    agent_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Load the last checkpoint/state for a given thread_id and agent.
    
    This retrieves the most recent graph state snapshot from the SQLite checkpoint
    for the specified thread. Returns None if no checkpoint exists for this thread.
    
    Args:
        graph: The compiled LangGraph instance
        thread_id: The thread/conversation ID
        agent_id: The agent ID (for debugging/logging purposes)
    
    Returns:
        Dict with the previous state values, or None if no checkpoint exists
    """
    from langgraph.types import RunnableConfig
    
    if not thread_id:
        return None
    
    try:
        cfg = RunnableConfig(configurable={"thread_id": thread_id})
        state = await graph.aget_state(cfg)
        
        if state and state.values:
            return dict(state.values)
        return None
    except Exception as e:
        print(f"Warning: Failed to load previous state for agent={agent_id}, thread={thread_id}: {e}")
        return None


def merge_with_new_messages(
    previous_state: Optional[Dict[str, Any]],
    new_state_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge a loaded previous checkpoint with new incoming state values.
    
    This function intelligently combines:
      - The full message history from the previous checkpoint
      - The new incoming messages or updates
    
    For the 'messages' field specifically:
      - Preserves all prior messages from the checkpoint
      - Removes duplicates
      - Appends only new messages
    
    For other fields:
      - New values override previous values (latest wins)
    
    Args:
        previous_state: The loaded checkpoint state (or None for first call)
        new_state_values: The new state values being passed in
    
    Returns:
        Dict: The merged state ready for graph execution
    """
    if previous_state is None:
        return new_state_values
    
    merged = dict(previous_state)
    
    # Merge messages specially: keep all old messages, add new ones
    previous_messages = previous_state.get("messages", [])
    new_messages = new_state_values.get("messages", [])
    
    if new_messages:
        # Collect IDs of messages already in previous state
        previous_ids = {
            getattr(m, "id", None) for m in previous_messages
            if hasattr(m, "id") and m.id
        }
        
        # Only add new messages that aren't already in the checkpoint
        unique_new_messages = [
            m for m in new_messages
            if not (hasattr(m, "id") and m.id and m.id in previous_ids)
        ]
        
        # Combine: old messages + unique new messages
        merged["messages"] = list(previous_messages) + unique_new_messages
    else:
        merged["messages"] = list(previous_messages)
    
    # For other fields, let new values override previous ones (except messages which we handled)
    for key, value in new_state_values.items():
        if key != "messages":
            merged[key] = value
    
    return merged
