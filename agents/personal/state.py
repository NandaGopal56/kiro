# agents/personal/state.py

from __future__ import annotations

from typing import Annotated, List, Optional, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class PersonalState(TypedDict, total=False):
    # Full conversation — LangGraph merges new messages automatically
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Which thread this conversation belongs to
    thread_id: str

    # Compact rolling summary of older messages
    # Prepended to prompts so the LLM has context without a huge message list
    summary: str

    # Which context-gathering steps the classifier decided to run
    # Possible values: "video_capture", "web_search", "document_search"
    steps_needed: List[str]

    # Storage IDs for the current turn — used to attach tool calls/results
    current_user_message_id: Optional[str]
    current_assistant_message_id: Optional[str]