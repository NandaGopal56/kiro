"""
LangGraph-compatible State schema with message-level IDs.
Tracks per-message steps, RAG/tool/LLM details, and assistant replies.
"""
from typing import List, Annotated
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import AnyMessage

class State(MessagesState):
    messages: Annotated[List[AnyMessage], add_messages]
    thread_id: str
    last_human_message_id: int
    last_ai_message_id: int
    tool_classifier_result: str