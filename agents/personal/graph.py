from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from agents.base import AgentInfo, BaseAgent
from agents.shared.checkpointer import get_checkpointer, load_previous_state, merge_with_new_messages

from .nodes import (
    call_llm,
    compress_history,
    decide_steps,
    fetch_doc_context,
    fetch_web_context,
    grab_video_frame,
    join_context,
    pick_context_steps,
    run_tools,
    what_next,
)
from .state import PersonalState


def build_personal_graph():
    """Build and compile the personal agent's LangGraph."""
    g = StateGraph(PersonalState)

    # -- Nodes ----------------------------------------------------------------
    g.add_node("decide_steps",       decide_steps)

    # Parallel context-gathering branches
    g.add_node("grab_video_frame",   grab_video_frame)
    g.add_node("fetch_web_context",  fetch_web_context)
    g.add_node("fetch_doc_context",  fetch_doc_context)

    g.add_node("join_context",       join_context)
    g.add_node("call_llm",           call_llm)
    g.add_node("run_tools",          run_tools)
    g.add_node("compress_history",   compress_history)

    # -- Edges ----------------------------------------------------------------
    g.add_edge(START, "decide_steps")

    # Fan-out: decide_steps → one or more context nodes (or straight to call_llm)
    g.add_conditional_edges(
        "decide_steps",
        pick_context_steps,
        path_map={
            "grab_video_frame":  "grab_video_frame",
            "fetch_web_context": "fetch_web_context",
            "fetch_doc_context": "fetch_doc_context",
            "call_llm":          "call_llm",
        },
    )

    # Fan-in: all context branches converge before calling the LLM
    g.add_edge("grab_video_frame",  "join_context")
    g.add_edge("fetch_web_context", "join_context")
    g.add_edge("fetch_doc_context", "join_context")
    g.add_edge("join_context",      "call_llm")

    # After the LLM responds: run tools, compress history, or finish
    g.add_conditional_edges(
        "call_llm",
        what_next,
        path_map={
            "run_tools":        "run_tools",
            "compress_history": "compress_history",
            "done":             END,
        },
    )

    # Tool loop: keep calling the LLM until it stops making tool calls
    g.add_edge("run_tools",       "call_llm")
    g.add_edge("compress_history", END)

    # Use persistent SQLite checkpointer for this agent
    checkpointer = get_checkpointer("personal")
    graph = g.compile(checkpointer=checkpointer)
    return graph


# ---------------------------------------------------------------------------
# PersonalAgent — the BaseAgent wrapper the supervisor talks to
# ---------------------------------------------------------------------------

class PersonalAgent(BaseAgent):
    """
    Everyday chat assistant with memory, vision, and tool use.
    Handles single-turn and short multi-turn conversations.
    """

    def __init__(self):
        self._graph = None   # built lazily on first call

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            agent_id="personal",
            name="Personal Assistant",
            description=(
                "Handles everyday chat, quick questions, task execution, "
                "web search, code running, and live camera vision. "
                "Best for short to medium conversations."
            ),
            can_handle_long_tasks=False,
        )

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_personal_graph()
        return self._graph

    async def invoke(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Invoke the personal agent with persistent state restoration.
        
        Loads previous checkpoint for the thread_id, merges with new message,
        then executes the graph.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        # Load previous checkpoint for this thread_id
        previous_state = await load_previous_state(self.graph, thread_id, "personal")

        if previous_state is None:
            # First call — start a new session
            initial_state = PersonalState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
            )
            result = await self.graph.ainvoke(initial_state, config=cfg)
        else:
            # Subsequent call — merge new message with previous checkpoint state
            new_state_values = {
                "messages": [HumanMessage(content=task)],
                "thread_id": thread_id,
            }
            merged_state = merge_with_new_messages(previous_state, new_state_values)
            result = await self.graph.ainvoke(merged_state, config=cfg)

        # Extract the last assistant message as the response
        messages = list(result.get("messages", []))
        from langchain_core.messages import AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""

    async def stream(
        self,
        task: str,
        thread_id: str,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream personal agent execution with persistent state restoration.
        
        Loads previous checkpoint for the thread_id, merges with new message,
        then streams all node updates.
        """
        cfg = config or RunnableConfig(configurable={"thread_id": thread_id})

        from agents.shared.memory import save_message_idempotent
        await save_message_idempotent(thread_id, "user", task)

        # Load previous checkpoint for this thread_id
        previous_state = await load_previous_state(self.graph, thread_id, "personal")

        if previous_state is None:
            # First call — start a new session
            initial_state = PersonalState(
                messages=[HumanMessage(content=task)],
                thread_id=thread_id,
            )
            async for update in self.graph.astream(
                initial_state, config=cfg, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    yield {"node": node_name, "update": node_update}
        else:
            # Subsequent call — merge new message with previous checkpoint state
            new_state_values = {
                "messages": [HumanMessage(content=task)],
                "thread_id": thread_id,
            }
            merged_state = merge_with_new_messages(previous_state, new_state_values)
            async for update in self.graph.astream(
                merged_state, config=cfg, stream_mode="updates"
            ):
                for node_name, node_update in update.items():
                    yield {"node": node_name, "update": node_update}
