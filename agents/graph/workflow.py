"""
Workflow definition for the conversation graph.
Builds and configures the state graph with all nodes and edges.
"""
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver as CheckpointMemorySaver
from .state import State
from .nodes import (
    tool_classifier_step,
    memory_state_update,
    call_model,
    retrieve_data_from_doc_RAG,
    retrieve_data_from_web_RAG,
    summarize_conversation,
    workflow_completion,
    tool_node_processor,
    video_capture,
    path_selector_post_llm_call,
    fanout_selector, 
    join_after_tools 
)

def build_workflow() -> StateGraph:
    workflow = StateGraph(State)

    # --------------------
    # Nodes
    # --------------------
    workflow.add_node("memory_state_update", memory_state_update)
    workflow.add_node("tool_classifier_step", tool_classifier_step)

    # Parallel-capable nodes
    workflow.add_node("video_capture", video_capture)
    workflow.add_node("internet_search", retrieve_data_from_web_RAG)
    workflow.add_node("document_rag_search", retrieve_data_from_doc_RAG)

    # Join node
    workflow.add_node("join_after_tools", join_after_tools)

    # LLM + rest
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools_execution", tool_node_processor)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("workflow_completion", workflow_completion)

    # --------------------
    # Edges
    # --------------------
    workflow.add_edge(START, "memory_state_update")
    workflow.add_edge("memory_state_update", "tool_classifier_step")

    # ---- FAN-OUT ----
    # classifier returns List[Enum]
    workflow.add_conditional_edges(
        source="tool_classifier_step",
        path=fanout_selector,   # returns list of node names
        path_map={
            "video_capture": "video_capture",
            "internet_search": "internet_search",
            "document_rag_search": "document_rag_search",
            "call_model": "call_model",
        }
    )

    # ---- JOIN ----
    workflow.add_edge("video_capture", "join_after_tools")
    workflow.add_edge("internet_search", "join_after_tools")
    workflow.add_edge("document_rag_search", "join_after_tools")

    # ---- CONTINUE ----
    workflow.add_edge("join_after_tools", "call_model")

    workflow.add_conditional_edges(
        source="call_model",
        path=path_selector_post_llm_call,
        path_map={
            "tools_execution": "tools_execution",
            "summarize_conversation": "summarize_conversation",
            "workflow_completion": "workflow_completion",
        }
    )

    workflow.add_edge("tools_execution", "call_model")
    workflow.add_edge("summarize_conversation", "workflow_completion")
    workflow.add_edge("workflow_completion", END)

    return workflow.compile(checkpointer=CheckpointMemorySaver())