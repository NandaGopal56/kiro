"""
Node definitions for the conversation graph.
Each function represents a node in the graph.
"""
import base64
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig

from ...shared.storage import (
    update_tool_result,
    add_tool_call,
    add_message,
    get_full_thread
)
from ...tools.basic_tools import basic_tools
from ..video_topic_buffer import video_buffer
from .prompts import (
    ASSISTANT_SYSTEM_PROMPT,
    CONVERSATION_SUMMARY_CONTEXT_PROMPT,
    summary_prompt,
    tool_classifier_prompt,
)
from .schemas import ToolClassifierOutput, ToolEnum
from .state import State

load_dotenv()

DEFAULT_CHAT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_CLASSIFIER_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_HISTORY_LENGTH = 2

# define the final list of tools
tools = basic_tools
tool_node = ToolNode(tools=tools)

llm_chat_model = None
llm_chat_model_with_tools = None
embeddings_generator = None


def get_chat_model():
    """Create the chat model lazily so importing this module stays testable."""
    global llm_chat_model
    if llm_chat_model is None:
        llm_chat_model = ChatOpenAI(model=DEFAULT_CHAT_MODEL)
    return llm_chat_model


def get_chat_model_with_tools():
    """Create the tool-bound chat model lazily."""
    global llm_chat_model_with_tools
    if llm_chat_model_with_tools is None:
        llm_chat_model_with_tools = get_chat_model().bind_tools(tools)
    return llm_chat_model_with_tools


def get_embeddings_generator():
    """Create the embedding model lazily so tests can import graph logic offline."""
    global embeddings_generator
    if embeddings_generator is None:
        embeddings_generator = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    return embeddings_generator


async def tool_node_processor(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process tool node safely with metadata tracking."""
    messages = state.get("messages", [])
    if not messages:
        # print("No messages found in state")
        return state

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        result = tool_node.invoke(tool_calls, config)

        tool_messages = result.get("messages", [])

        for tool_message in tool_messages:
            await update_tool_result(
                message_id=state.get("last_ai_message_id"),
                call_id=tool_message.tool_call_id,
                tool_output=tool_message.content
            )
        
        return {
            "messages": tool_messages
        }
    else:
        return state

def fanout_selector(state: State) -> list[str]:
    enum_to_node = {
        ToolEnum.INTERNET_SEARCH: "internet_search",
        ToolEnum.VIDEO_CAPTURE: "video_capture",
        ToolEnum.DOCUMENT_RAG: "document_rag_search",
    }

    tools = state.get("tool_classifier_result", [])

    selected = [enum_to_node[t] for t in tools if t in enum_to_node]

    # default path
    if not selected:
        return ["call_model"]

    return selected


def join_after_tools(state: State) -> State:
    """
    Acts as a synchronization barrier.
    All parallel nodes must reach here.
    """
    return state


async def tool_classifier_step(state: dict, config: RunnableConfig) -> Dict[str, Any]:
    last_message = state["messages"][-1].content

    llm = ChatOpenAI(
        model=DEFAULT_CLASSIFIER_MODEL,
        temperature=0,
    )

    parser = PydanticOutputParser(pydantic_object=ToolClassifierOutput)
    system_prompt = tool_classifier_prompt(parser.get_format_instructions())

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ])

    parsed: ToolClassifierOutput = parser.parse(response.content)

    return {
        "tool_classifier_result": parsed.tools
    }


async def video_capture(state: dict, config: RunnableConfig) -> Dict[str, Any]:
    
    latest_payload = video_buffer.latest()

    # Convert payload to a URL string that OpenAI vision expects
    data_url: str | None = None
    if isinstance(latest_payload, (bytes, bytearray)):
        b64 = base64.b64encode(latest_payload).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
    elif isinstance(latest_payload, str):
        # If it's already a data URL or a regular URL, pass through
        data_url = latest_payload

    if not data_url:
        return {
            "messages": [
                HumanMessage(content=[
                    {"type": "text", "text": "No recent camera frame available."}
                ])
            ]
        }

    return {
        "messages": [
            HumanMessage(content=[
                {"type": "text", "text": "Here is the image:"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ])
        ]
    }
    
def retrieve_data_from_doc_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for document RAG retrieval."""
    return {}


def retrieve_data_from_web_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for web RAG retrieval."""
    return {}

async def memory_state_update(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Rebuild message history from storage and update state."""

    thread_id=config.get("configurable", {}).get("thread_id")
    last_human_message = state.get("messages", [])[-1]

    human_message_id = await add_message(
        thread_id=thread_id, 
        role="user", 
        content=last_human_message.content
    )

    thread_id = config.get("metadata", {}).get("thread_id")
    thread_history: list[dict[str, str]] = await get_full_thread(thread_id)
    messages = state.get("messages", [])
    last_human_message = messages[-1] if messages else None

    existing_messages = []

    if thread_history:
        for message in thread_history:
            role = message['role']

            if role == 'user':
                msg = HumanMessage(
                    content=message['content']
                )
                existing_messages.append(msg)
            
            if role == 'assistant':
                ai_tool_calls = [ai_tool_call['tool_input_json'] for ai_tool_call in message['tool_calls']]
                msg = AIMessage(
                    content=message['content'],
                    tool_calls=ai_tool_calls
                )
                existing_messages.append(msg)

                tool_call_responses = {ai_tool_call['call_id']: ai_tool_call['tool_output'] for ai_tool_call in message['tool_calls']}

                for call_id, output in tool_call_responses.items():
                    existing_messages.append(ToolMessage(content=output, tool_call_id=call_id))

    # Rebuild message list: clear current messages, add old ones
    all_messages = [RemoveMessage(id=m.id) for m in messages] + existing_messages
    
    return {
        "messages": all_messages,
        "last_human_message_id": human_message_id
    }


def _last_human_message_index(messages: Sequence[Any]) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return None


def build_call_model_messages(
    messages: Sequence[Any],
    summary: str = "",
    max_history_length: int = MAX_HISTORY_LENGTH,
) -> List[Any]:
    """Build the model input from prompts, compact history, and current turn."""
    prompt_parts: List[Any] = [
        SystemMessage(content=ASSISTANT_SYSTEM_PROMPT)
    ]

    if summary:
        prompt_parts.append(
            SystemMessage(
                content=CONVERSATION_SUMMARY_CONTEXT_PROMPT.format(summary=summary)
            )
        )

    last_human_message_index = _last_human_message_index(messages)
    if last_human_message_index is None:
        prompt_parts.extend(messages)
        return prompt_parts

    recent_history = []
    human_message_count = 0

    for message in reversed(messages[:last_human_message_index]):
        recent_history.append(message)
        if isinstance(message, HumanMessage):
            human_message_count += 1
            if human_message_count == max_history_length:
                break

    prompt_parts.extend(reversed(recent_history))
    prompt_parts.extend(messages[last_human_message_index:])
    return prompt_parts


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process conversation through the language model with context."""
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    chat_messages = build_call_model_messages(messages, summary)

    response = await get_chat_model_with_tools().ainvoke(chat_messages, config=config)
    # print(f"LLM response: {response.content if response.content else response.tool_calls}")

    ai_message_id = await add_message(
        thread_id=config.get("configurable", {}).get("thread_id"), 
        role="assistant", 
        content=response.content
    )

    # if last message is AI message and tool calls are available, execute tools
    if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
        tool_calls = response.tool_calls

        for tool_call in tool_calls:
            await add_tool_call(
                message_id=ai_message_id,
                call_id=tool_call.get("id"),
                tool_input_json=tool_call
            )

    return {
        "messages": [response], 
        "last_ai_message_id": ai_message_id
    }



async def generate_embeddings_for_query(message: str) -> List[float]:
    """Generate embeddings for the given query text."""
    return await get_embeddings_generator().aembed_query(message)


async def path_selector_post_llm_call(state: State, config: RunnableConfig) -> str:
    """Decide which path to take after LLM call."""
    messages = state.get("messages", [])

    # fail-safe, if no messages, end the workflow
    if not messages:
        return "workflow_completion"

    # get the last message
    last_message = messages[-1]

    # if last message is AI message and tool calls are available, execute tools
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools_execution"

    # if len(messages) is more than 2, summarize conversation
    if len(messages) > 2:
        return "summarize_conversation"

    # else end the workflow
    return "workflow_completion"


async def summarize_conversation(state: State) -> Dict[str, Any]:
    """Summarize the conversation into a compact text."""
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    summary_message = summary_prompt(summary)

    chat_messages = [
        {"role": "system", "content": summary_message}] + [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": getattr(m, "content", "")}
        for m in messages
    ]

    config = RunnableConfig(
        metadata={
            "thread_id": state.get("thread_id", ""),
            "source_application": "summarize_conversation"
        }
    )

    response = await get_chat_model().ainvoke(chat_messages, config=config)

    return {"summary": getattr(response, "content", "")}


async def workflow_completion(state: State) -> Dict[str, Any]:
    """Workflow completion node."""
    return state
