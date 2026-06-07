"""
Node definitions for the conversation graph.
Each function represents a node in the graph.
"""
import logging
import base64
from typing import Dict, Any, List, Union
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, ToolMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.types import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from ..logger import logger
from .state import State
from ..tools.basic_tools import basic_tools
from ..storage import update_tool_result, add_tool_call, add_message, get_full_thread
from .schemas import ToolClassifierOutput, ToolEnum
from ..VideoTopicBuffer import video_buffer

load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# define the final list of tools
tools = basic_tools
tool_node = ToolNode(tools=tools)

llm_chat_model = ChatOpenAI(model="gpt-5-nano-2025-08-07")
llm_chat_model_with_tools = llm_chat_model.bind_tools(tools)

embeddings_generator = OpenAIEmbeddings(model="text-embedding-ada-002")



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
        model="gpt-4o-mini",
        temperature=0,
    )

    parser = PydanticOutputParser(pydantic_object=ToolClassifierOutput)

    system_prompt = f"""
        For every user request, decide whether camera footage (i.e., "video_capture") is potentially required. Classify every query as needing camera footage or not, even if it may only remotely need such footage.
        Allowed tools:
        - internet_search
        - video_capture
        Rules:
        - Return "video_capture" for any query that might need camera footage, even if the need is indirect or uncertain.
        - Only include tools that are potentially required for the query.
        - Return an empty list if no tools are needed.
        - Any query that requires real-time visual information, observation, or footage from the environment should include "video_capture".
        - If the user asks about their current action, state, behavior, or surroundings
          AND the system has access to a camera,
          ASSUME visual observation is required and return "video_capture".
        - When a query is ambiguous between conversational interpretation and visual observation,
          ALWAYS prefer visual observation and return "video_capture".
        - When in doubt, err on the side of caution and include "video_capture" if there's any possibility
          that the user might need visual context.
        - When the user asks about their environment, surroundings, or what they can see,
          ALWAYS include "video_capture" to provide accurate visual context.
        - When the user asks about their current location, position, or physical situation,
          ALWAYS include "video_capture" to provide accurate visual context.

        This includes but is not limited to queries like:
        - "Show me what the robot sees"
        - "Capture live video"
        - "Analyze the current scene"
        - "Check if there is an obstacle ahead"
        - "What is in front of me?"
        - "Detect people or objects nearby"
        - "Record the surroundings"
        - "Monitor activity in this area"
        - "Look around the room"
        - "Inspect this object"
        - "Track moving objects"
        - "Detect colors or shapes in view"
        - Queries that require online information, web search, or external data should include "internet_search".
        The output should be formatted as a JSON instance that conforms to the JSON schema below.

        {parser.get_format_instructions()}
    """

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ])

    parsed: ToolClassifierOutput = parser.parse(response.content)

    return {
        "tool_classifier_result": parsed.tools
    }


async def path_selector_post_tool_classifier(state: dict, config: RunnableConfig) -> Dict[str, Any]:
    tool_classifier_result = state.get("tool_classifier_result")

    if tool_classifier_result == "none":
        return "call_model"
    
    elif tool_classifier_result == 'video_capture':
        return "video_capture"
    
    elif tool_classifier_result == 'internet_search':
        return "internet_search"
    
    else:
        return "call_model"

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
    print("Retrieving data from document RAG")
    pass


def retrieve_data_from_web_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for web RAG retrieval."""
    print("Retrieving data from web RAG")
    pass

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


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process conversation through the language model with context."""
    
    messages = state.get("messages", [])
    summary = state.get("summary", "")

    prompt_parts = []

    # Build system context
    system_prompt = SystemMessage("""
        You are a universal personal assistant designed to understand and respond to any user request.
        You can process and reason over text, audio, images, video, and any other supported input.
        You can use tools when needed, such as search, calculations, file operations, vision models, audio models, or any custom tools provided.
        Your goals:
        Understand the user’s intent clearly, even when the request is ambiguous or conversational.
        Provide direct, concise, and practical responses without unnecessary verbosity.
        Take initiative when appropriate: offer helpful suggestions, detect missing details, and resolve tasks proactively.
        Use tools whenever they help produce a more accurate or complete result.
        Adapt to the user’s preferred style and context.
        Maintain consistent, reliable behavior across all types of inputs.
        Handle everyday tasks like answering questions, performing analysis, managing information, controlling devices, or guiding workflows.
        Be flexible, capable, and capable of dealing with both simple and complex requests.
        Operate with broad knowledge across all domains and reason logically when information is incomplete.
        Always act safely, respectfully, and in the user's best interest.
        General Behavior:
        Respond naturally and directly.
        Avoid overexplaining unless asked.
        Ask for missing details only when essential.
        Provide step-by-step reasoning only if the user requests it.
        When a tool is needed, call it cleanly and correctly.
        When no tool is needed, answer directly.
        You are always focused, helpful, and aligned with the user’s goals.
    """)
    prompt_parts.append(system_prompt)

    # if summary is available, add it to the system prompt
    if summary:
        rag_prompt = SystemMessage(f"""Here is the summary of the conversation so far: {summary}""")
        prompt_parts.append(rag_prompt)


    # extract last human message & the index of the last human message
    last_human_message = None
    last_human_message_index = None

    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_message = message
            last_human_message_index = i


    # Add last conversation history
    MAX_HISTORY_LENGTH = 2
    COUNT = 0
    recent_history = []

    # extract anything before last human_human_message_index till max_history_length considering the only the human messages will be covered when calculating the MAX_HISTORY_LENGTH
    for index, message in reversed(list(enumerate(messages[:last_human_message_index]))):
        
        if isinstance(message, HumanMessage):
            recent_history.append(message)
            COUNT += 1
            if COUNT == MAX_HISTORY_LENGTH:
                break
        else:
            recent_history.append(message)

    recent_history = reversed(recent_history)
    prompt_parts.extend(recent_history)



    # After adding the recent history, add the last human message and any flowwing messages like AI message containing tool calls or tool messages etc as well
    current_conversation = messages[last_human_message_index:]
    prompt_parts.extend(current_conversation)

    # format the prompt parts into a chat prompt
    chat_prompt = ChatPromptTemplate.from_messages(prompt_parts)
    # print(f"Chat prompt: {chat_prompt.format()}")
               
    # invoke the LLM with the chat prompt
    response = await llm_chat_model_with_tools.ainvoke(chat_prompt.messages, config=config)
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
    return await embeddings_generator.aembed_query(message)


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


def branch_selection_for_RAG(state: State) -> Union[str, List[str]]:
    """Decide which RAG branches to run based on state."""
    doc_enabled = state.get("is_doc_rag_enabled", False)
    web_enabled = state.get("is_search_enabled", False)

    if doc_enabled and web_enabled:
        return ["doc_rag_search", "web_rag_search"]

    if doc_enabled:
        return "doc_rag_search"

    if web_enabled:
        return "web_rag_search"

    return "call_model"





async def summarize_conversation(state: State) -> Dict[str, Any]:
    """Summarize the conversation into a compact text."""
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    if summary:
        summary_message = (
            f"This is the summary of the conversation so far: {summary}\n\n"
            "Extend the summary to include the new messages."
            "Do not end or add any questions or open-ended prompts."
            "Be concise and do not add any additional information."
            "End only with the summary and no additional text."
        )
    else:
        summary_message = "Create a summary of the conversation below."

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

    response = await llm_chat_model.ainvoke(chat_messages, config=config)

    return {"summary": getattr(response, "content", "")}


async def workflow_completion(state: State) -> Dict[str, Any]:
    """Workflow completion node."""
    return state