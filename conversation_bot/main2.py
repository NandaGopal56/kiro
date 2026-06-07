from datetime import datetime
from typing import Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from dotenv import load_dotenv

# ===============================
# 1. Load Environment
# ===============================
load_dotenv()


# ===============================
# 2. Global Config
# ===============================
USER_ID = "1"
NAMESPACE = (USER_ID, "conversation_memory")

llm = ChatOpenAI(
    model="gpt-5-nano-2025-08-07",
    temperature=0,
    streaming=True
)

store = InMemoryStore()


# ===============================
# 3. Define State
# ===============================
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# ===============================
# 4. Node Functions
# ===============================
def store_memory_node(state: State, store) -> State:
    """Store the latest user message into memory with timestamp."""
    if not state["messages"]:
        return state

    latest_message = state["messages"][-1].content

    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "content": latest_message,
        "context": "user_message"
    }

    memory_key = f"memory_{datetime.now().timestamp()}"
    store.put(NAMESPACE, memory_key, memory_entry)

    return state


def chat_node(state: State, store) -> State:
    """Generate LLM response using current and previous conversation context."""

    # Retrieve previous memories
    memory_context = ""
    if state["messages"]:
        latest_message = state["messages"][-1].content
        memories = store.search(NAMESPACE, query=latest_message, limit=5)

        if memories:
            memory_context = "\n".join(
                [f"Previous context: {memory.value['content']}" for memory in memories[-3:]]
            )

    system_prompt = f"""
You are a versatile and friendly conversational AI assistant. 
Your goal is to help the user with any query in a clear, engaging, and supportive way.

Context Handling:
- You may have access to previous conversation context and memory. 
- If relevant context exists, use it to personalize your response and maintain continuity.
- If little or no prior context is available, still provide a complete and helpful answer, 
  supplementing with general knowledge where needed.

Previous conversation context (if available):
{memory_context}

General Instructions:
- Be conversational, natural, and approachable.
- Reference previous topics or details when relevant.
- If you recall user preferences, facts, or goals, weave them naturally into your response.
- If memory is incomplete, gracefully bridge the gap instead of pointing it out.
- Always provide clear, useful, and accurate information.
- Ask thoughtful follow-up questions to keep the conversation engaging and discover more about the user.
- When uncertain or lacking memory, make reasonable suggestions or guide the user to clarify.

Role:
- You act as a generic personal assistant who can chat casually, explain concepts, 
  brainstorm ideas, help with work/study, or recall past conversations if context is available.
"""


    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages")
    ])

    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"messages": state["messages"]})

    # Append response
    new_messages = state["messages"] + [response]
    return {"messages": new_messages}


# ===============================
# 5. Graph Builder
# ===============================
def build_graph() -> StateGraph:
    graph_builder = StateGraph(State)

    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("store_memory_node", store_memory_node)

    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_edge("chat_node", "store_memory_node")
    graph_builder.add_edge("store_memory_node", END)

    return graph_builder.compile(checkpointer=MemorySaver(), store=store)


graph = build_graph()


# ===============================
# 6. Helper Functions
# ===============================
def chat_with_bot(message: str, config: Dict[str, Any]) -> str:
    """Send user message and return bot response."""
    input_state = {"messages": [{"role": "user", "content": message}]}
    response_state = graph.invoke(input_state, config=config)

    # Get the latest AI response (last item in messages list)
    ai_response = response_state["messages"][-1].content
    return ai_response


def print_conversation(message: str, response: str) -> None:
    """Pretty print conversation."""
    print("\n" + "=" * 50)
    print(f"User: {message}")
    print(f"Bot: {response}")
    print("=" * 50)


def show_memories(limit: int = 5) -> None:
    """Display recent stored memories."""
    all_items = list(store.list(NAMESPACE))
    print(f"Total memories stored: {len(all_items)}")

    for i, (key, memory) in enumerate(all_items[-limit:], 1):
        print(f"{i}. Key: {key}")
        print(f"   Content: {memory.value['content'][:100]}...")
        print(f"   Timestamp: {memory.value['timestamp']}")
        print()


# ===============================
# 7. Test Runner
# ===============================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_conversation_123"}}

    print("🤖 Starting Conversation Bot with Memory Test")
    print("=" * 60)

    tests = [
        ("Hi! My name is Alex and I'm a software engineer", "Introduction and Personal Information"),
        ("I love playing guitar and hiking on weekends", "Sharing Interests"),
        ("I work with Python and machine learning mostly", "Work-related Information"),
        ("What do you remember about my hobbies?", "Memory Recall Test"),
        ("Can you tell me what I mentioned about my work?", "Professional Background Recall"),
        ("I'm thinking of learning a new programming language", "Connecting New Information"),
        ("What's my name again?", "Name Recall Test"),
        ("Given my background and interests, what project would you recommend?", "Complex Memory Integration")
    ]

    for message, description in tests:
        print(f"\n📝 Test Case: {description}")
        response = chat_with_bot(message, config)
        print_conversation(message, response)

    print("\n🔍 Checking stored memories:")
    print("-" * 40)
    show_memories(limit=5)

    print("\n✅ Memory Test Complete!")
