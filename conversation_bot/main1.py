import asyncio
import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

# Define the state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

load_dotenv()

# Initialize the client with GitHub configuration
client = MultiServerMCPClient(
    {
        "github": {
            "url": "https://api.githubcopilot.com/mcp/x/repos/",
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                "User-Agent": "MCP-Client/1.0",
                "Accept": "application/json"
            }
        }
    }
)

async def load_github_tools():
    """Load and return GitHub tools from the MCP client."""
    try:
        tools = await client.get_tools()
        print(f"Loaded {len(tools)} GitHub tools")
        return tools
    except Exception as e:
        print(f"Error loading GitHub tools: {e}")
        return []

def create_github_agent(tools):
    """Create and return a LangGraph agent configured with GitHub tools."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-5-nano-2025-08-07",
        temperature=0,
        streaming=True
    )
    
    # Define the model node with enhanced debugging
    def call_model(state: AgentState):
        print(f"\n🤖 MODEL NODE: Processing {len(state['messages'])} messages")
        
        # Add system message if it's the first message
        if not any(isinstance(m, SystemMessage) for m in state["messages"]):
            system_message = SystemMessage(
                content="""You are a helpful AI assistant that can interact with GitHub. 
                You have access to various GitHub tools to help with repository management, 
                code search, and other GitHub operations. Use these tools when needed 
                to assist the user with their GitHub-related tasks."""
            )
            state["messages"] = [system_message] + list(state["messages"])
        
        # Call the model with tool binding
        response = llm.bind_tools(tools).invoke(state["messages"])
        
        # Check if the response contains tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 TOOL CALLS DETECTED: {len(response.tool_calls)} tool(s) will be called")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"   Tool {i+1}: {tool_call['name']} with args: {tool_call['args']}")
        else:
            print("💬 NO TOOL CALLS: AI responding directly")
        
        return {"messages": [response]}
    
    # Enhanced tools node with debugging
    def debug_tools_node(state: AgentState):
        print(f"\n🛠️ TOOLS NODE: Executing tools...")
        tool_node = ToolNode(tools)
        result = asyncio.run(tool_node.ainvoke(state))
        print(result)
        # Check for tool results
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    print(f"✅ Tool '{msg.name}' executed successfully")
                    print(f"   Result preview: {str(msg.content)[:100]}...")
        
        return result
    
    # Create the graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("agent", call_model)
    builder.add_node("tools", debug_tools_node)
    
    # Define edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "agent")
    
    # Compile the graph
    return builder.compile()

def analyze_conversation_flow(messages):
    """Analyze the conversation flow to understand tool usage patterns."""
    print(f"\n📊 CONVERSATION ANALYSIS:")
    print(f"Total messages: {len(messages)}")
    
    tool_calls_count = 0
    tool_results_count = 0
    
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_count += len(msg.tool_calls)
            print(f"Message {i}: AI made {len(msg.tool_calls)} tool call(s)")
        elif isinstance(msg, ToolMessage):
            tool_results_count += 1
            print(f"Message {i}: Tool result from '{msg.name}'")
        elif isinstance(msg, HumanMessage):
            print(f"Message {i}: Human input")
        elif isinstance(msg, SystemMessage):
            print(f"Message {i}: System message")
        elif isinstance(msg, AIMessage):
            print(f"Message {i}: AI response (no tools)")
    
    print(f"Total tool calls made: {tool_calls_count}")
    print(f"Total tool results received: {tool_results_count}")

async def main():
    # Load GitHub tools
    tools = await load_github_tools()
    if not tools:
        print("Failed to load GitHub tools. Exiting...")
        return
    
    print("Available GitHub Tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Create the agent
    agent = create_github_agent(tools)
    
    print(f"\nGitHub Assistant is ready with {len(tools)} tools! Type 'exit' to quit.")
    print("Try asking something like: 'List repositories for user octocat'")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'debug':
                print("Available debug commands:")
                print("- 'debug': Show this help")
                print("- 'tools': List available tools")
                continue
            
            if user_input.lower() == 'tools':
                print("\nAvailable tools:")
                for tool in tools:
                    print(f"- {tool.name}: {tool.description}")
                continue
                
            print(f"\n🚀 PROCESSING: '{user_input}'")
            
            # Process the input through the agent
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=user_input)]
            })
            
            # Analyze the conversation flow
            analyze_conversation_flow(response["messages"])
            
            # Get the last message from the agent
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"\n🤖 AI Response: {last_message.content}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())