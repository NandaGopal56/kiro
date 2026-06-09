import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.graph import nodes
from agents.graph.schemas import ToolEnum
from agents.graph.prompts import summary_prompt, tool_classifier_prompt


def test_fanout_selector_defaults_to_call_model():
    assert nodes.fanout_selector({"tool_classifier_result": []}) == ["call_model"]


def test_fanout_selector_maps_classifier_tools():
    state = {
        "tool_classifier_result": [
            ToolEnum.VIDEO_CAPTURE,
            ToolEnum.INTERNET_SEARCH,
            ToolEnum.DOCUMENT_RAG,
        ]
    }

    assert nodes.fanout_selector(state) == [
        "video_capture",
        "internet_search",
        "document_rag_search",
    ]


def test_build_call_model_messages_uses_prompt_summary_and_recent_history():
    messages = [
        HumanMessage(content="old 1"),
        AIMessage(content="answer 1"),
        HumanMessage(content="old 2"),
        AIMessage(content="answer 2"),
        HumanMessage(content="old 3"),
        HumanMessage(content="current"),
    ]

    result = nodes.build_call_model_messages(messages, summary="known context")

    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], SystemMessage)
    assert "known context" in result[1].content
    assert [m.content for m in result[2:]] == [
        "old 2",
        "answer 2",
        "old 3",
        "current",
    ]


@pytest.mark.asyncio
async def test_video_capture_returns_no_frame_message(monkeypatch):
    class EmptyBuffer:
        def latest(self):
            return None

    monkeypatch.setattr(nodes, "video_buffer", EmptyBuffer())

    result = await nodes.video_capture({}, {})

    assert result["messages"][0].content == [
        {"type": "text", "text": "No recent camera frame available."}
    ]


@pytest.mark.asyncio
async def test_video_capture_converts_bytes_to_data_url(monkeypatch):
    class BytesBuffer:
        def latest(self):
            return b"image-bytes"

    monkeypatch.setattr(nodes, "video_buffer", BytesBuffer())

    result = await nodes.video_capture({}, {})

    content = result["messages"][0].content
    assert content[0] == {"type": "text", "text": "Here is the image:"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_path_selector_routes_tool_calls_to_tool_execution():
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "get_current_weather",
                "args": {"location": "Bengaluru"},
            }
        ],
    )

    result = await nodes.path_selector_post_llm_call({"messages": [ai_message]}, {})

    assert result == "tools_execution"


@pytest.mark.asyncio
async def test_path_selector_ends_empty_state():
    result = await nodes.path_selector_post_llm_call({"messages": []}, {})

    assert result == "workflow_completion"


def test_prompt_builders_keep_parser_instructions_outside_nodes():
    classifier_prompt = tool_classifier_prompt("return json")

    assert "return json" in classifier_prompt
    assert summary_prompt("") == "Create a summary of the conversation below."
    assert "existing summary" in summary_prompt("existing summary")
