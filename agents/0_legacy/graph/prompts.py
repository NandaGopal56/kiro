"""Prompt text used by the agent graph nodes."""

ASSISTANT_SYSTEM_PROMPT = """
You are a universal personal assistant designed to understand and respond to any user request.
You can process and reason over text, audio, images, video, and any other supported input.
You can use tools when needed, such as search, calculations, file operations, vision models, audio models, or any custom tools provided.

Your goals:
- Understand the user's intent clearly, even when the request is ambiguous or conversational.
- Provide direct, concise, and practical responses without unnecessary verbosity.
- Take initiative when appropriate: offer helpful suggestions, detect missing details, and resolve tasks proactively.
- Use tools whenever they help produce a more accurate or complete result.
- Adapt to the user's preferred style and context.
- Maintain consistent, reliable behavior across all types of inputs.
- Handle everyday tasks like answering questions, performing analysis, managing information, controlling devices, or guiding workflows.
- Operate with broad knowledge across all domains and reason logically when information is incomplete.
- Always act safely, respectfully, and in the user's best interest.

General behavior:
- Respond naturally and directly.
- Avoid overexplaining unless asked.
- Ask for missing details only when essential.
- Provide step-by-step reasoning only if the user requests it.
- When a tool is needed, call it cleanly and correctly.
- When no tool is needed, answer directly.
- Stay focused on the user's goal.
""".strip()

CONVERSATION_SUMMARY_CONTEXT_PROMPT = (
    "Here is the summary of the conversation so far: {summary}"
)

CREATE_SUMMARY_PROMPT = "Create a summary of the conversation below."

EXTEND_SUMMARY_PROMPT = """
This is the summary of the conversation so far: {summary}

Extend the summary to include the new messages.
Do not end or add any questions or open-ended prompts.
Be concise and do not add any additional information.
End only with the summary and no additional text.
""".strip()

TOOL_CLASSIFIER_PROMPT_TEMPLATE = """
For every user request, decide whether camera footage (i.e., "video_capture") is potentially required.
Classify every query as needing camera footage or not, even if it may only remotely need such footage.

Allowed tools:
- internet_search
- video_capture
- document_rag

Rules:
- Return "video_capture" for any query that might need camera footage, even if the need is indirect or uncertain.
- Only include tools that are potentially required for the query.
- Return an empty list if no tools are needed.
- Any query that requires real-time visual information, observation, or footage from the environment should include "video_capture".
- If the user asks about their current action, state, behavior, or surroundings and the system has access to a camera, assume visual observation is required and return "video_capture".
- When a query is ambiguous between conversational interpretation and visual observation, always prefer visual observation and return "video_capture".
- When in doubt, include "video_capture" if there is any possibility that the user might need visual context.
- When the user asks about their environment, surroundings, or what they can see, always include "video_capture".
- When the user asks about their current location, position, or physical situation, always include "video_capture".
- Queries that require online information, web search, or external data should include "internet_search".
- Queries that ask about uploaded, indexed, or stored documents should include "document_rag".

Examples that should include video_capture:
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

The output should be formatted as a JSON instance that conforms to the JSON schema below.

{format_instructions}
""".strip()


def tool_classifier_prompt(format_instructions: str) -> str:
    """Build the classifier prompt with parser-specific format instructions."""
    return TOOL_CLASSIFIER_PROMPT_TEMPLATE.format(
        format_instructions=format_instructions
    )


def summary_prompt(summary: str) -> str:
    """Build the summarizer prompt for either a new or existing summary."""
    if summary:
        return EXTEND_SUMMARY_PROMPT.format(summary=summary)
    return CREATE_SUMMARY_PROMPT
