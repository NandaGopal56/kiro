from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.deep_research.state import ResearchState
    from agents.personal.state import PersonalState
    from agents.supervisor.state import SupervisorState


def get_formatted_recent_history(
    state: ResearchState | PersonalState | SupervisorState,
    max_messages: int = 10,
) -> str:
    """Render the last `max_messages` human/AI turns as a flat transcript.

    Used to give confirmation-stage prompts (goal, plan) enough recent context
    to interpret short replies like "yes" or "change the timeframe" correctly,
    without dumping the entire message history into every LLM call.

    System and tool messages are skipped since they add noise, not user intent.
    """
    messages = state.get("messages", [])

    turns: list[str] = []

    for msg in messages:
        role = getattr(msg, "type", None)

        if role == "human":
            speaker = "User"
        elif role == "ai":
            speaker = "Assistant"
        else:
            continue

        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)

        content = content.strip()
        if content:
            turns.append(f"{speaker}: {content}")

    return "\n".join(turns[-max_messages:]) if turns else "(no prior conversation)"