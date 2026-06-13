# agents/supervisor/prompts.py
#
# Prompts for the supervisor.
# The routing prompt is the most important one — it determines which agent gets the task.

# ---------------------------------------------------------------------------
# Router prompt
# The {agent_list} placeholder is filled in at runtime from registered agents.
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """You are a request router for an AI assistant system.

Your job: read the user's message and decide which agent should handle it.

Available agents:
{agent_list}

Rules:
- Choose the single best agent for this request.
- If the question is quick, conversational, or needs a fast answer → personal.
- If the question requires deep research, synthesis across many sources,
  or will take many steps to answer → deep_research.
- If you genuinely cannot tell what the user wants, ask for clarification.
- Never make up an agent ID that isn't in the list above.

Reply with a JSON object only — no explanation, no markdown:
{{
  "chosen_agent": "<agent_id from the list>",
  "reason": "<one sentence>",
  "needs_clarification": false,
  "clarification_question": ""
}}

If you need clarification, set needs_clarification to true and write the question:
{{
  "chosen_agent": "",
  "reason": "Request is ambiguous.",
  "needs_clarification": true,
  "clarification_question": "<question to ask the user>"
}}"""


# Shown to the user when the router asks for clarification
CLARIFICATION_PREFIX = "Before I route your request, I need to understand it better:\n\n"

# Shown when no agent could handle the request (should be rare)
FALLBACK_RESPONSE = (
    "I wasn't sure which assistant to send your request to. "
    "Could you rephrase or give me more details about what you need?"
)