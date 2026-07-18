"""Conversation orchestration — one turn, end to end.

Ties the pieces together for a single turn: take the OpenAI `messages` Bolna
sent (the transcript), enrich the prompt with any facts we've remembered for
this conversation, let the routing layer pick a provider/model, call Loom, and
persist lightweight derived facts for next time.

The endpoint is stateless in the OpenAI sense — Bolna's messages are the source
of truth for the dialogue. Memory is opportunistic: only used when a
conversation id is supplied, and only to carry small facts (like the caller's
name) across turns.
"""

import re

from backend.models.llm import LLMResult
from backend.models.openai import ChatMessage
from backend.services.loom import get_loom_service
from backend.services.memory import (
    ConversationState,
    Message,
    get_memory,
    render_prompt,
)
from backend.services.routing import RouteDecision, route

# Simple, high-precision name capture. Deliberately a heuristic — a smarter
# extractor could replace this without changing anything else.
_NAME_RE = re.compile(
    r"\b(?:my name is|i am|i'm|this is)\s+([A-Z][a-z]+)",
)

_DEFAULT_PREAMBLE = (
    "You are a helpful voice assistant. Keep replies short and natural for speech."
)


def _preamble(messages: list[ChatMessage]) -> str:
    system_texts = [m.text().strip() for m in messages if m.role == "system"]
    joined = "\n".join(t for t in system_texts if t)
    return joined or _DEFAULT_PREAMBLE


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.text()
    return ""


async def handle_turn(
    messages: list[ChatMessage],
    *,
    conversation_id: str | None = None,
    request_params: dict | None = None,
) -> tuple[LLMResult, RouteDecision]:
    user_text = _last_user_text(messages)

    memory = get_memory()
    facts: dict[str, str] = {}
    if conversation_id:
        facts = (await memory.get(conversation_id)).facts

    # Reuse the memory renderer by shaping Bolna's messages into a state object.
    state = ConversationState(
        conversation_id=conversation_id or "stateless",
        messages=[
            Message(role=m.role, content=m.text())
            for m in messages
            if m.role in ("user", "assistant")
        ],
        facts=facts,
    )
    prompt = render_prompt(state, system_preamble=_preamble(messages))

    decision = route(user_text, context_chars=len(prompt))

    # Routing owns provider/model + baseline params; honor generation knobs
    # (temperature/max_tokens) the caller explicitly set.
    params = {**decision.params, **(request_params or {})}
    result = await get_loom_service().generate(
        prompt,
        provider=decision.provider,
        model=decision.model,
        params=params,
    )

    if conversation_id:
        await memory.append_message(conversation_id, "user", user_text)
        await memory.append_message(conversation_id, "assistant", result.text)
        if "name" not in facts:
            match = _NAME_RE.search(user_text)
            if match:
                await memory.set_fact(conversation_id, "name", match.group(1))

    return result, decision
