"""Routing layer — decides which provider/model serves a given turn.

This is the ONLY place provider/model choices live. Callers ask for a *task*
(or let the classifier infer one from the message); they never name a provider.
Swapping a task onto a different provider/model — say routing REASONING to
`anthropic:claude-sonnet-4-6` — is a one-line edit to `ROUTES` and touches
nothing else in the app.

The classifier is deliberately a simple heuristic. It can be replaced with a
smarter one (even an LLM call) later without changing any call site, because
everything downstream depends only on `route()` returning a `RouteDecision`.
"""

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SMALL_TALK = "small_talk"
    REASONING = "reasoning"
    STRUCTURED = "structured"
    LONG_CONTEXT = "long_context"
    DEFAULT = "default"


class RouteTarget(BaseModel):
    provider: str
    model: str
    params: dict = Field(default_factory=dict)


class RouteDecision(BaseModel):
    task_type: TaskType
    provider: str
    model: str
    params: dict
    reason: str


# ── The routing table ──────────────────────────────────────────────────────
# One line per task. Provider-specific params (e.g. OpenAI's response_format)
# are allowed here — this is the layer permitted to know about providers.
# Models below are verified against Loom's catalog.
ROUTES: dict[TaskType, RouteTarget] = {
    TaskType.SMALL_TALK: RouteTarget(
        provider="openai",
        model="gpt-4o-mini",
        params={"temperature": 0.7, "max_tokens": 256},
    ),
    TaskType.REASONING: RouteTarget(
        provider="openai",
        model="gpt-4o",
        params={"temperature": 0.3, "max_tokens": 1024},
    ),
    TaskType.STRUCTURED: RouteTarget(
        provider="openai",
        model="gpt-4o-mini",
        params={
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        },
    ),
    TaskType.LONG_CONTEXT: RouteTarget(
        provider="openai",
        model="gpt-4.1",
        params={"temperature": 0.3, "max_tokens": 1024},
    ),
    TaskType.DEFAULT: RouteTarget(
        provider="openai",
        model="gpt-4o-mini",
        params={"temperature": 0.5, "max_tokens": 512},
    ),
}


# ── Classification heuristic (simple on purpose; swap freely) ───────────────
_LONG_CONTEXT_CHAR_THRESHOLD = 8000
_REASONING_CHAR_THRESHOLD = 400
_REASONING_KEYWORDS = (
    "why",
    "explain",
    "analy",
    "compare",
    "calculate",
    "step by step",
    "reason",
    "pros and cons",
    "trade-off",
    "tradeoff",
)
_SMALL_TALK_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "how are you",
    "good morning",
    "good evening",
    "nice to meet you",
)


def _is_small_talk(text: str) -> bool:
    return len(text) <= 40 and any(text.startswith(p) for p in _SMALL_TALK_PATTERNS)


def classify(
    message: str,
    *,
    wants_json: bool = False,
    context_chars: int = 0,
) -> TaskType:
    """Infer a task type from the message and a few caller-supplied hints.

    `wants_json` is set by callers that need structured output; `context_chars`
    is the size of any conversation history that will be prepended.
    """
    text = message.strip().lower()

    if wants_json:
        return TaskType.STRUCTURED
    if _is_small_talk(text):
        return TaskType.SMALL_TALK
    if context_chars >= _LONG_CONTEXT_CHAR_THRESHOLD or len(message) >= _LONG_CONTEXT_CHAR_THRESHOLD:
        return TaskType.LONG_CONTEXT
    if len(message) >= _REASONING_CHAR_THRESHOLD or any(k in text for k in _REASONING_KEYWORDS):
        return TaskType.REASONING
    return TaskType.DEFAULT


def route(
    message: str,
    *,
    task_type: TaskType | None = None,
    wants_json: bool = False,
    context_chars: int = 0,
) -> RouteDecision:
    """Resolve a message to a concrete provider/model decision.

    Pass `task_type` to force a route; otherwise it is inferred. Returns a copy
    of the target params so callers can mutate them freely.
    """
    resolved = task_type or classify(
        message, wants_json=wants_json, context_chars=context_chars
    )
    target = ROUTES.get(resolved, ROUTES[TaskType.DEFAULT])
    return RouteDecision(
        task_type=resolved,
        provider=target.provider,
        model=target.model,
        params=dict(target.params),
        reason=f"task '{resolved.value}' -> {target.provider}:{target.model}",
    )
