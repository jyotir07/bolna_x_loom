"""OpenAI-compatible request/response schemas.

Bolna's "custom LLM" speaks the OpenAI Chat Completions protocol, so our
Bolna-facing endpoint must accept and return these shapes exactly. The `model`
field Bolna sends is just the identifier it was configured with — provider/model
selection is our routing layer's job, not the caller's.
"""

import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any = ""

    def text(self) -> str:
        """OpenAI content may be a string or a list of parts; normalize to text."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts = [
                p.get("text", "")
                for p in self.content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "".join(parts)
        return "" if self.content is None else str(self.content)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None


# ── Non-streaming response ─────────────────────────────────────────────────
class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_new_id)
    object: str = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[Choice]
    usage: CompletionUsage


# ── Streaming response (chat.completion.chunk) ─────────────────────────────
class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=_new_id)
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[ChunkChoice]
