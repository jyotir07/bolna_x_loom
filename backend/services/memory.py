"""Conversation memory — history, facts, and pending questions per call.

The store sits behind an async `ConversationMemory` interface so the backing
implementation (in-memory now, Redis later) can be swapped without touching
callers. Because Loom's `generate()` accepts a single `prompt` string and has no
messages API, `render_prompt` flattens a conversation's state into a labeled
transcript with a system preamble — the format the model actually receives.

The in-memory implementation mutates plain dicts with no `await` inside its
critical sections, so within a single event loop those mutations are atomic and
need no lock. A Redis implementation would handle its own concurrency.
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ConversationState(BaseModel):
    conversation_id: str
    messages: list[Message] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    pending_question: str | None = None

    def context_chars(self) -> int:
        """Rough size of the stored dialogue, for routing's long-context hint."""
        return sum(len(m.content) for m in self.messages)


class ConversationMemory(ABC):
    """Async interface for conversation storage. Swap the implementation freely."""

    @abstractmethod
    async def get(self, conversation_id: str) -> ConversationState: ...

    @abstractmethod
    async def append_message(
        self, conversation_id: str, role: Role, content: str
    ) -> None: ...

    @abstractmethod
    async def set_fact(self, conversation_id: str, key: str, value: str) -> None: ...

    @abstractmethod
    async def set_pending_question(
        self, conversation_id: str, question: str | None
    ) -> None: ...

    @abstractmethod
    async def clear(self, conversation_id: str) -> None: ...


class InMemoryConversationMemory(ConversationMemory):
    def __init__(self) -> None:
        self._store: dict[str, ConversationState] = {}

    def _state(self, conversation_id: str) -> ConversationState:
        state = self._store.get(conversation_id)
        if state is None:
            state = ConversationState(conversation_id=conversation_id)
            self._store[conversation_id] = state
        return state

    async def get(self, conversation_id: str) -> ConversationState:
        # Return a copy so callers can't mutate stored state directly.
        return self._state(conversation_id).model_copy(deep=True)

    async def append_message(
        self, conversation_id: str, role: Role, content: str
    ) -> None:
        self._state(conversation_id).messages.append(Message(role=role, content=content))

    async def set_fact(self, conversation_id: str, key: str, value: str) -> None:
        self._state(conversation_id).facts[key] = value

    async def set_pending_question(
        self, conversation_id: str, question: str | None
    ) -> None:
        self._state(conversation_id).pending_question = question

    async def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)


_DEFAULT_PREAMBLE = "You are a helpful voice assistant. Keep replies short and natural for speech."


def render_prompt(state: ConversationState, *, system_preamble: str = _DEFAULT_PREAMBLE) -> str:
    """Flatten conversation state into a single prompt for Loom.

    Layout: system preamble → known facts → pending question → labeled
    transcript, ending with an open `Assistant:` turn to cue the reply. The
    caller is expected to have already appended the latest user message.
    """
    lines: list[str] = [system_preamble.strip()]

    if state.facts:
        lines.append("")
        lines.append("Known facts about the user:")
        lines.extend(f"- {key}: {value}" for key, value in state.facts.items())

    if state.pending_question:
        lines.append("")
        lines.append(f"You are waiting on an answer to: {state.pending_question}")

    lines.append("")
    lines.append("Conversation so far:")
    for message in state.messages:
        label = "User" if message.role == "user" else "Assistant"
        lines.append(f"{label}: {message.content}")

    lines.append("Assistant:")
    return "\n".join(lines)


_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    """Return a process-wide memory instance (in-memory for now)."""
    global _memory
    if _memory is None:
        _memory = InMemoryConversationMemory()
    return _memory
