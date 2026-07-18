"""Normalized LLM result models.

Loom already normalizes provider responses into a dict. We wrap that dict in a
typed model at the service boundary so nothing above `services/loom.py` deals
with raw dicts — and so a provider name / latency travel alongside the text for
logging.
"""

from typing import Any

from pydantic import BaseModel


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


class LLMCost(BaseModel):
    usd: float = 0.0
    local: float | None = None
    local_currency: str | None = None


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    latency_ms: float
    kind: str = "text"
    usage: LLMUsage = LLMUsage()
    cost: LLMCost = LLMCost()

    @classmethod
    def from_loom(
        cls,
        raw: dict[str, Any],
        *,
        provider: str,
        model: str,
        latency_ms: float,
    ) -> "LLMResult":
        """Build a typed result from Loom's normalized response dict.

        Parsed defensively — `usage`/`cost` may be absent or partial depending
        on the provider.
        """
        usage_raw = raw.get("usage") or {}
        cost_raw = raw.get("cost") or {}
        return cls(
            text=raw.get("text", ""),
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            kind=raw.get("kind", "text"),
            usage=LLMUsage(
                input_tokens=usage_raw.get("input_tokens", 0),
                output_tokens=usage_raw.get("output_tokens", 0),
                cached_tokens=usage_raw.get("cached_tokens", 0),
            ),
            cost=LLMCost(
                usd=cost_raw.get("usd", 0.0),
                local=cost_raw.get("local"),
                local_currency=cost_raw.get("local_currency"),
            ),
        )
