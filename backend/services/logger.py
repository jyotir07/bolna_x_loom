"""Structured, readable console logging.

`configure_logging()` is called once at app startup. `log_turn()` emits a single
line per LLM turn capturing everything worth seeing at a glance: the selected
provider/model, the routed task, latency, token usage, and cost.
"""

import logging

from backend.config import get_settings
from backend.models.llm import LLMResult
from backend.services.routing import RouteDecision

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers: httpx logs every provider call (and its
    # URL, which would leak the provider into our logs), and Loom emits its own
    # per-call line that duplicates our structured `log_turn`.
    for noisy in ("httpx", "loom"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_turn(
    logger: logging.Logger,
    *,
    decision: RouteDecision,
    result: LLMResult,
    conversation_id: str | None,
) -> None:
    u = result.usage
    logger.info(
        "turn cid=%s task=%s -> %s:%s | %.0fms | tokens in=%d out=%d | $%.6f (%.4f %s)",
        conversation_id or "-",
        decision.task_type.value,
        result.provider,
        result.model,
        result.latency_ms,
        u.input_tokens,
        u.output_tokens,
        result.cost.usd,
        result.cost.local or 0.0,
        result.cost.local_currency or "",
    )
