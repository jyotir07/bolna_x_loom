"""Loom service — the ONLY module that talks to an LLM provider.

Every LLM call in the backend goes through `LoomService.generate`. Callers pass
a task-shaped request (prompt + an already-decided provider/model from the
routing layer); they never import a provider SDK and never touch `AsyncLoom`
directly. Loom's provider-specific errors are translated into a single
`LLMServiceError` carrying a `retryable` flag, so the routing layer can later
fail over to another provider without any call site changing.
"""

import asyncio
import time

from loom import (
    AsyncLoom,
    AuthError,
    LoomError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

from backend.config import Settings, get_settings
from backend.models.llm import LLMResult


class LLMServiceError(Exception):
    """A provider/LLM failure, normalized for the backend.

    `retryable` marks failures a future failover layer could retry on another
    provider (rate limits, upstream/provider errors, timeouts) versus ones that
    won't fix themselves by retrying (bad key, unknown model).
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider
        self.model = model


class LoomService:
    def __init__(
        self,
        client: AsyncLoom,
        *,
        default_provider: str,
        default_model: str,
        timeout_seconds: float,
    ) -> None:
        self._client = client
        self._default_provider = default_provider
        self._default_model = default_model
        self._timeout = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> "LoomService":
        # Loom reads provider keys from the environment / .env itself.
        client = AsyncLoom.from_env(dotenv_path=".env")
        return cls(
            client,
            default_provider=settings.loom_default_provider,
            default_model=settings.loom_default_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    async def generate(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        params: dict | None = None,
        modality: str = "text",
    ) -> LLMResult:
        """Run one LLM turn through Loom and return a normalized result.

        `provider`/`model` default to the configured defaults; in normal
        operation the routing layer supplies them.
        """
        provider = provider or self._default_provider
        model = model or self._default_model

        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._client.generate(
                    provider=provider,
                    modality=modality,
                    model=model,
                    prompt=prompt,
                    params=params,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise LLMServiceError(
                f"LLM call timed out after {self._timeout}s",
                retryable=True,
                provider=provider,
                model=model,
            ) from exc
        except AuthError as exc:
            raise LLMServiceError(
                f"Authentication failed for provider '{provider}'",
                retryable=False,
                provider=provider,
                model=model,
            ) from exc
        except ModelNotFoundError as exc:
            raise LLMServiceError(
                f"Model '{model}' not found for provider '{provider}'",
                retryable=False,
                provider=provider,
                model=model,
            ) from exc
        except RateLimitError as exc:
            raise LLMServiceError(
                f"Rate limited by provider '{provider}'",
                retryable=True,
                provider=provider,
                model=model,
            ) from exc
        except ProviderError as exc:
            raise LLMServiceError(
                f"Provider '{provider}' error: {exc}",
                retryable=True,
                provider=provider,
                model=model,
            ) from exc
        except LoomError as exc:
            # Any other Loom-level failure we didn't specifically map.
            raise LLMServiceError(
                f"Loom error: {exc}",
                retryable=False,
                provider=provider,
                model=model,
            ) from exc
        except Exception as exc:
            # Safety net: Loom does not wrap every provider SDK error (e.g. an
            # Anthropic BadRequestError for billing propagates raw). Guarantee
            # no provider-specific exception escapes this boundary. Treated as
            # non-retryable since we can't confirm it's transient — only known
            # transient failures above are marked retryable. (CancelledError is
            # a BaseException and is intentionally not caught here.)
            raise LLMServiceError(
                f"Unexpected error from provider '{provider}': {exc}",
                retryable=False,
                provider=provider,
                model=model,
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        return LLMResult.from_loom(
            raw, provider=provider, model=model, latency_ms=latency_ms
        )


_service: LoomService | None = None


def get_loom_service() -> LoomService:
    """Return a process-wide LoomService (constructed once, lazily)."""
    global _service
    if _service is None:
        _service = LoomService.from_settings(get_settings())
    return _service
