"""Environment-driven application settings.

All configuration comes from environment variables (loaded from `.env` in
development). No secrets are hardcoded. Provider API keys are also read here
purely for a startup sanity check and logging — Loom itself reads them from the
environment via `AsyncLoom.from_env()`, so this module is not their consumer.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Provider keys (consumed by Loom; mirrored here for validation only) ──
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # ── Loom routing defaults ──────────────────────────────────────────────
    loom_default_provider: str = "openai"
    loom_default_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0

    # ── Bolna (voice layer) ────────────────────────────────────────────────
    bolna_api_key: str | None = None
    bolna_agent_id: str | None = None

    # ── Backend / app ──────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    @property
    def configured_providers(self) -> list[str]:
        """Providers that have a usable API key, in a stable order."""
        keyed = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }
        return [name for name, key in keyed.items() if key]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
