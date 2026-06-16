"""Runtime configuration, loaded from environment / .env (see .env.example).

Nothing here is secret by default — secrets (API keys) come from the environment on the
VPS and are never committed. The app reads a single `settings` singleton.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service ---
    app_name: str = "spendscope-backend"
    environment: str = "development"  # development | production
    api_prefix: str = "/api/v1"
    cors_allow_origins: str = "*"  # comma-separated; the app talks server-to-server, browsers rarely

    # --- Primary LLM provider (nan.builders, EU, zero prompt logging) ---
    # OpenAI-compatible Chat Completions endpoint.
    primary_base_url: str = "https://api.nan.builders/v1"
    primary_api_key: str = ""
    # Comma-separated model chain within this provider, tried in order (SPECS §11.5).
    # e.g. "gemma4,qwen3.6" → try gemma4 first, then qwen3.6, before the fallback provider.
    primary_model_text: str = "gemma4,qwen3.6"
    primary_model_vision: str = "gemma4,qwen3.6"

    # --- Fallback LLM provider (OpenAI) ---
    fallback_base_url: str = "https://api.openai.com/v1"
    fallback_api_key: str = ""
    fallback_model_text: str = "gpt-4o-mini"
    fallback_model_vision: str = "gpt-4o-mini"

    llm_timeout_seconds: float = 60.0
    llm_max_output_tokens: int = 4096
    # When true, log the model's raw output (truncated) on a parse failure, to debug LLMBadOutput.
    # Off by default: the raw output is user financial data and we don't want it in logs normally.
    llm_debug_raw: bool = False
    # Max chars of raw output to log/keep when debugging (cap so logs don't explode).
    llm_debug_raw_chars: int = 4000

    # --- Database (MySQL): cache + ops logging + rate limiting. No user data. ---
    # Empty string disables the DB entirely (cache/log/rate-limit degrade gracefully).
    database_url: str = ""  # e.g. mysql+pymysql://user:pass@127.0.0.1:3306/spendscope

    # --- Categorize cache ---
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days

    # --- Rate limiting (per user_id, fixed window) ---
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 60  # per window, per user, per endpoint

    # --- RevenueCat (premium validation for /enrich) ---
    require_premium: bool = True
    revenuecat_secret_key: str = ""  # sk_... (server secret, NEVER the public app key)
    revenuecat_entitlement_id: str = "premium"
    revenuecat_base_url: str = "https://api.revenuecat.com/v1"
    revenuecat_timeout_seconds: float = 10.0

    # --- Enrich validation ---
    enrich_total_tolerance: float = 0.05  # 5% tolerance when matching ticket total to amount

    @property
    def cors_origins_list(self) -> list[str]:
        value = self.cors_allow_origins.strip()
        if value == "*" or not value:
            return ["*"]
        return [o.strip() for o in value.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
