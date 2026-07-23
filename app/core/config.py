"""
Centralized application configuration.

Why this exists, instead of scattering os.environ.get() calls everywhere:

1. Single source of truth: every setting the app needs is declared once, here,
   with a type and (where safe) a default. If something is missing at startup,
   you get one clear error instead of a KeyError three layers deep at 2am.

2. Secrets never touch source code. Real values live in a local `.env` file
   that is explicitly git-ignored (see .gitignore). This file only defines
   *shape*, never actual keys.

3. Type safety: DB_URL is a str, QDRANT_PORT is an int, DEBUG is a bool --
   pydantic-settings parses and validates these from environment strings
   automatically, and will fail fast with a clear message if e.g. someone
   sets QDRANT_PORT=abc in their .env.

Usage elsewhere in the app:
    from app.core.config import settings
    settings.claude_api_key
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # --- Database ---
    # Defaults to local SQLite for development. Override with a real
    # postgresql:// URL via .env for production.
    database_url: str = Field(
        default="sqlite:///data/processed/medguard.db",
        description="SQLAlchemy-compatible database URL",
    )

    # --- Vector store (Qdrant) ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="fda_label_chunks")
    # Only required when using Qdrant Cloud rather than a local instance.
    qdrant_api_key: str | None = Field(default=None)

    # --- LLM generation (Claude API) ---
    anthropic_api_key: str | None = Field(
        default=None,
        description="Required for the generation layer. Get one at console.anthropic.com",
    )
    claude_model: str = Field(default="claude-sonnet-4-6")

    # --- openFDA ---
    # openFDA works without a key at low request volume, but a free key
    # raises the rate limit substantially -- register at open.fda.gov/apis/authentication
    openfda_api_key: str | None = Field(default=None)

    # --- App behavior ---
    environment: str = Field(default="development")  # development | staging | production
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # --- Retrieval tuning (we'll use these once we build retrieval this week) ---
    retrieval_top_k: int = Field(default=10, description="Chunks retrieved before reranking")
    rerank_top_n: int = Field(default=4, description="Chunks kept after reranking, sent to LLM")

    # --- App-level API protection ---
    # Optional shared-secret header required on cost-incurring endpoints
    # (e.g. /query, which calls the Claude API). This is a deterrent, not
    # real authentication -- since the frontend is static HTML, the key is
    # visible in client-side source to anyone who looks. It's meant to stop
    # casual bots/scrapers from accidentally burning API credits on a
    # publicly deployed demo, not to resist a determined attacker. Leave
    # unset (None) to disable the check entirely, e.g. for local dev.
    app_api_key: str | None = Field(default=None)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Cached so we parse the .env file once per process, not on every import.
    lru_cache with no args makes this a de facto singleton.
    """
    return Settings()


settings = get_settings()
