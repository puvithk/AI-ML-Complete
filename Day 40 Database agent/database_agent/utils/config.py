"""Central, env-driven configuration for the database agent.

Every tunable/secret is read from the environment (loaded from `.env` via
python-dotenv) in one place, so nothing is hardcoded across the codebase.
See `.env.example` for the full list.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Resolved application settings."""

    # --- Database ---------------------------------------------------------
    db_host: str = Field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = Field(default_factory=lambda: _get_int("DB_PORT", 5432))
    db_user: str = Field(default_factory=lambda: os.getenv("DB_USER", "layerbase"))
    db_password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    db_name: str = Field(default_factory=lambda: os.getenv("DB_NAME", "duckdb_test"))
    db_schema: str = Field(default_factory=lambda: os.getenv("DB_SCHEMA", "main"))

    # Connection pool sizing.
    db_pool_min: int = Field(default_factory=lambda: _get_int("DB_POOL_MIN", 1))
    db_pool_max: int = Field(default_factory=lambda: _get_int("DB_POOL_MAX", 10))

    # Per-statement timeout in milliseconds (0 disables).
    db_statement_timeout_ms: int = Field(
        default_factory=lambda: _get_int("DB_STATEMENT_TIMEOUT_MS", 15000)
    )
    # Hard cap on rows returned to the agent / LLM prompt.
    max_result_rows: int = Field(
        default_factory=lambda: _get_int("MAX_RESULT_ROWS", 200)
    )

    # --- LLM --------------------------------------------------------------
    mistral_model: str = Field(
        default_factory=lambda: os.getenv("MISTRAL_MODEL", "mistral-small-2506")
    )
    llm_timeout_s: int = Field(default_factory=lambda: _get_int("LLM_TIMEOUT_S", 60))
    llm_max_retries: int = Field(default_factory=lambda: _get_int("LLM_MAX_RETRIES", 2))

    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    )

    # --- Agent behaviour --------------------------------------------------
    sql_max_retries: int = Field(default_factory=lambda: _get_int("SQL_MAX_RETRIES", 3))
    max_revisions: int = Field(default_factory=lambda: _get_int("MAX_REVISIONS", 1))

    # --- Checkpointer -----------------------------------------------------
    # If set, a file-backed SQLite checkpointer is used (durable across
    # restarts). If empty, an in-memory checkpointer is used.
    checkpoint_db: str = Field(default_factory=lambda: os.getenv("CHECKPOINT_DB", ""))

    # --- API --------------------------------------------------------------
    api_cors_origins: str = Field(
        default_factory=lambda: os.getenv("API_CORS_ORIGINS", "")
    )
    memory_recall_k: int = Field(default_factory=lambda: _get_int("MEMORY_RECALL_K", 3))
    memory_max_per_session: int = Field(
        default_factory=lambda: _get_int("MEMORY_MAX_PER_SESSION", 50)
    )
    memory_max_sessions: int = Field(
        default_factory=lambda: _get_int("MEMORY_MAX_SESSIONS", 1000)
    )

    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
