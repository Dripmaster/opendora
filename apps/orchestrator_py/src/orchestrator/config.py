from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUntypedBaseClass=false, reportUnnecessaryTypeIgnoreComment=false

from pathlib import Path
from typing import Literal

from pydantic import Field  # pyright: ignore[reportMissingImports]
from pydantic_settings import BaseSettings, SettingsConfigDict  # pyright: ignore[reportMissingImports]

ROOT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


class AppEnv(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(  # pyright: ignore[reportIncompatibleVariableOverride]
        env_file=(str(ROOT_ENV_PATH), ".env"),
        extra="ignore",
    )

    LOG_LEVEL: str = "info"

    DISCORD_BOT_TOKEN: str | None = None
    NATURAL_CHAT_ENABLED: bool = True
    DISCORD_DM_POLICY: Literal["open", "pairing"] = "open"
    DISCORD_ALLOWLIST_USER_IDS: str = ""
    DISCORD_PAIRING_STORE: str = ".opendora/pairing.json"
    HITL_REQUIRED: bool = False
    HITL_TTL_SEC: int = Field(default=600, gt=0)

    DEFAULT_REPO_PATH: str = "."

    CONTEXT_OFFLOAD_ENABLED: bool = True
    CONTEXT_STORE_DIR: str = ".opendora/context"
    CONTEXT_MAX_ESTIMATED_TOKENS: int = Field(default=12000, gt=0)
    CONTEXT_KEEP_RECENT_MESSAGES: int = Field(default=10, gt=0)
    CONTEXT_RETRIEVE_TOP_K: int = Field(default=4, gt=0)
    CONTEXT_SESSION_CACHE_SIZE: int = Field(default=32, ge=0)
    CONTEXT_CHANNEL_ROTATION_ENABLED: bool = False
    CONTEXT_CHANNEL_ROTATION_MAX_OFFLOADS: int = Field(default=24, gt=0)
    CONTEXT_CHANNEL_ROTATION_MAX_LIVE_MESSAGES: int = Field(default=80, gt=0)
    CONTEXT_ACTIVE_CATEGORY_NAME: str = "opendora-active"
    CONTEXT_ARCHIVE_CATEGORY_NAME: str = "opendora-archive"
    CONTEXT_CHANNEL_ROUTER_ENABLED: bool = True

    RUN_ARTIFACTS_ENABLED: bool = True
    RUN_ARTIFACTS_DIR: str = ".opendora/runs"
    RUN_ARTIFACTS_REDACT: bool = True
    RUN_ARTIFACTS_MAX_BYTES: int = Field(default=2_000_000, gt=0)
    RUN_ARTIFACTS_RETENTION_DAYS: int = Field(default=7, ge=0)
    RUN_DEBUG_PROMPTS: bool = False
    DISCORD_PROGRESS_THROTTLE_MS: int = Field(default=1000, ge=0)
    DISCORD_PROGRESS_MAX_BUFFERED: int = Field(default=20, gt=0)

    DEEP_AGENT_ENABLED: bool = True
    DEEP_AGENT_MAX_SUBAGENTS: int = Field(default=3, gt=0)
    WARPGREP_MAX_FILES: int = Field(default=200, gt=0)
    WARPGREP_MAX_DEPTH: int = Field(default=4, ge=0)
    DEEP_AGENT_MAX_ROUNDS: int = Field(default=3, gt=0)

    CODEX_BIN: str = "codex"
    CODEX_TIMEOUT_MS: int = Field(default=900000, gt=0)
    CODEX_MODEL: str | None = None
    CODEX_MODEL_CANDIDATES: str = ""
    CODEX_SANDBOX: Literal["read-only", "workspace-write", "danger-full-access"] = (
        "workspace-write"
    )
    CODEX_RETRY_COUNT: int = Field(default=2, ge=0)
    CODEX_RETRY_BACKOFF_MS: int = Field(default=250, ge=0)

    MCP_ENABLED: bool = False
    MCP_SERVER_URLS: str = ""


def read_env() -> AppEnv:
    return AppEnv()
