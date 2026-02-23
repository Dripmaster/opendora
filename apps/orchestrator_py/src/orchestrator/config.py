from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


class AppEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ROOT_ENV_PATH), ".env"),
        extra="ignore",
    )

    LOG_LEVEL: str = "info"

    DISCORD_BOT_TOKEN: str | None = None
    NATURAL_CHAT_ENABLED: bool = True
    HITL_REQUIRED: bool = False
    HITL_TTL_SEC: int = Field(default=600, gt=0)

    DEFAULT_REPO_PATH: str = "."

    CONTEXT_OFFLOAD_ENABLED: bool = True
    CONTEXT_STORE_DIR: str = ".opendora/context"
    CONTEXT_MAX_ESTIMATED_TOKENS: int = Field(default=12000, gt=0)
    CONTEXT_KEEP_RECENT_MESSAGES: int = Field(default=10, gt=0)
    CONTEXT_RETRIEVE_TOP_K: int = Field(default=4, gt=0)
    CONTEXT_CHANNEL_ROTATION_ENABLED: bool = False
    CONTEXT_CHANNEL_ROTATION_MAX_OFFLOADS: int = Field(default=24, gt=0)
    CONTEXT_CHANNEL_ROTATION_MAX_LIVE_MESSAGES: int = Field(default=80, gt=0)
    CONTEXT_ACTIVE_CATEGORY_NAME: str = "opendora-active"
    CONTEXT_ARCHIVE_CATEGORY_NAME: str = "opendora-archive"
    CONTEXT_CHANNEL_ROUTER_ENABLED: bool = True

    DEEP_AGENT_ENABLED: bool = True
    DEEP_AGENT_MAX_SUBAGENTS: int = Field(default=3, gt=0)
    DEEP_AGENT_MAX_ROUNDS: int = Field(default=3, gt=0)

    CODEX_BIN: str = "codex"
    CODEX_TIMEOUT_MS: int = Field(default=900000, gt=0)
    CODEX_MODEL: str | None = None
    CODEX_SANDBOX: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"


def read_env() -> AppEnv:
    return AppEnv()
