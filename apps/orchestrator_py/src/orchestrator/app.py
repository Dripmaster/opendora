from __future__ import annotations

from orchestrator.adapters.discord_gateway import DiscordGateway
from orchestrator.config import read_env
from orchestrator.services.codex_cli_runtime import CodexCliRuntimeOptions, CodexCliRuntimeService
from orchestrator.services.context_offload import ContextOffloadOptions, ContextOffloadService
from orchestrator.services.deep_agent import DeepAgentOptions, DeepAgentService
from orchestrator.services.deep_agent_tools import DeepAgentToolsService
from orchestrator.services.logger import create_logger


class OrchestratorApp:
    def __init__(self) -> None:
        self.env = read_env()
        self.logger = create_logger(self.env.LOG_LEVEL)

        self.codex = CodexCliRuntimeService(
            logger=self.logger,
            options=CodexCliRuntimeOptions(
                binary=self.env.CODEX_BIN,
                timeout_ms=self.env.CODEX_TIMEOUT_MS,
                model=self.env.CODEX_MODEL,
                sandbox=self.env.CODEX_SANDBOX,
            ),
        )
        self.context_offload = ContextOffloadService(
            ContextOffloadOptions(
                enabled=self.env.CONTEXT_OFFLOAD_ENABLED,
                store_dir=self.env.CONTEXT_STORE_DIR,
                max_estimated_tokens=self.env.CONTEXT_MAX_ESTIMATED_TOKENS,
                keep_recent_messages=self.env.CONTEXT_KEEP_RECENT_MESSAGES,
                retrieve_top_k=self.env.CONTEXT_RETRIEVE_TOP_K,
            )
        )
        self.deep_agent_tools = DeepAgentToolsService(self.codex)
        self.deep_agent = DeepAgentService(
            codex=self.codex,
            context_offload=self.context_offload,
            tools=self.deep_agent_tools,
            options=DeepAgentOptions(
                enabled=self.env.DEEP_AGENT_ENABLED,
                max_subagents=self.env.DEEP_AGENT_MAX_SUBAGENTS,
            ),
        )
        self.discord = DiscordGateway(self.env, self.deep_agent, self.logger)

    async def start(self) -> None:
        self.logger.info("Starting remote Codex runner.")
        await self.discord.start()

    async def stop(self) -> None:
        self.logger.info("Stopping remote Codex runner.")
        await self.discord.stop()
