from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class McpAdapterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class McpToolDescriptor:
    name: str
    description: str = ""


class McpAdapter(Protocol):
    def list_tools(self) -> list[McpToolDescriptor]: ...

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class NullMcpAdapter:
    def list_tools(self) -> list[McpToolDescriptor]:
        return []

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise McpAdapterError("MCP is disabled (MCP_ENABLED=false).")


@dataclass(frozen=True, slots=True)
class MisconfiguredMcpAdapter:
    message: str

    def list_tools(self) -> list[McpToolDescriptor]:
        raise McpAdapterError(self.message)

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise McpAdapterError(self.message)


@dataclass(frozen=True, slots=True)
class StubEnabledMcpAdapter:
    server_urls: tuple[str, ...]

    def list_tools(self) -> list[McpToolDescriptor]:
        return []

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise McpAdapterError(
            f"MCP is enabled but no client implementation is configured for tool '{name}'."
        )


def build_mcp_adapter(enabled: bool, server_urls: list[str]) -> McpAdapter:
    if not enabled:
        return NullMcpAdapter()
    cleaned = [url.strip() for url in server_urls if url.strip()]
    if not cleaned:
        return MisconfiguredMcpAdapter(
            "MCP is enabled but MCP_SERVER_URLS is empty. Configure one or more server URLs."
        )
    return StubEnabledMcpAdapter(tuple(cleaned))
