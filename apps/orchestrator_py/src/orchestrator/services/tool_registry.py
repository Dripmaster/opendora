from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, Field

ToolMetricsHandler = Callable[[dict[str, Any]], Awaitable[None] | None]

InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    prompt_builder: Callable[[BaseModel], str]
    attempts: int = 2
    version: str = "v1"
    handler: Callable[[BaseModel], Awaitable[dict[str, Any]]] | None = None


class McpToolInput(BaseModel):
    repoPath: str
    args: dict[str, Any] = Field(default_factory=dict)


class McpToolOutput(BaseModel):
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class ToolRegistry:
    def __init__(
        self,
        invoker: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        self._invoker = invoker
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unknown tool spec: {name}")
        return spec

    def register_mcp_tools(
        self,
        adapter: Any,
        on_error: Callable[[str], None] | None = None,
    ) -> list[str]:
        try:
            descriptors = adapter.list_tools()
        except Exception as exc:
            if on_error:
                on_error(f"MCP tool registration skipped: {exc}")
            return []

        registered: list[str] = []
        for descriptor in descriptors:
            spec_name = f"mcp::{descriptor.name}"

            async def mcp_handler(
                parsed_input: BaseModel,
                mcp_tool_name: str = descriptor.name,
            ) -> dict[str, Any]:
                payload = McpToolInput.model_validate(parsed_input)
                try:
                    tool_result = await adapter.call_tool(mcp_tool_name, payload.args)
                    return {"ok": True, "result": tool_result, "error": None}
                except Exception as exc:
                    return {"ok": False, "result": None, "error": str(exc)}

            self.register(
                ToolSpec(
                    name=spec_name,
                    input_schema=McpToolInput,
                    output_schema=McpToolOutput,
                    prompt_builder=lambda _: "",
                    attempts=1,
                    version="mcp-v1",
                    handler=mcp_handler,
                )
            )
            registered.append(spec_name)
        return registered

    async def invoke(
        self,
        name: str,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> tuple[BaseModel, BaseModel]:
        spec = self.get(name)
        parsed_input = spec.input_schema.model_validate(input_data)
        if spec.handler is not None:
            raw = await spec.handler(parsed_input)
        else:
            repo_path = getattr(parsed_input, "repoPath")
            prompt = spec.prompt_builder(parsed_input)
            raw = await self._invoker(
                repo_path=repo_path,
                prompt=prompt,
                attempts=spec.attempts,
                schema=spec.output_schema,
                metrics_stage=spec.name,
                on_metrics=on_metrics,
                tool_version=spec.version,
            )
        return parsed_input, spec.output_schema.model_validate(raw)
