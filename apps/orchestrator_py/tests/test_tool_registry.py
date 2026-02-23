# pyright: reportMissingImports=false

from dataclasses import dataclass, field
import importlib
from typing import Any, cast

from orchestrator.config import AppEnv
from orchestrator.services.codex_cli_runtime import CodexRunResult, Usage
from orchestrator.services.deep_agent_tools import DeepAgentToolsService


@dataclass
class FakeLogger:
    messages: list[str] = field(default_factory=list)

    def warning(self, msg: str, **kwargs: Any) -> None:
        details = str(kwargs.get("message", ""))
        self.messages.append(f"{msg} {details}".strip())


@dataclass
class FakeCodex:
    outputs: list[str]
    prompts: list[str] = field(default_factory=list)
    logger: FakeLogger = field(default_factory=FakeLogger)

    async def run(self, repo_path: str, prompt: str):
        self.prompts.append(prompt)
        text = self.outputs.pop(0)
        return CodexRunResult(
            assistant_message=text,
            usage=Usage(),
            thread_id=None,
            events=[],
            duration_ms=1,
            prompt_chars=len(prompt),
        )


async def test_tool_registry_retries_bad_json_for_plan() -> None:
    codex = FakeCodex(
        outputs=[
            "not-json",
            '{"todos":[{"id":"T1","title":"Primary Task","instructions":"Do work","priority":"high","dependsOn":[],"doneDefinition":"Complete work"}]}',
        ]
    )
    tools = DeepAgentToolsService(codex=cast(Any, codex))

    todos = await tools.invoke_plan(
        {
            "repoPath": ".",
            "userMessage": "do work",
            "offloadedContext": [],
            "liveConversation": [],
            "maxTasks": 3,
        }
    )

    assert len(codex.prompts) == 2
    assert todos[0]["id"] == "T1"
    assert todos[0]["title"] == "Primary Task"


async def test_tool_registry_retries_wrong_shape_for_route_and_emits_tool_version() -> (
    None
):
    codex = FakeCodex(
        outputs=[
            '{"mode":"main_direct"}',
            '{"mode":"main_direct","reason":"ok"}',
        ]
    )
    tools = DeepAgentToolsService(codex=cast(Any, codex))
    metrics: list[dict[str, Any]] = []

    async def on_metrics(payload: dict[str, Any]) -> None:
        metrics.append(payload)

    decision = await tools.invoke_route(
        {
            "repoPath": ".",
            "userMessage": "hello",
            "offloadedContext": [],
            "liveConversation": [],
        },
        on_metrics=on_metrics,
    )

    assert len(codex.prompts) == 2
    assert decision == {"mode": "main_direct", "reason": "ok"}
    assert metrics
    assert all(m.get("stage") == "routing" for m in metrics)
    assert all(m.get("tool_version") == "v1" for m in metrics)


async def test_tool_registry_plan_keeps_existing_fallback_behavior() -> None:
    codex = FakeCodex(outputs=['{"todos":[{"id":"T1"}]}'])
    tools = DeepAgentToolsService(codex=cast(Any, codex))

    todos = await tools.invoke_plan(
        {
            "repoPath": ".",
            "userMessage": "create a plan",
            "offloadedContext": [],
            "liveConversation": [],
            "maxTasks": 2,
        }
    )

    assert len(todos) == 1
    assert todos[0]["id"] == "T1"
    assert todos[0]["title"] == "Primary Task"


def test_app_env_mcp_defaults_are_inert() -> None:
    env = AppEnv.model_validate({})
    assert env.MCP_ENABLED is False
    assert env.MCP_SERVER_URLS == ""


async def test_mcp_enabled_without_servers_is_graceful() -> None:
    codex = FakeCodex(outputs=['{"mode":"main_direct","reason":"ok"}'])
    tools = DeepAgentToolsService(
        codex=cast(Any, codex),
        mcp_enabled=True,
        mcp_server_urls=[],
    )

    decision = await tools.invoke_route(
        {
            "repoPath": ".",
            "userMessage": "hello",
            "offloadedContext": [],
            "liveConversation": [],
        }
    )

    assert decision == {"mode": "main_direct", "reason": "ok"}
    assert any(
        "MCP is enabled but MCP_SERVER_URLS is empty" in x
        for x in codex.logger.messages
    )


async def test_tool_registry_can_register_and_invoke_mcp_tools() -> None:
    mcp_adapter_module = importlib.import_module("orchestrator.services.mcp_adapter")
    tool_registry_module = importlib.import_module(
        "orchestrator.services.tool_registry"
    )
    mcp_tool_descriptor = getattr(mcp_adapter_module, "McpToolDescriptor")
    tool_registry_cls = getattr(tool_registry_module, "ToolRegistry")

    class FakeMcpAdapter:
        def list_tools(self) -> list[object]:
            return [mcp_tool_descriptor(name="inspect_repo")]

        async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"tool": name, "args": args}

    async def never_called(**_: Any) -> dict[str, Any]:
        raise AssertionError("codex invoker should not be called for MCP tools")

    registry = tool_registry_cls(invoker=never_called)
    names = registry.register_mcp_tools(adapter=FakeMcpAdapter())
    assert names == ["mcp::inspect_repo"]

    _, out = await registry.invoke(
        name="mcp::inspect_repo",
        input_data={"repoPath": ".", "args": {"path": "src"}},
    )
    assert out.ok is True
    assert out.result == {"tool": "inspect_repo", "args": {"path": "src"}}
