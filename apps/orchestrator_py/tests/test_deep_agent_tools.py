from dataclasses import dataclass

import pytest

from orchestrator.services.codex_cli_runtime import CodexRunResult, Usage
from orchestrator.services.deep_agent_tools import (
    DeepAgentToolsService,
    RouteDecision,
    extract_json_line,
)


@dataclass
class FakeCodex:
    outputs: list[str]

    async def run(self, repo_path: str, prompt: str):
        text = self.outputs.pop(0)
        return CodexRunResult(
            assistant_message=text,
            usage=Usage(),
            thread_id=None,
            events=[],
            duration_ms=1,
            prompt_chars=len(prompt),
        )


async def test_tools_retry_json_then_succeed() -> None:
    codex = FakeCodex(outputs=["not-json", '{"mode":"main_direct","reason":"short"}'])
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]

    result = await tools.invoke_route(
        {
            "repoPath": ".",
            "userMessage": "hi",
            "offloadedContext": [],
            "liveConversation": [],
        }
    )
    assert result["mode"] == "main_direct"
    assert result["reason"] == "short"


async def test_external_context_routing_selects_only_allowed_sessions() -> None:
    codex = FakeCodex(
        outputs=[
            '{"useExternalContext":true,"reason":"needs previous project context","selectedSessionKeys":["s1:u1","invalid"]}'
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    result = await tools.invoke_external_context_routing(
        {
            "repoPath": ".",
            "userMessage": "전에 하던 프로젝트 이어서 하자",
            "currentSessionKey": "s2:u1",
            "offloadedContext": [],
            "liveConversation": [],
            "candidates": [
                {
                    "sessionKey": "s1:u1",
                    "channelId": "s1",
                    "updatedAt": "",
                    "summary": "project a summary",
                }
            ],
            "maxSelect": 3,
        }
    )
    assert result["useExternalContext"] is True
    assert result["selectedSessionKeys"] == ["s1:u1"]


def test_extract_json_line_accepts_multiline_json_object() -> None:
    text = '{\n  "mode": "main_direct",\n  "reason": "multiline"\n}'
    parsed = extract_json_line(text)
    assert parsed == {"mode": "main_direct", "reason": "multiline"}


def test_extract_json_line_accepts_json_code_fence() -> None:
    text = '```json\n{"mode":"subagent_pipeline","reason":"fenced"}\n```'
    parsed = extract_json_line(text)
    assert parsed == {"mode": "subagent_pipeline", "reason": "fenced"}


def test_extract_json_line_accepts_embedded_json_object() -> None:
    text = '설명 먼저\n{"mode":"main_direct","reason":"embedded"}\n설명 끝'
    parsed = extract_json_line(text)
    assert parsed == {"mode": "main_direct", "reason": "embedded"}


async def test_run_json_with_retry_includes_last_output_on_failure() -> None:
    codex = FakeCodex(outputs=['{"unexpected":"shape"}'])
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]

    with pytest.raises(ValueError) as exc_info:
        await tools._run_json_with_retry(
            repo_path=".", prompt="x", attempts=1, schema=RouteDecision
        )

    message = str(exc_info.value)
    assert "last_error=schema_failed" in message
    assert '{"unexpected":"shape"}' in message
