from dataclasses import dataclass

from orchestrator.services.codex_cli_runtime import CodexRunResult, Usage
from orchestrator.services.deep_agent_tools import DeepAgentToolsService


@dataclass
class FakeCodex:
    outputs: list[str]

    async def run(self, repo_path: str, prompt: str):
        text = self.outputs.pop(0)
        return CodexRunResult(assistant_message=text, usage=Usage(), thread_id=None, events=[])


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
