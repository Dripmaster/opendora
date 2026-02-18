import inspect
from dataclasses import dataclass

from orchestrator.services.context_offload import ContextOffloadOptions, ContextOffloadService
from orchestrator.services.deep_agent import DeepAgentOptions, DeepAgentService
from orchestrator.services.deep_agent_tools import DeepAgentToolsService


@dataclass
class FakeRunResult:
    assistant_message: str


class FakeCodex:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    async def run(self, repo_path: str, prompt: str):
        text = self.outputs.pop(0)
        return FakeRunResult(assistant_message=text)

    async def run_streaming(self, repo_path: str, prompt: str, on_event=None):
        if on_event:
            for event in [
                {"type": "thread.started", "thread_id": "th-1"},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "worker step"}},
                {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
            ]:
                maybe = on_event(event)
                if inspect.isawaitable(maybe):
                    await maybe
        text = self.outputs.pop(0)
        return FakeRunResult(assistant_message=text)


async def test_deep_agent_main_direct(tmp_path):
    codex = FakeCodex(['{"mode":"main_direct","reason":"simple"}', "direct-answer"])
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(codex=codex, context_offload=context, tools=tools, options=DeepAgentOptions(True, 3))  # type: ignore[arg-type]

    result = await agent.execute("c:u", "hello", ".")
    assert result.mode == "main_direct"
    assert result.final_response == "direct-answer"


async def test_deep_agent_subagent_blocked_and_aggregate(tmp_path):
    codex = FakeCodex(
        [
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"},{"id":"T2","title":"second","instructions":"do-2","priority":"medium","dependsOn":["TX"],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "subagent-output",
            '{"done":true,"reason":"complete","nextTodos":[]}',
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(codex=codex, context_offload=context, tools=tools, options=DeepAgentOptions(True, 3))  # type: ignore[arg-type]

    result = await agent.execute("c:u", "build this", ".")
    assert result.mode == "subagent_pipeline"
    assert result.subagent_count == 1
    assert result.final_response == "final-aggregate"


async def test_deep_agent_replans_and_runs_additional_round(tmp_path):
    codex = FakeCodex(
        [
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "round1-output",
            '{"done":false,"reason":"missing verification","nextTodos":[{"id":"T2","title":"verify","instructions":"do-2","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "round2-output",
            '{"done":true,"reason":"now complete","nextTodos":[]}',
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(codex=codex, context_offload=context, tools=tools, options=DeepAgentOptions(True, 3))  # type: ignore[arg-type]

    result = await agent.execute("c:u", "build this", ".")
    assert result.mode == "subagent_pipeline"
    assert result.subagent_count == 2
    assert result.final_response == "final-aggregate"
