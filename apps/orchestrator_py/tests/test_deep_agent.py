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
        self.prompts: list[str] = []

    async def run(self, repo_path: str, prompt: str):
        self.prompts.append(prompt)
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
    codex = FakeCodex(
        ['{"offloadIds":[],"liveMessageIds":[]}', '{"mode":"main_direct","reason":"simple"}', "direct-answer"]
    )
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
            '{"offloadIds":[],"liveMessageIds":[]}',
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
            '{"offloadIds":[],"liveMessageIds":[]}',
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


async def test_deep_agent_filesystem_tool_node_injects_warpgrep_context(tmp_path):
    (tmp_path / "warpgrep_pattern_docs.txt").write_text("notes")
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "round1-output",
            '{"done":true,"reason":"complete","nextTodos":[]}',
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(codex=codex, context_offload=context, tools=tools, options=DeepAgentOptions(True, 3))  # type: ignore[arg-type]

    result = await agent.execute("c:u", "warpgrep pattern 파일 찾아줘", str(tmp_path))

    assert result.mode == "subagent_pipeline"
    plan_prompt = codex.prompts[2]  # 0=select_context, 1=route, 2=plan
    assert "[warpgrep] file=warpgrep_pattern_docs.txt" in plan_prompt


async def test_deep_agent_skip_and_stop_round_control(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"},{"id":"T2","title":"second","instructions":"do-2","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "round1-output",
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(codex=codex, context_offload=context, tools=tools, options=DeepAgentOptions(True, 3))  # type: ignore[arg-type]

    events: list[dict[str, str]] = []

    async def progress(event: dict[str, str]):
        events.append(event)

    async def inputs(todo_id: str):
        if todo_id == "T2":
            return ["__control__:skip", "__control__:stop-round"]
        return []

    result = await agent.execute("c:u", "build this", ".", on_progress=progress, todo_input_provider=inputs)

    assert result.final_response == "final-aggregate"
    assert result.subagent_count == 1
    assert any(e.get("message") == "TODO T2 skipped by user" for e in events)
    assert any("현재 라운드 종료 후 집계" in str(e.get("message", "")) for e in events)


async def test_deep_agent_abort_control(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(codex=codex, context_offload=context, tools=tools, options=DeepAgentOptions(True, 3))  # type: ignore[arg-type]

    events: list[dict[str, str]] = []

    async def progress(event: dict[str, str]):
        events.append(event)

    async def inputs(_todo_id: str):
        return ["__control__:abort"]

    result = await agent.execute("c:u", "build this", ".", on_progress=progress, todo_input_provider=inputs)

    assert result.final_response == "final-aggregate"
    assert result.subagent_count == 0
    assert any(e.get("message") == "TODO T1 aborted by user" for e in events)
