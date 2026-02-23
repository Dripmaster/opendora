import inspect
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orchestrator.services.codex_cli_runtime import Usage
from orchestrator.services.context_offload import (
    ContextOffloadOptions,
    ContextOffloadService,
)
from orchestrator.services.deep_agent import (
    DeepAgentOptions,
    DeepAgentService,
    collect_warpgrep_filesystem_context,
    validate_todo_plan,
)
from orchestrator.services.deep_agent_tools import DeepAgentToolsService
from orchestrator.services.run_artifacts import ArtifactWriter, RunContext
from orchestrator.services.warpgrep_cache import clear_warpgrep_inventory_cache


@dataclass
class FakeRunResult:
    assistant_message: str
    usage: Usage = field(default_factory=Usage)


class FakeCodex:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.prompts: list[str] = []

    async def run(self, repo_path: str, prompt: str):
        self.prompts.append(prompt)
        text = self.outputs.pop(0)
        return FakeRunResult(assistant_message=text, usage=Usage(total=11))

    async def run_streaming(self, repo_path: str, prompt: str, on_event=None):
        if on_event:
            for event in [
                {"type": "thread.started", "thread_id": "th-1"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "worker step"},
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            ]:
                maybe = on_event(event)
                if inspect.isawaitable(maybe):
                    await maybe
        text = self.outputs.pop(0)
        return FakeRunResult(assistant_message=text, usage=Usage(total=2))


async def test_deep_agent_main_direct(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"main_direct","reason":"simple"}',
            "direct-answer",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

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
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

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
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    result = await agent.execute("c:u", "build this", ".")
    assert result.mode == "subagent_pipeline"
    assert result.subagent_count == 2
    assert result.final_response == "final-aggregate"


async def test_deep_agent_stops_when_max_rounds_reached(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "round1-output",
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=1),
    )  # type: ignore[arg-type]

    result = await agent.execute("c:u", "build this", ".")

    assert result.mode == "subagent_pipeline"
    assert result.final_response == "final-aggregate"
    aggregate_prompt = codex.prompts[-1]
    assert "[Completion Reason]" in aggregate_prompt
    assert "max-rounds-reached:1" in aggregate_prompt


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
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    result = await agent.execute("c:u", "warpgrep pattern 파일 찾아줘", str(tmp_path))

    assert result.mode == "subagent_pipeline"
    plan_prompt = codex.prompts[2]  # 0=select_context, 1=route, 2=plan
    assert "[warpgrep] file=warpgrep_pattern_docs.txt" in plan_prompt


async def test_deep_agent_warpgrep_limits_reflected_in_plan_prompt(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (tmp_path / "warpgrep_root_target.txt").write_text("root")
    (nested / "warpgrep_deep_target.txt").write_text("deep")

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
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(
            enabled=True,
            max_subagents=3,
            warpgrep_max_files=1,
            warpgrep_max_depth=1,
            max_rounds=3,
        ),
    )  # type: ignore[arg-type]

    result = await agent.execute("c:u", "warpgrep target 파일 찾아줘", str(tmp_path))

    assert result.mode == "subagent_pipeline"
    plan_prompt = codex.prompts[2]  # 0=select_context, 1=route, 2=plan
    assert "[warpgrep] limits max_files=1 max_depth=1" in plan_prompt
    assert "[warpgrep] file=warpgrep_root_target.txt" in plan_prompt
    assert "warpgrep_deep_target.txt" not in plan_prompt


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
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    events: list[dict[str, str]] = []

    async def progress(event: dict[str, str]):
        events.append(event)

    async def inputs(todo_id: str):
        if todo_id == "T2":
            return ["__control__:skip", "__control__:stop-round"]
        return []

    result = await agent.execute(
        "c:u", "build this", ".", on_progress=progress, todo_input_provider=inputs
    )

    assert result.final_response == "final-aggregate"
    assert result.subagent_count == 1
    assert any(e.get("message") == "TODO T2 skipped by user" for e in events)
    assert any("현재 라운드 종료 후 집계" in str(e.get("message", "")) for e in events)


async def test_deep_agent_invalid_plan_replans_before_running_subagents(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"},{"id":"T1","title":"dup","instructions":"do-dup","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"done":false,"reason":"fix invalid plan","nextTodos":[{"id":"T2","title":"valid","instructions":"do-2","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "replanned-output",
            '{"done":true,"reason":"complete","nextTodos":[]}',
            "final-aggregate",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    result = await agent.execute("c:u", "build this", ".")

    assert result.mode == "subagent_pipeline"
    assert result.subagent_count == 1
    assert result.final_response == "final-aggregate"


async def test_deep_agent_invalid_plan_and_invalid_replan_finishes_safely(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":["T9"],"doneDefinition":"done"}]}',
            "safe-final",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    result = await agent.execute("c:u", "build this", ".")

    assert result.mode == "subagent_pipeline"
    assert result.subagent_count == 0
    assert result.final_response == "safe-final"
    assert "all-todos-blocked" in codex.prompts[-1]


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
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    events: list[dict[str, str]] = []

    async def progress(event: dict[str, str]):
        events.append(event)

    async def inputs(_todo_id: str):
        return ["__control__:abort"]

    result = await agent.execute(
        "c:u", "build this", ".", on_progress=progress, todo_input_provider=inputs
    )

    assert result.final_response == "final-aggregate"
    assert result.subagent_count == 0
    assert any(e.get("message") == "TODO T1 aborted by user" for e in events)


def test_validate_todo_plan_detects_duplicate_undefined_and_cycle():
    errors = validate_todo_plan(
        [
            {"id": "T1", "dependsOn": ["T2"]},
            {"id": "T2", "dependsOn": ["T1", "T9"]},
            {"id": "T1", "dependsOn": []},
        ]
    )

    assert any("duplicate ids" in item for item in errors)
    assert any("undefined dependsOn references" in item for item in errors)
    assert any("cyclic dependency detected" in item for item in errors)


async def test_deep_agent_timing_events_are_written_to_artifacts(tmp_path):
    codex = FakeCodex(
        [
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"main_direct","reason":"simple"}',
            "direct-answer",
        ]
    )
    tools = DeepAgentToolsService(codex=codex)  # type: ignore[arg-type]
    context = ContextOffloadService(
        ContextOffloadOptions(True, str(tmp_path / "ctx"), 12000, 10, 4)
    )
    agent = DeepAgentService(
        codex=codex,
        context_offload=context,
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )  # type: ignore[arg-type]

    writer = ArtifactWriter(
        RunContext(
            run_id="run-timing",
            request_id="req-timing",
            session_key="c:u",
            repo_path=str(tmp_path),
            started_at=datetime.now(tz=timezone.utc),
            artifacts_dir=tmp_path / "artifacts",
            debug_enabled=False,
        ),
        max_bytes=1024 * 1024,
        retention_days=7,
        redact=True,
    )

    async def progress(event: dict[str, object]):
        writer.append_event(event)

    result = await agent.execute("c:u", "hello", str(tmp_path), on_progress=progress)

    assert result.final_response == "direct-answer"
    events_path = writer.run_dir / "events.jsonl"
    raw_lines = events_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in raw_lines if line.strip()]
    assert any(
        ("duration_ms" in event and "prompt_chars" in event) for event in payloads
    )
    assert all("prompt" not in event for event in payloads)


async def test_warpgrep_cache_reuses_inventory_walk_on_cache_hit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    head_ref = repo / ".git" / "refs" / "heads" / "main"
    head_ref.parent.mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    head_ref.write_text("abc123\n", encoding="utf-8")
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "warpgrep_target_one.md").write_text("x", encoding="utf-8")
    (repo / "docs" / "warpgrep_target_two.md").write_text("x", encoding="utf-8")

    clear_warpgrep_inventory_cache()

    import orchestrator.services.warpgrep_cache as warpgrep_cache

    walk_calls = 0
    original_walk = warpgrep_cache.os.walk

    def counting_walk(*args, **kwargs):
        nonlocal walk_calls
        walk_calls += 1
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(warpgrep_cache.os, "walk", counting_walk)

    first = await collect_warpgrep_filesystem_context(
        str(repo), "warpgrep target", max_files=200, max_depth=4
    )
    second = await collect_warpgrep_filesystem_context(
        str(repo), "target", max_files=200, max_depth=4
    )

    assert walk_calls == 1
    assert any("warpgrep_target_one.md" in line for line in first)
    assert any("warpgrep_target_two.md" in line for line in second)


async def test_warpgrep_cache_invalidates_when_git_ref_mtime_changes(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    head_ref = repo / ".git" / "refs" / "heads" / "main"
    head_ref.parent.mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    head_ref.write_text("abc123\n", encoding="utf-8")
    (repo / "warpgrep_target.md").write_text("x", encoding="utf-8")

    clear_warpgrep_inventory_cache()

    import orchestrator.services.warpgrep_cache as warpgrep_cache

    walk_calls = 0
    original_walk = warpgrep_cache.os.walk

    def counting_walk(*args, **kwargs):
        nonlocal walk_calls
        walk_calls += 1
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(warpgrep_cache.os, "walk", counting_walk)

    await collect_warpgrep_filesystem_context(
        str(repo), "warpgrep target", max_files=200, max_depth=4
    )

    ref_stat = head_ref.stat()
    os.utime(
        head_ref,
        ns=(ref_stat.st_atime_ns, ref_stat.st_mtime_ns + 1_000_000),
    )

    await collect_warpgrep_filesystem_context(
        str(repo), "warpgrep target", max_files=200, max_depth=4
    )

    assert walk_calls == 2
