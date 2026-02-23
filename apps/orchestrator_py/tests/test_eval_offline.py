from __future__ import annotations

# pyright: reportMissingImports=false

import asyncio
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.services import codex_cli_runtime as codex_runtime
from orchestrator.services.codex_cli_runtime import (
    CodexCliRuntimeOptions,
    CodexCliRuntimeService,
    CodexRunResult,
    Usage,
)
from orchestrator.services.context_offload import (
    ContextCapsule,
    ExternalSessionCandidate,
)
from orchestrator.services.deep_agent import (
    DeepAgentOptions,
    DeepAgentResult,
    DeepAgentService,
)
from orchestrator.services.deep_agent_tools import DeepAgentToolsService
from orchestrator.services.run_artifacts import ArtifactWriter, RunContext


class FakeCodex:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.prompts: list[str] = []
        self.run_calls = 0
        self.streaming_calls = 0

    async def run(self, repo_path: str, prompt: str):
        del repo_path
        self.run_calls += 1
        self.prompts.append(prompt)
        return CodexRunResult(
            assistant_message=self.outputs.pop(0),
            usage=Usage(),
            thread_id=None,
            events=[],
            duration_ms=0,
            prompt_chars=len(prompt),
        )

    async def run_streaming(self, repo_path: str, prompt: str, on_event=None):
        del repo_path
        self.streaming_calls += 1
        self.prompts.append(prompt)
        if on_event:
            for event in [
                {"type": "thread.started", "thread_id": f"th-{self.streaming_calls}"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": f"worker step {self.streaming_calls}",
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ]:
                maybe = on_event(event)
                if inspect.isawaitable(maybe):
                    await maybe
        return CodexRunResult(
            assistant_message=self.outputs.pop(0),
            usage=Usage(prompt=1, completion=1, total=2),
            thread_id=f"th-{self.streaming_calls}",
            events=[],
            duration_ms=0,
            prompt_chars=len(prompt),
        )


class FakeContextOffload:
    def __init__(
        self,
        *,
        capsule: ContextCapsule | None = None,
        candidates: list[ExternalSessionCandidate] | None = None,
        external_capsules: dict[str, ContextCapsule] | None = None,
    ) -> None:
        self._capsule = capsule or ContextCapsule([], [], 0, 0)
        self._candidates = candidates or []
        self._external_capsules = external_capsules or {}

    async def persist_user_request(self, **_: Any) -> None:
        return None

    async def compact_session(self, **_: Any) -> None:
        return None

    async def persist_turn(self, **_: Any) -> None:
        return None

    async def build_capsule(self, **_: Any) -> ContextCapsule:
        return self._capsule

    def list_related_session_candidates(
        self, **_: Any
    ) -> list[ExternalSessionCandidate]:
        return self._candidates

    def build_capsule_from_session(
        self, session_key: str, query: str
    ) -> ContextCapsule:
        del query
        return self._external_capsules.get(session_key, ContextCapsule([], [], 0, 0))


def make_writer(tmp_path: Path, scenario: str) -> ArtifactWriter:
    context = RunContext(
        run_id=f"eval-{scenario}",
        request_id=f"req-{scenario}",
        session_key="channel:user",
        repo_path=str(tmp_path),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        artifacts_dir=tmp_path / "RUN_ARTIFACTS_DIR",
        debug_enabled=False,
    )
    return ArtifactWriter(context, max_bytes=300_000, retention_days=7)


def assert_manifest_structure(path: Path, scenario: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text("utf-8"))
    required = {
        "run_id",
        "scenario",
        "status",
        "mode",
        "final_response",
        "subagent_count",
        "event_count",
        "used_offloads",
        "live_messages",
    }
    assert required.issubset(manifest.keys())
    assert manifest["run_id"] == f"eval-{scenario}"
    assert manifest["scenario"] == scenario
    assert manifest["status"] == "ok"
    return manifest


async def run_eval_scenario(
    *,
    scenario: str,
    tmp_path: Path,
    codex_outputs: list[str],
    context: FakeContextOffload | None = None,
    todo_input_provider=None,
) -> tuple[DeepAgentResult, list[dict[str, Any]], dict[str, Any], FakeCodex]:
    writer = make_writer(tmp_path, scenario)
    codex = FakeCodex(codex_outputs)
    tools = DeepAgentToolsService(codex=codex)  # pyright: ignore[reportArgumentType]
    agent = DeepAgentService(
        codex=codex,  # pyright: ignore[reportArgumentType]
        context_offload=context or FakeContextOffload(),  # pyright: ignore[reportArgumentType]
        tools=tools,
        options=DeepAgentOptions(enabled=True, max_subagents=3, max_rounds=3),
    )

    events: list[dict[str, Any]] = []

    async def on_progress(event: dict[str, Any]) -> None:
        payload = {"run_id": writer.context.run_id, **event}
        events.append(payload)
        writer.append_event(payload)

    result = await agent.execute(
        "channel:user",
        "offline eval request",
        str(tmp_path),
        on_progress=on_progress,
        todo_input_provider=todo_input_provider,
    )
    manifest_path = writer.write_manifest(
        {
            "run_id": writer.context.run_id,
            "scenario": scenario,
            "status": "ok",
            "mode": result.mode,
            "final_response": result.final_response,
            "subagent_count": result.subagent_count,
            "event_count": len(events),
            "used_offloads": result.used_offloads,
            "live_messages": result.live_messages,
        }
    )
    manifest = assert_manifest_structure(manifest_path, scenario)
    return result, events, manifest, codex


async def test_eval_main_direct_routing_writes_artifacts(tmp_path: Path) -> None:
    result, events, manifest, _ = await run_eval_scenario(
        scenario="main-direct",
        tmp_path=tmp_path,
        codex_outputs=[
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"main_direct","reason":"simple"}',
            "direct-answer",
        ],
    )
    assert result.mode == "main_direct"
    assert result.final_response == "direct-answer"
    assert manifest["subagent_count"] == 0
    assert len(events) >= 1


async def test_eval_invalid_plan_replan_path_writes_artifacts(tmp_path: Path) -> None:
    result, _, manifest, _ = await run_eval_scenario(
        scenario="invalid-plan-replan",
        tmp_path=tmp_path,
        codex_outputs=[
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"dup-1","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"},{"id":"T1","title":"dup-2","instructions":"do-2","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"done":false,"reason":"fix invalid plan","nextTodos":[{"id":"T2","title":"valid","instructions":"do-2","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "replanned-output",
            '{"done":true,"reason":"complete","nextTodos":[]}',
            "final-aggregate",
        ],
    )
    assert result.mode == "subagent_pipeline"
    assert result.final_response == "final-aggregate"
    assert manifest["subagent_count"] == 1


async def test_eval_skip_and_stop_round_control_writes_artifacts(
    tmp_path: Path,
) -> None:
    async def todo_inputs(todo_id: str) -> list[str]:
        if todo_id == "T2":
            return ["__control__:skip", "__control__:stop-round"]
        return []

    result, events, manifest, _ = await run_eval_scenario(
        scenario="skip-stop-round",
        tmp_path=tmp_path,
        codex_outputs=[
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"},{"id":"T2","title":"second","instructions":"do-2","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            '{"offloadIds":[],"liveMessageIds":[]}',
            "round1-output",
            "final-aggregate",
        ],
        todo_input_provider=todo_inputs,
    )
    assert result.final_response == "final-aggregate"
    assert result.subagent_count == 1
    assert manifest["subagent_count"] == 1
    assert any(event.get("message") == "TODO T2 skipped by user" for event in events)
    assert any(
        "현재 라운드 종료 후 집계" in str(event.get("message", "")) for event in events
    )


async def test_eval_abort_control_writes_artifacts(tmp_path: Path) -> None:
    async def todo_inputs(_todo_id: str) -> list[str]:
        return ["__control__:abort"]

    result, events, manifest, _ = await run_eval_scenario(
        scenario="abort",
        tmp_path=tmp_path,
        codex_outputs=[
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"mode":"subagent_pipeline","reason":"complex"}',
            '{"todos":[{"id":"T1","title":"first","instructions":"do-1","priority":"high","dependsOn":[],"doneDefinition":"done"}]}',
            "final-aggregate",
        ],
        todo_input_provider=todo_inputs,
    )
    assert result.final_response == "final-aggregate"
    assert result.subagent_count == 0
    assert manifest["subagent_count"] == 0
    assert any(event.get("message") == "TODO T1 aborted by user" for event in events)


async def test_eval_external_context_selection_constraints_writes_artifacts(
    tmp_path: Path,
) -> None:
    context = FakeContextOffload(
        capsule=ContextCapsule(
            offloaded_context=["current offload"],
            live_conversation=["[user] continue previous work"],
            used_offloads=1,
            live_messages=1,
        ),
        candidates=[
            ExternalSessionCandidate(
                session_key="ext-1:user",
                channel_id="ext-1",
                updated_at="2026-01-01T00:00:00+00:00",
                offload_count=2,
                live_message_count=1,
                summary="legacy implementation decisions",
            )
        ],
        external_capsules={
            "ext-1:user": ContextCapsule(
                offloaded_context=["external context summary"],
                live_conversation=["[assistant] external note"],
                used_offloads=1,
                live_messages=1,
            )
        },
    )

    result, events, manifest, codex = await run_eval_scenario(
        scenario="external-context-routing",
        tmp_path=tmp_path,
        context=context,
        codex_outputs=[
            '{"offloadIds":[],"liveMessageIds":[]}',
            '{"useExternalContext":true,"reason":"needs old channel","selectedSessionKeys":["ext-1:user","invalid:key"]}',
            '{"mode":"main_direct","reason":"context complete"}',
            "final-direct",
        ],
    )
    assert result.mode == "main_direct"
    assert result.final_response == "final-direct"
    assert manifest["used_offloads"] == 1
    assert any(
        "외부 채널 컨텍스트 1개 세션" in str(event.get("message", ""))
        for event in events
    )
    assert any("[External Session: ext-1:user]" in prompt for prompt in codex.prompts)


class _RetryLogger:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.messages.append({"args": args, "kwargs": kwargs})


class _RetryRuntime(CodexCliRuntimeService):
    def __init__(self, logger: _RetryLogger, options: CodexCliRuntimeOptions) -> None:
        super().__init__(logger=logger, options=options)
        self.attempts = 0

    async def _run_once(
        self,
        repo_path: str,
        prompt: str,
        sandbox: str | None = None,
        timeout_ms: int | None = None,
        model: str | None = None,
    ) -> CodexRunResult:
        del repo_path, prompt, sandbox, timeout_ms, model
        self.attempts += 1
        if self.attempts < 3:
            raise codex_runtime._RetryableCodexError(
                "codex exited with retryable code 1",
                stderr_summary=f"retry-{self.attempts}",
            )
        return CodexRunResult(
            assistant_message="retry-ok",
            usage=Usage(total=3),
            thread_id="thread-retry",
            events=[{"type": "turn.completed"}],
            duration_ms=0,
            prompt_chars=16,
        )


async def test_eval_codex_runtime_retry_chain_writes_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    writer = make_writer(tmp_path, "codex-runtime-retry")
    logger = _RetryLogger()
    runtime = _RetryRuntime(
        logger=logger,
        options=CodexCliRuntimeOptions(
            binary="codex",
            timeout_ms=100,
            sandbox="workspace-write",
            retry_count=2,
            retry_backoff_ms=0,
        ),
    )

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await runtime.run(repo_path=".", prompt="retry chain eval")
    writer.append_event(
        {
            "run_id": writer.context.run_id,
            "stage": "runtime",
            "attempts": runtime.attempts,
        }
    )
    writer.append_event(
        {
            "run_id": writer.context.run_id,
            "stage": "runtime",
            "warning_count": len(logger.messages),
        }
    )
    manifest_path = writer.write_manifest(
        {
            "run_id": writer.context.run_id,
            "scenario": "codex-runtime-retry",
            "status": "ok",
            "mode": "runtime_retry",
            "final_response": result.assistant_message,
            "subagent_count": 0,
            "event_count": 2,
            "used_offloads": 0,
            "live_messages": 0,
        }
    )
    _ = assert_manifest_structure(manifest_path, "codex-runtime-retry")

    assert result.assistant_message == "retry-ok"
    assert runtime.attempts == 3
    assert sleep_calls == [0.0, 0.0]
    assert len(logger.messages) == 2
