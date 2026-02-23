from __future__ import annotations

import asyncio

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator.services.codex_cli_runtime import (  # pyright: ignore[reportMissingImports]
    CodexCliRuntimeOptions,
    CodexCliRuntimeService,
    parse_codex_json_events,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def warning(self, *args, **kwargs) -> None:
        self.warnings.append((args, kwargs))


def _extract_model_arg(args: tuple[object, ...]) -> str | None:
    string_args = [arg for arg in args if isinstance(arg, str)]
    if "--model" not in string_args:
        return None
    index = string_args.index("--model")
    if index + 1 >= len(string_args):
        return None
    return string_args[index + 1]


class _FakeProcess:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


def test_parse_codex_json_events_extracts_text_and_usage() -> None:
    out = "\n".join(
        [
            "noise",
            '{"type":"thread.started","thread_id":"t-1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}',
            '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":5,"total_tokens":17}}',
        ]
    )
    parsed = parse_codex_json_events(out)
    assert parsed.thread_id == "t-1"
    assert parsed.assistant_message == "hello"
    assert parsed.usage.prompt == 12
    assert parsed.usage.completion == 5
    assert parsed.usage.total == 17


def test_parse_codex_json_events_handles_missing_usage() -> None:
    out = '{"type":"turn.completed"}'
    parsed = parse_codex_json_events(out)
    assert parsed.usage.prompt == 0
    assert parsed.usage.completion == 0
    assert parsed.usage.total == 0


def test_run_retries_once_and_then_succeeds(monkeypatch) -> None:
    attempts = 0

    async def _fake_create_subprocess_exec(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _FakeProcess(returncode=1, stdout=b"", stderr=b"temporary issue")
        success_stdout = b"\n".join(
            [
                b'{"type":"thread.started","thread_id":"thread-1"}',
                b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
                b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}',
            ]
        )
        return _FakeProcess(returncode=0, stdout=success_stdout, stderr=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    service = CodexCliRuntimeService(
        logger=_FakeLogger(),
        options=CodexCliRuntimeOptions(
            binary="codex",
            timeout_ms=500,
            sandbox="workspace-write",
            retry_count=1,
            retry_backoff_ms=0,
        ),
    )

    result = asyncio.run(service.run(repo_path=".", prompt="hello"))

    assert attempts == 2
    assert result.assistant_message == "ok"
    assert result.thread_id == "thread-1"
    assert result.usage.total == 3


def test_model_rotation_run_rotates_candidates_on_retryable_failures(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    attempts = 0

    async def _fake_create_subprocess_exec(*args, **_kwargs):
        nonlocal attempts
        calls.append(args)
        attempts += 1
        if attempts < 3:
            return _FakeProcess(returncode=1, stdout=b"", stderr=b"temporary issue")
        success_stdout = b"\n".join(
            [
                b'{"type":"thread.started","thread_id":"thread-1"}',
                b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
                b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}',
            ]
        )
        return _FakeProcess(returncode=0, stdout=success_stdout, stderr=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    logger = _FakeLogger()
    service = CodexCliRuntimeService(
        logger=logger,
        options=CodexCliRuntimeOptions(
            binary="codex",
            timeout_ms=500,
            sandbox="workspace-write",
            model="model-primary",
            model_candidates=["model-fallback-1", "model-fallback-2"],
            retry_count=2,
            retry_backoff_ms=0,
        ),
    )

    result = asyncio.run(service.run(repo_path=".", prompt="hello"))

    assert result.assistant_message == "ok"
    assert [_extract_model_arg(call) for call in calls] == [
        "model-primary",
        "model-fallback-1",
        "model-fallback-2",
    ]
    retry_logs = [
        warning
        for warning in logger.warnings
        if warning[0] and warning[0][0] == "Codex CLI transient failure. Retrying."
    ]
    assert len(retry_logs) == 2
    assert retry_logs[0][1]["attempt"] == 1
    assert retry_logs[0][1]["model"] == "model-primary"
    assert retry_logs[0][1]["reason"] == "codex exited with retryable code 1"
    assert retry_logs[0][1]["stderr_summary"] == "temporary issue"


def test_model_rotation_run_does_not_retry_or_rotate_on_nonretryable_failure(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    async def _fake_create_subprocess_exec(*args, **_kwargs):
        calls.append(args)
        return _FakeProcess(returncode=2, stdout=b"", stderr=b"invalid argument")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    service = CodexCliRuntimeService(
        logger=_FakeLogger(),
        options=CodexCliRuntimeOptions(
            binary="codex",
            timeout_ms=500,
            sandbox="workspace-write",
            model="model-primary",
            model_candidates=["model-fallback-1", "model-fallback-2"],
            retry_count=3,
            retry_backoff_ms=0,
        ),
    )

    with pytest.raises(RuntimeError, match="non-retryable"):
        asyncio.run(service.run(repo_path=".", prompt="hello"))

    assert len(calls) == 1
    assert _extract_model_arg(calls[0]) == "model-primary"
