from __future__ import annotations

import asyncio

from orchestrator.services.codex_cli_runtime import (
    CodexCliRuntimeOptions,
    CodexCliRuntimeService,
    parse_codex_json_events,
)


class _FakeLogger:
    def warning(self, *args, **kwargs) -> None:
        return None


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
        success_stdout = b'\n'.join(
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
