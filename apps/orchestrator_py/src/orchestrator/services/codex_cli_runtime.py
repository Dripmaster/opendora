from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class CodexCliRuntimeOptions:
    binary: str
    timeout_ms: int
    sandbox: str
    model: str | None = None
    retry_count: int = 0
    retry_backoff_ms: int = 250


@dataclass(slots=True)
class _RetryAttempt:
    attempt: int
    reason: str
    stderr_summary: str


class _RetryableCodexError(RuntimeError):
    def __init__(self, reason: str, stderr_summary: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.stderr_summary = stderr_summary


class _NonRetryableCodexError(RuntimeError):
    pass


@dataclass(slots=True)
class Usage:
    prompt: int = 0
    completion: int = 0
    total: int = 0


@dataclass(slots=True)
class CodexRunResult:
    assistant_message: str
    usage: Usage
    thread_id: str | None
    events: list[dict[str, Any]]


class CodexCliRuntimeService:
    def __init__(self, logger: Any, options: CodexCliRuntimeOptions):
        self.logger = logger
        self.options = options

    async def run(self, repo_path: str, prompt: str, sandbox: str | None = None, timeout_ms: int | None = None) -> CodexRunResult:
        return await self._run_with_retries(
            operation=lambda: self._run_once(repo_path=repo_path, prompt=prompt, sandbox=sandbox, timeout_ms=timeout_ms),
            operation_name="codex run",
        )

    async def _run_once(self, repo_path: str, prompt: str, sandbox: str | None = None, timeout_ms: int | None = None) -> CodexRunResult:
        args = self._build_args(repo_path=repo_path, prompt=prompt, sandbox=sandbox)
        proc = await asyncio.create_subprocess_exec(
            self.options.binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=(timeout_ms or self.options.timeout_ms) / 1000)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise _RetryableCodexError("codex execution timed out", stderr_summary="timeout")

        stderr_text = (stderr or b"").decode("utf-8", errors="ignore").strip()
        if stderr_text:
            self.logger.warning("Codex CLI wrote to stderr.", stderr=stderr_text[:2000])

        if proc.returncode != 0:
            self._raise_for_exit_failure(proc.returncode, stderr_text)

        parsed = parse_codex_json_events((stdout or b"").decode("utf-8", errors="ignore"))
        if not parsed.events:
            raise _RetryableCodexError("failed to parse codex json events", stderr_summary=(stderr_text or "no json events")[:300])
        return parsed

    async def run_streaming(
        self,
        repo_path: str,
        prompt: str,
        sandbox: str | None = None,
        timeout_ms: int | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> CodexRunResult:
        return await self._run_with_retries(
            operation=lambda: self._run_streaming_once(
                repo_path=repo_path,
                prompt=prompt,
                sandbox=sandbox,
                timeout_ms=timeout_ms,
                on_event=on_event,
            ),
            operation_name="codex run_streaming",
        )

    async def _run_streaming_once(
        self,
        repo_path: str,
        prompt: str,
        sandbox: str | None = None,
        timeout_ms: int | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> CodexRunResult:
        args = self._build_args(repo_path=repo_path, prompt=prompt, sandbox=sandbox)
        proc = await asyncio.create_subprocess_exec(
            self.options.binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        deadline = (timeout_ms or self.options.timeout_ms) / 1000
        events: list[dict[str, Any]] = []
        assistant_message = ""
        thread_id: str | None = None
        usage = Usage()

        async def _read_stdout() -> None:
            nonlocal assistant_message, thread_id, usage
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").strip()
                if not (text.startswith("{") and text.endswith("}")):
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                events.append(parsed)
                if on_event:
                    maybe_awaited = on_event(parsed)
                    if inspect.isawaitable(maybe_awaited):
                        await maybe_awaited
                if parsed.get("type") == "thread.started" and isinstance(parsed.get("thread_id"), str):
                    thread_id = parsed["thread_id"]
                if parsed.get("type") == "item.completed":
                    item = parsed.get("item") if isinstance(parsed.get("item"), dict) else {}
                    if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                        assistant_message = item["text"]
                if parsed.get("type") == "turn.completed":
                    raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
                    inp = int(raw_usage.get("input_tokens", 0) or 0)
                    out = int(raw_usage.get("output_tokens", 0) or 0)
                    total = int(raw_usage.get("total_tokens", inp + out) or (inp + out))
                    usage = Usage(prompt=inp, completion=out, total=total)

        try:
            await asyncio.wait_for(_read_stdout(), timeout=deadline)
            stderr = await proc.stderr.read() if proc.stderr else b""
            await asyncio.wait_for(proc.wait(), timeout=1)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise _RetryableCodexError("codex streaming execution timed out", stderr_summary="timeout")

        stderr_text = (stderr or b"").decode("utf-8", errors="ignore").strip()
        if stderr_text:
            self.logger.warning("Codex CLI wrote to stderr.", stderr=stderr_text[:2000])
        if proc.returncode != 0:
            self._raise_for_exit_failure(proc.returncode, stderr_text)
        if not events:
            raise _RetryableCodexError("failed to parse codex json events", stderr_summary=(stderr_text or "no json events")[:300])

        return CodexRunResult(
            assistant_message=assistant_message,
            usage=usage,
            thread_id=thread_id,
            events=events,
        )

    async def _run_with_retries(self, operation: Callable[[], Any], operation_name: str) -> CodexRunResult:
        max_attempts = max(1, self.options.retry_count + 1)
        attempts: list[_RetryAttempt] = []
        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except _NonRetryableCodexError as exc:
                raise RuntimeError(f"{operation_name} failed with non-retryable error: {exc}") from exc
            except _RetryableCodexError as exc:
                summary = exc.stderr_summary or "unknown"
                attempts.append(_RetryAttempt(attempt=attempt, reason=exc.reason, stderr_summary=summary))
                if attempt >= max_attempts:
                    last = attempts[-1]
                    raise RuntimeError(
                        f"{operation_name} failed after {len(attempts)} attempt(s); "
                        f"last_reason={last.reason}; last_stderr={last.stderr_summary}"
                    ) from exc
                backoff_sec = (self.options.retry_backoff_ms * (2 ** (attempt - 1))) / 1000
                self.logger.warning(
                    "Codex CLI transient failure. Retrying.",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=exc.reason,
                )
                await asyncio.sleep(backoff_sec)

        raise RuntimeError(f"{operation_name} failed before starting")

    def _raise_for_exit_failure(self, returncode: int, stderr_text: str) -> None:
        err_excerpt = (stderr_text or "unknown error")[:1000]
        lower_excerpt = err_excerpt.lower()
        if returncode in {2, 126, 127} or any(
            marker in lower_excerpt
            for marker in ("unknown option", "invalid option", "invalid argument", "permission denied", "access denied")
        ):
            raise _NonRetryableCodexError(f"codex exited with code {returncode}: {err_excerpt}")
        if returncode in {1, 124, 137}:
            raise _RetryableCodexError(f"codex exited with retryable code {returncode}", stderr_summary=err_excerpt[:300])
        raise _NonRetryableCodexError(f"codex exited with code {returncode}: {err_excerpt}")

    def _build_args(self, repo_path: str, prompt: str, sandbox: str | None) -> list[str]:
        args = ["exec", "--json", "-C", repo_path, "--sandbox", sandbox or self.options.sandbox]
        if self.options.model:
            args += ["--model", self.options.model]
        args.append(prompt)
        return args


def parse_codex_json_events(stdout: str) -> CodexRunResult:
    assistant_message = ""
    thread_id: str | None = None
    usage = Usage()
    events: list[dict[str, Any]] = []

    for raw in stdout.splitlines():
        line = raw.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        events.append(parsed)
        if parsed.get("type") == "thread.started" and isinstance(parsed.get("thread_id"), str):
            thread_id = parsed["thread_id"]
        if parsed.get("type") == "item.completed":
            item = parsed.get("item") if isinstance(parsed.get("item"), dict) else {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                assistant_message = item["text"]
        if parsed.get("type") == "turn.completed":
            raw_usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
            inp = int(raw_usage.get("input_tokens", 0) or 0)
            out = int(raw_usage.get("output_tokens", 0) or 0)
            total = int(raw_usage.get("total_tokens", inp + out) or (inp + out))
            usage = Usage(prompt=inp, completion=out, total=total)

    return CodexRunResult(
        assistant_message=assistant_message,
        usage=usage,
        thread_id=thread_id,
        events=events,
    )
