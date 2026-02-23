from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]

from orchestrator.services.run_artifacts import (  # type: ignore
    ArtifactLimitError,
    ArtifactWriter,
    RunContext,
)


def make_context(tmp_path: Path, run_id: str = "run-1") -> RunContext:
    return RunContext(
        run_id=run_id,
        request_id="req-1",
        session_key="chan:user",
        repo_path=str((tmp_path / "repo").resolve()),
        started_at=datetime.now(tz=timezone.utc),
        artifacts_dir=tmp_path / "runs",
        debug_enabled=False,
    )


def test_writer_keeps_paths_under_run_dir(tmp_path: Path) -> None:
    writer = ArtifactWriter(make_context(tmp_path), max_bytes=4_096, retention_days=7)

    path = writer.write_bounded_text("logs/result.txt", "ok")
    assert path == (tmp_path / "runs" / "run-1" / "logs" / "result.txt").resolve()
    assert path.read_text("utf-8") == "ok"

    with pytest.raises(ValueError):
        _ = writer.write_bounded_text("../escape.txt", "blocked")


def test_writer_redacts_by_default(tmp_path: Path) -> None:
    writer = ArtifactWriter(make_context(tmp_path), max_bytes=8_192, retention_days=7)
    repo_path = str((tmp_path / "repo").resolve())

    text = f"user=123456789012345678 repo={repo_path}/src/main.py"
    text_path = writer.write_bounded_text("trace.txt", text)
    content = text_path.read_text("utf-8")
    assert "123456789012345678" not in content
    assert repo_path not in content
    assert "[REDACTED_DISCORD_ID]" in content
    assert "[REDACTED_REPO_PATH]" in content

    _ = writer.append_event({"prompt": "super secret prompt", "api_token": "abc123"})
    event_line = (tmp_path / "runs" / "run-1" / "events.jsonl").read_text("utf-8")
    assert "super secret prompt" not in event_line
    assert "abc123" not in event_line
    assert "[REDACTED_PROMPT]" in event_line
    assert "[REDACTED_TOKEN]" in event_line


def test_writer_enforces_size_limit(tmp_path: Path) -> None:
    writer = ArtifactWriter(make_context(tmp_path), max_bytes=20, retention_days=7)
    _ = writer.write_bounded_text("a.txt", "1234567890")

    with pytest.raises(ArtifactLimitError):
        _ = writer.write_bounded_text("b.txt", "12345678901")


def test_writer_prunes_old_run_dirs_on_create(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    old_dir = runs_dir / "old-run"
    recent_dir = runs_dir / "recent-run"
    old_dir.mkdir(parents=True)
    recent_dir.mkdir(parents=True)

    now = datetime.now(tz=timezone.utc)
    three_days_ago = (now - timedelta(days=3)).timestamp()
    two_hours_ago = (now - timedelta(hours=2)).timestamp()
    os.utime(old_dir, (three_days_ago, three_days_ago))
    os.utime(recent_dir, (two_hours_ago, two_hours_ago))

    context = make_context(tmp_path, run_id="new-run")
    _ = ArtifactWriter(context, max_bytes=4_096, retention_days=1)

    assert not old_dir.exists()
    assert recent_dir.exists()
    assert (runs_dir / "new-run").exists()


def test_writer_prunes_oldest_run_dirs_by_max_count(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    base_time = datetime.now(tz=timezone.utc)
    for idx in range(1, 6):
        run_dir = runs_dir / f"run-{idx}"
        run_dir.mkdir()
        modified = (base_time - timedelta(minutes=6 - idx)).timestamp()
        os.utime(run_dir, (modified, modified))

    context = make_context(tmp_path, run_id="new-run")
    _ = ArtifactWriter(
        context,
        max_bytes=4_096,
        retention_days=365,
        max_runs_to_keep=3,
    )

    remaining = sorted(path.name for path in runs_dir.iterdir() if path.is_dir())
    assert remaining == ["new-run", "run-4", "run-5"]
