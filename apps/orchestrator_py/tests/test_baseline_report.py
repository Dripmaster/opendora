from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false

import json
from pathlib import Path
from typing import cast

from orchestrator.benchmarks.baseline_report import build_baseline_report, run


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def test_build_baseline_report_schema_without_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RUN_ARTIFACTS_ENABLED", "false")
    monkeypatch.setenv("RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))

    report = build_baseline_report()

    assert report["schema_version"] == "1.0"
    assert isinstance(report["generated_at"], str)

    fixture = _as_dict(report["fixture"])
    assert fixture["sessions"] == 200
    assert isinstance(fixture["repo_files"], int)

    durations = _as_dict(report["durations_ms"])
    for key in (
        "list_related_session_candidates",
        "build_capsule",
        "warpgrep_inventory_build",
    ):
        value = durations[key]
        assert isinstance(value, int | float)
        assert value >= 0

    results = _as_dict(report["results"])
    assert isinstance(results["candidate_count"], int)
    assert isinstance(results["warpgrep_inventory_count"], int)
    capsule = _as_dict(results["capsule"])
    assert isinstance(capsule["used_offloads"], int)
    assert isinstance(capsule["live_messages"], int)
    assert isinstance(capsule["offloaded_context_count"], int)
    assert isinstance(capsule["live_conversation_count"], int)

    artifacts = _as_dict(report["artifacts"])
    assert artifacts["enabled"] is False
    assert artifacts["written"] is False
    assert artifacts["path"] is None


def test_build_baseline_report_writes_artifact_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RUN_ARTIFACTS_ENABLED", "true")
    monkeypatch.setenv("RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RUN_ARTIFACTS_REDACT", "true")

    report = build_baseline_report()

    artifacts = _as_dict(report["artifacts"])
    assert artifacts["enabled"] is True
    assert artifacts["written"] is True
    assert isinstance(artifacts["path"], str)

    artifact_path = Path(artifacts["path"])
    assert artifact_path.exists()
    persisted_raw = cast(object, json.loads(artifact_path.read_text(encoding="utf-8")))
    persisted = _as_dict(persisted_raw)
    assert persisted["schema_version"] == "1.0"
    assert "durations_ms" in persisted


def test_baseline_run_prints_valid_json(capsys, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_ARTIFACTS_ENABLED", "false")
    monkeypatch.setenv("RUN_ARTIFACTS_DIR", str(tmp_path / "runs"))

    run()

    captured = capsys.readouterr()
    payload_raw = cast(object, json.loads(captured.out))
    payload = _as_dict(payload_raw)
    assert payload["schema_version"] == "1.0"
    assert "durations_ms" in payload
