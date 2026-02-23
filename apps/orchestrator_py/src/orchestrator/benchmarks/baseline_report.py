from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportPrivateUsage=false

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]

from orchestrator.config import AppEnv, read_env
from orchestrator.services.context_offload import (
    ContextOffloadOptions,
    ContextOffloadService,
)
from orchestrator.services.run_artifacts import ArtifactWriter, RunContext
from orchestrator.services.warpgrep_cache import (
    clear_warpgrep_inventory_cache,
    get_warpgrep_inventory,
)


def build_baseline_report(env: AppEnv | None = None) -> dict[str, JSONValue]:
    settings = env or read_env()
    now = datetime.now(tz=timezone.utc)

    with tempfile.TemporaryDirectory(prefix="orchestrator-baseline-") as work_dir_raw:
        work_dir = Path(work_dir_raw)
        context_dir = work_dir / "context"
        repo_dir = work_dir / "repo"
        _populate_context_store(context_dir)
        repo_file_count = _populate_repo_tree(repo_dir)

        service = ContextOffloadService(
            ContextOffloadOptions(
                enabled=True,
                store_dir=str(context_dir),
                max_estimated_tokens=12_000,
                keep_recent_messages=10,
                retrieve_top_k=4,
            )
        )

        related_started = perf_counter()
        candidates = service.list_related_session_candidates(
            current_session_key="bench-main:user-1",
            query="payment timeout lock contention",
            limit=12,
        )
        related_duration_ms = _duration_ms(related_started)

        capsule_started = perf_counter()
        capsule = asyncio.run(
            service.build_capsule(
                session_key="bench-main:user-1",
                query="payment timeout lock contention",
            )
        )
        capsule_duration_ms = _duration_ms(capsule_started)

        clear_warpgrep_inventory_cache()
        inventory_started = perf_counter()
        inventory_paths = get_warpgrep_inventory(str(repo_dir))
        inventory_duration_ms = _duration_ms(inventory_started)

        report: dict[str, JSONValue] = {
            "schema_version": "1.0",
            "generated_at": now.isoformat(),
            "fixture": {
                "sessions": 200,
                "repo_files": repo_file_count,
            },
            "durations_ms": {
                "list_related_session_candidates": related_duration_ms,
                "build_capsule": capsule_duration_ms,
                "warpgrep_inventory_build": inventory_duration_ms,
            },
            "results": {
                "candidate_count": len(candidates),
                "capsule": {
                    "used_offloads": capsule.used_offloads,
                    "live_messages": capsule.live_messages,
                    "offloaded_context_count": len(capsule.offloaded_context),
                    "live_conversation_count": len(capsule.live_conversation),
                },
                "warpgrep_inventory_count": len(inventory_paths),
            },
            "artifacts": {
                "enabled": bool(settings.RUN_ARTIFACTS_ENABLED),
                "written": False,
                "path": None,
            },
        }

        artifact_path = _write_report_artifact_if_enabled(
            report=report,
            env=settings,
            now=now,
            repo_path=str(repo_dir.resolve()),
        )
        if artifact_path is not None:
            artifacts = report.get("artifacts")
            if isinstance(artifacts, dict):
                artifacts["written"] = True
                artifacts["path"] = str(artifact_path)

        return report


def run() -> None:
    report = build_baseline_report()
    print(json.dumps(report, ensure_ascii=True, indent=2))


def _populate_context_store(context_dir: Path) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(context_dir),
            max_estimated_tokens=12_000,
            keep_recent_messages=10,
            retrieve_top_k=4,
        )
    )

    service._save_session(
        "bench-main:user-1",
        {
            "sessionKey": "bench-main:user-1",
            "updatedAt": "2026-01-10T12:00:00+00:00",
            "offloads": [
                {
                    "id": "main-offload-1",
                    "summary": "Incident note: payment timeout lock contention mitigation checklist",
                    "createdAt": "2026-01-10T11:00:00+00:00",
                }
            ],
            "messages": [
                {
                    "id": "main-msg-1",
                    "role": "user",
                    "content": "Please review payment timeout lock contention logs",
                    "createdAt": "2026-01-10T11:30:00+00:00",
                },
                {
                    "id": "main-msg-2",
                    "role": "assistant",
                    "content": "I will analyze lock contention and timeout traces",
                    "createdAt": "2026-01-10T11:31:00+00:00",
                },
            ],
        },
    )

    for idx in range(200):
        session_key = f"bench-{idx:03d}:user-1"
        relevant = idx % 5 == 0
        summary = (
            "payment timeout lock contention postmortem notes"
            if relevant
            else "general planning notes and status updates"
        )
        service._save_session(
            session_key,
            {
                "sessionKey": session_key,
                "updatedAt": f"2026-01-09T10:{idx % 60:02d}:00+00:00",
                "offloads": [
                    {
                        "id": f"offload-{idx}",
                        "summary": summary,
                        "createdAt": "2026-01-09T10:00:00+00:00",
                    }
                ],
                "messages": [
                    {
                        "id": f"msg-{idx}",
                        "role": "assistant",
                        "content": summary,
                        "createdAt": "2026-01-09T10:30:00+00:00",
                    }
                ],
            },
        )


def _populate_repo_tree(repo_dir: Path) -> int:
    repo_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for idx in range(240):
        top = f"module_{idx % 12:02d}"
        nested = f"part_{idx % 5:02d}"
        path = repo_dir / "src" / top / nested / f"file_{idx:03d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(f"fixture file {idx}\n", encoding="utf-8")
        files.append(path)
    return len(files)


def _write_report_artifact_if_enabled(
    report: dict[str, JSONValue],
    env: AppEnv,
    now: datetime,
    repo_path: str,
) -> Path | None:
    if not env.RUN_ARTIFACTS_ENABLED:
        return None

    run_context = RunContext(
        run_id=f"baseline-{now.strftime('%Y%m%d%H%M%S')}",
        request_id="baseline-report",
        session_key="benchmark:baseline",
        repo_path=repo_path,
        started_at=now,
        artifacts_dir=Path(env.RUN_ARTIFACTS_DIR),
        debug_enabled=env.RUN_DEBUG_PROMPTS,
    )
    writer = ArtifactWriter(
        run_context,
        max_bytes=env.RUN_ARTIFACTS_MAX_BYTES,
        retention_days=env.RUN_ARTIFACTS_RETENTION_DAYS,
        redact=env.RUN_ARTIFACTS_REDACT,
    )
    payload = json.dumps(report, ensure_ascii=True, indent=2) + "\n"
    return writer.write_bounded_text("benchmarks/baseline_report.json", payload)


def _duration_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
