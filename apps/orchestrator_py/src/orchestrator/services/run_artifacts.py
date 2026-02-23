from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]


DISCORD_ID_PATTERN = re.compile(r"(?<!\d)\d{16,22}(?!\d)")
REDACTED_DISCORD_ID = "[REDACTED_DISCORD_ID]"
REDACTED_REPO_PATH = "[REDACTED_REPO_PATH]"
REDACTED_PROMPT = "[REDACTED_PROMPT]"
REDACTED_TOKEN = "[REDACTED_TOKEN]"
MAX_RUN_DIRS_DEFAULT = 200


class ArtifactLimitError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class RunContext:
    run_id: str
    request_id: str
    session_key: str
    repo_path: str
    started_at: datetime
    artifacts_dir: Path
    debug_enabled: bool


class ArtifactWriter:
    def __init__(
        self,
        context: RunContext,
        *,
        max_bytes: int,
        retention_days: int,
        max_runs_to_keep: int = MAX_RUN_DIRS_DEFAULT,
        redact: bool = True,
    ) -> None:
        self.context: RunContext = context
        self.max_bytes: int = max_bytes
        self.retention_days: int = retention_days
        self.max_runs_to_keep: int = max_runs_to_keep
        self.redact: bool = redact

        self.base_dir: Path = context.artifacts_dir.resolve()
        self.run_dir: Path = self._resolve_under_base(context.run_id)
        self._bytes_written: int
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._prune_old_runs(protected_dir=self.run_dir)
        self._bytes_written = self._compute_total_bytes()

    def write_manifest(self, data: Mapping[str, object]) -> Path:
        payload = self._serialize_json(data, pretty=True)
        path = self._resolve_under_run("manifest.json")
        self._write_bytes(path, payload)
        return path

    def append_event(self, event: Mapping[str, object]) -> Path:
        payload = self._serialize_json(event, pretty=False)
        line = payload + b"\n"
        path = self._resolve_under_run("events.jsonl")
        self._write_bytes(path, line, append=True)
        return path

    def write_bounded_text(self, relative_path: str, text: str) -> Path:
        redacted = self._redact_text(text)
        path = self._resolve_under_run(relative_path)
        self._write_bytes(path, redacted.encode("utf-8"))
        return path

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    def _serialize_json(self, data: Mapping[str, object], *, pretty: bool) -> bytes:
        scrubbed = self._scrub_payload(data)
        if pretty:
            text = json.dumps(scrubbed, ensure_ascii=True, indent=2) + "\n"
        else:
            text = json.dumps(scrubbed, ensure_ascii=True, separators=(",", ":"))
        return text.encode("utf-8")

    def _scrub_payload(self, value: object, *, key_name: str = "") -> JSONValue:
        lowered_key = key_name.lower()
        if isinstance(value, Mapping):
            result: dict[str, JSONValue] = {}
            for key, item in cast(dict[object, object], value).items():
                key_obj: object = key
                item_obj: object = item
                key_text = str(key_obj)
                result[key_text] = self._scrub_payload(item_obj, key_name=key_text)
            return result
        if isinstance(value, list):
            scrubbed_items: list[JSONValue] = []
            for item in cast(list[object], value):
                list_item_obj: object = item
                scrubbed_items.append(
                    self._scrub_payload(list_item_obj, key_name=key_name)
                )
            return scrubbed_items
        if isinstance(value, str):
            if "token" in lowered_key:
                return REDACTED_TOKEN
            if "prompt" in lowered_key and not self.context.debug_enabled:
                return REDACTED_PROMPT
            return self._redact_text(value)
        if value is None or isinstance(value, bool | int | float):
            return value
        return self._redact_text(str(value))

    def _redact_text(self, text: str) -> str:
        out = text
        if self.redact:
            out = DISCORD_ID_PATTERN.sub(REDACTED_DISCORD_ID, out)
            out = self._redact_repo_path(out)
        return out

    def _redact_repo_path(self, text: str) -> str:
        repo = Path(self.context.repo_path)
        if not repo.is_absolute():
            return text
        repo_path = str(repo.resolve())
        escaped = re.escape(repo_path)
        return re.sub(rf"{escaped}(?=$|[\\/])", REDACTED_REPO_PATH, text)

    def _write_bytes(self, path: Path, data: bytes, *, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            delta = len(data)
        else:
            current_size = path.stat().st_size if path.exists() else 0
            delta = len(data) - current_size

        projected = self._bytes_written + delta
        if projected > self.max_bytes:
            raise ArtifactLimitError(
                f"Run artifacts exceeded byte limit ({projected} > {self.max_bytes})"
            )

        if append:
            with path.open("ab") as fp:
                _ = fp.write(data)
        else:
            _ = path.write_bytes(data)

        self._bytes_written = projected

    def _resolve_under_base(self, relative_path: str) -> Path:
        target = (self.base_dir / relative_path).resolve()
        if self.base_dir != target and self.base_dir not in target.parents:
            raise ValueError(f"Path escapes artifacts dir: {relative_path}")
        return target

    def _resolve_under_run(self, relative_path: str) -> Path:
        target = (self.run_dir / relative_path).resolve()
        if self.run_dir != target and self.run_dir not in target.parents:
            raise ValueError(f"Path escapes run dir: {relative_path}")
        return target

    def _compute_total_bytes(self) -> int:
        total = 0
        for file_path in self.run_dir.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total

    def _prune_old_runs(self, *, protected_dir: Path) -> None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.retention_days)
        run_dirs: list[tuple[Path, float]] = []
        for candidate in self.base_dir.iterdir():
            if not candidate.is_dir():
                continue
            try:
                modified_ts = candidate.stat().st_mtime
            except OSError:
                continue

            if (
                self.retention_days >= 0
                and candidate != protected_dir
                and datetime.fromtimestamp(modified_ts, tz=timezone.utc) < cutoff
            ):
                try:
                    shutil.rmtree(candidate)
                except OSError:
                    continue
                continue

            run_dirs.append((candidate, modified_ts))

        if len(run_dirs) <= self.max_runs_to_keep:
            return

        for candidate, _modified_ts in sorted(
            run_dirs, key=lambda item: (item[1], item[0].name)
        ):
            if len(run_dirs) <= self.max_runs_to_keep:
                break
            if candidate == protected_dir:
                continue
            try:
                shutil.rmtree(candidate)
            except OSError:
                continue
            run_dirs = [item for item in run_dirs if item[0] != candidate]
