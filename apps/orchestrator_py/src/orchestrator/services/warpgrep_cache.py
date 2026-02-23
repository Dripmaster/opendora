from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass(slots=True)
class _InventoryCacheEntry:
    fingerprint: str
    paths: list[str]


_INVENTORY_CACHE: dict[str, _InventoryCacheEntry] = {}


def clear_warpgrep_inventory_cache() -> None:
    _INVENTORY_CACHE.clear()


def get_warpgrep_inventory(repo_path: str) -> list[str]:
    root = Path(repo_path)
    if not root.exists() or not root.is_dir():
        return []

    cache_key = str(root.resolve())
    fingerprint = compute_repo_fingerprint(root)
    cached = _INVENTORY_CACHE.get(cache_key)
    if cached and cached.fingerprint == fingerprint:
        return cached.paths

    paths = _walk_inventory(root)
    _INVENTORY_CACHE[cache_key] = _InventoryCacheEntry(
        fingerprint=fingerprint,
        paths=paths,
    )
    return paths


def compute_repo_fingerprint(repo_path: Path) -> str:
    root = repo_path.resolve()
    root_mtime = root.stat().st_mtime_ns
    git_dir = root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return f"nogit:{root_mtime}"

    try:
        head_value = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return f"head-unreadable:{root_mtime}"

    marker_mtime = root_mtime
    if head_value.startswith("ref: "):
        ref_name = head_value.split(":", 1)[1].strip()
        ref_path = git_dir / ref_name
        if ref_path.exists():
            marker_mtime = ref_path.stat().st_mtime_ns

    return f"head={head_value}|marker_mtime_ns={marker_mtime}"


def _walk_inventory(root: Path) -> list[str]:
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for fname in filenames:
            rel_path = (rel_dir / fname).as_posix() if str(rel_dir) != "." else fname
            paths.append(rel_path)
    return paths
