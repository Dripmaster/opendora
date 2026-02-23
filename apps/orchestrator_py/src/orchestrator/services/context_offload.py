from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol


class ContextSelector(Protocol):
    async def __call__(self, input_data: dict[str, Any]) -> dict[str, list[str]]: ...


class OffloadSummarizer(Protocol):
    async def __call__(self, input_data: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(slots=True)
class ContextOffloadOptions:
    enabled: bool
    store_dir: str
    max_estimated_tokens: int
    keep_recent_messages: int
    retrieve_top_k: int
    session_cache_size: int = 32


@dataclass(slots=True)
class ContextCapsule:
    offloaded_context: list[str]
    live_conversation: list[str]
    used_offloads: int
    live_messages: int


@dataclass(slots=True)
class PersistedTurnStats:
    offloads_created: int
    live_messages: int
    total_offloads: int


@dataclass(slots=True)
class ExternalSessionCandidate:
    session_key: str
    channel_id: str
    updated_at: str
    offload_count: int
    live_message_count: int
    summary: str


class ContextOffloadService:
    def __init__(self, options: ContextOffloadOptions):
        self.options = options
        self._session_cache_size = max(0, options.session_cache_size)
        self._session_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()
        self._index_lock = threading.Lock()

    def _get_session_lock(self, session_key: str) -> threading.Lock:
        with self._session_locks_guard:
            lock = self._session_locks.get(session_key)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_key] = lock
            return lock

    def _cache_get(self, session_key: str) -> dict[str, Any] | None:
        if self._session_cache_size <= 0:
            return None
        with self._cache_lock:
            cached = self._session_cache.get(session_key)
            if cached is None:
                return None
            self._session_cache.move_to_end(session_key)
            return copy.deepcopy(cached)

    def _cache_put(self, session_key: str, state: dict[str, Any]) -> None:
        if self._session_cache_size <= 0:
            return
        with self._cache_lock:
            self._session_cache[session_key] = copy.deepcopy(state)
            self._session_cache.move_to_end(session_key)
            while len(self._session_cache) > self._session_cache_size:
                self._session_cache.popitem(last=False)

    def _cache_invalidate(self, session_key: str) -> None:
        with self._cache_lock:
            self._session_cache.pop(session_key, None)

    def _new_session_state(self, session_key: str) -> dict[str, Any]:
        return {
            "sessionKey": session_key,
            "messages": [],
            "offloads": [],
            "updatedAt": now_iso(),
        }

    async def build_capsule(
        self,
        session_key: str,
        query: str,
        select_context: ContextSelector | None = None,
    ) -> ContextCapsule:
        if not self.options.enabled:
            return ContextCapsule([], [], 0, 0)

        state = self._load_session(session_key)
        if not state["offloads"] and not state["messages"]:
            return ContextCapsule([], [], 0, 0)

        if select_context:
            selected = await select_context(
                {
                    "query": query,
                    "offloads": [
                        {
                            "id": x["id"],
                            "summary": x["summary"],
                            "createdAt": x["createdAt"],
                        }
                        for x in state["offloads"]
                    ],
                    "liveMessages": [
                        {
                            "id": x["id"],
                            "role": x["role"],
                            "content": x["content"],
                            "createdAt": x["createdAt"],
                        }
                        for x in state["messages"]
                    ],
                    "limits": {
                        "offloads": self.options.retrieve_top_k,
                        "liveMessages": self.options.keep_recent_messages,
                    },
                }
            )
            offload_ids = selected.get("offloadIds", [])
            live_ids = selected.get("liveMessageIds", [])
        else:
            offload_ids = [
                x["id"] for x in state["offloads"][-self.options.retrieve_top_k :]
            ]
            live_ids = [
                x["id"] for x in state["messages"][-self.options.keep_recent_messages :]
            ]

        picked_offloads = [x for x in state["offloads"] if x["id"] in offload_ids]
        picked_live = [x for x in state["messages"] if x["id"] in live_ids]

        return ContextCapsule(
            offloaded_context=[
                sanitize_for_prompt(x["summary"], 1000) for x in picked_offloads
            ],
            live_conversation=[
                f"[{x['role']}] {sanitize_for_prompt(x['content'], 1000)}"
                for x in picked_live
            ],
            used_offloads=len(picked_offloads),
            live_messages=len(picked_live),
        )

    async def _append_and_compact(
        self,
        state: dict[str, Any],
        summarize_offload: OffloadSummarizer | None,
    ) -> int:
        offloads_created = 0
        while len(state["messages"]) > self.options.keep_recent_messages:
            overflow = len(state["messages"]) - self.options.keep_recent_messages
            chunk_size = max(2, min(12, overflow))
            chunk = state["messages"][:chunk_size]
            state["messages"] = state["messages"][chunk_size:]
            offload = await summarize_chunk(chunk, summarize_offload)
            state["offloads"].append(offload)
            offloads_created += 1
        return offloads_created

    async def persist_user_request(
        self,
        session_key: str,
        user_message: str,
        summarize_offload: OffloadSummarizer | None = None,
    ) -> PersistedTurnStats:
        """유저 요청 저장 시: user 메시지만 append 후 오프로드."""
        if not self.options.enabled:
            return PersistedTurnStats(0, 0, 0)
        state = self._load_session(session_key)
        now = now_iso()
        state["messages"].append(
            {"id": make_id(), "role": "user", "content": user_message, "createdAt": now}
        )
        offloads_created = await self._append_and_compact(state, summarize_offload)
        state["updatedAt"] = now_iso()
        self._save_session(session_key, state)
        return PersistedTurnStats(
            offloads_created=offloads_created,
            live_messages=len(state["messages"]),
            total_offloads=len(state["offloads"]),
        )

    async def compact_session(
        self,
        session_key: str,
        summarize_offload: OffloadSummarizer | None = None,
    ) -> PersistedTurnStats:
        """append 없이 현재 세션만 live 초과분 오프로드 (일 할당 전/서브에이전트 결과 후 등)."""
        if not self.options.enabled:
            return PersistedTurnStats(0, 0, 0)
        state = self._load_session(session_key)
        offloads_created = await self._append_and_compact(state, summarize_offload)
        state["updatedAt"] = now_iso()
        self._save_session(session_key, state)
        return PersistedTurnStats(
            offloads_created=offloads_created,
            live_messages=len(state["messages"]),
            total_offloads=len(state["offloads"]),
        )

    async def persist_turn(
        self,
        session_key: str,
        assistant_message: str,
        summarize_offload: OffloadSummarizer | None = None,
    ) -> PersistedTurnStats:
        """턴 종료 시: assistant 메시지만 append 후 오프로드 (user는 이미 persist_user_request로 저장됨)."""
        if not self.options.enabled:
            return PersistedTurnStats(0, 0, 0)
        state = self._load_session(session_key)
        now = now_iso()
        state["messages"].append(
            {
                "id": make_id(),
                "role": "assistant",
                "content": assistant_message,
                "createdAt": now,
            }
        )
        offloads_created = await self._append_and_compact(state, summarize_offload)
        state["updatedAt"] = now_iso()
        self._save_session(session_key, state)
        return PersistedTurnStats(
            offloads_created=offloads_created,
            live_messages=len(state["messages"]),
            total_offloads=len(state["offloads"]),
        )

    def _session_file(self, session_key: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", session_key)
        return Path(self.options.store_dir).resolve() / "sessions" / f"{safe}.json"

    def _sessions_index_file(self) -> Path:
        return Path(self.options.store_dir).resolve() / "sessions_index.json"

    def _to_index_entry(self, state: dict[str, Any]) -> dict[str, Any] | None:
        session_key = str(state.get("sessionKey") or "")
        if not session_key:
            return None

        raw_messages = state.get("messages")
        messages = list(raw_messages) if isinstance(raw_messages, list) else []
        raw_offloads = state.get("offloads")
        offloads = list(raw_offloads) if isinstance(raw_offloads, list) else []
        summary = summarize_session_for_candidate(messages=messages, offloads=offloads)

        return {
            "sessionKey": session_key,
            "updatedAt": str(state.get("updatedAt") or ""),
            "offload_count": len(offloads),
            "live_message_count": len(messages),
            "summary": summary,
        }

    def _normalize_index_entry(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        session_key = str(raw.get("sessionKey") or "")
        if not session_key:
            return None
        try:
            offload_count = int(raw.get("offload_count") or 0)
        except (TypeError, ValueError):
            offload_count = 0
        try:
            live_message_count = int(raw.get("live_message_count") or 0)
        except (TypeError, ValueError):
            live_message_count = 0
        return {
            "sessionKey": session_key,
            "updatedAt": str(raw.get("updatedAt") or ""),
            "offload_count": max(0, offload_count),
            "live_message_count": max(0, live_message_count),
            "summary": str(raw.get("summary") or ""),
        }

    def _write_sessions_index_unlocked(
        self, entries: dict[str, dict[str, Any]]
    ) -> None:
        index_path = self._sessions_index_file()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload_entries = sorted(entries.values(), key=lambda x: x["sessionKey"])
        payload = json.dumps(payload_entries, ensure_ascii=False, indent=2) + "\n"
        tmp_path = Path(str(index_path) + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, index_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _rebuild_sessions_index_unlocked(self) -> dict[str, dict[str, Any]]:
        sessions_dir = Path(self.options.store_dir).resolve() / "sessions"
        rebuilt: dict[str, dict[str, Any]] = {}
        if sessions_dir.exists():
            for path in sessions_dir.glob("*.json"):
                try:
                    parsed = json.loads(path.read_text("utf-8"))
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                entry = self._to_index_entry(parsed)
                if entry is None:
                    continue
                rebuilt[str(entry["sessionKey"])] = entry
        self._write_sessions_index_unlocked(rebuilt)
        return rebuilt

    def _load_sessions_index(self) -> dict[str, dict[str, Any]]:
        with self._index_lock:
            index_path = self._sessions_index_file()
            if index_path.exists():
                try:
                    parsed = json.loads(index_path.read_text("utf-8"))
                    if isinstance(parsed, list):
                        loaded: dict[str, dict[str, Any]] = {}
                        for item in parsed:
                            entry = self._normalize_index_entry(item)
                            if entry is not None:
                                loaded[str(entry["sessionKey"])] = entry
                        return loaded
                except Exception:
                    pass
            return self._rebuild_sessions_index_unlocked()

    def _update_sessions_index_entry(self, state: dict[str, Any]) -> None:
        entry = self._to_index_entry(state)
        if entry is None:
            return
        with self._index_lock:
            index_path = self._sessions_index_file()
            existing: dict[str, dict[str, Any]] = {}
            if index_path.exists():
                try:
                    parsed = json.loads(index_path.read_text("utf-8"))
                    if isinstance(parsed, list):
                        for item in parsed:
                            normalized = self._normalize_index_entry(item)
                            if normalized is not None:
                                existing[str(normalized["sessionKey"])] = normalized
                except Exception:
                    existing = self._rebuild_sessions_index_unlocked()
            existing[str(entry["sessionKey"])] = entry
            self._write_sessions_index_unlocked(existing)

    def _load_session_summary_from_file(
        self, session_key: str
    ) -> dict[str, Any] | None:
        try:
            raw = json.loads(self._session_file(session_key).read_text("utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        entry = self._to_index_entry(raw)
        if entry is None:
            return None
        return entry

    def _load_session(self, session_key: str) -> dict[str, Any]:
        cached = self._cache_get(session_key)
        if cached is not None:
            return cached

        session_lock = self._get_session_lock(session_key)
        with session_lock:
            cached = self._cache_get(session_key)
            if cached is not None:
                return cached

            path = self._session_file(session_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            state = self._new_session_state(session_key)
            if path.exists():
                try:
                    parsed = json.loads(path.read_text("utf-8"))
                    if (
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("messages"), list)
                        and isinstance(parsed.get("offloads"), list)
                    ):
                        state = parsed
                except json.JSONDecodeError:
                    pass
            self._cache_put(session_key, state)
            return copy.deepcopy(state)

    def _save_session(self, session_key: str, state: dict[str, Any]) -> None:
        session_lock = self._get_session_lock(session_key)
        with session_lock:
            self._cache_invalidate(session_key)

            path = self._session_file(session_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"

            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f"{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as tmp_file:
                    tmp_file.write(payload)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                    tmp_path = Path(tmp_file.name)
                os.replace(tmp_path, path)
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            self._update_sessions_index_entry(state)

    def list_related_session_candidates(
        self,
        current_session_key: str,
        query: str = "",
        limit: int = 12,
    ) -> list[ExternalSessionCandidate]:
        if not self.options.enabled:
            return []
        current_user_id = session_user_id(current_session_key)
        if not current_user_id:
            return []
        index_entries = self._load_sessions_index()
        if not index_entries:
            return []

        query_keywords = extract_keywords(query)
        scored_items: list[tuple[int, ExternalSessionCandidate]] = []
        pending_details: list[tuple[int, str, str]] = []
        for raw in index_entries.values():
            session_key = str(raw.get("sessionKey") or "")
            if not session_key or session_key == current_session_key:
                continue
            if session_user_id(session_key) != current_user_id:
                continue
            summary = str(raw.get("summary") or "")
            updated_at = str(raw.get("updatedAt") or "")
            try:
                offload_count = int(raw.get("offload_count") or 0)
            except (TypeError, ValueError):
                offload_count = 0
            try:
                live_message_count = int(raw.get("live_message_count") or 0)
            except (TypeError, ValueError):
                live_message_count = 0
            candidate = ExternalSessionCandidate(
                session_key=session_key,
                channel_id=session_channel_id(session_key),
                updated_at=updated_at,
                offload_count=max(0, offload_count),
                live_message_count=max(0, live_message_count),
                summary=summary,
            )
            score = score_by_query(summary, query_keywords)
            scored_items.append((score, candidate))

            if not summary:
                pending_details.append((score, updated_at, session_key))

        if pending_details:
            pending_details.sort(key=lambda x: (x[0], x[1]), reverse=True)
            detail_budget = max(1, self.options.retrieve_top_k)
            details_by_key: dict[str, dict[str, Any]] = {}
            for _, _, session_key in pending_details[:detail_budget]:
                detail = self._load_session_summary_from_file(session_key)
                if detail is not None and str(detail.get("summary") or ""):
                    details_by_key[session_key] = detail

            refreshed: list[tuple[int, ExternalSessionCandidate]] = []
            for score, candidate in scored_items:
                detail = details_by_key.get(candidate.session_key)
                if detail is not None:
                    updated_summary = str(detail.get("summary") or "")
                    updated_candidate = ExternalSessionCandidate(
                        session_key=candidate.session_key,
                        channel_id=candidate.channel_id,
                        updated_at=str(detail.get("updatedAt") or candidate.updated_at),
                        offload_count=int(
                            detail.get("offload_count") or candidate.offload_count
                        ),
                        live_message_count=int(
                            detail.get("live_message_count")
                            or candidate.live_message_count
                        ),
                        summary=updated_summary,
                    )
                    refreshed.append(
                        (
                            score_by_query(updated_summary, query_keywords),
                            updated_candidate,
                        )
                    )
                else:
                    refreshed.append((score, candidate))
            scored_items = refreshed

        scored_items = [x for x in scored_items if x[1].summary]
        scored_items.sort(key=lambda x: (x[0], x[1].updated_at), reverse=True)
        return [x[1] for x in scored_items[:limit]]

    def build_capsule_from_session(
        self,
        session_key: str,
        query: str,
    ) -> ContextCapsule:
        # Reuse the same rules but without model selection to keep retrieval deterministic.
        state = self._load_session(session_key)
        if not state["offloads"] and not state["messages"]:
            return ContextCapsule([], [], 0, 0)
        query_keywords = extract_keywords(query)

        offload_ids = select_top_k_ids(
            items=state["offloads"],
            query_keywords=query_keywords,
            text_selector=lambda x: str(x.get("summary") or ""),
            limit=self.options.retrieve_top_k,
        )
        live_ids = select_top_k_ids(
            items=state["messages"],
            query_keywords=query_keywords,
            text_selector=lambda x: str(x.get("content") or ""),
            limit=self.options.keep_recent_messages,
        )
        picked_offloads = [x for x in state["offloads"] if x["id"] in offload_ids]
        picked_live = [x for x in state["messages"] if x["id"] in live_ids]
        return ContextCapsule(
            offloaded_context=[
                sanitize_for_prompt(x["summary"], 1000) for x in picked_offloads
            ],
            live_conversation=[
                f"[{x['role']}] {sanitize_for_prompt(x['content'], 1000)}"
                for x in picked_live
            ],
            used_offloads=len(picked_offloads),
            live_messages=len(picked_live),
        )


def score_by_query(text: str, query_keywords: list[str]) -> int:
    if not query_keywords:
        return 0
    lowered = text.lower()
    text_keywords = set(extract_keywords(text))
    score = 0
    for keyword in query_keywords:
        if keyword in text_keywords:
            score += 3
        elif keyword in lowered:
            score += 1
    return score


def select_top_k_ids(
    items: list[dict[str, Any]],
    query_keywords: list[str],
    text_selector: Callable[[dict[str, Any]], str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    if not query_keywords:
        return [str(x["id"]) for x in items[-limit:]]

    scored: list[tuple[int, int, str]] = []
    for idx, item in enumerate(items):
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        score = score_by_query(text_selector(item), query_keywords)
        scored.append((score, idx, item_id))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [item_id for score, _, item_id in scored[:limit] if score > 0]
    if selected:
        return selected
    return [str(x["id"]) for x in items[-limit:]]


async def summarize_chunk(
    chunk: list[dict[str, Any]], summarize_offload: OffloadSummarizer | None
) -> dict[str, Any]:
    if summarize_offload:
        model_result = await summarize_offload(
            {
                "messages": [
                    {"role": x["role"], "content": x["content"]} for x in chunk
                ],
            }
        )
        summary = sanitize_for_prompt(
            str(model_result.get("summary", "")).strip(), 1400
        )
        keywords = list(model_result.get("keywords") or [])[:24]
        if not keywords:
            keywords = extract_keywords(summary)
    else:
        joined = " | ".join(
            [
                f"{'U' if x['role'] == 'user' else 'A'}: {sanitize_for_prompt(str(x['content']), 280)}"
                for x in chunk
            ]
        )
        summary = joined if len(joined) <= 1400 else joined[:1399] + "..."
        keywords = extract_keywords(summary)

    return {
        "id": make_id(),
        "summary": summary,
        "keywords": keywords,
        "createdAt": now_iso(),
        "sourceMessageIds": [x["id"] for x in chunk],
    }


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = sum(len(str(x.get("content", ""))) for x in messages)
    return (chars + 3) // 4


def sanitize_for_prompt(text: str, max_len: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= max_len else compact[: max_len - 1] + "..."


def make_id() -> str:
    return f"{int(datetime.now(tz=timezone.utc).timestamp() * 1000):x}"


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9가-힣_/-]{2,}", text.lower())
    freq: dict[str, int] = {}
    for token in tokens:
        if token in STOPWORDS:
            continue
        freq[token] = freq.get(token, 0) + 1
    ordered = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ordered[:24]]


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "have",
    "will",
    "you",
    "are",
    "http",
    "https",
    "www",
    "있는",
    "해서",
    "그냥",
    "그리고",
    "대한",
    "요청",
    "실행",
}


def session_user_id(session_key: str) -> str:
    parts = session_key.split(":")
    return parts[-1] if len(parts) >= 2 else ""


def session_channel_id(session_key: str) -> str:
    parts = session_key.split(":")
    return parts[0] if len(parts) >= 2 else ""


def summarize_session_for_candidate(messages: list[Any], offloads: list[Any]) -> str:
    blocks: list[str] = []
    if offloads:
        tail = offloads[-2:]
        for item in tail:
            if isinstance(item, dict):
                text = str(item.get("summary") or "").strip()
                if text:
                    blocks.append(f"offload: {sanitize_for_prompt(text, 220)}")
    if messages:
        tail = messages[-4:]
        for item in tail:
            if isinstance(item, dict):
                role = str(item.get("role") or "user")
                content = str(item.get("content") or "").strip()
                if content:
                    blocks.append(f"{role}: {sanitize_for_prompt(content, 180)}")
    return " | ".join(blocks)[:1200]
