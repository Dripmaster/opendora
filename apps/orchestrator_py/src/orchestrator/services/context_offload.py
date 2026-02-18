from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


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

    async def build_capsule(self, session_key: str, query: str, select_context: ContextSelector | None = None) -> ContextCapsule:
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
                        {"id": x["id"], "summary": x["summary"], "createdAt": x["createdAt"]}
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
            offload_ids = [x["id"] for x in state["offloads"][-self.options.retrieve_top_k :]]
            live_ids = [x["id"] for x in state["messages"][-self.options.keep_recent_messages :]]

        picked_offloads = [x for x in state["offloads"] if x["id"] in offload_ids]
        picked_live = [x for x in state["messages"] if x["id"] in live_ids]

        return ContextCapsule(
            offloaded_context=[sanitize_for_prompt(x["summary"], 1000) for x in picked_offloads],
            live_conversation=[f"[{x['role']}] {sanitize_for_prompt(x['content'], 1000)}" for x in picked_live],
            used_offloads=len(picked_offloads),
            live_messages=len(picked_live),
        )

    async def persist_turn(
        self,
        session_key: str,
        user_message: str,
        assistant_message: str,
        summarize_offload: OffloadSummarizer | None = None,
    ) -> PersistedTurnStats:
        if not self.options.enabled:
            return PersistedTurnStats(0, 0, 0)

        state = self._load_session(session_key)
        now = now_iso()
        state["messages"].append({"id": make_id(), "role": "user", "content": user_message, "createdAt": now})
        state["messages"].append(
            {"id": make_id(), "role": "assistant", "content": assistant_message, "createdAt": now_iso()}
        )

        offloads_created = 0
        while estimate_tokens(state["messages"]) > self.options.max_estimated_tokens:
            overflow = len(state["messages"]) - self.options.keep_recent_messages
            if overflow < 2:
                break
            chunk_size = max(2, min(12, overflow))
            chunk = state["messages"][:chunk_size]
            state["messages"] = state["messages"][chunk_size:]
            offload = await summarize_chunk(chunk, summarize_offload)
            state["offloads"].append(offload)
            offloads_created += 1

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

    def _load_session(self, session_key: str) -> dict[str, Any]:
        path = self._session_file(session_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                parsed = json.loads(path.read_text("utf-8"))
                if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list) and isinstance(parsed.get("offloads"), list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {
            "sessionKey": session_key,
            "messages": [],
            "offloads": [],
            "updatedAt": now_iso(),
        }

    def _save_session(self, session_key: str, state: dict[str, Any]) -> None:
        path = self._session_file(session_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", "utf-8")

    def list_related_session_candidates(
        self,
        current_session_key: str,
        limit: int = 12,
    ) -> list[ExternalSessionCandidate]:
        if not self.options.enabled:
            return []
        current_user_id = session_user_id(current_session_key)
        if not current_user_id:
            return []
        sessions_dir = Path(self.options.store_dir).resolve() / "sessions"
        if not sessions_dir.exists():
            return []

        items: list[ExternalSessionCandidate] = []
        for path in sessions_dir.glob("*.json"):
            try:
                raw = json.loads(path.read_text("utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            session_key = str(raw.get("sessionKey") or "")
            if not session_key or session_key == current_session_key:
                continue
            if session_user_id(session_key) != current_user_id:
                continue
            messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
            offloads = raw.get("offloads") if isinstance(raw.get("offloads"), list) else []
            summary = summarize_session_for_candidate(messages=messages, offloads=offloads)
            if not summary:
                continue
            items.append(
                ExternalSessionCandidate(
                    session_key=session_key,
                    channel_id=session_channel_id(session_key),
                    updated_at=str(raw.get("updatedAt") or ""),
                    offload_count=len(offloads),
                    live_message_count=len(messages),
                    summary=summary,
                )
            )
        items.sort(key=lambda x: x.updated_at, reverse=True)
        return items[:limit]

    def build_capsule_from_session(
        self,
        session_key: str,
        query: str,
    ) -> ContextCapsule:
        # Reuse the same rules but without model selection to keep retrieval deterministic.
        state = self._load_session(session_key)
        if not state["offloads"] and not state["messages"]:
            return ContextCapsule([], [], 0, 0)
        offload_ids = [x["id"] for x in state["offloads"][-self.options.retrieve_top_k :]]
        live_ids = [x["id"] for x in state["messages"][-self.options.keep_recent_messages :]]
        picked_offloads = [x for x in state["offloads"] if x["id"] in offload_ids]
        picked_live = [x for x in state["messages"] if x["id"] in live_ids]
        _ = query
        return ContextCapsule(
            offloaded_context=[sanitize_for_prompt(x["summary"], 1000) for x in picked_offloads],
            live_conversation=[f"[{x['role']}] {sanitize_for_prompt(x['content'], 1000)}" for x in picked_live],
            used_offloads=len(picked_offloads),
            live_messages=len(picked_live),
        )


async def summarize_chunk(chunk: list[dict[str, Any]], summarize_offload: OffloadSummarizer | None) -> dict[str, Any]:
    if summarize_offload:
        model_result = await summarize_offload(
            {
                "messages": [{"role": x["role"], "content": x["content"]} for x in chunk],
            }
        )
        summary = sanitize_for_prompt(str(model_result.get("summary", "")).strip(), 1400)
        keywords = list(model_result.get("keywords") or [])[:24]
        if not keywords:
            keywords = extract_keywords(summary)
    else:
        joined = " | ".join(
            [f"{'U' if x['role'] == 'user' else 'A'}: {sanitize_for_prompt(str(x['content']), 280)}" for x in chunk]
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
