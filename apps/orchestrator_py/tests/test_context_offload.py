# pyright: reportMissingImports=false

import json
import os
from pathlib import Path

from orchestrator.services.context_offload import (  # pyright: ignore[reportMissingImports]
    ContextOffloadOptions,
    ContextOffloadService,
    session_user_id,
)


async def test_context_offload_compacts_old_messages(tmp_path: Path) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=2,
            retrieve_top_k=2,
        )
    )

    await service.persist_user_request(session_key="ch:u", user_message="a" * 200)
    await service.persist_turn(session_key="ch:u", assistant_message="b" * 200)
    await service.persist_user_request(session_key="ch:u", user_message="c" * 200)
    stats = await service.persist_turn(session_key="ch:u", assistant_message="d" * 200)

    assert stats.total_offloads >= 1

    capsule = await service.build_capsule("ch:u", "test")
    assert capsule.used_offloads >= 1


def test_build_capsule_from_session_prefers_relevant_items_over_recency(
    tmp_path: Path,
) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=1,
            retrieve_top_k=1,
        )
    )

    service._save_session(
        "ch:target-user",
        {
            "sessionKey": "ch:target-user",
            "updatedAt": "2026-01-01T00:00:00+00:00",
            "offloads": [
                {
                    "id": "old-relevant",
                    "summary": "Root cause analysis for payment timeout incident and DB lock contention",
                    "createdAt": "2025-12-30T00:00:00+00:00",
                },
                {
                    "id": "new-irrelevant",
                    "summary": "Team lunch menu and office event announcement",
                    "createdAt": "2026-01-01T00:00:00+00:00",
                },
            ],
            "messages": [
                {
                    "id": "old-msg-relevant",
                    "role": "assistant",
                    "content": "Investigate payment timeout caused by transaction lock contention",
                    "createdAt": "2025-12-30T00:00:00+00:00",
                },
                {
                    "id": "new-msg-irrelevant",
                    "role": "assistant",
                    "content": "By the way, the office snack restock is complete",
                    "createdAt": "2026-01-01T00:00:00+00:00",
                },
            ],
        },
    )

    capsule = service.build_capsule_from_session(
        "ch:target-user", "payment timeout lock"
    )

    assert capsule.used_offloads == 1
    assert "payment timeout incident" in capsule.offloaded_context[0]
    assert capsule.live_messages == 1
    assert "payment timeout caused by transaction lock" in capsule.live_conversation[0]


def test_list_related_session_candidates_prefers_query_relevant_sessions(
    tmp_path: Path,
) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=2,
            retrieve_top_k=2,
        )
    )

    service._save_session(
        "main:user-1",
        {
            "sessionKey": "main:user-1",
            "updatedAt": "2026-01-01T12:00:00+00:00",
            "offloads": [],
            "messages": [],
        },
    )
    service._save_session(
        "recent:user-1",
        {
            "sessionKey": "recent:user-1",
            "updatedAt": "2026-01-03T12:00:00+00:00",
            "offloads": [],
            "messages": [
                {
                    "id": "r1",
                    "role": "assistant",
                    "content": "Weekly social event planning and lunch vote",
                    "createdAt": "2026-01-03T11:00:00+00:00",
                }
            ],
        },
    )
    service._save_session(
        "older:user-1",
        {
            "sessionKey": "older:user-1",
            "updatedAt": "2026-01-02T12:00:00+00:00",
            "offloads": [
                {
                    "id": "o1",
                    "summary": "Incident report: payment timeout due to lock contention in orders DB",
                    "createdAt": "2026-01-02T10:00:00+00:00",
                }
            ],
            "messages": [],
        },
    )

    candidates = service.list_related_session_candidates(
        current_session_key="main:user-1",
        query="payment timeout lock contention",
        limit=2,
    )

    assert len(candidates) == 2
    assert candidates[0].session_key == "older:user-1"


def test_save_session_uses_atomic_replace_with_valid_json(
    tmp_path: Path, monkeypatch
) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=2,
            retrieve_top_k=2,
        )
    )

    session_key = "ch:user"
    expected_path = service._session_file(session_key)
    index_path = Path(service.options.store_dir).resolve() / "sessions_index.json"
    called = {"session": False, "index": False}
    original_replace = os.replace

    def spy_replace(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        source_path = Path(src)
        destination_path = Path(dst)
        assert source_path.suffix == ".tmp"
        if destination_path == expected_path:
            called["session"] = True
            assert source_path.parent == expected_path.parent
            payload = json.loads(source_path.read_text("utf-8"))
            assert payload["sessionKey"] == session_key
        if destination_path == index_path:
            called["index"] = True
            assert source_path == Path(str(index_path) + ".tmp")
            payload = json.loads(source_path.read_text("utf-8"))
            assert isinstance(payload, list)
            assert any(item.get("sessionKey") == session_key for item in payload)

        original_replace(src, dst)

    monkeypatch.setattr("orchestrator.services.context_offload.os.replace", spy_replace)

    service._save_session(
        session_key,
        {
            "sessionKey": session_key,
            "messages": [
                {"id": "1", "role": "user", "content": "hello", "createdAt": "now"}
            ],
            "offloads": [],
            "updatedAt": "now",
        },
    )

    assert called["session"] is True
    assert called["index"] is True
    persisted = json.loads(expected_path.read_text("utf-8"))
    assert persisted["messages"][0]["content"] == "hello"


def test_list_related_session_candidates_session_index_bounds_file_reads(
    tmp_path: Path, monkeypatch
) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=2,
            retrieve_top_k=6,
        )
    )

    service._save_session(
        "main:user-1",
        {
            "sessionKey": "main:user-1",
            "updatedAt": "2026-01-01T00:00:00+00:00",
            "offloads": [],
            "messages": [],
        },
    )
    for i in range(200):
        session_key = f"s{i:03d}:user-1"
        content = "payment timeout lock contention" if i % 7 == 0 else "general chatter"
        service._save_session(
            session_key,
            {
                "sessionKey": session_key,
                "updatedAt": f"2026-01-02T00:{i % 60:02d}:00+00:00",
                "offloads": [],
                "messages": [
                    {
                        "id": f"m-{i}",
                        "role": "assistant",
                        "content": content,
                        "createdAt": "2026-01-02T00:00:00+00:00",
                    }
                ],
            },
        )

    index_path = Path(service.options.store_dir).resolve() / "sessions_index.json"
    sessions_dir = Path(service.options.store_dir).resolve() / "sessions"
    original_read_text = Path.read_text
    read_count = {"index": 0, "session": 0}

    def spy_read_text(self: Path, *args, **kwargs) -> str:
        if self == index_path:
            read_count["index"] += 1
        elif self.parent == sessions_dir and self.suffix == ".json":
            read_count["session"] += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    candidates = service.list_related_session_candidates(
        current_session_key="main:user-1",
        query="payment timeout lock contention",
        limit=5,
    )

    assert len(candidates) == 5
    assert read_count["index"] == 1
    assert read_count["session"] <= service.options.retrieve_top_k


def test_list_related_session_candidates_session_index_prevents_cross_user_leakage(
    tmp_path: Path,
) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=2,
            retrieve_top_k=4,
        )
    )

    service._save_session(
        "main:user-1",
        {
            "sessionKey": "main:user-1",
            "updatedAt": "2026-01-03T12:00:00+00:00",
            "offloads": [],
            "messages": [],
        },
    )
    service._save_session(
        "safe:user-1",
        {
            "sessionKey": "safe:user-1",
            "updatedAt": "2026-01-03T12:01:00+00:00",
            "offloads": [
                {
                    "id": "s1",
                    "summary": "payment timeout recovery checklist",
                    "createdAt": "2026-01-03T12:01:00+00:00",
                }
            ],
            "messages": [],
        },
    )
    service._save_session(
        "leak:user-2",
        {
            "sessionKey": "leak:user-2",
            "updatedAt": "2026-01-03T12:02:00+00:00",
            "offloads": [
                {
                    "id": "l1",
                    "summary": "payment timeout lock contention exact match",
                    "createdAt": "2026-01-03T12:02:00+00:00",
                }
            ],
            "messages": [],
        },
    )

    candidates = service.list_related_session_candidates(
        current_session_key="main:user-1",
        query="payment timeout lock contention",
        limit=5,
    )

    assert candidates
    assert all(session_user_id(item.session_key) == "user-1" for item in candidates)
    assert all(item.session_key != "leak:user-2" for item in candidates)


def test_load_session_uses_cache_and_invalidates_on_save(
    tmp_path: Path, monkeypatch
) -> None:
    service = ContextOffloadService(
        ContextOffloadOptions(
            enabled=True,
            store_dir=str(tmp_path / "ctx"),
            max_estimated_tokens=30,
            keep_recent_messages=2,
            retrieve_top_k=2,
            session_cache_size=8,
        )
    )

    session_key = "ch:user"
    session_path = service._session_file(session_key)
    service._save_session(
        session_key,
        {
            "sessionKey": session_key,
            "messages": [
                {"id": "1", "role": "user", "content": "one", "createdAt": "now"}
            ],
            "offloads": [],
            "updatedAt": "now",
        },
    )

    read_count = {"value": 0}
    original_read_text = Path.read_text

    def spy_read_text(self: Path, *args, **kwargs) -> str:
        if self == session_path:
            read_count["value"] += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    first = service._load_session(session_key)
    second = service._load_session(session_key)

    assert first["messages"][0]["content"] == "one"
    assert second["messages"][0]["content"] == "one"
    assert read_count["value"] == 1

    service._save_session(
        session_key,
        {
            "sessionKey": session_key,
            "messages": [
                {"id": "2", "role": "user", "content": "two", "createdAt": "later"}
            ],
            "offloads": [],
            "updatedAt": "later",
        },
    )

    third = service._load_session(session_key)
    fourth = service._load_session(session_key)

    assert third["messages"][0]["content"] == "two"
    assert fourth["messages"][0]["content"] == "two"
    assert read_count["value"] == 2
