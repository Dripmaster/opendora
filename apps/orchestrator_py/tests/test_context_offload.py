from pathlib import Path

from orchestrator.services.context_offload import ContextOffloadOptions, ContextOffloadService


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


def test_build_capsule_from_session_prefers_relevant_items_over_recency(tmp_path: Path) -> None:
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

    capsule = service.build_capsule_from_session("ch:target-user", "payment timeout lock")

    assert capsule.used_offloads == 1
    assert "payment timeout incident" in capsule.offloaded_context[0]
    assert capsule.live_messages == 1
    assert "payment timeout caused by transaction lock" in capsule.live_conversation[0]


def test_list_related_session_candidates_prefers_query_relevant_sessions(tmp_path: Path) -> None:
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
