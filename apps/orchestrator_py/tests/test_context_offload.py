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

    await service.persist_turn(
        session_key="ch:u",
        user_message="a" * 200,
        assistant_message="b" * 200,
    )
    stats = await service.persist_turn(
        session_key="ch:u",
        user_message="c" * 200,
        assistant_message="d" * 200,
    )

    assert stats.offloads_created >= 1
    assert stats.total_offloads >= 1

    capsule = await service.build_capsule("ch:u", "test")
    assert capsule.used_offloads >= 1
