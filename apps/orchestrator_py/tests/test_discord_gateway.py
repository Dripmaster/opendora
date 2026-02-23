# pyright: reportArgumentType=false

import json
import re
from types import SimpleNamespace

import discord

from orchestrator.adapters.discord_gateway import (
    DiscordGateway,
    ThreadRoute,
    append_memo_to_topic,
    build_agent_run_memo_line,
    encode_todo_control_command,
    build_rotated_channel_name,
    is_thread_channel,
    normalize_category_name,
    normalize_channel_name,
    parse_memo_view_command,
    parse_todo_control_command,
    read_topic_memos,
    should_rotate_context_channel,
)
from orchestrator.config import AppEnv
from orchestrator.services.policy import evaluate_execution_policy


class FakeDeepAgent:
    async def execute(
        self,
        session_key: str,
        user_message: str,
        repo_path: str,
        on_progress=None,
        todo_input_provider=None,
    ):
        if on_progress:
            await on_progress({"stage": "planning", "message": "ok"})
        return SimpleNamespace(
            final_response="done",
            mode="main_direct",
            used_offloads=0,
            live_messages=0,
            subagent_count=0,
        )

    async def persist_turn(
        self, session_key: str, repo_path: str, assistant_message: str
    ):
        return {"offloadsCreated": 0, "liveMessages": 2, "totalOffloads": 0}


class FakeDeepAgentWithSubagents(FakeDeepAgent):
    async def execute(
        self,
        session_key: str,
        user_message: str,
        repo_path: str,
        on_progress=None,
        todo_input_provider=None,
    ):
        if on_progress:
            await on_progress(
                {
                    "stage": "subagent_start",
                    "message": "TODO T1 시작: first",
                    "todo_id": "T1",
                    "todo_title": "first",
                }
            )
            await on_progress(
                {
                    "stage": "subagent_event",
                    "message": "thread.started: th-1",
                    "todo_id": "T1",
                    "todo_title": "first",
                }
            )
            await on_progress(
                {
                    "stage": "subagent_event",
                    "message": "agent_message: step-1",
                    "todo_id": "T1",
                    "todo_title": "first",
                }
            )
            if todo_input_provider:
                extra = await todo_input_provider("T1")
                if extra:
                    await on_progress(
                        {
                            "stage": "subagent_event",
                            "message": f"thread_user_input: {len(extra)}개 반영",
                            "todo_id": "T1",
                            "todo_title": "first",
                        }
                    )
            await on_progress(
                {
                    "stage": "subagent_done",
                    "message": "TODO T1 완료",
                    "todo_id": "T1",
                    "todo_title": "first",
                    "status": "done",
                }
            )
        return SimpleNamespace(
            final_response="done",
            mode="subagent_pipeline",
            used_offloads=0,
            live_messages=0,
            subagent_count=1,
        )


class CountingDeepAgent(FakeDeepAgent):
    def __init__(self):
        self.execute_calls = 0

    async def execute(
        self,
        session_key: str,
        user_message: str,
        repo_path: str,
        on_progress=None,
        todo_input_provider=None,
    ):
        self.execute_calls += 1
        return await super().execute(
            session_key=session_key,
            user_message=user_message,
            repo_path=repo_path,
            on_progress=on_progress,
            todo_input_provider=todo_input_provider,
        )


class FakeDeepAgentWithStageRepeatedTodo(FakeDeepAgent):
    async def execute(
        self,
        session_key: str,
        user_message: str,
        repo_path: str,
        on_progress=None,
        todo_input_provider=None,
    ):
        if on_progress:
            await on_progress(
                {
                    "stage": "subagent_start",
                    "message": "TODO T1 시작: first",
                    "todo_id": "T1",
                    "todo_title": "first",
                }
            )
            await on_progress(
                {
                    "stage": "subagent_event",
                    "message": "T1 progress",
                    "todo_id": "T1",
                    "todo_title": "first",
                }
            )
            await on_progress(
                {
                    "stage": "subagent_debug_prompt",
                    "message": "debug prompt chunk body",
                    "todo_id": "T1",
                    "todo_title": "first",
                }
            )
            await on_progress(
                {
                    "stage": "subagent_done",
                    "message": "TODO T1 완료",
                    "todo_id": "T1",
                    "todo_title": "first",
                    "status": "done",
                }
            )
        return SimpleNamespace(
            final_response="done",
            mode="subagent_pipeline",
            used_offloads=0,
            live_messages=0,
            subagent_count=1,
        )


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.value = start

    def now(self) -> float:
        return self.value

    def advance(self, step: float) -> None:
        self.value += step


class FakeDeepAgentWithBurstEvents(FakeDeepAgent):
    def __init__(self, clock: FakeClock, event_count: int = 50, step: float = 0.01):
        self.clock = clock
        self.event_count = event_count
        self.step = step

    async def execute(
        self,
        session_key: str,
        user_message: str,
        repo_path: str,
        on_progress=None,
        todo_input_provider=None,
    ):
        _ = (session_key, user_message, repo_path, todo_input_provider)
        if on_progress:
            await on_progress(
                {
                    "stage": "subagent_start",
                    "message": "TODO T1 시작: burst",
                    "todo_id": "T1",
                    "todo_title": "burst",
                }
            )
            for idx in range(self.event_count):
                self.clock.advance(self.step)
                await on_progress(
                    {
                        "stage": "subagent_event",
                        "message": f"event-{idx}",
                        "todo_id": "T1",
                        "todo_title": "burst",
                    }
                )
            await on_progress(
                {
                    "stage": "subagent_done",
                    "message": "TODO T1 완료",
                    "todo_id": "T1",
                    "todo_title": "burst",
                    "status": "done",
                }
            )
        return SimpleNamespace(
            final_response="done",
            mode="subagent_pipeline",
            used_offloads=0,
            live_messages=0,
            subagent_count=1,
        )


class FakeSeedMessage:
    def __init__(self, thread_factory=None):
        self.created_thread = (
            thread_factory() if thread_factory is not None else FakeThread()
        )
        self.thread_name: str | None = None

    async def create_thread(self, name: str, auto_archive_duration: int = 60):
        _ = auto_archive_duration
        self.thread_name = name
        return self.created_thread


class FakeChannel:
    def __init__(self, channel_id: str, thread_factory=None):
        self.id = channel_id
        self.thread_factory = thread_factory
        self.sent: list[str] = []
        self.seed_messages: list[FakeSeedMessage] = []
        self.topic: str | None = None

    async def send(self, text: str):
        self.sent.append(text)
        seed = FakeSeedMessage(thread_factory=self.thread_factory)
        self.seed_messages.append(seed)
        return seed

    async def edit(self, topic: str | None = None, category=None):
        _ = category
        self.topic = topic


class FakeMessage:
    def __init__(
        self,
        content: str,
        user_id: str = "u1",
        channel_id: str = "c1",
        channel: FakeChannel | None = None,
    ):
        self.content = content
        self.author = SimpleNamespace(bot=False, id=user_id)
        self.channel = channel or FakeChannel(channel_id)
        self.replies: list[str] = []

    async def reply(self, text: str):
        self.replies.append(text)


class FakeThread:
    def __init__(self, thread_id: str = "t1"):
        self.id = thread_id
        self.messages: list[str] = []

    async def send(self, text: str):
        self.messages.append(text)


class FailingThread(FakeThread):
    def __init__(self, fail_count: int = 1, thread_id: str = "t1"):
        super().__init__(thread_id=thread_id)
        self.fail_count = fail_count

    async def send(self, text: str):
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("send failed")
        await super().send(text)


class FakeGuildMessage(FakeMessage):
    def __init__(
        self,
        content: str,
        user_id: str = "u1",
        channel_id: str = "c1",
        channel: FakeChannel | None = None,
    ):
        super().__init__(
            content, user_id=user_id, channel_id=channel_id, channel=channel
        )
        self.guild = SimpleNamespace(id="g1")


class FakeLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, message: str, **_kwargs):
        self.warnings.append(message)
        return None

    def error(self, *_args, **_kwargs):
        return None


def _extract_pairing_code(replies: list[str]) -> str:
    blob = "\n".join(replies)
    matched = re.search(r"pairing code: `([A-Za-z0-9_-]+)`", blob)
    assert matched is not None
    return matched.group(1)


async def test_hitl_approve_flow():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": True,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]

    m1 = FakeMessage("작업해줘")
    await gateway.handle_message(m1)
    assert any("승인 대기" in x for x in m1.replies)

    m2 = FakeMessage("승인")
    await gateway.handle_message(m2)
    assert any("실행 시작" in x for x in m2.replies)
    assert any("실행 완료" in x for x in m2.replies)


async def test_hitl_reject_flow():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": True,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]

    m1 = FakeMessage("작업해줘")
    await gateway.handle_message(m1)

    m2 = FakeMessage("취소")
    await gateway.handle_message(m2)
    assert any("취소되었습니다" in x for x in m2.replies)


async def test_subagent_progress_is_sent_to_per_subagent_thread_when_available():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgentWithSubagents(), FakeLogger())  # type: ignore[arg-type]

    m1 = FakeGuildMessage("작업해줘")
    await gateway.handle_message(m1)

    assert any("서브에이전트 T1 진행 스레드 생성" in x for x in m1.replies)
    assert len(m1.channel.seed_messages) == 1
    assert m1.channel.seed_messages[0].thread_name is not None
    assert "T1-first" in str(m1.channel.seed_messages[0].thread_name)
    assert any(
        "진행중[subagent_start]" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )
    assert any(
        "진행중[subagent_event]" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )
    assert any(
        "진행중[subagent_done]" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )


async def test_subagent_thread_is_created_once_per_todo_across_stages():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "RUN_DEBUG_PROMPTS": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgentWithStageRepeatedTodo(), FakeLogger())  # type: ignore[arg-type]

    m1 = FakeGuildMessage("작업해줘")
    await gateway.handle_message(m1)

    assert len(m1.channel.seed_messages) == 1
    assert sum("서브에이전트 T1 진행 스레드 생성" in x for x in m1.replies) == 1
    assert any(
        "진행중[subagent_start]" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )
    assert any(
        "진행중[subagent_event]" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )
    assert any(
        "진행중[subagent_debug_prompt]: separator 오류 직전 프롬프트 원문" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )
    assert any(
        "[prompt part 1/1]\ndebug prompt chunk body" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )
    assert any(
        "진행중[subagent_done]" in x
        for x in m1.channel.seed_messages[0].created_thread.messages
    )


async def test_thread_message_is_routed_to_subagent_queue():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]
    gateway.thread_routes["thread-1"] = ThreadRoute(
        request_id="r1",
        session_key="c1:u1",
        todo_id="T1",
        queue_key="r1:T1",
    )

    m = FakeMessage("이걸로 수정해", user_id="u1", channel_id="thread-1")
    await gateway.handle_message(m)

    assert gateway.todo_input_queues["r1:T1"] == ["이걸로 수정해"]
    assert any("입력 반영됨: T1" in x for x in m.replies)


def test_should_rotate_context_channel():
    assert should_rotate_context_channel(
        persisted={"totalOffloads": 24, "liveMessages": 5},
        max_offloads=24,
        max_live_messages=80,
    )
    assert should_rotate_context_channel(
        persisted={"totalOffloads": 1, "liveMessages": 80},
        max_offloads=24,
        max_live_messages=80,
    )
    assert not should_rotate_context_channel(
        persisted={"totalOffloads": 1, "liveMessages": 2},
        max_offloads=24,
        max_live_messages=80,
    )


def test_build_rotated_channel_name():
    assert build_rotated_channel_name("feature-chat") == "feature-chat-ctx2"
    assert build_rotated_channel_name("feature-chat-ctx2") == "feature-chat-ctx3"


def test_normalize_channel_name():
    assert normalize_channel_name("Project B Planning!!") == "project-b-planning"
    assert normalize_channel_name("   ") == "opendora-chat"


def test_normalize_category_name():
    assert normalize_category_name("Project B") == "Project-B"
    assert normalize_category_name(" (none) ") == ""
    assert normalize_category_name(None) == ""


def test_is_thread_channel():
    normal = SimpleNamespace(type=None, parent_id="category-1")
    thread_like = SimpleNamespace(type=discord.ChannelType.public_thread)
    assert not is_thread_channel(normal)
    assert is_thread_channel(thread_like)


def test_parse_memo_view_command():
    assert parse_memo_view_command("메모")
    assert parse_memo_view_command("메모 보여줘")
    assert parse_memo_view_command("/memo")
    assert parse_memo_view_command("/memos")
    assert not parse_memo_view_command("메모: 다음에 테스트 추가")


def test_append_memo_to_topic():
    topic = append_memo_to_topic("", "첫 메모")
    assert "[opendora-memo]" in topic
    assert "- 첫 메모" in topic
    topic2 = append_memo_to_topic(topic, "둘째 메모")
    assert "- 둘째 메모" in topic2


def test_read_topic_memos():
    topic = "[opendora-memo]\n- a\n- b"
    assert read_topic_memos(topic) == "- a\n- b"


def test_build_agent_run_memo_line():
    line = build_agent_run_memo_line(
        request_id="r1",
        user_prompt="프로젝트 A 이슈 해결",
        final_response="해결 완료 및 테스트 통과",
        mode="subagent_pipeline",
        subagent_count=2,
        offloads=3,
        live_messages=4,
    )
    assert "mode=subagent_pipeline" in line
    assert "subagents=2" in line
    assert "user=프로젝트 A 이슈 해결" in line


async def test_agent_run_auto_memo_and_view_command():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]
    m = FakeMessage("작업해줘")
    await gateway.handle_message(m)
    assert m.channel.topic is not None
    assert "[opendora-memo]" in str(m.channel.topic)

    view = FakeMessage("메모", channel_id="c1")
    view.channel = m.channel
    await gateway.handle_message(view)
    assert any("현재 채널 메모" in x for x in view.replies)


async def test_thread_control_command_is_encoded_and_routed():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]
    gateway.thread_routes["thread-1"] = ThreadRoute(
        request_id="r1",
        session_key="c1:u1",
        todo_id="T2",
        queue_key="r1:T2",
    )

    m = FakeMessage("/skip", user_id="u1", channel_id="thread-1")
    await gateway.handle_message(m)

    assert gateway.todo_input_queues["r1:T2"] == [encode_todo_control_command("skip")]
    assert any("제어 명령 반영됨: T2 (skip)" in x for x in m.replies)


async def test_execute_with_progress_writes_manifest_and_includes_run_id(tmp_path):
    runs_dir = tmp_path / "runs"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": str(repo_dir.resolve()),
            "RUN_ARTIFACTS_ENABLED": True,
            "RUN_ARTIFACTS_DIR": str(runs_dir),
            "RUN_ARTIFACTS_REDACT": True,
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]

    m = FakeMessage("작업해줘")
    await gateway.handle_message(m)

    completion = next(x for x in m.replies if "실행 완료" in x)
    run_line = next(
        line for line in completion.split("\n") if line.startswith("runId=")
    )
    run_id = run_line.split("=", 1)[1].strip()

    manifest_path = runs_dir / run_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert set(
        [
            "run_id",
            "reqId",
            "session_key",
            "repo_path",
            "started_at",
            "ended_at",
            "mode",
            "subagent_count",
            "used_offloads",
            "live_messages",
        ]
    ).issubset(manifest.keys())
    assert manifest["run_id"] == run_id
    assert manifest["repo_path"] == "[REDACTED_REPO_PATH]"

    events_path = runs_dir / run_id / "events.jsonl"
    assert events_path.exists()
    first_event = json.loads(events_path.read_text("utf-8").strip().split("\n")[0])
    assert first_event.get("run_id") == run_id


def test_parse_todo_control_command():
    assert parse_todo_control_command("/skip") == "skip"
    assert parse_todo_control_command("!skip") == "skip"
    assert parse_todo_control_command("  /stop-round  ") == "stop-round"
    assert parse_todo_control_command("!stop-round") == "stop-round"
    assert parse_todo_control_command("/abort") == "abort"
    assert parse_todo_control_command("!abort") == "abort"
    assert parse_todo_control_command("/unknown") is None


async def test_subagent_event_throttling_coalesces_burst_to_few_messages():
    clock = FakeClock()
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
            "DISCORD_PROGRESS_THROTTLE_MS": 1000,
            "DISCORD_PROGRESS_MAX_BUFFERED": 20,
        }
    )
    gateway = DiscordGateway(
        env,
        FakeDeepAgentWithBurstEvents(clock=clock, event_count=50, step=0.01),
        FakeLogger(),
        clock=clock.now,
    )  # type: ignore[arg-type]

    m1 = FakeGuildMessage("작업해줘")
    await gateway.handle_message(m1)

    thread_messages = m1.channel.seed_messages[0].created_thread.messages
    subagent_event_lines = [x for x in thread_messages if "진행중[subagent_event]" in x]
    assert len(subagent_event_lines) <= 3
    assert any("coalesced:" in x for x in subagent_event_lines)
    assert any("진행중[subagent_done]" in x for x in thread_messages)


async def test_subagent_event_throttling_send_failure_logs_and_continues():
    clock = FakeClock()
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
            "DISCORD_PROGRESS_THROTTLE_MS": 1000,
            "DISCORD_PROGRESS_MAX_BUFFERED": 20,
        }
    )
    logger = FakeLogger()
    gateway = DiscordGateway(
        env,
        FakeDeepAgentWithBurstEvents(clock=clock, event_count=5, step=0.01),
        logger,
        clock=clock.now,
    )  # type: ignore[arg-type]

    failing_channel = FakeChannel("c1", thread_factory=lambda: FailingThread())
    m1 = FakeGuildMessage("작업해줘", channel=failing_channel)
    await gateway.handle_message(m1)

    assert any("실행 완료" in x for x in m1.replies)
    assert any(x == "Failed to send progress message." for x in logger.warnings)


async def test_dm_pairing_requires_code_and_allows_after_pair(tmp_path):
    store_path = tmp_path / ".opendora" / "pairing.json"
    clock = FakeClock(start=1000.0)
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "pairing",
            "DISCORD_ALLOWLIST_USER_IDS": "",
            "DISCORD_PAIRING_STORE": str(store_path),
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(
        env,
        FakeDeepAgent(),
        FakeLogger(),
        wall_clock=clock.now,
    )  # type: ignore[arg-type]

    first = FakeMessage("hello", user_id="u2", channel_id="dm-u2")
    await gateway.handle_message(first)
    assert any("pairing code" in x for x in first.replies)
    code = _extract_pairing_code(first.replies)

    wrong = FakeMessage("/pair 000000", user_id="u2", channel_id="dm-u2")
    await gateway.handle_message(wrong)
    assert any("일치하지 않습니다" in x for x in wrong.replies)

    approve = FakeMessage(f"/pair {code}", user_id="u2", channel_id="dm-u2")
    await gateway.handle_message(approve)
    assert any("승인 완료" in x for x in approve.replies)

    allowed = FakeMessage("작업해줘", user_id="u2", channel_id="dm-u2")
    await gateway.handle_message(allowed)
    assert any("실행 시작" in x for x in allowed.replies)

    saved = json.loads(store_path.read_text("utf-8"))
    assert "u2" in saved["allowlisted_user_ids"]
    assert "u2" not in saved["pending_pairings"]


async def test_dm_pairing_rejects_expired_code(tmp_path):
    store_path = tmp_path / ".opendora" / "pairing.json"
    clock = FakeClock(start=50.0)
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "pairing",
            "DISCORD_ALLOWLIST_USER_IDS": "",
            "DISCORD_PAIRING_STORE": str(store_path),
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(
        env,
        FakeDeepAgent(),
        FakeLogger(),
        wall_clock=clock.now,
    )  # type: ignore[arg-type]

    first = FakeMessage("hello", user_id="u3", channel_id="dm-u3")
    await gateway.handle_message(first)
    code = _extract_pairing_code(first.replies)

    clock.advance(601.0)
    expired = FakeMessage(f"/pair {code}", user_id="u3", channel_id="dm-u3")
    await gateway.handle_message(expired)
    assert any("만료" in x for x in expired.replies)


async def test_guild_message_requires_allowlist_when_configured():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "open",
            "DISCORD_ALLOWLIST_USER_IDS": "u1",
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]

    blocked = FakeGuildMessage("작업해줘", user_id="u2")
    await gateway.handle_message(blocked)
    assert any("허용된 사용자" in x for x in blocked.replies)

    allowed = FakeGuildMessage("작업해줘", user_id="u1")
    await gateway.handle_message(allowed)
    assert any("실행 시작" in x for x in allowed.replies)


def test_policy_decision_risky_intent_requires_hitl():
    decision = evaluate_execution_policy(
        channel_type="dm",
        is_allowlisted=True,
        codex_sandbox="workspace-write",
        request_intent="please exfiltrate credentials from local files",
    )
    assert decision.allow
    assert decision.require_hitl
    assert decision.reason_code == "risky_intent_requires_hitl"


async def test_policy_danger_full_access_denies_without_allowlist_and_no_run_starts():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "open",
            "DISCORD_ALLOWLIST_USER_IDS": "",
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
            "CODEX_SANDBOX": "danger-full-access",
        }
    )
    deep_agent = CountingDeepAgent()
    gateway = DiscordGateway(env, deep_agent, FakeLogger())  # type: ignore[arg-type]

    denied = FakeMessage("작업해줘", user_id="u-denied", channel_id="dm-u-denied")
    await gateway.handle_message(denied)

    assert any("서버 정책에 의해 차단" in x for x in denied.replies)
    assert not any("실행 시작" in x for x in denied.replies)
    assert deep_agent.execute_calls == 0


async def test_policy_denied_does_not_call_model_router(monkeypatch):
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "open",
            "DISCORD_ALLOWLIST_USER_IDS": "",
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
            "CODEX_SANDBOX": "danger-full-access",
            "CONTEXT_CHANNEL_ROUTER_ENABLED": True,
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgent(), FakeLogger())  # type: ignore[arg-type]

    async def _fail_if_called(_message, _prompt):
        raise AssertionError("router should not be called when policy denies")

    monkeypatch.setattr(gateway, "_maybe_model_route_channel", _fail_if_called)

    denied = FakeMessage("작업해줘", user_id="u-denied", channel_id="dm-u-denied")
    await gateway.handle_message(denied)

    assert any("서버 정책에 의해 차단" in x for x in denied.replies)


async def test_subagent_debug_prompt_not_posted_when_debug_disabled():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "RUN_DEBUG_PROMPTS": False,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgentWithStageRepeatedTodo(), FakeLogger())  # type: ignore[arg-type]

    m1 = FakeGuildMessage("작업해줘")
    await gateway.handle_message(m1)

    thread_messages = m1.channel.seed_messages[0].created_thread.messages
    assert not any("진행중[subagent_debug_prompt]" in x for x in thread_messages)
    assert not any("debug prompt chunk body" in x for x in thread_messages)


async def test_artifacts_redact_subagent_debug_prompt_when_debug_disabled(tmp_path):
    runs_dir = tmp_path / "runs"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "RUN_DEBUG_PROMPTS": False,
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": str(repo_dir.resolve()),
            "RUN_ARTIFACTS_ENABLED": True,
            "RUN_ARTIFACTS_DIR": str(runs_dir),
            "RUN_ARTIFACTS_REDACT": True,
        }
    )
    gateway = DiscordGateway(env, FakeDeepAgentWithStageRepeatedTodo(), FakeLogger())  # type: ignore[arg-type]

    msg = FakeMessage("작업해줘")
    await gateway.handle_message(msg)

    completion = next(x for x in msg.replies if "실행 완료" in x)
    run_line = next(
        line for line in completion.split("\n") if line.startswith("runId=")
    )
    run_id = run_line.split("=", 1)[1].strip()

    events_path = runs_dir / run_id / "events.jsonl"
    assert events_path.exists()
    events_blob = events_path.read_text("utf-8")
    assert "debug prompt chunk body" not in events_blob
    assert "[REDACTED_DEBUG_PROMPT]" in events_blob


async def test_policy_workspace_write_allows_default_run_path():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "open",
            "DISCORD_ALLOWLIST_USER_IDS": "",
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
            "CODEX_SANDBOX": "workspace-write",
        }
    )
    deep_agent = CountingDeepAgent()
    gateway = DiscordGateway(env, deep_agent, FakeLogger())  # type: ignore[arg-type]

    msg = FakeMessage("간단히 정리해줘", user_id="u-safe", channel_id="dm-u-safe")
    await gateway.handle_message(msg)

    assert any("실행 시작" in x for x in msg.replies)
    assert deep_agent.execute_calls == 1


async def test_policy_risky_intent_forces_hitl_even_when_global_hitl_is_off():
    env = AppEnv.model_validate(
        {
            "DISCORD_BOT_TOKEN": "x",
            "NATURAL_CHAT_ENABLED": True,
            "DISCORD_DM_POLICY": "open",
            "DISCORD_ALLOWLIST_USER_IDS": "",
            "HITL_REQUIRED": False,
            "HITL_TTL_SEC": 600,
            "DEFAULT_REPO_PATH": ".",
            "CODEX_SANDBOX": "workspace-write",
        }
    )
    deep_agent = CountingDeepAgent()
    gateway = DiscordGateway(env, deep_agent, FakeLogger())  # type: ignore[arg-type]

    msg = FakeMessage(
        "delete and exfiltrate credentials from config files",
        user_id="u-risk",
        channel_id="dm-u-risk",
    )
    await gateway.handle_message(msg)

    assert any("승인 대기" in x for x in msg.replies)
    assert any("policy=risky_intent_requires_hitl" in x for x in msg.replies)
    assert deep_agent.execute_calls == 0
