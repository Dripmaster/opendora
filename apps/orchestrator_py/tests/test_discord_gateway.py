from types import SimpleNamespace

import discord

from orchestrator.adapters.discord_gateway import (
    DiscordGateway,
    ThreadRoute,
    append_memo_to_topic,
    build_agent_run_memo_line,
    build_rotated_channel_name,
    is_thread_channel,
    normalize_category_name,
    normalize_channel_name,
    parse_memo_view_command,
    read_topic_memos,
    should_rotate_context_channel,
)
from orchestrator.config import AppEnv


class FakeDeepAgent:
    async def execute(self, session_key: str, user_message: str, repo_path: str, on_progress=None, todo_input_provider=None):
        if on_progress:
            await on_progress({"stage": "planning", "message": "ok"})
        return SimpleNamespace(
            final_response="done",
            mode="main_direct",
            used_offloads=0,
            live_messages=0,
            subagent_count=0,
        )

    async def persist_turn(self, session_key: str, repo_path: str, user_message: str, assistant_message: str):
        return {"offloadsCreated": 0, "liveMessages": 2, "totalOffloads": 0}


class FakeDeepAgentWithSubagents(FakeDeepAgent):
    async def execute(self, session_key: str, user_message: str, repo_path: str, on_progress=None, todo_input_provider=None):
        if on_progress:
            await on_progress({"stage": "subagent_start", "message": "TODO T1 시작: first", "todo_id": "T1", "todo_title": "first"})
            await on_progress({"stage": "subagent_event", "message": "thread.started: th-1", "todo_id": "T1", "todo_title": "first"})
            await on_progress({"stage": "subagent_event", "message": "agent_message: step-1", "todo_id": "T1", "todo_title": "first"})
            if todo_input_provider:
                extra = await todo_input_provider("T1")
                if extra:
                    await on_progress({"stage": "subagent_event", "message": f"thread_user_input: {len(extra)}개 반영", "todo_id": "T1", "todo_title": "first"})
            await on_progress({"stage": "subagent_done", "message": "TODO T1 완료", "todo_id": "T1", "todo_title": "first", "status": "done"})
        return SimpleNamespace(
            final_response="done",
            mode="subagent_pipeline",
            used_offloads=0,
            live_messages=0,
            subagent_count=1,
        )


class FakeSeedMessage:
    def __init__(self):
        self.created_thread = FakeThread()
        self.thread_name: str | None = None

    async def create_thread(self, name: str, auto_archive_duration: int = 60):
        _ = auto_archive_duration
        self.thread_name = name
        return self.created_thread


class FakeChannel:
    def __init__(self, channel_id: str):
        self.id = channel_id
        self.sent: list[str] = []
        self.seed_messages: list[FakeSeedMessage] = []
        self.topic: str | None = None

    async def send(self, text: str):
        self.sent.append(text)
        seed = FakeSeedMessage()
        self.seed_messages.append(seed)
        return seed

    async def edit(self, topic: str | None = None, category=None):
        _ = category
        self.topic = topic


class FakeMessage:
    def __init__(self, content: str, user_id: str = "u1", channel_id: str = "c1"):
        self.content = content
        self.author = SimpleNamespace(bot=False, id=user_id)
        self.channel = FakeChannel(channel_id)
        self.replies: list[str] = []

    async def reply(self, text: str):
        self.replies.append(text)


class FakeThread:
    def __init__(self, thread_id: str = "t1"):
        self.id = thread_id
        self.messages: list[str] = []

    async def send(self, text: str):
        self.messages.append(text)


class FakeGuildMessage(FakeMessage):
    def __init__(self, content: str, user_id: str = "u1", channel_id: str = "c1"):
        super().__init__(content, user_id=user_id, channel_id=channel_id)
        self.guild = SimpleNamespace(id="g1")


class FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


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
    assert any("진행중[subagent_start]" in x for x in m1.channel.seed_messages[0].created_thread.messages)
    assert any("진행중[subagent_event]" in x for x in m1.channel.seed_messages[0].created_thread.messages)
    assert any("진행중[subagent_done]" in x for x in m1.channel.seed_messages[0].created_thread.messages)


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
