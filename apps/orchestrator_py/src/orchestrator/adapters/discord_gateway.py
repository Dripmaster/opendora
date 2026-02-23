from __future__ import annotations

import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import discord

from orchestrator.config import AppEnv
from orchestrator.services.deep_agent import DeepAgentService
from orchestrator.services.policy import evaluate_execution_policy
from orchestrator.services.run_artifacts import ArtifactWriter, RunContext


PAIRING_CODE_TTL_SEC = 600
PAIR_COMMAND_PATTERN = re.compile(r"^/pair\s+([A-Za-z0-9_-]+)$", re.IGNORECASE)
REDACTED_DEBUG_PROMPT_PLACEHOLDER = "[REDACTED_DEBUG_PROMPT]"


@dataclass(slots=True)
class PendingApproval:
    id: str
    user_id: str
    channel_id: str
    prompt: str
    created_at: float


@dataclass(slots=True)
class ThreadRoute:
    request_id: str
    session_key: str
    todo_id: str
    queue_key: str


@dataclass(slots=True)
class _ThrottleBucket:
    last_sent_at: float | None = None
    buffered: list[str] | None = None
    dropped: int = 0


class ProgressThrottler:
    def __init__(
        self,
        interval_ms: int,
        max_buffered: int,
        now_fn: Any,
    ):
        self.interval_s = max(interval_ms, 0) / 1000.0
        self.max_buffered = max(max_buffered, 1)
        self.now_fn = now_fn
        self._buckets: dict[str, _ThrottleBucket] = {}

    def push(self, key: str, text: str) -> str | None:
        now = float(self.now_fn())
        bucket = self._buckets.setdefault(key, _ThrottleBucket(buffered=[]))
        if bucket.last_sent_at is None or now - bucket.last_sent_at >= self.interval_s:
            out = self._build_buffered_message(bucket, text)
            bucket.last_sent_at = now
            bucket.buffered = []
            bucket.dropped = 0
            return out

        assert bucket.buffered is not None
        bucket.buffered.append(text)
        if len(bucket.buffered) > self.max_buffered:
            bucket.buffered.pop(0)
            bucket.dropped += 1
        return None

    def flush(self, key: str) -> str | None:
        bucket = self._buckets.get(key)
        if bucket is None:
            return None
        text = self._build_buffered_message(bucket, None)
        self._buckets.pop(key, None)
        return text

    def _build_buffered_message(
        self, bucket: _ThrottleBucket, latest_text: str | None
    ) -> str | None:
        buffered = list(bucket.buffered or [])
        if latest_text is not None:
            buffered.append(latest_text)
        if not buffered:
            return None
        if len(buffered) == 1 and bucket.dropped == 0:
            return buffered[0]

        tail = buffered[-1]
        collapsed = len(buffered) - 1
        suffix_parts: list[str] = []
        if collapsed > 0:
            suffix_parts.append(f"{collapsed} buffered")
        if bucket.dropped > 0:
            suffix_parts.append(f"{bucket.dropped} dropped")
        return f"{tail} (coalesced: {', '.join(suffix_parts)})"


class RoutedExecutionMessage:
    def __init__(self, original: Any, target_channel: Any):
        self.author = getattr(original, "author", None)
        self.guild = getattr(original, "guild", None)
        self.content = getattr(original, "content", "")
        self.channel = target_channel

    async def reply(self, text: str):
        if hasattr(self.channel, "send"):
            return await self.channel.send(text)
        if hasattr(self.channel, "reply"):
            return await self.channel.reply(text)
        return None


class DiscordGateway:
    def __init__(
        self,
        env: AppEnv,
        deep_agent: DeepAgentService,
        logger: Any,
        clock: Any | None = None,
        wall_clock: Any | None = None,
    ):
        self.env = env
        self.deep_agent = deep_agent
        self.logger = logger
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.pending: dict[str, PendingApproval] = {}
        self.running: set[str] = set()
        self.thread_routes: dict[str, ThreadRoute] = {}
        self.todo_input_queues: dict[str, list[str]] = {}
        self._env_allowlist_user_ids = parse_allowlist_user_ids(
            self.env.DISCORD_ALLOWLIST_USER_IDS
        )
        self._pairing_store_path = resolve_pairing_store_path(
            self.env.DISCORD_PAIRING_STORE
        )

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.dm_messages = True

        self.client = discord.Client(intents=intents)
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

    async def start(self) -> None:
        if not self.env.DISCORD_BOT_TOKEN:
            self.logger.warning(
                "DISCORD_BOT_TOKEN is not set. Discord gateway disabled."
            )
            return
        await self.client.start(self.env.DISCORD_BOT_TOKEN)

    async def stop(self) -> None:
        if not self.client.is_closed():
            await self.client.close()

    async def on_ready(self) -> None:
        self.logger.info("Discord client ready.", user=str(self.client.user))

    async def on_message(self, message: discord.Message) -> None:
        await self.handle_message(message)

    async def handle_message(self, message: Any) -> None:
        try:
            if not self.env.NATURAL_CHAT_ENABLED:
                return
            if getattr(message.author, "bot", False):
                return
            user_id = str(getattr(getattr(message, "author", None), "id", ""))

            if self._is_guild_message(message) and self._is_guild_user_blocked(user_id):
                await message.reply(
                    "허용된 사용자만 사용할 수 있습니다. 관리자에게 allowlist 등록을 요청하세요."
                )
                return

            channel_id = str(getattr(getattr(message, "channel", None), "id", ""))
            thread_route = self.thread_routes.get(channel_id)
            if thread_route:
                prompt = sanitize_incoming_content(
                    str(getattr(message, "content", "")),
                    getattr(self.client.user, "id", None),
                )
                if not prompt:
                    return
                queue = self.todo_input_queues.setdefault(thread_route.queue_key, [])
                control = parse_todo_control_command(prompt)
                if control:
                    queue.append(encode_todo_control_command(control))
                    await message.reply(
                        f"제어 명령 반영됨: {thread_route.todo_id} ({control})"
                    )
                    return
                queue.append(prompt)
                await message.reply(
                    f"입력 반영됨: {thread_route.todo_id} (대기 입력 {len(queue)}개)"
                )
                return

            if parse_memo_view_command(str(getattr(message, "content", ""))):
                memo_text = read_topic_memos(
                    str(getattr(getattr(message, "channel", None), "topic", "") or "")
                )
                if memo_text:
                    await message.reply(
                        "현재 채널 메모:\n" + truncate_for_discord(memo_text, 1800)
                    )
                else:
                    await message.reply("저장된 메모가 없습니다.")
                return

            prompt = sanitize_incoming_content(
                str(getattr(message, "content", "")),
                getattr(self.client.user, "id", None),
            )
            if not prompt:
                return

            if self._is_dm_message(message):
                pair_code = parse_pair_command(prompt)
                if pair_code is not None:
                    await self._handle_pair_command(message, user_id, pair_code)
                    return
                if (
                    self.env.DISCORD_DM_POLICY == "pairing"
                    and not self._is_user_allowlisted(user_id)
                ):
                    await self._reply_pairing_required(message, user_id)
                    return

            policy_decision = evaluate_execution_policy(
                channel_type="dm" if self._is_dm_message(message) else "guild",
                is_allowlisted=self._is_user_allowlisted(user_id),
                codex_sandbox=self.env.CODEX_SANDBOX,
                request_intent=prompt,
            )
            if not policy_decision.allow:
                await message.reply(
                    "\n".join(
                        [
                            "요청이 서버 정책에 의해 차단되었습니다.",
                            f"reason={policy_decision.reason_code}",
                            "CODEX_SANDBOX=danger-full-access 에서는 allowlist 사용자 + HITL 승인 조건이 필요합니다.",
                        ]
                    )
                )
                return

            message = await self._maybe_model_route_channel(message, prompt)
            key = get_session_key(str(message.channel.id), str(message.author.id))
            now = float(self.wall_clock())
            existing = self.pending.get(key)
            if existing and now - existing.created_at > self.env.HITL_TTL_SEC:
                self.pending.pop(key, None)

            if key in self.running:
                await message.reply(
                    "현재 실행 중입니다. 완료 후 다음 요청을 보내주세요."
                )
                return

            decision = parse_decision(prompt)
            if decision and key in self.pending:
                pending = self.pending[key]
                if decision == "reject":
                    self.pending.pop(key, None)
                    await message.reply(f"요청이 취소되었습니다. reqId={pending.id}")
                    return
                self.pending.pop(key, None)
                await self.execute_with_progress(message, key, pending)
                return

            force_hitl = bool(policy_decision.require_hitl)
            effective_hitl_required = bool(self.env.HITL_REQUIRED or force_hitl)
            if effective_hitl_required:
                if key in self.pending:
                    pending = self.pending[key]
                    await message.reply(
                        "\n".join(
                            [
                                f"이미 승인 대기 중인 요청이 있습니다. reqId={pending.id}",
                                "답장으로 `승인` 또는 `취소`를 보내주세요.",
                            ]
                        )
                    )
                    return
                request = create_pending_request(
                    str(message.author.id), str(message.channel.id), prompt
                )
                self.pending[key] = request
                await message.reply(
                    "\n".join(
                        [
                            f"요청 접수 (승인 대기). reqId={request.id}",
                            f"repo={self._display_repo_path(self.env.DEFAULT_REPO_PATH)}",
                            f"sandbox={self.env.CODEX_SANDBOX}",
                            f"prompt={truncate_for_discord(request.prompt, 240)}",
                            (
                                f"policy={policy_decision.reason_code}"
                                if force_hitl
                                else ""
                            ),
                            "",
                            "실행하려면 `승인`, 취소하려면 `취소`라고 답장하세요.",
                        ]
                    )
                )
                return

            await self.execute_with_progress(
                message,
                key,
                create_pending_request(
                    str(message.author.id), str(message.channel.id), prompt
                ),
            )
        except Exception as exc:
            self.logger.error("Failed to run remote Codex request.", error=str(exc))
            await message.reply(
                f"실행 중 오류가 발생했습니다: {truncate_for_discord(str(exc), 300)}"
            )

    def _is_dm_message(self, message: Any) -> bool:
        return getattr(message, "guild", None) is None

    def _is_guild_message(self, message: Any) -> bool:
        return getattr(message, "guild", None) is not None

    def _is_guild_user_blocked(self, user_id: str) -> bool:
        effective = self._effective_allowlist_user_ids()
        if not effective:
            return False
        return user_id not in effective

    def _is_user_allowlisted(self, user_id: str) -> bool:
        return user_id in self._effective_allowlist_user_ids()

    def _effective_allowlist_user_ids(self) -> set[str]:
        state = self._load_pairing_state()
        stored = {
            str(x).strip()
            for x in state.get("allowlisted_user_ids", [])
            if str(x).strip()
        }
        return set(self._env_allowlist_user_ids).union(stored)

    async def _reply_pairing_required(self, message: Any, user_id: str) -> None:
        state = self._load_pairing_state()
        now = float(self.wall_clock())
        changed = self._cleanup_expired_pairings(state, now)
        pending = state.get("pending_pairings", {}).get(user_id)
        code = ""
        expires_at = now + float(PAIRING_CODE_TTL_SEC)
        if pending:
            code = str(pending.get("code", "")).strip()
            expires_at = float(pending.get("expires_at", expires_at) or expires_at)
        if not code:
            code = self._generate_pairing_code()
            pending = {
                "code": code,
                "expires_at": expires_at,
                "expires_at_iso": datetime.fromtimestamp(
                    expires_at, tz=timezone.utc
                ).isoformat(),
                "created_at": now,
            }
            state.setdefault("pending_pairings", {})[user_id] = pending
            changed = True
        if changed:
            self._save_pairing_state(state)
        ttl_sec = max(0, int(expires_at - now))
        await message.reply(
            "\n".join(
                [
                    "보안 모드(pairing)가 활성화되어 있습니다.",
                    f"pairing code: `{code}`",
                    f"DM에서 `/pair {code}` 를 보내면 승인됩니다. (유효 {ttl_sec}초)",
                ]
            )
        )

    async def _handle_pair_command(self, message: Any, user_id: str, code: str) -> None:
        if not self._is_dm_message(message):
            await message.reply("`/pair` 명령은 DM에서만 사용할 수 있습니다.")
            return
        if self._is_user_allowlisted(user_id):
            await message.reply("이미 허용된 사용자입니다.")
            return

        state = self._load_pairing_state()
        now = float(self.wall_clock())
        pending_pairings = state.setdefault("pending_pairings", {})
        pending = pending_pairings.get(user_id)
        if not pending:
            if self._cleanup_expired_pairings(state, now):
                self._save_pairing_state(state)
            await message.reply(
                "유효한 pairing 요청이 없습니다. 먼저 일반 DM을 보내 code를 받으세요."
            )
            return

        expected = str(pending.get("code", "")).strip()
        expires_at = float(pending.get("expires_at", 0.0) or 0.0)
        if now > expires_at:
            pending_pairings.pop(user_id, None)
            self._save_pairing_state(state)
            await message.reply(
                "pairing code가 만료되었습니다. 다시 DM을 보내 새 code를 받으세요."
            )
            return
        if code.strip() != expected:
            await message.reply(
                "pairing code가 일치하지 않습니다. `/pair <code>` 형식으로 다시 시도하세요."
            )
            return

        allowlisted_user_ids = {
            str(x).strip()
            for x in state.get("allowlisted_user_ids", [])
            if str(x).strip()
        }
        allowlisted_user_ids.add(user_id)
        state["allowlisted_user_ids"] = sorted(allowlisted_user_ids)
        pending_pairings.pop(user_id, None)
        self._save_pairing_state(state)
        await message.reply("pairing 승인 완료: allowlist에 추가되었습니다.")

    def _generate_pairing_code(self) -> str:
        return f"{secrets.randbelow(900000) + 100000}"

    def _load_pairing_state(self) -> dict[str, Any]:
        path = self._pairing_store_path
        if not path.exists():
            return {"allowlisted_user_ids": [], "pending_pairings": {}}
        try:
            raw = json.loads(path.read_text("utf-8"))
            if not isinstance(raw, dict):
                return {"allowlisted_user_ids": [], "pending_pairings": {}}
            allowlisted = raw.get("allowlisted_user_ids", [])
            pending = raw.get("pending_pairings", {})
            if not isinstance(allowlisted, list):
                allowlisted = []
            if not isinstance(pending, dict):
                pending = {}
            return {
                "allowlisted_user_ids": allowlisted,
                "pending_pairings": pending,
            }
        except Exception as exc:
            self.logger.warning("Failed to load pairing store.", error=str(exc))
            return {"allowlisted_user_ids": [], "pending_pairings": {}}

    def _save_pairing_state(self, state: dict[str, Any]) -> None:
        path = self._pairing_store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(payload, "utf-8")
        os.replace(tmp_path, path)

    def _cleanup_expired_pairings(self, state: dict[str, Any], now: float) -> bool:
        pending = state.get("pending_pairings", {})
        if not isinstance(pending, dict):
            state["pending_pairings"] = {}
            return True
        stale_user_ids = [
            user_id
            for user_id, item in pending.items()
            if now > float((item or {}).get("expires_at", 0.0) or 0.0)
        ]
        if not stale_user_ids:
            return False
        for user_id in stale_user_ids:
            pending.pop(user_id, None)
        return True

    async def execute_with_progress(
        self, message: Any, key: str, request: PendingApproval
    ) -> None:
        self.running.add(key)
        started_at = time.time()
        started_at_dt = datetime.now(tz=timezone.utc)
        run_context = RunContext(
            run_id=build_run_id(request.id),
            request_id=request.id,
            session_key=key,
            repo_path=self.env.DEFAULT_REPO_PATH,
            started_at=started_at_dt,
            artifacts_dir=Path(self.env.RUN_ARTIFACTS_DIR),
            debug_enabled=self.env.RUN_DEBUG_PROMPTS,
        )
        artifact_writer: ArtifactWriter | None = None
        if self.env.RUN_ARTIFACTS_ENABLED:
            try:
                artifact_writer = ArtifactWriter(
                    run_context,
                    max_bytes=self.env.RUN_ARTIFACTS_MAX_BYTES,
                    retention_days=self.env.RUN_ARTIFACTS_RETENTION_DAYS,
                    redact=self.env.RUN_ARTIFACTS_REDACT,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to initialize run artifact writer.", error=str(exc)
                )

        run_thread_ids: list[str] = []
        run_queue_keys: set[str] = set()
        run_mode = ""
        run_subagent_count = 0
        run_used_offloads = 0
        run_live_messages = 0
        try:
            await message.reply(f"실행 시작. reqId={request.id}")
            subagent_threads: dict[str, Any] = {}
            subagent_thread_refs: dict[str, str] = {}
            progress_throttler = ProgressThrottler(
                interval_ms=self.env.DISCORD_PROGRESS_THROTTLE_MS,
                max_buffered=self.env.DISCORD_PROGRESS_MAX_BUFFERED,
                now_fn=self.clock,
            )

            async def ensure_subagent_thread(todo_id: str, todo_title: str) -> Any:
                thread_target = subagent_threads.get(todo_id)
                if thread_target is not None:
                    return thread_target

                (
                    thread_target,
                    thread_ref,
                    thread_id,
                ) = await self._create_subagent_thread(
                    message,
                    request.id,
                    todo_id,
                    todo_title,
                )
                subagent_threads[todo_id] = thread_target

                queue_key = self._queue_key(request.id, todo_id)
                run_queue_keys.add(queue_key)
                self.todo_input_queues.setdefault(queue_key, [])

                if thread_id:
                    self.thread_routes[str(thread_id)] = ThreadRoute(
                        request_id=request.id,
                        session_key=key,
                        todo_id=todo_id,
                        queue_key=queue_key,
                    )
                    run_thread_ids.append(str(thread_id))

                if thread_ref:
                    subagent_thread_refs[todo_id] = thread_ref
                    await message.reply(
                        f"서브에이전트 {todo_id} 진행 스레드 생성: {thread_ref}"
                    )

                return thread_target

            async def progress(event: dict[str, Any]) -> None:
                payload = dict(event)
                payload["run_id"] = run_context.run_id
                stage = str(payload.get("stage", "unknown"))
                if stage == "subagent_debug_prompt" and not self.env.RUN_DEBUG_PROMPTS:
                    payload["message"] = REDACTED_DEBUG_PROMPT_PLACEHOLDER
                if artifact_writer is not None:
                    try:
                        artifact_writer.append_event(payload)
                    except Exception as exc:
                        self.logger.warning(
                            "Failed to append run progress event.", error=str(exc)
                        )
                todo_id = str(payload.get("todo_id", "")).strip()
                todo_title = str(payload.get("todo_title", "")).strip()
                raw_message = str(payload.get("message", ""))
                if (
                    "duration_ms" in payload
                    and "prompt_chars" in payload
                    and not raw_message.strip()
                ):
                    return
                if stage == "subagent_debug_prompt":
                    if not self.env.RUN_DEBUG_PROMPTS:
                        return
                    chunks = split_for_discord(raw_message, max_len=1700)
                    target = message
                    if todo_id:
                        target = await ensure_subagent_thread(todo_id, todo_title)
                    await self._send_message(
                        target,
                        "진행중[subagent_debug_prompt]: separator 오류 직전 프롬프트 원문",
                    )
                    for idx, chunk in enumerate(chunks, start=1):
                        await self._send_message(
                            target, f"[prompt part {idx}/{len(chunks)}]\n{chunk}"
                        )
                    return
                text = f"진행중[{stage}]: {truncate_for_discord(raw_message, 300)}"
                is_subagent_error = (
                    todo_id and stage.startswith("subagent") and ("error" in stage)
                )
                if (
                    stage in {"subagent_start", "subagent_done", "subagent_event"}
                    or is_subagent_error
                ) and todo_id:
                    target = await ensure_subagent_thread(todo_id, todo_title)
                    throttle_key = self._queue_key(request.id, todo_id)
                    if stage == "subagent_event":
                        throttled = progress_throttler.push(throttle_key, text)
                        if throttled:
                            await self._send_message(target, throttled)
                        return
                    if stage == "subagent_done" or is_subagent_error:
                        flushed = progress_throttler.flush(throttle_key)
                        if flushed:
                            await self._send_message(target, flushed)
                    await self._send_message(target, text)
                    if stage == "subagent_done":
                        self._clear_todo_thread_route(request.id, todo_id)
                    return
                await self._send_message(message, text)

            result = await self.deep_agent.execute(
                session_key=key,
                user_message=request.prompt,
                repo_path=self.env.DEFAULT_REPO_PATH,
                on_progress=progress,
                todo_input_provider=lambda todo_id: self._consume_todo_inputs(
                    request.id, todo_id
                ),
            )
            run_mode = str(getattr(result, "mode", "") or "")
            run_subagent_count = int(getattr(result, "subagent_count", 0) or 0)
            run_used_offloads = int(getattr(result, "used_offloads", 0) or 0)
            run_live_messages = int(getattr(result, "live_messages", 0) or 0)
            await message.reply(
                truncate_for_discord(result.final_response or "(empty response)", 1900)
            )
            persisted = await self.deep_agent.persist_turn(
                session_key=key,
                repo_path=self.env.DEFAULT_REPO_PATH,
                assistant_message=result.final_response,
            )
            elapsed_ms = int((time.time() - started_at) * 1000)
            await message.reply(
                "\n".join(
                    [
                        "실행 완료",
                        f"reqId={request.id}",
                        f"runId={run_context.run_id}",
                        f"mode={result.mode}",
                        f"subagents={result.subagent_count}",
                        f"contextLive={result.live_messages}",
                        f"contextOffloaded={result.used_offloads}",
                        f"durationMs={elapsed_ms}",
                        f"offloadsCreated={persisted['offloadsCreated']}",
                        f"liveMessages={persisted['liveMessages']}",
                        f"totalOffloads={persisted['totalOffloads']}",
                    ]
                )
            )
            await self._append_agent_run_memo(
                message=message,
                request_id=request.id,
                user_prompt=request.prompt,
                result=result,
                persisted=persisted,
            )
            if subagent_thread_refs:
                await message.reply(
                    "서브에이전트 스레드 요약:\n"
                    + "\n".join(
                        [
                            f"- {todo_id}: {ref}"
                            for todo_id, ref in subagent_thread_refs.items()
                        ]
                    )
                )
            await self._maybe_rotate_context_channel(message, request.id, persisted)
        finally:
            if artifact_writer is not None:
                try:
                    artifact_writer.write_manifest(
                        {
                            "run_id": run_context.run_id,
                            "reqId": request.id,
                            "session_key": key,
                            "repo_path": self.env.DEFAULT_REPO_PATH,
                            "started_at": started_at_dt.isoformat(),
                            "ended_at": datetime.now(tz=timezone.utc).isoformat(),
                            "mode": run_mode,
                            "subagent_count": run_subagent_count,
                            "used_offloads": run_used_offloads,
                            "live_messages": run_live_messages,
                        }
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to write run artifact manifest.", error=str(exc)
                    )
            for tid in run_thread_ids:
                self.thread_routes.pop(tid, None)
            for qk in run_queue_keys:
                self.todo_input_queues.pop(qk, None)
            self.running.discard(key)

    def _display_repo_path(self, repo_path: str) -> str:
        if self.env.RUN_ARTIFACTS_REDACT and Path(repo_path).is_absolute():
            return "[REDACTED_REPO_PATH]"
        return repo_path

    async def _create_subagent_thread(
        self,
        message: Any,
        request_id: str,
        todo_id: str,
        todo_title: str,
    ) -> tuple[Any, str | None, str | None]:
        if not getattr(message, "guild", None):
            return message, None, None
        channel = getattr(message, "channel", None)
        if channel is None or not hasattr(channel, "send"):
            return message, None, None
        try:
            seed = await channel.send(
                f"서브에이전트 {todo_id} 시작: {truncate_for_discord(todo_title or 'task', 80)}"
            )
            if not hasattr(seed, "create_thread"):
                return message, None, None
            thread_name = build_subagent_thread_name(
                request_id=request_id, todo_id=todo_id, todo_title=todo_title
            )
            thread = await seed.create_thread(
                name=thread_name, auto_archive_duration=60
            )
            thread_id = getattr(thread, "id", None)
            if thread_id is None:
                return thread, None, None
            return thread, f"<#{thread_id}>", str(thread_id)
        except Exception as exc:
            self.logger.warning(
                "Failed to create subagent thread; falling back to channel reply.",
                error=str(exc),
            )
            return message, None, None

    async def _send_message(self, target: Any, text: str) -> None:
        try:
            if hasattr(target, "send"):
                await target.send(text)
                return
            await target.reply(text)
        except Exception as exc:
            self.logger.warning("Failed to send progress message.", error=str(exc))

    async def _consume_todo_inputs(self, request_id: str, todo_id: str) -> list[str]:
        qk = self._queue_key(request_id, todo_id)
        items = self.todo_input_queues.get(qk, [])
        self.todo_input_queues[qk] = []
        return [x for x in items if x.strip()]

    def _clear_todo_thread_route(self, request_id: str, todo_id: str) -> None:
        qk = self._queue_key(request_id, todo_id)
        stale = [
            tid
            for tid, route in self.thread_routes.items()
            if route.request_id == request_id and route.queue_key == qk
        ]
        for tid in stale:
            self.thread_routes.pop(tid, None)

    def _queue_key(self, request_id: str, todo_id: str) -> str:
        return f"{request_id}:{todo_id}"

    async def _append_channel_topic_memo(self, message: Any, memo_text: str) -> bool:
        channel = getattr(message, "channel", None)
        if channel is None or not hasattr(channel, "edit"):
            return False
        current = str(getattr(channel, "topic", "") or "")
        updated = append_memo_to_topic(current, memo_text)
        try:
            await channel.edit(topic=updated)
            return True
        except Exception as exc:
            self.logger.warning("Failed to update channel topic memo.", error=str(exc))
            return False

    async def _append_agent_run_memo(
        self,
        message: Any,
        request_id: str,
        user_prompt: str,
        result: Any,
        persisted: dict[str, int],
    ) -> None:
        memo = build_agent_run_memo_line(
            request_id=request_id,
            user_prompt=user_prompt,
            final_response=str(getattr(result, "final_response", "") or ""),
            mode=str(getattr(result, "mode", "")),
            subagent_count=int(getattr(result, "subagent_count", 0) or 0),
            offloads=int(persisted.get("totalOffloads", 0) or 0),
            live_messages=int(persisted.get("liveMessages", 0) or 0),
        )
        await self._append_channel_topic_memo(message, memo)

    async def _maybe_rotate_context_channel(
        self, message: Any, request_id: str, persisted: dict[str, int]
    ) -> None:
        if not self.env.CONTEXT_CHANNEL_ROTATION_ENABLED:
            return
        if not should_rotate_context_channel(
            persisted=persisted,
            max_offloads=self.env.CONTEXT_CHANNEL_ROTATION_MAX_OFFLOADS,
            max_live_messages=self.env.CONTEXT_CHANNEL_ROTATION_MAX_LIVE_MESSAGES,
        ):
            return
        if not getattr(message, "guild", None):
            return
        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)
        if guild is None or channel is None:
            return
        if not hasattr(guild, "create_text_channel"):
            return
        if not hasattr(channel, "name"):
            return
        try:
            active_category = await self._get_or_create_category(
                guild, self.env.CONTEXT_ACTIVE_CATEGORY_NAME
            )
            archive_category = await self._get_or_create_category(
                guild, self.env.CONTEXT_ARCHIVE_CATEGORY_NAME
            )
            new_name = build_rotated_channel_name(str(channel.name))
            topic = (
                f"Auto-rotated from #{channel.name} "
                f"(reqId={request_id}, offloads={persisted.get('totalOffloads', 0)}, live={persisted.get('liveMessages', 0)})"
            )
            new_channel = await guild.create_text_channel(
                name=new_name,
                category=active_category or getattr(channel, "category", None),
                topic=topic[:1000],
            )
            if hasattr(new_channel, "send"):
                await new_channel.send(
                    "\n".join(
                        [
                            f"이 채널은 컨텍스트 분리를 위해 자동 생성되었습니다. (from: <#{getattr(channel, 'id', '')}>)",
                            f"reqId={request_id}",
                            "이후 대화는 여기서 계속 진행하세요.",
                        ]
                    )
                )
            await message.reply(
                "\n".join(
                    [
                        "컨텍스트 길이 누적으로 채널을 분리했습니다.",
                        f"새 채널: <#{getattr(new_channel, 'id', '')}>",
                        f"기준: offloads={persisted.get('totalOffloads', 0)}, liveMessages={persisted.get('liveMessages', 0)}",
                    ]
                )
            )
            if archive_category and hasattr(channel, "edit"):
                await channel.edit(category=archive_category)
                if hasattr(channel, "send"):
                    await channel.send(
                        f"이 채널은 컨텍스트 아카이브로 이동했습니다. 새 대화는 <#{getattr(new_channel, 'id', '')}> 에서 진행하세요."
                    )
        except Exception as exc:
            self.logger.warning("Failed to rotate context channel.", error=str(exc))

    async def _maybe_model_route_channel(self, message: Any, prompt: str) -> Any:
        if not self.env.CONTEXT_CHANNEL_ROUTER_ENABLED:
            return message
        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)
        author = getattr(message, "author", None)
        if guild is None or channel is None or author is None:
            return message
        if is_thread_channel(channel):
            return message
        if str(getattr(channel, "id", "")) in self.thread_routes:
            return message
        if not hasattr(self.deep_agent, "tools") or not hasattr(
            self.deep_agent, "context_offload"
        ):
            return message
        try:
            session_key = get_session_key(str(channel.id), str(author.id))
            capsule = await self.deep_agent.context_offload.build_capsule(
                session_key=session_key,
                query=prompt,
                select_context=lambda input_data: (
                    self.deep_agent.tools.invoke_select_context(  # type: ignore[attr-defined]
                        {
                            "repoPath": self.env.DEFAULT_REPO_PATH,
                            "taskInstructions": prompt,
                            **input_data,
                        }
                    )
                ),
            )
            decision = await self.deep_agent.tools.invoke_channel_routing(  # type: ignore[attr-defined]
                {
                    "repoPath": self.env.DEFAULT_REPO_PATH,
                    "userMessage": prompt,
                    "currentChannelName": str(getattr(channel, "name", "channel")),
                    "currentCategoryName": str(
                        getattr(getattr(channel, "category", None), "name", "")
                    )
                    or None,
                    "offloadedContext": capsule.offloaded_context,
                    "liveConversation": capsule.live_conversation,
                }
            )
            await message.reply(
                "\n".join(
                    [
                        f"[router] action={decision.get('action', 'stay')}",
                        f"[router] reason={truncate_for_discord(str(decision.get('reason', '')), 200)}",
                        f"[router] newChannel={truncate_for_discord(str(decision.get('newChannelName', '') or '(none)'), 80)}",
                        f"[router] targetCategory={truncate_for_discord(str(decision.get('targetCategoryName', '') or '(none)'), 80)}",
                    ]
                )
            )
            if decision.get("action") != "split":
                return message

            desired_name = str(decision.get("newChannelName", "")).strip()
            target_category_name = normalize_category_name(
                decision.get("targetCategoryName")
            )
            channel_name = normalize_channel_name(desired_name)
            if not channel_name or not target_category_name:
                await message.reply(
                    "[router] split rejected: 모델이 newChannelName/targetCategoryName을 유효하게 채우지 않았습니다."
                )
                return message
            target_category = await self._get_or_create_category(
                guild, target_category_name
            )
            if not hasattr(guild, "create_text_channel"):
                return message
            new_channel = await guild.create_text_channel(
                name=channel_name,
                category=target_category or getattr(channel, "category", None),
                topic=truncate_for_discord(
                    f"Auto-routed from #{getattr(channel, 'name', 'channel')}: {decision.get('reason', '')}",
                    1000,
                ),
            )
            user_id = str(getattr(author, "id", ""))
            if hasattr(new_channel, "send"):
                await new_channel.send(
                    "\n".join(
                        [
                            f"<@{user_id}> 새 주제로 자동 분리된 채널입니다.",
                            f"이관 사유: {decision.get('reason', '')}",
                            f"원 채널: <#{getattr(channel, 'id', '')}>",
                            "요청을 이 채널에서 바로 이어서 실행합니다.",
                        ]
                    )
                )
            await message.reply(
                f"주제/프로젝트 전환으로 새 채널로 이관합니다: <#{getattr(new_channel, 'id', '')}>"
            )
            await message.reply(
                f"[router] created channel={channel_name} category={target_category_name}"
            )
            return RoutedExecutionMessage(message, new_channel)
        except Exception as exc:
            self.logger.warning(
                "Model-based channel routing failed; keeping current channel.",
                error=str(exc),
                channel_id=str(getattr(channel, "id", "")),
                channel_name=str(getattr(channel, "name", "")),
            )
            return message

    async def _get_or_create_category(self, guild: Any, name: str) -> Any | None:
        cat_name = (name or "").strip()
        if not cat_name:
            return None
        try:
            categories = list(getattr(guild, "categories", []) or [])
            for cat in categories:
                if str(getattr(cat, "name", "")).strip().lower() == cat_name.lower():
                    return cat
            if hasattr(guild, "create_category"):
                return await guild.create_category(cat_name)
        except Exception as exc:
            self.logger.warning(
                "Failed to create/find category.", error=str(exc), category=cat_name
            )
        return None


def sanitize_incoming_content(content: str, bot_id: int | None) -> str:
    stripped = content
    if bot_id is not None:
        stripped = re.sub(rf"<@!?{bot_id}>", "", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def parse_allowlist_user_ids(raw: str) -> set[str]:
    return {x.strip() for x in str(raw or "").split(",") if x.strip()}


def parse_pair_command(content: str) -> str | None:
    normalized = re.sub(r"\s+", " ", content).strip()
    matched = PAIR_COMMAND_PATTERN.match(normalized)
    if not matched:
        return None
    return matched.group(1).strip()


def resolve_pairing_store_path(raw_path: str) -> Path:
    fallback = (Path.cwd() / ".opendora" / "pairing.json").resolve()
    candidate = Path(str(raw_path or "").strip() or ".opendora/pairing.json")
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if ".opendora" not in candidate.parts:
        return fallback
    return candidate


TODO_CONTROL_COMMANDS = {
    "/skip": "skip",
    "!skip": "skip",
    "/stop-round": "stop-round",
    "!stop-round": "stop-round",
    "/abort": "abort",
    "!abort": "abort",
}


def parse_todo_control_command(content: str) -> str | None:
    normalized = re.sub(r"\s+", " ", content).strip().lower()
    return TODO_CONTROL_COMMANDS.get(normalized)


def encode_todo_control_command(command: str) -> str:
    return f"__control__:{command}"


def truncate_for_discord(input_text: str, max_len: int) -> str:
    return (
        input_text if len(input_text) <= max_len else input_text[: max_len - 1] + "..."
    )


def create_pending_request(
    user_id: str, channel_id: str, prompt: str
) -> PendingApproval:
    suffix = hex(int(time.time() * 1000000))[-6:]
    return PendingApproval(
        id=f"{int(time.time()):x}-{suffix}",
        user_id=user_id,
        channel_id=channel_id,
        prompt=prompt,
        created_at=time.time(),
    )


def build_run_id(request_id: str) -> str:
    return f"{request_id}-{hex(int(time.time() * 1000000))[-6:]}"


def get_session_key(channel_id: str, user_id: str) -> str:
    return f"{channel_id}:{user_id}"


def parse_decision(text: str) -> str | None:
    if re.match(r"^(승인|approve|yes|y|ok)$", text, re.IGNORECASE):
        return "approve"
    if re.match(r"^(취소|거절|reject|no|n|cancel)$", text, re.IGNORECASE):
        return "reject"
    return None


def build_subagent_thread_name(request_id: str, todo_id: str, todo_title: str) -> str:
    normalized = re.sub(r"\s+", "-", todo_title.strip())
    normalized = re.sub(r"[^0-9A-Za-z가-힣_-]", "", normalized)
    normalized = normalized[:50] if normalized else "task"
    return f"{request_id}-{todo_id}-{normalized}"[:90]


def should_rotate_context_channel(
    persisted: dict[str, int],
    max_offloads: int,
    max_live_messages: int,
) -> bool:
    return int(persisted.get("totalOffloads", 0)) >= int(max_offloads) or int(
        persisted.get("liveMessages", 0)
    ) >= int(max_live_messages)


def build_rotated_channel_name(current_name: str) -> str:
    clean = current_name.strip().lower() or "opendora"
    matched = re.match(r"^(.*)-ctx(\d+)$", clean)
    if matched:
        base = matched.group(1)
        idx = int(matched.group(2)) + 1
        return f"{base}-ctx{idx}"[:100]
    return f"{clean}-ctx2"[:100]


def normalize_channel_name(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9a-z_-]", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return (text or "opendora-chat")[:100]


def split_for_discord(text: str, max_len: int = 1700) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        parts.append(text[cursor : cursor + max_len])
        cursor += max_len
    return parts


def normalize_category_name(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "(none)", "n/a", "na"}:
        return ""
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9A-Za-z가-힣_-]", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:100]


def is_thread_channel(channel: Any) -> bool:
    if isinstance(channel, discord.Thread):
        return True
    channel_type = getattr(channel, "type", None)
    thread_types = {
        discord.ChannelType.public_thread,
        discord.ChannelType.private_thread,
        discord.ChannelType.news_thread,
    }
    return channel_type in thread_types


def parse_memo_view_command(content: str) -> bool:
    raw = content.strip()
    if not raw:
        return False
    return bool(re.match(r"^(/memos?|메모(\s*보여줘)?)$", raw, re.IGNORECASE))


def append_memo_to_topic(
    current_topic: str, memo_text: str, max_len: int = 1024
) -> str:
    prefix = "[opendora-memo]"
    lines = [x.strip() for x in (current_topic or "").split("\n") if x.strip()]
    if not lines or lines[0] != prefix:
        lines = [prefix]
    memo_line = f"- {memo_text}"
    lines.append(memo_line)
    while len("\n".join(lines)) > max_len and len(lines) > 2:
        lines.pop(1)
    topic = "\n".join(lines)
    if len(topic) > max_len:
        topic = topic[: max_len - 1] + "…"
    return topic


def read_topic_memos(current_topic: str) -> str:
    lines = [x.rstrip() for x in (current_topic or "").split("\n")]
    if not lines or lines[0].strip() != "[opendora-memo]":
        return ""
    body = [x for x in lines[1:] if x.strip()]
    return "\n".join(body)


def build_agent_run_memo_line(
    request_id: str,
    user_prompt: str,
    final_response: str,
    mode: str,
    subagent_count: int,
    offloads: int,
    live_messages: int,
) -> str:
    prompt_head = truncate_for_discord(re.sub(r"\s+", " ", user_prompt).strip(), 70)
    result_head = truncate_for_discord(re.sub(r"\s+", " ", final_response).strip(), 90)
    return (
        f"[{request_id}] mode={mode} subagents={subagent_count} offloads={offloads} live={live_messages} "
        f"| user={prompt_head} | result={result_head}"
    )
