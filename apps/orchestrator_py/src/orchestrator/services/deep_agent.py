from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .codex_cli_runtime import CodexCliRuntimeService
from .context_offload import ContextOffloadService
from .deep_agent_tools import DeepAgentToolsService


class DeepAgentState(TypedDict, total=False):
    session_key: str
    repo_path: str
    user_message: str
    offloaded_context: list[str]
    live_conversation: list[str]
    used_offloads: int
    live_messages: int
    external_context_count: int
    filesystem_context: list[str]
    mode: str
    route_reason: str
    todo_list: list[dict[str, Any]]
    todo_results: list[dict[str, Any]]
    latest_todo_results: list[dict[str, Any]]
    subagent_outputs: list[str]
    latest_subagent_outputs: list[str]
    subagent_count: int
    round: int
    done: bool
    completion_reason: str
    final_response: str


@dataclass(slots=True)
class DeepAgentOptions:
    enabled: bool
    max_subagents: int


@dataclass(slots=True)
class DeepAgentResult:
    final_response: str
    mode: str
    used_offloads: int
    live_messages: int
    subagent_count: int


ProgressHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
TodoInputProvider = Callable[[str], Awaitable[list[str]] | list[str]]


@dataclass(slots=True)
class DeepAgentService:
    codex: CodexCliRuntimeService
    context_offload: ContextOffloadService
    tools: DeepAgentToolsService
    options: DeepAgentOptions

    async def execute(
        self,
        session_key: str,
        user_message: str,
        repo_path: str,
        on_progress: ProgressHandler | None = None,
        todo_input_provider: TodoInputProvider | None = None,
    ) -> DeepAgentResult:
        max_rounds = 3
        graph = StateGraph(DeepAgentState)

        async def save_user_request(state: DeepAgentState) -> DeepAgentState:
            """유저 요청 저장 시 오프로드."""
            await self.context_offload.persist_user_request(
                session_key=state["session_key"],
                user_message=state["user_message"],
                summarize_offload=lambda offload: self.tools.invoke_summarize_offload(
                    {"repoPath": state["repo_path"], "messages": offload["messages"]}
                ),
            )
            return {}

        async def prepare_context(state: DeepAgentState) -> DeepAgentState:
            capsule = await self.context_offload.build_capsule(
                session_key=state["session_key"],
                query=state["user_message"],
                select_context=lambda x: self.tools.invoke_select_context(
                    {
                        "repoPath": state["repo_path"],
                        "taskInstructions": state["user_message"],
                        **x,
                    }
                ),
            )
            merged_offloads = list(capsule.offloaded_context)
            external_count = 0
            candidates = self.context_offload.list_related_session_candidates(
                current_session_key=state["session_key"],
                query=state["user_message"],
                limit=12,
            )
            if candidates:
                external_decision = await self.tools.invoke_external_context_routing(
                    {
                        "repoPath": state["repo_path"],
                        "userMessage": state["user_message"],
                        "currentSessionKey": state["session_key"],
                        "offloadedContext": capsule.offloaded_context,
                        "liveConversation": capsule.live_conversation,
                        "candidates": [
                            {
                                "sessionKey": x.session_key,
                                "channelId": x.channel_id,
                                "updatedAt": x.updated_at,
                                "summary": x.summary,
                            }
                            for x in candidates
                        ],
                        "maxSelect": 3,
                    }
                )
                selected = external_decision.get("selectedSessionKeys", []) if external_decision.get("useExternalContext") else []
                for session_key in selected:
                    ext_capsule = self.context_offload.build_capsule_from_session(
                        session_key=session_key,
                        query=state["user_message"],
                    )
                    if ext_capsule.offloaded_context or ext_capsule.live_conversation:
                        merged_offloads.append(
                            "\n".join(
                                [
                                    f"[External Session: {session_key}]",
                                    *ext_capsule.offloaded_context[:2],
                                    *ext_capsule.live_conversation[:2],
                                ]
                            )
                        )
                        external_count += 1
                if external_count > 0 and on_progress:
                    await maybe_await(
                        on_progress(
                            {
                                "stage": "planning",
                                "message": f"외부 채널 컨텍스트 {external_count}개 세션을 추가했습니다. ({external_decision.get('reason', '')})",
                            }
                        )
                    )
            return {
                "offloaded_context": merged_offloads,
                "live_conversation": capsule.live_conversation,
                "used_offloads": capsule.used_offloads,
                "live_messages": capsule.live_messages,
                "external_context_count": external_count,
            }

        async def route(state: DeepAgentState) -> DeepAgentState:
            if not self.options.enabled:
                return {"mode": "main_direct", "route_reason": "deep-agent-disabled"}
            route_result = await self.tools.invoke_route(
                {
                    "repoPath": state["repo_path"],
                    "userMessage": state["user_message"],
                    "offloadedContext": state.get("offloaded_context", []),
                    "liveConversation": state.get("live_conversation", []),
                }
            )
            return {"mode": route_result["mode"], "route_reason": route_result["reason"]}

        async def main_direct(state: DeepAgentState) -> DeepAgentState:
            if on_progress:
                await maybe_await(on_progress({"stage": "main_direct", "message": f"모델 라우팅: main_direct ({state.get('route_reason', '')})"}))
            result = await self.codex.run(
                repo_path=state["repo_path"],
                prompt=build_main_direct_prompt(state),
            )
            return {"final_response": result.assistant_message.strip() or "(empty response)", "subagent_count": 0}

        async def plan(state: DeepAgentState) -> DeepAgentState:
            if on_progress:
                await maybe_await(on_progress({"stage": "planning", "message": f"모델 라우팅: subagent_pipeline ({state.get('route_reason', '')})"}))
            tasks = await self.tools.invoke_plan(
                {
                    "repoPath": state["repo_path"],
                    "userMessage": state["user_message"],
                    "offloadedContext": [
                        *state.get("offloaded_context", []),
                        *state.get("filesystem_context", []),
                    ],
                    "liveConversation": state.get("live_conversation", []),
                    "maxTasks": self.options.max_subagents,
                }
            )
            if on_progress:
                await maybe_await(
                    on_progress(
                        {
                            "stage": "planning",
                            "message": build_todo_list_message(tasks),
                        }
                    )
                )
            return {"todo_list": tasks, "todo_results": [], "subagent_outputs": [], "round": 1}

        async def run_subagents(state: DeepAgentState) -> DeepAgentState:
            # 서브에이전트한테 일 할당 전 오프로드
            await self.context_offload.compact_session(
                session_key=state["session_key"],
                summarize_offload=lambda offload: self.tools.invoke_summarize_offload(
                    {"repoPath": state["repo_path"], "messages": offload["messages"]}
                ),
            )
            tasks = state.get("todo_list", [])
            outputs: list[str] = []
            results: list[dict[str, Any]] = []
            prev_results = state.get("todo_results", [])
            prev_outputs = state.get("subagent_outputs", [])
            for i, task in enumerate(tasks):
                unmet = [
                    dep
                    for dep in task.get("dependsOn", [])
                    if not any(r["todoId"] == dep and r["status"] == "done" for r in [*prev_results, *results])
                ]
                if unmet:
                    blocker = f"의존 TODO 미완료: {', '.join(unmet)}"
                    results.append({"todoId": task["id"], "status": "blocked", "summary": blocker, "blocker": blocker})
                    if on_progress:
                        await maybe_await(
                            on_progress(
                                {
                                    "stage": "subagent_done",
                                    "message": f"TODO {task['id']} blocked ({blocker})",
                                    "todo_id": task["id"],
                                    "todo_title": task.get("title", ""),
                                    "status": "blocked",
                                }
                            )
                        )
                    continue

                if on_progress:
                    await maybe_await(
                        on_progress(
                            {
                                "stage": "subagent_start",
                                "message": f"TODO {task['id']} 시작: {task['title']}",
                                "todo_id": task["id"],
                                "todo_title": task.get("title", ""),
                            }
                        )
                    )

                selected = await self.tools.invoke_select_context(
                    {
                        "repoPath": state["repo_path"],
                        "taskInstructions": task["instructions"],
                        "query": task["instructions"],
                        "offloads": [
                            {"id": f"offload-{idx}", "summary": summary, "createdAt": ""}
                            for idx, summary in enumerate(state.get("offloaded_context", []))
                        ],
                        "liveMessages": [
                            {
                                "id": f"live-{idx}",
                                "role": "assistant" if line.startswith("[assistant]") else "user",
                                "content": line,
                                "createdAt": "",
                            }
                            for idx, line in enumerate(state.get("live_conversation", []))
                        ],
                        "limits": {"offloads": 2, "liveMessages": 6},
                    }
                )
                relevant_offloads = map_selected(state.get("offloaded_context", []), selected["offloadIds"], "offload")
                relevant_live = map_selected(state.get("live_conversation", []), selected["liveMessageIds"], "live")
                thread_inputs: list[str] = []
                if todo_input_provider:
                    consumed = todo_input_provider(task["id"])
                    thread_inputs = await maybe_await(consumed) or []
                    if thread_inputs and on_progress:
                        await maybe_await(
                            on_progress(
                                {
                                    "stage": "subagent_event",
                                    "message": f"thread_user_input: {len(thread_inputs)}개 반영",
                                    "todo_id": task["id"],
                                    "todo_title": task.get("title", ""),
                                }
                            )
                        )

                worker_prompt = build_subagent_prompt(
                    i + 1,
                    len(tasks),
                    task,
                    relevant_offloads,
                    relevant_live,
                    thread_inputs,
                    state["user_message"],
                )
                try:
                    worker_result = await self.codex.run_streaming(
                        repo_path=state["repo_path"],
                        prompt=worker_prompt,
                        on_event=(
                            lambda raw_event, task_id=task["id"], task_title=task.get("title", ""): (
                                maybe_await(
                                    on_progress(
                                        {
                                            "stage": "subagent_event",
                                            "message": formatted,
                                            "todo_id": task_id,
                                            "todo_title": task_title,
                                        }
                                    )
                                )
                                if on_progress and (formatted := format_subagent_event(raw_event))
                                else None
                            )
                        ),
                    )
                except Exception as exc:
                    if on_progress and is_separator_chunk_error(exc):
                        await maybe_await(
                            on_progress(
                                {
                                    "stage": "subagent_debug_prompt",
                                    "message": worker_prompt,
                                    "todo_id": task["id"],
                                    "todo_title": task.get("title", ""),
                                }
                            )
                        )
                    raise
                summary = worker_result.assistant_message.strip() or "(empty)"
                outputs.append(f"[{task['id']}] {summary}")
                results.append({"todoId": task["id"], "status": "done", "summary": summary})
                if on_progress:
                    await maybe_await(
                        on_progress(
                            {
                                "stage": "subagent_done",
                                "message": f"TODO {task['id']} 완료",
                                "todo_id": task["id"],
                                "todo_title": task.get("title", ""),
                                "status": "done",
                            }
                        )
                    )

            # 서브에이전트 결과 후 오프로드
            await self.context_offload.compact_session(
                session_key=state["session_key"],
                summarize_offload=lambda offload: self.tools.invoke_summarize_offload(
                    {"repoPath": state["repo_path"], "messages": offload["messages"]}
                ),
            )
            all_results = [*prev_results, *results]
            all_outputs = [*prev_outputs, *outputs]
            return {
                "todo_results": all_results,
                "latest_todo_results": results,
                "subagent_outputs": all_outputs,
                "latest_subagent_outputs": outputs,
                "subagent_count": sum(1 for r in all_results if r["status"] == "done"),
            }

        async def review_and_replan(state: DeepAgentState) -> DeepAgentState:
            current_round = state.get("round", 1)
            if current_round >= max_rounds:
                if on_progress:
                    await maybe_await(
                        on_progress(
                            {
                                "stage": "planning",
                                "message": f"최대 라운드({max_rounds}) 도달. 현재 결과를 최종 집계합니다.",
                            }
                        )
                    )
                return {"done": True, "completion_reason": f"max-rounds-reached:{max_rounds}"}

            decision = await self.tools.invoke_replan(
                {
                    "repoPath": state["repo_path"],
                    "userMessage": state["user_message"],
                    "offloadedContext": state.get("offloaded_context", []),
                    "liveConversation": state.get("live_conversation", []),
                    "currentTodoList": state.get("todo_list", []),
                    "allTodoResults": state.get("todo_results", []),
                    "latestTodoResults": state.get("latest_todo_results", []),
                    "allOutputs": state.get("subagent_outputs", []),
                    "currentRound": current_round,
                    "maxRounds": max_rounds,
                    "maxTasks": self.options.max_subagents,
                }
            )
            if decision.get("done"):
                if on_progress:
                    await maybe_await(
                        on_progress(
                            {
                                "stage": "planning",
                                "message": f"품질 게이트 완료 판단: {decision.get('reason', 'complete')}",
                            }
                        )
                    )
                return {"done": True, "completion_reason": decision.get("reason", "complete")}

            next_todos = decision.get("nextTodos", [])
            if not next_todos:
                return {"done": True, "completion_reason": "no-followup-todos"}

            if on_progress:
                await maybe_await(
                    on_progress(
                        {
                            "stage": "planning",
                            "message": f"추가 TODO {len(next_todos)}개 생성. 라운드 {current_round + 1} 진행.",
                        }
                    )
                )
            return {
                "done": False,
                "completion_reason": decision.get("reason", "needs-more-work"),
                "todo_list": next_todos,
                "round": current_round + 1,
            }

        async def aggregate(state: DeepAgentState) -> DeepAgentState:
            if on_progress:
                await maybe_await(on_progress({"stage": "aggregation", "message": "메인 에이전트가 서브에이전트 결과를 집계합니다."}))
            aggregated = await self.codex.run(
                repo_path=state["repo_path"],
                prompt=build_aggregation_prompt(
                    state["user_message"],
                    state.get("todo_list", []),
                    state.get("todo_results", []),
                    state.get("subagent_outputs", []),
                    state.get("completion_reason", ""),
                ),
            )
            return {"final_response": aggregated.assistant_message.strip() or "(empty response)"}

        async def filesystem_tool_node(state: DeepAgentState) -> DeepAgentState:
            summaries = build_warpgrep_filesystem_context(
                repo_path=state["repo_path"],
                user_message=state["user_message"],
            )
            if summaries and on_progress:
                await maybe_await(
                    on_progress(
                        {
                            "stage": "planning",
                            "message": f"warpgrep 파일시스템 컨텍스트 {len(summaries)}개를 추가했습니다.",
                        }
                    )
                )
            return {"filesystem_context": summaries}


        graph.add_node("save_user_request", save_user_request)
        graph.add_node("prepare_context", prepare_context)
        graph.add_node("route", route)
        graph.add_node("filesystem_tool", filesystem_tool_node)
        graph.add_node("main_direct", main_direct)
        graph.add_node("plan", plan)
        graph.add_node("run_subagents", run_subagents)
        graph.add_node("review_and_replan", review_and_replan)
        graph.add_node("aggregate", aggregate)

        graph.add_edge(START, "save_user_request")
        graph.add_edge("save_user_request", "prepare_context")
        graph.add_edge("prepare_context", "route")
        graph.add_conditional_edges("route", lambda s: "main_direct" if s.get("mode") == "main_direct" else "filesystem_tool")
        graph.add_edge("main_direct", END)
        graph.add_edge("filesystem_tool", "plan")
        graph.add_edge("plan", "run_subagents")
        graph.add_edge("run_subagents", "review_and_replan")
        graph.add_conditional_edges("review_and_replan", lambda s: "aggregate" if s.get("done") else "run_subagents")
        graph.add_edge("aggregate", END)

        chain = graph.compile()
        state = await chain.ainvoke(
            {
                "session_key": session_key,
                "repo_path": repo_path,
                "user_message": user_message,
            }
        )

        return DeepAgentResult(
            final_response=state.get("final_response", "(empty response)"),
            mode=state.get("mode", "main_direct"),
            used_offloads=state.get("used_offloads", 0),
            live_messages=state.get("live_messages", 0),
            subagent_count=state.get("subagent_count", 0),
        )

    async def persist_turn(self, session_key: str, repo_path: str, assistant_message: str) -> dict[str, int]:
        """턴 종료 시 assistant 응답만 저장 후 오프로드 (user 요청은 이미 save_user_request에서 저장됨)."""
        stats = await self.context_offload.persist_turn(
            session_key=session_key,
            assistant_message=assistant_message,
            summarize_offload=lambda offload: self.tools.invoke_summarize_offload(
                {
                    "repoPath": repo_path,
                    "messages": offload["messages"],
                }
            ),
        )
        return {
            "offloadsCreated": stats.offloads_created,
            "liveMessages": stats.live_messages,
            "totalOffloads": stats.total_offloads,
        }


async def maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def build_warpgrep_filesystem_context(repo_path: str, user_message: str, limit: int = 8) -> list[str]:
    keywords = [token.strip(" .,!?()[]{}\"'\n\t") for token in user_message.split()]
    keywords = [x.lower() for x in keywords if len(x) >= 3]
    if not keywords:
        return []

    ranked: list[tuple[int, str, str]] = []
    root = Path(repo_path)
    if not root.exists() or not root.is_dir():
        return []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}]
        for fname in filenames:
            rel_path = (rel_dir / fname).as_posix() if str(rel_dir) != "." else fname
            lower = rel_path.lower()
            score = sum(1 for token in keywords if token in lower)
            if score <= 0:
                continue
            ranked.append((score, rel_path, f"[warpgrep] file={rel_path} score={score}"))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [item[2] for item in ranked[:limit]]


def map_selected(items: list[str], selected_ids: list[str], prefix: str) -> list[str]:
    mapped = []
    for sid in selected_ids:
        if not sid.startswith(prefix + "-"):
            continue
        try:
            idx = int(sid.split("-", 1)[1])
        except ValueError:
            continue
        if 0 <= idx < len(items):
            mapped.append(items[idx])
    return mapped


def build_main_direct_prompt(state: DeepAgentState) -> str:
    return "\n".join(
        [
            "Role: main agent (direct response mode).",
            "Goal: answer the user request directly without decomposition.",
            "Language: Korean unless user explicitly requests another language.",
            "Style: concise, practical, no fluff.",
            f"Repository: {state['repo_path']}",
            "",
            "[Offloaded Context Summary]",
            *state.get("offloaded_context", []),
            "",
            "[Live Conversation Excerpts]",
            *state.get("live_conversation", []),
            "",
            "[User Request]",
            state["user_message"],
        ]
    )


def build_subagent_prompt(
    index: int,
    total: int,
    task: dict[str, Any],
    offloaded_context: list[str],
    live_conversation: list[str],
    thread_user_inputs: list[str],
    user_message: str,
) -> str:
    safe_task = {
        "id": sanitize_prompt_segment(str(task.get("id", "")), 80),
        "title": sanitize_prompt_segment(str(task.get("title", "")), 240),
        "instructions": sanitize_prompt_segment(str(task.get("instructions", "")), 2400),
        "doneDefinition": sanitize_prompt_segment(str(task.get("doneDefinition", "")), 600),
    }
    safe_offloads = [sanitize_prompt_segment(x, 900) for x in offloaded_context[:6]]
    safe_live = [sanitize_prompt_segment(x, 700) for x in live_conversation[:10]]
    safe_thread_inputs = [sanitize_prompt_segment(x, 700) for x in thread_user_inputs[:10]]
    safe_user_message = sanitize_prompt_segment(user_message, 1200)
    return "\n".join(
        [
            "Role: subagent worker.",
            "Goal: execute exactly one TODO and report concise result.",
            f"Task index: {index}/{total}",
            f"[TODO ID] {safe_task['id']}",
            f"[TODO TITLE] {safe_task['title']}",
            f"[TODO INSTRUCTIONS] {safe_task['instructions']}",
            f"[DONE DEFINITION] {safe_task['doneDefinition']}",
            "",
            "[Relevant Offloaded Context]",
            *safe_offloads,
            "",
            "[Relevant Live Conversation]",
            *safe_live,
            "",
            "[Thread User Inputs For This TODO]",
            *safe_thread_inputs,
            "",
            "[Original User Request]",
            safe_user_message,
        ]
    )


def build_aggregation_prompt(
    user_message: str,
    todos: list[dict[str, Any]],
    todo_results: list[dict[str, Any]],
    outputs: list[str],
    completion_reason: str,
) -> str:
    return "\n".join(
        [
            "Role: main aggregator agent.",
            "Goal: produce final user-facing answer from TODO execution results.",
            "Language: Korean unless user explicitly requested another language.",
            "Do not finalize if major gaps remain; clearly state what is completed.",
            "",
            "[User Request]",
            user_message,
            "",
            "[Completion Reason]",
            completion_reason or "(none)",
            "",
            "[TODO Plan]",
            *[f"{x.get('id')}: {x.get('title')}" for x in todos],
            "",
            "[TODO Execution Results]",
            *[str(x) for x in todo_results],
            "",
            "[Subagent Outputs]",
            *outputs,
        ]
    )


def build_todo_list_message(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "TODO 생성 결과: (empty)"
    lines = ["TODO 생성 결과:"]
    for task in tasks:
        depends = ",".join(task.get("dependsOn", [])) or "-"
        lines.append(f"- {task.get('id','?')} | {task.get('title','(no-title)')} | dependsOn={depends}")
    return "\n".join(lines)


def format_subagent_event(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("type", "unknown"))
    if event_type == "item.completed":
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type", "unknown"))
        if item_type == "agent_message":
            text = str(item.get("text", "")).strip()
            return f"agent_message: {truncate(text, 260)}" if text else "agent_message: (empty)"
        return None
    return None


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


def is_separator_chunk_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "Separator is found, but chunk is longer than limit" in text
        or "Separator is not found, and chunk exceed the limit" in text
    )


def sanitize_prompt_segment(value: str, max_len: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Prevent very long unbroken chunks that can break downstream splitters.
    text = break_long_tokens(text, token_len=240)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def break_long_tokens(value: str, token_len: int = 240) -> str:
    parts: list[str] = []
    for line in value.split("\n"):
        if len(line) <= token_len:
            parts.append(line)
            continue
        cursor = 0
        while cursor < len(line):
            parts.append(line[cursor : cursor + token_len])
            cursor += token_len
    return "\n".join(parts)
