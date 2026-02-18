from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .codex_cli_runtime import CodexCliRuntimeService


class RouteInput(BaseModel):
    repoPath: str
    userMessage: str
    offloadedContext: list[str]
    liveConversation: list[str]


class PlanInput(BaseModel):
    repoPath: str
    userMessage: str
    offloadedContext: list[str]
    liveConversation: list[str]
    maxTasks: int = Field(gt=0)


class SelectContextInput(BaseModel):
    repoPath: str
    taskInstructions: str
    query: str
    offloads: list[dict[str, str]]
    liveMessages: list[dict[str, str]]
    limits: dict[str, int]


class SummarizeOffloadInput(BaseModel):
    repoPath: str
    messages: list[dict[str, str]]


class ChannelRoutingInput(BaseModel):
    repoPath: str
    userMessage: str
    currentChannelName: str
    currentCategoryName: str | None = None
    offloadedContext: list[str]
    liveConversation: list[str]


class ExternalContextRoutingInput(BaseModel):
    repoPath: str
    userMessage: str
    currentSessionKey: str
    offloadedContext: list[str]
    liveConversation: list[str]
    candidates: list[dict[str, str]]
    maxSelect: int = Field(gt=0)


class ReplanInput(BaseModel):
    repoPath: str
    userMessage: str
    offloadedContext: list[str]
    liveConversation: list[str]
    currentTodoList: list[dict[str, Any]]
    allTodoResults: list[dict[str, Any]]
    latestTodoResults: list[dict[str, Any]]
    allOutputs: list[str]
    currentRound: int = Field(gt=0)
    maxRounds: int = Field(gt=0)
    maxTasks: int = Field(gt=0)


class RouteDecision(BaseModel):
    mode: str
    reason: str


class TodoItem(BaseModel):
    id: str
    title: str
    instructions: str
    priority: str = "medium"
    dependsOn: list[str] = []
    doneDefinition: str


class ContextSelectionOutput(BaseModel):
    offloadIds: list[str]
    liveMessageIds: list[str]


class OffloadSummaryOutput(BaseModel):
    summary: str
    keywords: list[str]


class ReplanDecision(BaseModel):
    done: bool
    reason: str
    nextTodos: list[TodoItem] = []


class ChannelRoutingDecision(BaseModel):
    action: str
    reason: str
    newChannelName: str = ""
    targetCategoryName: str = ""


class ExternalContextRoutingDecision(BaseModel):
    useExternalContext: bool
    reason: str
    selectedSessionKeys: list[str] = []


@dataclass(slots=True)
class DeepAgentToolsService:
    codex: CodexCliRuntimeService

    async def invoke_route(self, input_data: dict[str, Any]) -> dict[str, str]:
        parsed = RouteInput.model_validate(input_data)
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt="\n".join(
                [
                    "Role: routing model for a deep-agent system.",
                    "Goal: choose execution mode for the current request.",
                    "Output contract: return ONLY one-line JSON.",
                    '{"mode":"main_direct|subagent_pipeline","reason":"string"}',
                    "Decision policy:",
                    "- Use main_direct only for truly simple conversational replies.",
                    "- Use subagent_pipeline for coding, analysis, planning, debugging, automation, or multi-step work.",
                    "- If uncertain, choose subagent_pipeline.",
                    "",
                    "[Offloaded Context Summary]",
                    *parsed.offloadedContext,
                    "",
                    "[Live Conversation Excerpts]",
                    *parsed.liveConversation,
                    "",
                    "[User Request]",
                    parsed.userMessage,
                ]
            ),
        )
        decision = RouteDecision.model_validate(raw)
        mode = "main_direct" if decision.mode == "main_direct" else "subagent_pipeline"
        reason = decision.reason.strip()[:120] or "model-route"
        return {"mode": mode, "reason": reason}

    async def invoke_plan(self, input_data: dict[str, Any]) -> list[dict[str, Any]]:
        parsed = PlanInput.model_validate(input_data)
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt="\n".join(
                [
                    "Role: main orchestrator agent.",
                    "Goal: create a TODO list and delegate each TODO to subagents.",
                    "Output contract: return ONLY one-line JSON.",
                    '{"todos":[{"id":"string","title":"string","instructions":"string","priority":"high|medium|low","dependsOn":["id"],"doneDefinition":"string"}]}',
                    f"- Max {parsed.maxTasks} TODOs.",
                    "- If request is simple, still output one TODO.",
                    "",
                    "[Offloaded Context Summary]",
                    *parsed.offloadedContext,
                    "",
                    "[Live Conversation Excerpts]",
                    *parsed.liveConversation,
                    "",
                    "[User Request]",
                    parsed.userMessage,
                ]
            ),
        )
        todos = raw.get("todos") if isinstance(raw, dict) else None
        valid: list[dict[str, Any]] = []
        if isinstance(todos, list):
            for idx, item in enumerate(todos[: parsed.maxTasks]):
                try:
                    todo = TodoItem.model_validate(item)
                    valid.append(todo.model_dump())
                except ValidationError:
                    continue
        if valid:
            return valid
        return [
            TodoItem(
                id="T1",
                title="Primary Task",
                instructions=parsed.userMessage,
                priority="high",
                dependsOn=[],
                doneDefinition="요청한 핵심 결과를 사용자에게 전달 가능한 형태로 완성한다.",
            ).model_dump()
        ]

    async def invoke_select_context(self, input_data: dict[str, Any]) -> dict[str, list[str]]:
        parsed = SelectContextInput.model_validate(input_data)
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt="\n".join(
                [
                    "Role: context selector for one subagent task.",
                    "Goal: choose the smallest useful context subset.",
                    "Output contract: return ONLY one-line JSON.",
                    '{"offloadIds":["id"],"liveMessageIds":["id"]}',
                    f"- Select at most {parsed.limits.get('offloads', 2)} offload IDs.",
                    f"- Select at most {parsed.limits.get('liveMessages', 6)} live message IDs.",
                    "",
                    "[Task Instructions]",
                    parsed.taskInstructions,
                    "",
                    "[Offload Candidates]",
                    *[f"id={x.get('id')} summary={truncate(str(x.get('summary', '')), 500)}" for x in parsed.offloads],
                    "",
                    "[Live Message Candidates]",
                    *[
                        f"id={x.get('id')} role={x.get('role')} content={truncate(str(x.get('content', '')), 500)}"
                        for x in parsed.liveMessages
                    ],
                ]
            ),
        )
        out = ContextSelectionOutput.model_validate(raw)
        return {
            "offloadIds": out.offloadIds[: parsed.limits.get("offloads", 2)],
            "liveMessageIds": out.liveMessageIds[: parsed.limits.get("liveMessages", 6)],
        }

    async def invoke_summarize_offload(self, input_data: dict[str, Any]) -> dict[str, Any]:
        parsed = SummarizeOffloadInput.model_validate(input_data)
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt="\n".join(
                [
                    "Role: long-context memory compressor.",
                    "Goal: compress old dialogue chunk for future retrieval.",
                    "Output contract: return ONLY one-line JSON.",
                    '{"summary":"string","keywords":["string"]}',
                    "- Keep summary dense and factual (<= 180 words).",
                    "",
                    "[Messages]",
                    *[
                        f"{idx + 1}. role={m.get('role')} content={truncate(str(m.get('content', '')), 700)}"
                        for idx, m in enumerate(parsed.messages)
                    ],
                ]
            ),
        )
        out = OffloadSummaryOutput.model_validate(raw)
        return out.model_dump()

    async def invoke_channel_routing(self, input_data: dict[str, Any]) -> dict[str, Any]:
        parsed = ChannelRoutingInput.model_validate(input_data)
        base_prompt = "\n".join(
            [
                "Role: channel and category router for long-running chat operations.",
                "Goal: decide whether to keep conversation in current Discord channel or split into a new channel/category.",
                "Output contract: return ONLY one-line JSON.",
                '{"action":"stay|split","reason":"string","newChannelName":"string","targetCategoryName":"string"}',
                "Hard rules:",
                "- If action=split, both newChannelName and targetCategoryName MUST be non-empty.",
                "- If action=stay, set newChannelName=\"\" and targetCategoryName=\"\".",
                "Policy:",
                "- split when topic/project/domain clearly shifts (e.g., Project A -> Project B, coding -> food chat).",
                "- split when separation improves context quality and retrieval precision.",
                "- stay when request is a clear continuation of ongoing work.",
                "- If uncertain, prefer stay.",
                "Naming rules:",
                "- newChannelName: short kebab-case label (3-40 chars) if split.",
                "- targetCategoryName: concise topic/project category if split.",
                "- Do NOT use generic categories like '채팅 채널', 'general', 'chat'.",
                "",
                f"[Current Channel] name={parsed.currentChannelName}",
                f"[Current Category] name={parsed.currentCategoryName or '(none)'}",
                "",
                "[Offloaded Context Summary]",
                *parsed.offloadedContext,
                "",
                "[Live Conversation Excerpts]",
                *parsed.liveConversation,
                "",
                "[Incoming User Message]",
                parsed.userMessage,
            ]
        )
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt=base_prompt,
        )
        out = ChannelRoutingDecision.model_validate(raw)
        if out.action == "split" and (not out.newChannelName.strip() or not out.targetCategoryName.strip()):
            raw_retry = await self._run_json_with_retry(
                repo_path=parsed.repoPath,
                prompt="\n".join(
                    [
                        base_prompt,
                        "",
                        "Previous output was invalid because action=split requires non-empty newChannelName and targetCategoryName.",
                        "Retry with valid JSON that satisfies all hard rules.",
                    ]
                ),
                attempts=1,
            )
            out = ChannelRoutingDecision.model_validate(raw_retry)
            if out.action == "split" and (not out.newChannelName.strip() or not out.targetCategoryName.strip()):
                raise ValueError("channel routing model returned split without required channel/category names")
        action = "split" if out.action == "split" else "stay"
        return {
            "action": action,
            "reason": out.reason.strip()[:200] or "channel-router",
            "newChannelName": out.newChannelName,
            "targetCategoryName": out.targetCategoryName,
        }

    async def invoke_external_context_routing(self, input_data: dict[str, Any]) -> dict[str, Any]:
        parsed = ExternalContextRoutingInput.model_validate(input_data)
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt="\n".join(
                [
                    "Role: cross-channel context router.",
                    "Goal: decide whether current request needs context from other Discord channels/threads.",
                    "Output contract: return ONLY one-line JSON.",
                    '{"useExternalContext":true|false,"reason":"string","selectedSessionKeys":["sessionKey"]}',
                    "Hard rules:",
                    "- selectedSessionKeys must come only from [Candidate Sessions].",
                    f"- Select at most {parsed.maxSelect} sessions.",
                    "- Prefer precision over quantity.",
                    "- If current context is sufficient, useExternalContext=false and selectedSessionKeys=[].",
                    "When to select:",
                    "- User explicitly references previous project/channel/thread/conversation.",
                    "- Current request appears to continue work from another channel.",
                    "- Missing requirements/decisions likely live in a different channel history.",
                    "",
                    f"[Current Session] {parsed.currentSessionKey}",
                    "",
                    "[Current Offloaded Context]",
                    *parsed.offloadedContext,
                    "",
                    "[Current Live Conversation]",
                    *parsed.liveConversation,
                    "",
                    "[Candidate Sessions]",
                    *[
                        f"sessionKey={x.get('sessionKey','')} channelId={x.get('channelId','')} updatedAt={x.get('updatedAt','')} summary={truncate(str(x.get('summary','')), 260)}"
                        for x in parsed.candidates
                    ],
                    "",
                    "[User Request]",
                    parsed.userMessage,
                ]
            ),
        )
        out = ExternalContextRoutingDecision.model_validate(raw)
        allowed = {str(x.get("sessionKey", "")) for x in parsed.candidates}
        selected = [x for x in out.selectedSessionKeys if x in allowed][: parsed.maxSelect]
        return {
            "useExternalContext": bool(out.useExternalContext and selected),
            "reason": out.reason.strip()[:200] or "external-context-router",
            "selectedSessionKeys": selected,
        }

    async def invoke_replan(self, input_data: dict[str, Any]) -> dict[str, Any]:
        parsed = ReplanInput.model_validate(input_data)
        raw = await self._run_json_with_retry(
            repo_path=parsed.repoPath,
            prompt="\n".join(
                [
                    "Role: main agent quality gate and replanner.",
                    "Goal: determine if user request is fully complete after subagent execution.",
                    "If incomplete, produce additional TODOs for the next subagent round.",
                    "Output contract: return ONLY one-line JSON.",
                    '{"done":true|false,"reason":"string","nextTodos":[{"id":"string","title":"string","instructions":"string","priority":"high|medium|low","dependsOn":["id"],"doneDefinition":"string"}]}',
                    "Policy:",
                    "- done=true only when output is complete, correct, and directly deliverable.",
                    "- If any missing work, unresolved blocker, or quality gap exists, done=false.",
                    "- If done=false, nextTodos must contain actionable tasks (max limit).",
                    f"- currentRound={parsed.currentRound}, maxRounds={parsed.maxRounds}.",
                    f"- nextTodos max {parsed.maxTasks}.",
                    "- Prefer fewer, high-impact TODOs.",
                    "",
                    "[User Request]",
                    parsed.userMessage,
                    "",
                    "[Current Round Todo List]",
                    *[str(x) for x in parsed.currentTodoList],
                    "",
                    "[Latest Round Results]",
                    *[str(x) for x in parsed.latestTodoResults],
                    "",
                    "[All Todo Results]",
                    *[str(x) for x in parsed.allTodoResults],
                    "",
                    "[All Subagent Outputs]",
                    *parsed.allOutputs,
                    "",
                    "[Offloaded Context Summary]",
                    *parsed.offloadedContext,
                    "",
                    "[Live Conversation Excerpts]",
                    *parsed.liveConversation,
                ]
            ),
        )
        decision = ReplanDecision.model_validate(raw)
        next_todos = [x.model_dump() for x in decision.nextTodos[: parsed.maxTasks]]
        return {
            "done": bool(decision.done),
            "reason": decision.reason.strip()[:200] or "quality-gate",
            "nextTodos": next_todos,
        }

    async def _run_json_with_retry(self, repo_path: str, prompt: str, attempts: int = 2) -> dict[str, Any]:
        last_raw = ""
        for i in range(attempts):
            actual_prompt = prompt
            if i > 0:
                actual_prompt = "\n".join(
                    [
                        prompt,
                        "",
                        "[Previous invalid output]",
                        truncate(last_raw or "(none)", 1200),
                        "",
                        "Retry now and return ONLY valid one-line JSON following the schema exactly.",
                    ]
                )
            result = await self.codex.run(repo_path=repo_path, prompt=actual_prompt)
            last_raw = result.assistant_message.strip()
            parsed = extract_json_line(last_raw)
            if parsed is not None:
                return parsed
        raise ValueError("failed to parse valid JSON from model")


def extract_json_line(text: str) -> dict[str, Any] | None:
    candidates = [x.strip() for x in text.splitlines() if x.strip()]
    for line in candidates:
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."
