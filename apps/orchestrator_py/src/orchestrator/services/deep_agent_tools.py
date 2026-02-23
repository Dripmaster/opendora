from __future__ import annotations

import json
import importlib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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


class PlanDecision(BaseModel):
    todos: list[dict[str, Any]]


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


ToolMetricsHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class DeepAgentToolsService:
    codex: CodexCliRuntimeService
    mcp_enabled: bool = False
    mcp_server_urls: list[str] = field(default_factory=list)
    registry: Any = field(init=False)

    def __post_init__(self) -> None:
        tool_registry = importlib.import_module("orchestrator.services.tool_registry")
        mcp_adapter_module = importlib.import_module(
            "orchestrator.services.mcp_adapter"
        )
        tool_registry_cls = getattr(tool_registry, "ToolRegistry")
        tool_spec_cls = getattr(tool_registry, "ToolSpec")
        build_mcp_adapter = getattr(mcp_adapter_module, "build_mcp_adapter")
        mcp_adapter = build_mcp_adapter(
            enabled=self.mcp_enabled,
            server_urls=self.mcp_server_urls,
        )
        self.registry = tool_registry_cls(invoker=self._run_json_with_retry)
        self.registry.register(
            tool_spec_cls(
                name="routing",
                input_schema=RouteInput,
                output_schema=RouteDecision,
                prompt_builder=self._build_route_prompt,
                version="v1",
            )
        )
        self.registry.register(
            tool_spec_cls(
                name="planning",
                input_schema=PlanInput,
                output_schema=PlanDecision,
                prompt_builder=self._build_plan_prompt,
                version="v1",
            )
        )
        self.registry.register(
            tool_spec_cls(
                name="context_select",
                input_schema=SelectContextInput,
                output_schema=ContextSelectionOutput,
                prompt_builder=self._build_select_context_prompt,
                version="v1",
            )
        )
        self.registry.register(
            tool_spec_cls(
                name="context_summarize",
                input_schema=SummarizeOffloadInput,
                output_schema=OffloadSummaryOutput,
                prompt_builder=self._build_summarize_offload_prompt,
                version="v1",
            )
        )
        self.registry.register(
            tool_spec_cls(
                name="channel_routing",
                input_schema=ChannelRoutingInput,
                output_schema=ChannelRoutingDecision,
                prompt_builder=self._build_channel_routing_prompt,
                version="v1",
            )
        )
        self.registry.register(
            tool_spec_cls(
                name="external_context_routing",
                input_schema=ExternalContextRoutingInput,
                output_schema=ExternalContextRoutingDecision,
                prompt_builder=self._build_external_context_routing_prompt,
                version="v1",
            )
        )
        self.registry.register(
            tool_spec_cls(
                name="replan",
                input_schema=ReplanInput,
                output_schema=ReplanDecision,
                prompt_builder=self._build_replan_prompt,
                version="v1",
            )
        )
        self.registry.register_mcp_tools(
            adapter=mcp_adapter,
            on_error=self._log_mcp_issue,
        )

    async def invoke_route(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> dict[str, str]:
        _, output = await self.registry.invoke(
            name="routing",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        decision = RouteDecision.model_validate(output)
        mode = "main_direct" if decision.mode == "main_direct" else "subagent_pipeline"
        reason = decision.reason.strip()[:120] or "model-route"
        return {"mode": mode, "reason": reason}

    async def invoke_plan(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> list[dict[str, Any]]:
        parsed_input, output = await self.registry.invoke(
            name="planning",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        parsed = PlanInput.model_validate(parsed_input)
        decision = PlanDecision.model_validate(output)
        todos = decision.todos
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

    async def invoke_select_context(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> dict[str, list[str]]:
        parsed_input, output = await self.registry.invoke(
            name="context_select",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        parsed = SelectContextInput.model_validate(parsed_input)
        out = ContextSelectionOutput.model_validate(output)
        return {
            "offloadIds": out.offloadIds[: parsed.limits.get("offloads", 2)],
            "liveMessageIds": out.liveMessageIds[
                : parsed.limits.get("liveMessages", 6)
            ],
        }

    async def invoke_summarize_offload(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> dict[str, Any]:
        _, output = await self.registry.invoke(
            name="context_summarize",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        out = OffloadSummaryOutput.model_validate(output)
        return out.model_dump()

    async def invoke_channel_routing(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> dict[str, Any]:
        parsed_input, output = await self.registry.invoke(
            name="channel_routing",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        parsed = ChannelRoutingInput.model_validate(parsed_input)
        out = ChannelRoutingDecision.model_validate(output)
        spec = self.registry.get("channel_routing")
        base_prompt = spec.prompt_builder(parsed)
        if out.action == "split" and (
            not out.newChannelName.strip() or not out.targetCategoryName.strip()
        ):
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
                schema=ChannelRoutingDecision,
                metrics_stage=spec.name,
                on_metrics=on_metrics,
                tool_version=spec.version,
            )
            out = ChannelRoutingDecision.model_validate(raw_retry)
            if out.action == "split" and (
                not out.newChannelName.strip() or not out.targetCategoryName.strip()
            ):
                raise ValueError(
                    "channel routing model returned split without required channel/category names"
                )
        action = "split" if out.action == "split" else "stay"
        return {
            "action": action,
            "reason": out.reason.strip()[:200] or "channel-router",
            "newChannelName": out.newChannelName,
            "targetCategoryName": out.targetCategoryName,
        }

    async def invoke_external_context_routing(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> dict[str, Any]:
        parsed_input, output = await self.registry.invoke(
            name="external_context_routing",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        parsed = ExternalContextRoutingInput.model_validate(parsed_input)
        out = ExternalContextRoutingDecision.model_validate(output)
        allowed = {str(x.get("sessionKey", "")) for x in parsed.candidates}
        selected = [x for x in out.selectedSessionKeys if x in allowed][
            : parsed.maxSelect
        ]
        return {
            "useExternalContext": bool(out.useExternalContext and selected),
            "reason": out.reason.strip()[:200] or "external-context-router",
            "selectedSessionKeys": selected,
        }

    async def invoke_replan(
        self,
        input_data: dict[str, Any],
        on_metrics: ToolMetricsHandler | None = None,
    ) -> dict[str, Any]:
        parsed_input, output = await self.registry.invoke(
            name="replan",
            input_data=input_data,
            on_metrics=on_metrics,
        )
        parsed = ReplanInput.model_validate(parsed_input)
        decision = ReplanDecision.model_validate(output)
        next_todos = [x.model_dump() for x in decision.nextTodos[: parsed.maxTasks]]
        return {
            "done": bool(decision.done),
            "reason": decision.reason.strip()[:200] or "quality-gate",
            "nextTodos": next_todos,
        }

    def _build_route_prompt(self, parsed: BaseModel) -> str:
        route = RouteInput.model_validate(parsed)
        return "\n".join(
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
                *route.offloadedContext,
                "",
                "[Live Conversation Excerpts]",
                *route.liveConversation,
                "",
                "[User Request]",
                route.userMessage,
            ]
        )

    def _build_plan_prompt(self, parsed: BaseModel) -> str:
        plan = PlanInput.model_validate(parsed)
        return "\n".join(
            [
                "Role: main orchestrator agent.",
                "Goal: create a TODO list and delegate each TODO to subagents.",
                "Output contract: return ONLY one-line JSON.",
                '{"todos":[{"id":"string","title":"string","instructions":"string","priority":"high|medium|low","dependsOn":["id"],"doneDefinition":"string"}]}',
                f"- Max {plan.maxTasks} TODOs.",
                "- If request is simple, still output one TODO.",
                "",
                "[Offloaded Context Summary]",
                *plan.offloadedContext,
                "",
                "[Live Conversation Excerpts]",
                *plan.liveConversation,
                "",
                "[User Request]",
                plan.userMessage,
            ]
        )

    def _build_select_context_prompt(self, parsed: BaseModel) -> str:
        context = SelectContextInput.model_validate(parsed)
        return "\n".join(
            [
                "Role: context selector for one subagent task.",
                "Goal: choose the smallest useful context subset.",
                "Output contract: return ONLY one-line JSON.",
                '{"offloadIds":["id"],"liveMessageIds":["id"]}',
                f"- Select at most {context.limits.get('offloads', 2)} offload IDs.",
                f"- Select at most {context.limits.get('liveMessages', 6)} live message IDs.",
                "",
                "[Task Instructions]",
                context.taskInstructions,
                "",
                "[Offload Candidates]",
                *[
                    f"id={x.get('id')} summary={truncate(str(x.get('summary', '')), 500)}"
                    for x in context.offloads
                ],
                "",
                "[Live Message Candidates]",
                *[
                    f"id={x.get('id')} role={x.get('role')} content={truncate(str(x.get('content', '')), 500)}"
                    for x in context.liveMessages
                ],
            ]
        )

    def _build_summarize_offload_prompt(self, parsed: BaseModel) -> str:
        summarize = SummarizeOffloadInput.model_validate(parsed)
        return "\n".join(
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
                    for idx, m in enumerate(summarize.messages)
                ],
            ]
        )

    def _build_channel_routing_prompt(self, parsed: BaseModel) -> str:
        channel = ChannelRoutingInput.model_validate(parsed)
        return "\n".join(
            [
                "Role: channel and category router for long-running chat operations.",
                "Goal: decide whether to keep conversation in current Discord channel or split into a new channel/category.",
                "Output contract: return ONLY one-line JSON.",
                '{"action":"stay|split","reason":"string","newChannelName":"string","targetCategoryName":"string"}',
                "Hard rules:",
                "- If action=split, both newChannelName and targetCategoryName MUST be non-empty.",
                '- If action=stay, set newChannelName="" and targetCategoryName="".',
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
                f"[Current Channel] name={channel.currentChannelName}",
                f"[Current Category] name={channel.currentCategoryName or '(none)'}",
                "",
                "[Offloaded Context Summary]",
                *channel.offloadedContext,
                "",
                "[Live Conversation Excerpts]",
                *channel.liveConversation,
                "",
                "[Incoming User Message]",
                channel.userMessage,
            ]
        )

    def _build_external_context_routing_prompt(self, parsed: BaseModel) -> str:
        context = ExternalContextRoutingInput.model_validate(parsed)
        return "\n".join(
            [
                "Role: cross-channel context router.",
                "Goal: decide whether current request needs context from other Discord channels/threads.",
                "Output contract: return ONLY one-line JSON.",
                '{"useExternalContext":true|false,"reason":"string","selectedSessionKeys":["sessionKey"]}',
                "Hard rules:",
                "- selectedSessionKeys must come only from [Candidate Sessions].",
                f"- Select at most {context.maxSelect} sessions.",
                "- Prefer precision over quantity.",
                "- If current context is sufficient, useExternalContext=false and selectedSessionKeys=[].",
                "When to select:",
                "- User explicitly references previous project/channel/thread/conversation.",
                "- Current request appears to continue work from another channel.",
                "- Missing requirements/decisions likely live in a different channel history.",
                "",
                f"[Current Session] {context.currentSessionKey}",
                "",
                "[Current Offloaded Context]",
                *context.offloadedContext,
                "",
                "[Current Live Conversation]",
                *context.liveConversation,
                "",
                "[Candidate Sessions]",
                *[
                    f"sessionKey={x.get('sessionKey', '')} channelId={x.get('channelId', '')} updatedAt={x.get('updatedAt', '')} summary={truncate(str(x.get('summary', '')), 260)}"
                    for x in context.candidates
                ],
                "",
                "[User Request]",
                context.userMessage,
            ]
        )

    def _build_replan_prompt(self, parsed: BaseModel) -> str:
        replan = ReplanInput.model_validate(parsed)
        return "\n".join(
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
                f"- currentRound={replan.currentRound}, maxRounds={replan.maxRounds}.",
                f"- nextTodos max {replan.maxTasks}.",
                "- Prefer fewer, high-impact TODOs.",
                "",
                "[User Request]",
                replan.userMessage,
                "",
                "[Current Round Todo List]",
                *[str(x) for x in replan.currentTodoList],
                "",
                "[Latest Round Results]",
                *[str(x) for x in replan.latestTodoResults],
                "",
                "[All Todo Results]",
                *[str(x) for x in replan.allTodoResults],
                "",
                "[All Subagent Outputs]",
                *replan.allOutputs,
                "",
                "[Offloaded Context Summary]",
                *replan.offloadedContext,
                "",
                "[Live Conversation Excerpts]",
                *replan.liveConversation,
            ]
        )

    async def _run_json_with_retry(
        self,
        repo_path: str,
        prompt: str,
        attempts: int = 2,
        schema: type[BaseModel] | None = None,
        metrics_stage: str | None = None,
        on_metrics: ToolMetricsHandler | None = None,
        tool_version: str | None = None,
    ) -> dict[str, Any]:
        last_raw = ""
        last_error = ""
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
            started = time.perf_counter()
            result = await self.codex.run(repo_path=repo_path, prompt=actual_prompt)
            duration_ms = int((time.perf_counter() - started) * 1000)
            if on_metrics and metrics_stage:
                payload: dict[str, Any] = {
                    "stage": metrics_stage,
                    "duration_ms": duration_ms,
                    "prompt_chars": len(actual_prompt),
                }
                if tool_version:
                    payload["tool_version"] = tool_version
                usage_total = int(getattr(result.usage, "total", 0) or 0)
                if usage_total > 0:
                    payload["usage_total_tokens"] = usage_total
                await maybe_await(on_metrics(payload))
            last_raw = result.assistant_message.strip()
            parsed = extract_json_line(last_raw)
            if parsed is None:
                last_error = "parse_failed"
                self._log_retry_issue(
                    attempt=i + 1,
                    attempts=attempts,
                    reason="JSON parsing failed",
                    raw=last_raw,
                )
                continue
            if schema is not None:
                try:
                    schema.model_validate(parsed)
                except ValidationError as exc:
                    last_error = f"schema_failed:{exc.errors()[:2]}"
                    self._log_retry_issue(
                        attempt=i + 1,
                        attempts=attempts,
                        reason="Schema validation failed",
                        raw=last_raw,
                    )
                    continue
            return parsed
        raise ValueError(
            "failed to parse valid JSON from model"
            f" (last_error={last_error or 'unknown'}, last_output={truncate(last_raw or '(empty)', 240)})"
        )

    def _log_retry_issue(
        self, attempt: int, attempts: int, reason: str, raw: str
    ) -> None:
        logger = getattr(self.codex, "logger", None)
        if logger is None:
            return
        logger.warning(
            "DeepAgentTools JSON attempt failed.",
            attempt=attempt,
            attempts=attempts,
            reason=reason,
            output_preview=truncate(raw or "(empty)", 400),
        )

    def _log_mcp_issue(self, message: str) -> None:
        logger = getattr(self.codex, "logger", None)
        if logger is None:
            return
        logger.warning("MCP adapter issue.", message=message)


def extract_json_line(text: str) -> dict[str, Any] | None:
    # Prompt contract remains "one-line JSON", but parser accepts relaxed formats
    # to recover from common model formatting drift.
    try:
        parsed_direct = json.loads(text.strip())
        if isinstance(parsed_direct, dict):
            return parsed_direct
    except json.JSONDecodeError:
        pass

    for block in re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    ):
        try:
            parsed_block = json.loads(block.strip())
            if isinstance(parsed_block, dict):
                return parsed_block
        except json.JSONDecodeError:
            continue

    extracted = _extract_first_json_object(text)
    if extracted is not None:
        try:
            parsed_extracted = json.loads(extracted)
            if isinstance(parsed_extracted, dict):
                return parsed_extracted
        except json.JSONDecodeError:
            pass

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


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


async def maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
