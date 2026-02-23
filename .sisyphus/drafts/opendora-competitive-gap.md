# Draft: Opendora Competitive Gap + Upgrade Plan

## Requirements (confirmed)
- "openclaw나 deepagents 등 최신 ai비서 프로젝트들과 우리의 프로젝트를 비교해서 우리가 기능면, 성능면에서 더 추가해야 할 것들을 찾아서 구현 계획 세워"

## Current Product Snapshot (repo-grounded)
- Codebase: Python 3.11+ Discord-driven Codex orchestrator (`apps/orchestrator_py`).
- Core loop: LangGraph-based Deep Agent pipeline (`apps/orchestrator_py/src/orchestrator/services/deep_agent.py`).
- Tools layer: JSON-schema tool calls via Codex (`apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`).
- Memory/context: filesystem JSON session store + offload summaries + keyword relevance scoring (`apps/orchestrator_py/src/orchestrator/services/context_offload.py`).
- Discord UX: DM/mention intake, optional HITL approval, per-TODO Discord threads, model-based channel/category routing, context rotation to active/archive categories (`apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`).
- Execution runtime: Codex CLI wrapper with sandbox/timeout + retry/backoff; streaming events parser (`apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`).
- Tests: pytest + pytest-asyncio; no CI workflow currently in repo (`apps/orchestrator_py/tests/*.py`).

## Research Findings (external anchors)
- OpenClaw (personal AI assistant) highlights: multi-channel inbox, gateway control plane + control UI, skills platform, onboarding wizard, security defaults for DMs, model failover, docker-based sandbox images.
  - Source: https://github.com/openclaw/openclaw (README + docs links)
- DeepAgents (LangChain) highlights: deep agent harness with planning/task decomposition, filesystem tools for context management, pluggable backends, subagent spawning (`task` tool), long-term memory via LangGraph store, sandboxes.
  - Source: https://docs.langchain.com/oss/python/deepagents/overview
- OpenHands evaluation harness: benchmark-driven development, multi-LLM config, docker/remote runtime, parallel eval workers, rich logging.
  - Source: https://docs.openhands.dev/openhands/usage/developers/evaluation-harness
  - Benchmarks repo (examples of benchmark organization/workspace strategy): https://github.com/OpenHands/benchmarks

## Gap Hypotheses (to validate/implement)
### Feature gaps vs peers
- Skills/plugin ecosystem (OpenClaw-style “skills” + DeepAgents/MCP tool ecosystem).
- Robust sandboxing options beyond Codex CLI flags (Docker/remote sandboxes).
- Model provider/model fallback orchestration (OpenClaw model failover patterns).
- Ops UX: dashboards/control UI, run inspection, persistent run artifacts.
- Formal eval harness + repeatable benchmarks for regression tracking (OpenHands-style).
- Security posture: DM pairing / allowlist defaults; tool policy gating.

### Performance/quality gaps vs peers
- Observability: structured logs exist, but no metrics/tracing or run-level artifacts.
- Discord rate-limit risk: high-frequency progress messages without explicit throttling.
- Filesystem scan cost: Warpgrep is bounded but still expensive; lacks indexing/caching.
- Context offload storage: JSON file per session; global session candidate scan can be heavy.

## Technical Decisions (defaults unless user overrides)
- Primary goal: improve opendora’s reliability, extensibility, and measurable performance in its existing Discord+Cortex/Codex architecture (NOT rebuilding into a full multi-channel OpenClaw clone).
- Prioritization default: (1) observability+eval harness (prove improvements), (2) performance fixes (Discord throttling + caching/indexing), (3) extensibility via tool/skills registry + MCP integration, (4) safety hardening.
- Out of scope for the first plan wave: Voice/mobile nodes, building a full web UI dashboard, adding 10+ messaging connectors.

## Open Questions (only if required)
- If we must pick a single north-star: "coding agent quality" vs "personal assistant breadth" vs "ops/enterprise governance".
- Where should new run artifacts live: `.opendora/runs/` vs `.opendora/logs/` vs configurable path.

## Scope Boundaries
- INCLUDE: features/perf improvements that fit the existing Python Discord orchestrator architecture.
- EXCLUDE: full OpenClaw parity (multi-channel support, voice), unless explicitly selected.
