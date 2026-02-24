# Opendora Competitive Upgrade Plan (vs OpenClaw, DeepAgents, OpenHands Eval)

## TL;DR
> **Summary**: Add run artifacts + deterministic eval/CI first, then ship high-ROI performance and safety upgrades, and finally introduce a thin tool/skills registry (optionally MCP) to keep feature growth sustainable.
> **Deliverables**:
> - RunContext + per-run artifacts (debuggable, replayable)
> - Deterministic offline eval harness + GitHub Actions CI
> - Performance fixes: Discord progress throttling, context store caching/indexing, warpgrep cache
> - Safety gates: allowlist/pairing defaults + tool policy enforcement
> - Reliability: model rotation on transient failures
> - Extensibility: ToolRegistry + optional MCP adapter boundary
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Run artifacts → deterministic eval harness → CI → perf/safety changes → tool registry

## Context
### Original Request
- Compare recent AI assistant projects (OpenClaw / DeepAgents, etc.) with our project and find feature/performance improvements we should add; produce an implementation plan.

### Repo Snapshot (ground truth)
- Project: Discord-driven Codex orchestrator (Python 3.11+), app at `apps/orchestrator_py/`.
- Key files:
  - Discord gateway: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`
  - Deep agent (LangGraph): `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`
  - Tool prompts/contracts: `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`
  - Context offload store: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`
  - Codex CLI runtime: `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`
  - Wiring/config: `apps/orchestrator_py/src/orchestrator/app.py`, `apps/orchestrator_py/src/orchestrator/config.py`
  - Tests: `apps/orchestrator_py/tests/*.py` (pytest + pytest-asyncio)
- Notable current capabilities:
  - HITL approval, per-TODO threads, model-based channel/category routing, context rotation.
  - Deep-agent flow with TODO planning, validation, multi-round replan, warpgrep filesystem context.
  - Codex CLI streaming events parse + retry/backoff.

### External Reference Points (why these upgrades)
- OpenClaw (personal assistant): multi-channel inbox, gateway control plane, onboarding, security defaults, model failover, skills.
  - Source: `https://github.com/openclaw/openclaw`
- DeepAgents (LangChain): planning + filesystem tools + pluggable backends + `task` subagents + sandboxes + long-term memory.
  - Source: `https://docs.langchain.com/oss/python/deepagents/overview`
- OpenHands evaluation harness: benchmark-driven eval, runtime abstraction, parallel workers, rich logging.
  - Source: `https://docs.openhands.dev/openhands/usage/developers/evaluation-harness`
  - Benchmarks repo patterns: `https://github.com/OpenHands/benchmarks`

### Metis Review (gaps addressed)
- Prevent scope creep: explicitly exclude “full OpenClaw parity” (multi-channel/voice/UI) from MVP.
- Require measurable baselines before performance refactors.
- Deterministic eval + CI must not depend on Discord tokens or live models.
- Run artifacts must have redaction + retention limits to avoid PII/secrets leaks.

## Work Objectives
### Core Objective
- Make opendora measurably more reliable, debuggable, safe-by-default, and extensible while staying within the current Discord + Codex + LangGraph architecture.

### Deliverables
- Run artifacts + run IDs (end-to-end).
- Deterministic offline eval harness (pytest-native) + GitHub Actions CI.
- Performance fixes with measurable targets.
- Safety policy engine + allowlist/pairing defaults.
- Model rotation on transient failures.
- Thin ToolRegistry (and optional MCP adapter) to support a skills/tool ecosystem.

### Definition of Done (verifiable)
- `uv sync --directory apps/orchestrator_py` succeeds.
- `uv run --directory apps/orchestrator_py run --extra dev pytest -q` passes.
- New deterministic eval suite passes in CI without secrets.
- Each orchestrator run produces a run artifact directory with manifest + key events (with redaction).

### Must Have
- Offline deterministic evals (no Discord API calls).
- Artifacts are bounded (retention + max size) and redact sensitive fields by default.
- Discord progress throttling preserves milestones (start/done/errors) and prevents message spam.
- Context store writes are safe (atomic write) and cached/indexed to avoid directory-wide scans.

### Must NOT Have (guardrails)
- No full re-architecture away from LangGraph.
- No adding 10+ messaging connectors / voice nodes / full web UI as part of MVP.
- No high-cardinality metrics labels (no user/channel/session keys in labels).
- No CI requirement for external tokens/secrets.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (existing pytest suite + new deterministic eval suite).
- Evidence policy: store artifacts/logs under `.sisyphus/evidence/` for each task; when running commands, pipe output to the Evidence file via `| tee <path>`.
- Baseline measurement: add a small local benchmark script or test that measures:
  - outbound Discord messages per run (mocked)
  - warpgrep scan time (tmp repo)
  - context candidate selection time (synthetic sessions)

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Foundations): Tasks 1–6.
Wave 2 (Performance/Reliability): Tasks 7–11.
Wave 3 (Safety): Tasks 12–13.
Wave 4 (Extensibility + Docs): Tasks 14–17.

### Dependency Matrix (high-level)
- Artifacts + RunContext blocks: eval harness artifacts, CI diagnostics, perf measurement.
- Deterministic eval harness blocks: safe refactors for perf/safety.
- Context store atomic writes blocks: caching/indexing.

### Agent Dispatch Summary (suggested)
- Wave 1: 3 agents parallel (unspecified-high, quick, unspecified-high)
- Wave 2: 3 agents parallel (unspecified-high, unspecified-high, unspecified-high)
- Wave 3: 2 agents parallel (unspecified-high, unspecified-high)
- Wave 4: 2 agents parallel (unspecified-high, quick)

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task includes QA scenarios and writes evidence under `.sisyphus/evidence/`.

- [ ] 1. Introduce `RunContext` + artifact writer (redacted, bounded)

  **What to do**: Create a small cross-cutting run-lifecycle module that generates a `run_id` per request and writes redacted, size-bounded artifacts under `.opendora/runs/<run_id>/`.
  - Add new module `apps/orchestrator_py/src/orchestrator/services/run_artifacts.py` with:
    - `RunContext` (dataclass): `run_id`, `request_id`, `session_key`, `repo_path`, `started_at`, `artifacts_dir`, `debug_enabled`.
    - `ArtifactWriter`: methods `write_manifest()`, `append_event_jsonl()`, `write_text_file()`.
    - Redaction helpers (mask Discord IDs, repo absolute paths, and optionally prompt content) with env toggles.
    - Size + retention controls: max bytes per run, max runs to keep (delete oldest), TTL days (best-effort).
  - Wire env knobs into `apps/orchestrator_py/src/orchestrator/config.py` (new fields):
    - `RUN_ARTIFACTS_ENABLED` (default `true`)
    - `RUN_ARTIFACTS_DIR` (default `.opendora/runs`)
    - `RUN_ARTIFACTS_REDACT` (default `true`)
    - `RUN_ARTIFACTS_MAX_BYTES` (default `2_000_000`)
    - `RUN_ARTIFACTS_RETENTION_DAYS` (default `7`)
    - `RUN_DEBUG_PROMPTS` (default `false`)
  - Update `.env.example` to include the new env vars.

  **Must NOT do**: Store raw prompts, tokens, or Discord IDs unredacted when `RUN_ARTIFACTS_REDACT=true`. Do not write artifacts outside `.opendora/`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-cutting design + correctness.
  - Skills: []
  - Omitted: `playwright` — no browser/UI.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 7, 8 | Blocked By: (none)

  **References**:
  - Wiring root: `apps/orchestrator_py/src/orchestrator/app.py`
  - Logging style: `apps/orchestrator_py/src/orchestrator/services/logger.py`
  - Runtime storage already ignored: `.gitignore`
  - Metis guardrail (redaction/retention): `.sisyphus/drafts/opendora-competitive-gap.md`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q` passes after adding unit tests for redaction + retention behavior.
  - [ ] A new unit test proves artifact output path is under `.opendora/runs/` and capped by `RUN_ARTIFACTS_MAX_BYTES`.

  **QA Scenarios**:
  ```
  Scenario: Artifact writer redacts and bounds output
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k run_artifacts
    Expected: Exit code 0; test asserts redaction enabled and size limit enforced
    Evidence: .sisyphus/evidence/task-1-run-artifacts.txt

  Scenario: Retention removes oldest runs
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k run_artifacts_retention
    Expected: Exit code 0; test asserts only N newest run dirs remain
    Evidence: .sisyphus/evidence/task-1-run-retention.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add run artifacts with redaction` | Files: `apps/orchestrator_py/src/orchestrator/services/run_artifacts.py`, `apps/orchestrator_py/src/orchestrator/config.py`, `.env.example`, tests

- [ ] 2. Plumb `RunContext` through Discord gateway execution flow

  **What to do**: Generate a `run_id` at request start and propagate it through progress events and final summary; persist minimal run metadata.
  - In `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`:
    - Create `RunContext` at the start of `execute_with_progress()`.
    - Add `run_id` to all `on_progress` event payloads (stage events) and final “실행 완료” message (include `runId=`).
    - Use `ArtifactWriter` to write a `manifest.json` with: run_id, reqId, session_key, repo_path, start/end timestamps, mode, subagent_count, used_offloads/live_messages.
  - Ensure that per-TODO thread routing includes `run_id` in thread messages when debug enabled.

  **Must NOT do**: Do not include raw `DEFAULT_REPO_PATH` absolute path in Discord messages when redaction enabled.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: touches user-visible behavior and artifacts.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3, 7 | Blocked By: 1

  **References**:
  - Execution entry: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`
  - Deep agent invocation: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q -k discord_gateway` passes and asserts `runId=` appears in completion summary.
  - [ ] Artifacts are written for a fake run (unit test uses tmp_path + env override for `RUN_ARTIFACTS_DIR`).

  **QA Scenarios**:
  ```
  Scenario: Run ID appears in completion summary
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k runId
    Expected: Exit code 0; test asserts completion reply contains runId
    Evidence: .sisyphus/evidence/task-2-runid-discord.txt

  Scenario: Artifact manifest created
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k artifact_manifest
    Expected: Exit code 0; manifest.json exists and is valid JSON
    Evidence: .sisyphus/evidence/task-2-manifest.txt
  ```

  **Commit**: YES | Message: `feat(discord): attach run_id and persist manifest` | Files: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`, tests

- [ ] 3. Instrument timings + prompt sizing at key boundaries (logs + artifacts)

  **What to do**: Add lightweight timings and prompt size counters to improve debuggability and baseline performance measurement.
  - Add timing wrappers around:
    - context capsule build (`ContextOffloadService.build_capsule`) and candidate scan.
    - warpgrep filesystem node (`collect_warpgrep_filesystem_context`).
    - planning (`invoke_plan`), replan (`invoke_replan`), aggregation prompt.
    - each Codex run/streaming call.
  - Emit as structured logs (structlog) AND append to artifacts event stream: `stage`, `duration_ms`, `prompt_chars`, `usage_total_tokens` where available.

  **Must NOT do**: Do not add metrics labels using `session_key`, channel name, or user IDs.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-cutting, must avoid perf regressions.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7, 8, 9 | Blocked By: 1, 2

  **References**:
  - Deep agent stages: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`
  - Tool invocations: `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`
  - Codex usage parse: `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`

  **Acceptance Criteria**:
  - [ ] Tests assert that at least one artifact event includes `duration_ms` and `prompt_chars` for a run.
  - [ ] No test failures; no new logs include raw prompts when `RUN_DEBUG_PROMPTS=false`.

  **QA Scenarios**:
  ```
  Scenario: Artifact event includes timing fields
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k timing
    Expected: Exit code 0; artifact JSONL includes duration_ms and stage
    Evidence: .sisyphus/evidence/task-3-timing.txt

  Scenario: No raw prompt logged when debug disabled
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k no_raw_prompt
    Expected: Exit code 0; tests confirm redaction policy
    Evidence: .sisyphus/evidence/task-3-redaction.txt
  ```

  **Commit**: YES | Message: `feat(obs): add stage timings and prompt sizing` | Files: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`, `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`, `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`, tests

- [ ] 4. Create deterministic offline eval suite (pytest-native) + artifact outputs

  **What to do**: Add a set of deterministic “eval scenarios” that run without Discord and without live model calls.
  - Implement a `RecordedCodex`/`FakeCodex`-style harness (pattern: `apps/orchestrator_py/tests/test_deep_agent.py`) that drives:
    - route → plan → subagent runs → replan → aggregation
  - Add 5–10 eval cases that cover:
    - main_direct routing
    - invalid plan → validate → replan path
    - stop/skip/abort control handling
    - external context routing selection constraints
    - codex runtime retry chain (simulated)
  - Each eval case must emit run artifacts to a temp `RUN_ARTIFACTS_DIR` and assert manifest structure.

  **Must NOT do**: Do not call Discord API; do not require `DISCORD_BOT_TOKEN`; do not require network.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: needs high rigor and stable determinism.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 6, 7, 8, 9 | Blocked By: 1

  **References**:
  - Existing fake runtime patterns: `apps/orchestrator_py/tests/test_deep_agent.py`
  - OpenHands harness concept: `https://docs.openhands.dev/openhands/usage/developers/evaluation-harness`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q -k eval` passes.
  - [ ] At least 5 eval cases exist and assert stable outputs.

  **QA Scenarios**:
  ```
  Scenario: Offline eval suite executes deterministically
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k eval
    Expected: Exit code 0; outputs stable across 2 consecutive runs
    Evidence: .sisyphus/evidence/task-4-eval.txt

  Scenario: Eval writes artifacts
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k eval_artifacts
    Expected: Exit code 0; manifest.json exists for each case
    Evidence: .sisyphus/evidence/task-4-eval-artifacts.txt
  ```

  **Commit**: YES | Message: `test(eval): add deterministic offline scenarios` | Files: `apps/orchestrator_py/tests/*`

- [ ] 5. Add GitHub Actions CI for orchestrator tests (no secrets)

  **What to do**: Add `.github/workflows/ci.yml` to run pytest on PR/push, using Python 3.11, and ensure it does not require Discord tokens.
  - Use uv (preferred) or pip to install `apps/orchestrator_py` + `dev` extras.
  - Run: `uv run --directory apps/orchestrator_py run --extra dev pytest -q`.
  - Upload `.opendora/runs/` artifacts only if created in CI temp dir (avoid leaking secrets).

  **Must NOT do**: Do not require repository secrets; do not run live model calls.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: workflow YAML is isolated.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: all future refactors | Blocked By: 4

  **References**:
  - Test command: `apps/orchestrator_py/pyproject.toml`

  **Acceptance Criteria**:
  - [ ] CI workflow exists and runs on push + pull_request.
  - [ ] Workflow executes pytest successfully on a clean runner.
  - [ ] Workflow does not reference secrets for test execution (grep-based assertion).

  **QA Scenarios**:
  ```
  Scenario: CI file present and validates
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q | tee .sisyphus/evidence/task-5-ci-local.txt
    Expected: Exit code 0 locally; workflow mirrors this
    Evidence: .sisyphus/evidence/task-5-ci-local.txt

  Scenario: CI does not require secrets
    Tool: Bash
    Steps: test -f .github/workflows/ci.yml && ! grep -E "secrets\\.|DISCORD_BOT_TOKEN" .github/workflows/ci.yml | tee .sisyphus/evidence/task-5-ci-no-secrets.txt
    Expected: Exit code 0; grep finds no forbidden secret references
    Evidence: .sisyphus/evidence/task-5-ci-no-secrets.txt
  ```

  **Commit**: YES | Message: `ci: run orchestrator pytest suite` | Files: `.github/workflows/ci.yml`

- [ ] 6. Align Discord control command UX (docs + supported aliases)

  **What to do**: Resolve mismatch between documented thread control commands and the actual control flags in code.
  - Canonical commands in code today:
    - thread input control: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py` supports `/skip`, `/stop-round`, `/abort` and encodes them as `__control__:{command}`.
    - deep-agent parses control flags via `parse_control_input()` in `apps/orchestrator_py/src/orchestrator/services/deep_agent.py` (`skip`, `stop-round`, `abort`).
  - Implement *aliases only* (no new semantics):
    - Accept `!skip`, `!stop-round`, `!abort` as aliases to the slash commands in `parse_todo_control_command()`.
  - Update `README.md` to list the canonical commands and remove/clarify unsupported `!pause`/`!resume` language.
  - Update/extend tests in `apps/orchestrator_py/tests/test_discord_gateway.py` to cover alias parsing.

  **Must NOT do**: Do not implement pause/resume semantics in this plan wave (requires stateful suspension and is a separate feature).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: localized changes and tests.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7 (throttling messaging UX) | Blocked By: (none)

  **References**:
  - Control commands map: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`
  - Control parsing: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`
  - Existing tests: `apps/orchestrator_py/tests/test_discord_gateway.py`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q -k control_command` passes.
  - [ ] `README.md` lists the supported commands and matches tests.

  **QA Scenarios**:
  ```
  Scenario: Alias control commands are accepted
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k control_command | tee .sisyphus/evidence/task-6-control-alias.txt
    Expected: Exit code 0; tests confirm !skip maps to skip
    Evidence: .sisyphus/evidence/task-6-control-alias.txt

  Scenario: README matches implementation
    Tool: Bash
    Steps: (grep -q "/skip" README.md && grep -q "/stop-round" README.md && grep -q "/abort" README.md && ! grep -q "!pause" README.md && ! grep -q "!resume" README.md) | tee .sisyphus/evidence/task-6-readme-alignment.txt
    Expected: Exit code 0; README contains supported commands and not unsupported ones
    Evidence: .sisyphus/evidence/task-6-readme-alignment.txt
  ```

  **Commit**: YES | Message: `docs(discord): align TODO control commands` | Files: `README.md`, `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`, tests

- [ ] 7. Throttle/batch Discord progress events to avoid rate-limit spam

  **What to do**: Introduce a throttling layer so streaming `subagent_event` updates do not flood Discord.
  - Add a `ProgressThrottler` helper in `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`:
    - Per-target-channel (main channel + per-TODO thread) bucket.
    - Rules:
      - Always send immediately: `subagent_start`, `subagent_done`, `subagent_debug_prompt`, error stages.
      - Throttle `subagent_event` to at most 1 message/second/target; coalesce latest message.
      - Preserve a final flush on `subagent_done`.
    - On Discord send failure, log warning and continue without aborting the run.
  - Add env knobs in `apps/orchestrator_py/src/orchestrator/config.py`:
    - `DISCORD_PROGRESS_THROTTLE_MS` default `1000`
    - `DISCORD_PROGRESS_MAX_BUFFERED` default `20` (prevent unbounded memory)
  - Add tests in `apps/orchestrator_py/tests/test_discord_gateway.py` verifying N events produce <= M outbound messages.

  **Must NOT do**: Do not drop milestone messages; do not make throttling dependent on real time in unit tests (use fake clock / deterministic scheduling).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: correctness + concurrency + user-visible behavior.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: (none) | Blocked By: 2, 6

  **References**:
  - Progress callback: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`
  - Streaming event emission: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`

  **Acceptance Criteria**:
  - [ ] Unit test proves throttling: 50 `subagent_event` updates → <= 3 outbound messages (with throttle=1000ms and simulated time).
  - [ ] Existing gateway tests continue to pass.

  **QA Scenarios**:
  ```
  Scenario: Streaming progress is throttled
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k throttling
    Expected: Exit code 0; test asserts message count bound
    Evidence: .sisyphus/evidence/task-7-throttle.txt

  Scenario: Milestones are not throttled
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k milestone
    Expected: Exit code 0; start/done always emitted
    Evidence: .sisyphus/evidence/task-7-milestones.txt
  ```

  **Commit**: YES | Message: `perf(discord): throttle streaming progress updates` | Files: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`, `apps/orchestrator_py/src/orchestrator/config.py`, tests

- [ ] 8. Make context store writes atomic + add session cache to cut disk I/O

  **What to do**: Improve correctness and performance of session persistence.
  - In `apps/orchestrator_py/src/orchestrator/services/context_offload.py`:
    - Change `_save_session()` to atomic write: write to `*.tmp` then `os.replace()`.
    - Add an in-memory LRU cache for `_load_session()` results keyed by `session_key` with max size (new env: `CONTEXT_SESSION_CACHE_SIZE` default `32`).
    - Ensure cache invalidation on `_save_session()`.
    - Add a per-session `asyncio.Lock` (or `threading.Lock`) to serialize load/save within process.
  - Add tests:
    - atomic write produces valid JSON even if interrupted (simulate by writing partial temp and ensuring replace semantics).
    - cache hit path does not reread from disk (can spy on Path.read_text via monkeypatch).

  **Must NOT do**: Do not introduce a new DB dependency in this iteration (SQLite migration is a separate track).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: data integrity + perf.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9 | Blocked By: 4

  **References**:
  - Current store logic: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`
  - Candidate scan: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`
  - Existing tests: `apps/orchestrator_py/tests/test_context_offload.py`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q -k context_offload` passes.
  - [ ] New tests prove atomic writes + cache behavior.

  **QA Scenarios**:
  ```
  Scenario: Atomic write preserves JSON validity
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k atomic_write
    Expected: Exit code 0; session file loads correctly
    Evidence: .sisyphus/evidence/task-8-atomic.txt

  Scenario: Cache reduces disk reads
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k session_cache
    Expected: Exit code 0; spy asserts read_text called <= 1
    Evidence: .sisyphus/evidence/task-8-cache.txt
  ```

  **Commit**: YES | Message: `perf(memory): atomic session writes and LRU cache` | Files: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`, tests

- [ ] 9. Speed up external context routing by indexing sessions (avoid full directory scans)

  **What to do**: Replace `list_related_session_candidates()` full scan with an index file that is maintained incrementally.
  - Add `sessions_index.json` under `.opendora/context/` (or under `CONTEXT_STORE_DIR`) containing:
    - session_key, updatedAt, offload_count, live_message_count, lightweight summary.
  - Update `_save_session()` to update the index entry.
  - Update `list_related_session_candidates()` to:
    - read the index once,
    - filter by `session_user_id()` (current invariant),
    - only read the top-K candidate session files (if deeper detail is needed).
  - Add tests with 200 synthetic session files to prove candidate selection does not read all files.

  **Must NOT do**: Do not leak cross-user sessions; keep the invariant that external context only routes within the same user ID parsed from `session_key`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: correctness/security + perf.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 12 (pairing safety) | Blocked By: 8

  **References**:
  - Candidate scan: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`
  - External routing tool: `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`
  - Deep agent external routing call site: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`

  **Acceptance Criteria**:
  - [ ] Test proves for N=200 sessions, only O(1) files are read to return candidates (index read + limited session reads).
  - [ ] Existing `test_list_related_session_candidates_prefers_query_relevant_sessions` still passes.

  **QA Scenarios**:
  ```
  Scenario: Session index used instead of full scan
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k session_index
    Expected: Exit code 0; spy asserts glob/read count bounded
    Evidence: .sisyphus/evidence/task-9-index.txt

  Scenario: Cross-user leakage prevented
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k no_cross_user
    Expected: Exit code 0; tests confirm filtering
    Evidence: .sisyphus/evidence/task-9-no-leak.txt
  ```

  **Commit**: YES | Message: `perf(memory): index sessions for external context routing` | Files: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`, tests

- [ ] 10. Add warpgrep inventory cache (reuse directory walk across runs)

  **What to do**: Reduce repeated `os.walk` cost by caching the file inventory per repo fingerprint.
  - Add helper module `apps/orchestrator_py/src/orchestrator/services/warpgrep_cache.py`:
    - `get_repo_fingerprint(repo_path)` uses:
      - `.git/HEAD` and referenced ref mtime if present, else repo root mtime.
    - `get_cached_inventory(repo_path, max_depth)` stores list of relative paths and returns cached result when fingerprint unchanged.
  - Update `collect_warpgrep_filesystem_context()` in `apps/orchestrator_py/src/orchestrator/services/deep_agent.py` to use cached inventory, then score/filter by query.
  - Add tests using tmp_path that change a file and verify fingerprint invalidates cache.

  **Must NOT do**: Do not cache full file contents; cache file paths only.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: performance + correctness.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: (none) | Blocked By: 4

  **References**:
  - Warpgrep functions: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`
  - Existing warpgrep tests: `apps/orchestrator_py/tests/test_deep_agent.py`

  **Acceptance Criteria**:
  - [ ] New tests pass: inventory cache hit avoids second `os.walk`.
  - [ ] Existing warpgrep limit tests still pass.

  **QA Scenarios**:
  ```
  Scenario: Inventory cache hit avoids repeated walk
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k warpgrep_cache
    Expected: Exit code 0; spy asserts walk called once
    Evidence: .sisyphus/evidence/task-10-warpgrep-cache.txt

  Scenario: Fingerprint invalidation on repo change
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k warpgrep_invalidate
    Expected: Exit code 0; cache invalidated after file change
    Evidence: .sisyphus/evidence/task-10-warpgrep-invalidate.txt
  ```

  **Commit**: YES | Message: `perf(search): cache warpgrep inventory by repo fingerprint` | Files: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`, `apps/orchestrator_py/src/orchestrator/services/warpgrep_cache.py`, tests

- [ ] 11. Implement model rotation on transient Codex failures (record attempt chain)

  **What to do**: Add multi-model fallback similar to “model failover” patterns.
  - In `apps/orchestrator_py/src/orchestrator/config.py` add:
    - `CODEX_MODEL_CANDIDATES` (comma-separated; default empty)
  - In `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`:
    - Extend options to hold `model_candidates: list[str]`.
    - On `_RetryableCodexError` retry path, rotate to next model candidate (if any) before re-attempt.
    - Emit artifact/log event per attempt: attempt number, model, reason, stderr_summary.
  - Update wiring: `apps/orchestrator_py/src/orchestrator/app.py` passes candidates.
  - Add unit tests for args building and attempt chain recording (using monkeypatch fake subprocess like `apps/orchestrator_py/tests/test_codex_cli_runtime.py`).

  **Must NOT do**: Do not rotate models on “bad answers”; rotate only on explicit transient failures (timeout / retryable exit codes).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: failure modes + retries.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: (none) | Blocked By: 1, 3

  **References**:
  - Retry mechanism: `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`
  - Existing retry test: `apps/orchestrator_py/tests/test_codex_cli_runtime.py`
  - OpenClaw README references model failover (concept): `https://github.com/openclaw/openclaw`

  **Acceptance Criteria**:
  - [ ] Unit test proves model candidate rotation occurs on retryable failure.
  - [ ] Attempt chain fields appear in artifact stream when artifacts enabled.

  **QA Scenarios**:
  ```
  Scenario: Retry rotates model candidates
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k model_rotation
    Expected: Exit code 0; test asserts second attempt uses next model
    Evidence: .sisyphus/evidence/task-11-rotation.txt

  Scenario: Non-retryable error does not rotate
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k non_retryable
    Expected: Exit code 0; test asserts immediate failure
    Evidence: .sisyphus/evidence/task-11-nonretryable.txt
  ```

  **Commit**: YES | Message: `feat(codex): rotate models on transient failures` | Files: `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`, `apps/orchestrator_py/src/orchestrator/config.py`, `apps/orchestrator_py/src/orchestrator/app.py`, tests

- [ ] 12. Add allowlist + DM pairing mode (safe-by-default inbound messages)

  **What to do**: Introduce OpenClaw-like “pairing” as an optional security default.
  - Add env fields in `apps/orchestrator_py/src/orchestrator/config.py`:
    - `DISCORD_DM_POLICY` = `open|pairing` (default `open` to avoid breaking current users)
    - `DISCORD_ALLOWLIST_USER_IDS` = comma-separated user IDs (default empty)
    - `DISCORD_PAIRING_STORE` = `.opendora/pairing.json` (default)
  - In `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`:
    - If DM policy is `pairing` and user not allowlisted:
      - respond with pairing code, store pending pairing with TTL.
      - accept `/pair <code>` to approve and add user to allowlist store.
    - In guild channels: if allowlist is set, require allowlisted user.
  - Add unit tests covering pairing flow with fake messages.

  **Must NOT do**: Do not store pairing secrets in git-tracked files; use `.opendora/` only.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: security + user-facing behavior.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 13 | Blocked By: 5

  **References**:
  - HITL + pending request store patterns: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`
  - OpenClaw DM pairing concept (reference): `https://github.com/openclaw/openclaw` (README “Security defaults”)

  **Acceptance Criteria**:
  - [ ] Unit tests cover: unpaired DM gets code; `/pair <code>` enables execution; wrong code rejected.
  - [ ] Allowlist is enforced when configured.

  **QA Scenarios**:
  ```
  Scenario: Pairing blocks unknown DM
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k pairing
    Expected: Exit code 0; test asserts response includes pairing code
    Evidence: .sisyphus/evidence/task-12-pairing.txt

  Scenario: Allowlisted user can run
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k allowlist
    Expected: Exit code 0; test asserts execute_with_progress called
    Evidence: .sisyphus/evidence/task-12-allowlist.txt
  ```

  **Commit**: YES | Message: `feat(security): add DM pairing and allowlist` | Files: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`, `apps/orchestrator_py/src/orchestrator/config.py`, `.env.example`, tests

- [ ] 13. Enforce tool/sandbox policy gates (deny/require HITL for risky execution)

  **What to do**: Add a small policy engine that runs server-side before Codex execution.
  - Create `apps/orchestrator_py/src/orchestrator/services/policy.py` with `PolicyDecision {allow: bool, require_hitl: bool, reason_code: str}`.
  - Inputs to policy: channel type (dm/guild), is_allowlisted, configured `CODEX_SANDBOX`, request intent (best-effort heuristic: keyword match for “delete”, “exfiltrate”, “credentials”, etc.).
  - Enforce in `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`:
    - If `CODEX_SANDBOX == 'danger-full-access'` then require HITL + allowlist; else deny.
    - If policy denies: reply with reason and do not start run.
  - Add unit tests for policy decisions.

  **Must NOT do**: Do not let the model decide sandbox level or override policy.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: security correctness.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: (none) | Blocked By: 12

  **References**:
  - Sandbox config: `apps/orchestrator_py/src/orchestrator/config.py`
  - HITL flow: `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`

  **Acceptance Criteria**:
  - [ ] Tests prove `danger-full-access` is denied without HITL+allowlist.
  - [ ] Normal `workspace-write` continues to work.

  **QA Scenarios**:
  ```
  Scenario: Dangerous sandbox denied by policy
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k policy
    Expected: Exit code 0; tests assert denial message
    Evidence: .sisyphus/evidence/task-13-policy.txt

  Scenario: Safe sandbox allowed
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k policy_safe
    Expected: Exit code 0; tests assert execution proceeds
    Evidence: .sisyphus/evidence/task-13-policy-safe.txt
  ```

  **Commit**: YES | Message: `feat(security): enforce sandbox policy gates` | Files: `apps/orchestrator_py/src/orchestrator/services/policy.py`, `apps/orchestrator_py/src/orchestrator/adapters/discord_gateway.py`, tests

- [ ] 14. Introduce `ToolRegistry` to make tools/skills extensible (thin refactor)

  **What to do**: Create a small registry abstraction over the existing `DeepAgentToolsService` tool prompts so new tools can be added without rewriting the agent loop.
  - Add `apps/orchestrator_py/src/orchestrator/services/tool_registry.py`:
    - `ToolSpec`: name, input_schema (pydantic model), output_schema (pydantic model), prompt_builder (callable), attempts default, version string.
    - `ToolRegistry`: register/get tool specs; `invoke(tool_name, input_data)` uses Codex runtime and shared JSON parsing + schema validation.
  - Refactor `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`:
    - Keep public methods (`invoke_route`, `invoke_plan`, etc.) but implement them via `ToolRegistry` to reduce duplication.
    - Add `tool_version` fields into artifact events to keep eval runs comparable.
  - Add tests ensuring:
    - registry invocation enforces schema and retries.
    - outputs match previous behavior for at least `invoke_route` and `invoke_plan`.

  **Must NOT do**: Do not change the JSON contracts returned to `DeepAgentService`; this is a refactor + extension surface, not a behavior change.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: refactor under test constraints.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 15 | Blocked By: 4

  **References**:
  - Current tool implementations: `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`
  - Codex runtime: `apps/orchestrator_py/src/orchestrator/services/codex_cli_runtime.py`
  - Deep agent call sites: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q` passes.
  - [ ] New tests verify schema enforcement and retry behavior through registry.

  **QA Scenarios**:
  ```
  Scenario: ToolRegistry preserves existing behavior
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k deep_agent_tools
    Expected: Exit code 0; existing tests pass unchanged
    Evidence: .sisyphus/evidence/task-14-registry-compat.txt

  Scenario: Registry enforces schema and retries
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k tool_registry
    Expected: Exit code 0; tests assert retry and schema validation
    Evidence: .sisyphus/evidence/task-14-registry-schema.txt
  ```

  **Commit**: YES | Message: `refactor(tools): introduce ToolRegistry for extensibility` | Files: `apps/orchestrator_py/src/orchestrator/services/tool_registry.py`, `apps/orchestrator_py/src/orchestrator/services/deep_agent_tools.py`, tests

- [ ] 15. Optional: add MCP tool adapter boundary (design + minimal PoC)

  **What to do**: Add a minimal, *off-by-default* MCP integration point so external tool servers can be attached later.
  - Decision default for this plan: implement boundary + a fake/stub adapter; do not ship a full MCP client unless explicitly enabled.
  - Add `apps/orchestrator_py/src/orchestrator/services/mcp_adapter.py`:
    - Interface: `list_tools()` → tool specs; `call_tool(name, args)` → result.
    - Provide `NullMcpAdapter` (default).
  - Add config/env fields in `apps/orchestrator_py/src/orchestrator/config.py`:
    - `MCP_ENABLED` (default `false`)
    - `MCP_SERVER_URLS` (default empty)
  - If `MCP_ENABLED=true`, ToolRegistry can register MCP-exposed tools as additional ToolSpecs (still requiring strict schema + redaction).
  - Add tests that ensure MCP is off by default and enabling it without servers fails gracefully.

  **Must NOT do**: Do not allow MCP tools to bypass sandbox policy; do not log raw MCP payloads when redaction is enabled.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: interface design + safety.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: (none) | Blocked By: 14

  **References**:
  - DeepAgents mentions MCP as an integration approach: `https://docs.langchain.com/oss/python/deepagents/overview`

  **Acceptance Criteria**:
  - [ ] With defaults, existing tests pass and MCP code path is not exercised.
  - [ ] Enabling MCP with no servers returns a clear error without crashing the orchestrator.

  **QA Scenarios**:
  ```
  Scenario: MCP is off by default
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q
    Expected: Exit code 0; no MCP-related tests fail
    Evidence: .sisyphus/evidence/task-15-mcp-default.txt

  Scenario: MCP enabled fails gracefully with no servers
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k mcp
    Expected: Exit code 0; tests assert clear error handling
    Evidence: .sisyphus/evidence/task-15-mcp-graceful.txt
  ```

  **Commit**: YES | Message: `feat(ext): add optional MCP adapter boundary` | Files: `apps/orchestrator_py/src/orchestrator/services/mcp_adapter.py`, `apps/orchestrator_py/src/orchestrator/config.py`, tests

- [ ] 16. Documentation + config tables: keep README in sync with new env vars

  **What to do**: Update `README.md` configuration table to include all newly introduced env vars (run artifacts, throttling, allowlist/pairing, model candidates, MCP).
  - Add a pytest test `apps/orchestrator_py/tests/test_docs_env_vars.py` that extracts env var field names from `AppEnv` (via `orchestrator.config.AppEnv.model_fields`) and asserts each appears in both `README.md` and `.env.example`.

  **Must NOT do**: Do not document features not implemented (e.g., pause/resume) as supported.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: docs-only.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: (none) | Blocked By: 15

  **References**:
  - Existing config table: `README.md`
  - Env definitions: `apps/orchestrator_py/src/orchestrator/config.py`
  - Example env: `.env.example`

  **Acceptance Criteria**:
  - [ ] `README.md` lists the new env vars with correct defaults.
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q` still passes.
  - [ ] New test `test_docs_env_vars.py` passes and enforces README/.env coverage.

  **QA Scenarios**:
  ```
  Scenario: README and .env.example match config
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q | tee .sisyphus/evidence/task-16-readme.txt
    Expected: Exit code 0; reviewers can cross-check env names
    Evidence: .sisyphus/evidence/task-16-readme.txt

  Scenario: No undocumented env usage
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k docs_env_vars | tee .sisyphus/evidence/task-16-env-coverage.txt
    Expected: Exit code 0; test asserts env vars are documented
    Evidence: .sisyphus/evidence/task-16-env-coverage.txt
  ```

  **Commit**: YES | Message: `docs: document new runtime and safety configuration` | Files: `README.md`, `.env.example`

- [ ] 17. Add an agent-executed baseline report command (no assertions; just evidence)

  **What to do**: Create a deterministic “baseline report” command that runs core operations on synthetic fixtures and writes a timing report to artifacts/evidence.
  - Add `apps/orchestrator_py/src/orchestrator/benchmarks/baseline_report.py` (and `__init__.py`) that:
    - creates a temp context store with N synthetic session files (e.g., 200) and measures:
      - `list_related_session_candidates` duration
      - `build_capsule` duration
    - creates a temp repo tree and measures:
      - warpgrep inventory build duration
    - prints a JSON report to stdout and writes it via ArtifactWriter when enabled.
  - Add a script entrypoint in `apps/orchestrator_py/pyproject.toml`:
    - `orchestrator-baseline = "orchestrator.benchmarks.baseline_report:run"`
  - Add a pytest test that runs the command and validates JSON schema (but does NOT assert absolute time thresholds).

  **Must NOT do**: Do not enforce time thresholds in CI (flaky). Do not require network.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: adds a reproducible measurement loop.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: (none) | Blocked By: 8, 10

  **References**:
  - Context candidate selection: `apps/orchestrator_py/src/orchestrator/services/context_offload.py`
  - Warpgrep scanning: `apps/orchestrator_py/src/orchestrator/services/deep_agent.py`

  **Acceptance Criteria**:
  - [ ] `uv run --directory apps/orchestrator_py orchestrator-baseline` prints valid JSON.
  - [ ] `uv run --directory apps/orchestrator_py run --extra dev pytest -q -k baseline_report` passes.

  **QA Scenarios**:
  ```
  Scenario: Baseline report runs and produces JSON
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py orchestrator-baseline
    Expected: Exit code 0; stdout is valid JSON report
    Evidence: .sisyphus/evidence/task-17-baseline.json

  Scenario: Baseline report test validates schema
    Tool: Bash
    Steps: uv run --directory apps/orchestrator_py run --extra dev pytest -q -k baseline_report
    Expected: Exit code 0
    Evidence: .sisyphus/evidence/task-17-baseline-test.txt
  ```

  **Commit**: YES | Message: `feat(perf): add baseline report command` | Files: `apps/orchestrator_py/src/orchestrator/benchmarks/*`, `apps/orchestrator_py/pyproject.toml`, tests

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. QA Run (tests + eval harness) — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Target: atomic commits by theme (artifacts/eval, perf, safety, extensibility).
- No commits required by this plan itself; executor decides based on repo workflow.

## Success Criteria
- Regression-resistant development loop exists: deterministic eval suite + CI.
- Debuggability: every run produces a redacted manifest + stage timings + attempt history.
- Performance: reduced Discord message spam; faster context candidate routing; fewer repeated scans.
- Security: unsafe sandbox/tool usage is gated by policy (deny/require HITL) and allowlist defaults.
