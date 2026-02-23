# Learnings

- Added `RunContext` + `ArtifactWriter` in `apps/orchestrator_py/src/orchestrator/services/run_artifacts.py` with safe path resolution that rejects `..` escapes and constrains writes to `<RUN_ARTIFACTS_DIR>/<run_id>/`.
- Artifact redaction defaults ON via `RUN_ARTIFACTS_REDACT=true`; writer masks Discord-like numeric IDs and absolute `repo_path` prefixes in text and JSON payloads.
- Bounded artifact writes are enforced as total per-run byte budget (`RUN_ARTIFACTS_MAX_BYTES`), and retention pruning removes old run dirs by mtime when a new writer is created (`RUN_ARTIFACTS_RETENTION_DAYS`).
- `RUN_DEBUG_PROMPTS=false` now suppresses raw prompt fields in JSON artifacts by replacing keys containing `prompt`; token-like keys are always redacted.
- In Deep Agent TODO validation, duplicate `id` entries must merge dependency edges in the cycle graph (set-union semantics) so cycle detection still works even when later duplicate tasks would otherwise overwrite earlier deps.
- For thread control commands, parse `todo_input_provider` inputs before `invoke_select_context` so `skip`/`abort` TODOs do not consume model context-selection calls.
- Undefined `dependsOn` references are now treated as non-fatal plan warnings at validation-node time; execution continues and dependency resolution is deferred to runtime `blocked` status handling.
- Discord gateway now creates a `RunContext` at the start of `execute_with_progress()` and injects `run_id` into every progress payload before artifact event append, so per-stage traces can be correlated with the final run summary.
- Gateway completion messages now include `runId=<id>`, and a per-run `manifest.json` is emitted with request/session/run metadata plus execution stats (`mode`, `subagent_count`, `used_offloads`, `live_messages`) and UTC start/end timestamps.
- For redaction-safe Discord replies, HITL approval preview now formats `repo=` through a display helper that hides absolute `DEFAULT_REPO_PATH` as `[REDACTED_REPO_PATH]` when artifact redaction is enabled.
- Aligned Discord control command UX by adding aliases (!skip, !stop-round, !abort) to existing slash commands.
- Updated README.md to reflect supported control commands and remove unsupported ones.
- Offline eval coverage now lives in `apps/orchestrator_py/tests/test_eval_offline.py` with six deterministic scenarios (main_direct, invalid-plan replan, skip+stop-round, abort, external-context constraints, codex runtime retry), all isolated from Discord/network.
- Deterministic artifact assertions use per-test `RunContext` + `ArtifactWriter` rooted at `tmp_path / "RUN_ARTIFACTS_DIR"`, progress events written to `events.jsonl`, and required `manifest.json` keys validated in a shared helper.
- Deep-agent timing instrumentation can stay redaction-safe by emitting only `stage`, `duration_ms`, `prompt_chars`, and optional `usage_total_tokens`; these fields survive artifact scrubbing with `RUN_DEBUG_PROMPTS=false` because they do not include raw prompt payloads.
- To cover planning/replan prompt sizing without duplicating prompt construction logic, attach an `on_metrics` callback inside `DeepAgentToolsService._run_json_with_retry` so each Codex JSON call reports prompt length and elapsed time at the actual prompt string used (including retry variants).
- `CodexRunResult` now carrying `duration_ms` and `prompt_chars` makes runtime-level timing reusable across deep-agent flows and tests; fake runtimes should return these fields to avoid interface drift.
- Discord gateway progress handler now suppresses metrics-only events (`duration_ms` + `prompt_chars` with empty `message`) from chat replies while still appending payloads to `events.jsonl`.
- Added GitHub Actions CI in `.github/workflows/ci.yml` using `astral-sh/setup-uv` to run orchestrator tests on `push` and `pull_request`.
- CI uses `uv sync --directory apps/orchestrator_py --extra dev` and `uv run --directory apps/orchestrator_py --extra dev pytest -q` for clean, secret-free verification.
- Codex runtime now supports fallback rotation with `CODEX_MODEL_CANDIDATES` (comma-separated in env, parsed in app wiring); retries advance model per attempt while preserving old behavior when candidates are empty.
- Retry telemetry is now structured for artifacts/logs via `codex.retry.attempt` payloads (`attempt`, `model`, `reason`, `stderr_summary`) and terminal failures include JSON `attempt_chain` for postmortem tracing.
- `ContextOffloadService` session persistence now uses atomic file replace (`NamedTemporaryFile` in the same directory, `flush` + `os.fsync`, then `os.replace`) to avoid exposing partially written JSON files.
- Added a per-session in-process lock and a bounded `OrderedDict` LRU cache for `_load_session`, with `CONTEXT_SESSION_CACHE_SIZE` (default 32) as the cache-size control point.
- Cache behavior is verified by spying `Path.read_text`: repeated `_load_session` calls hit memory after first read, and `_save_session` invalidates cache so the next load re-reads disk exactly once.
- Warpgrep inventory caching now works best as a two-step flow: cache only relative file paths keyed by repo fingerprint, then apply existing `max_depth`/`max_files` scoring filters per query to preserve prompt output behavior while avoiding repeated `os.walk`.

- Added per-target `ProgressThrottler` in `discord_gateway` so `subagent_event` emits at most one immediate message per throttle window, buffers intermediate events, and sends a coalesced tail (with buffered/dropped counts) on flush.
- Milestone progress stages now bypass throttling (`subagent_start`, `subagent_done`, debug prompt, and subagent error stages), while `subagent_done` first flushes any buffered progress to preserve final context ordering.
- Discord gateway now reads `DISCORD_PROGRESS_THROTTLE_MS` (default 1000) and `DISCORD_PROGRESS_MAX_BUFFERED` (default 20), and throttling tests use an injected fake clock for deterministic no-sleep verification.
Wired CONTEXT_SESSION_CACHE_SIZE from AppEnv into ContextOffloadOptions in OrchestratorApp.

- Added session index at CONTEXT_STORE_DIR/sessions_index.json with per-session lightweight routing metadata (sessionKey, updatedAt, offload_count, live_message_count, summary) to avoid full session scans.
- list_related_session_candidates now reads the index once and only falls back to reading up to retrieve_top_k session JSON files when an indexed summary is missing; same-user filtering remains enforced before candidate creation.
- Index lifecycle is resilient: missing/corrupt index triggers best-effort one-pass rebuild from sessions/*.json, then proceeds using rebuilt data.
- Added DM pairing mode controls in `discord_gateway`: `DISCORD_DM_POLICY=pairing` now blocks non-allowlisted DM execution, issues `/pair <code>` challenges with TTL-backed pending entries, and persists approvals into a `.opendora` pairing store.
- Effective allowlist now merges static env IDs and runtime-approved IDs from pairing storage, enabling guild-channel enforcement when any allowlist entries exist while preserving legacy open behavior by default.
- Added server-side execution policy gate in `orchestrator/services/policy.py` with `PolicyDecision {allow, require_hitl, reason_code}`; `danger-full-access` now enforces allowlist membership and always forces HITL, while safer sandboxes remain allowed by default.
- Gateway now evaluates policy after pending approve/reject handling but before request creation/run start, so denied requests exit early with a reason and risky-intent prompts in safe sandboxes can force HITL even when `HITL_REQUIRED=false`.

- Introduced `ToolRegistry`/`ToolSpec` (`apps/orchestrator_py/src/orchestrator/services/tool_registry.py`) so DeepAgent tools share one schema-validated JSON invocation path while preserving per-tool post-processing contracts.
- Tool metrics emitted from `DeepAgentToolsService._run_json_with_retry` now include `tool_version` (from each `ToolSpec`) alongside existing stage/duration/prompt token telemetry.
- Added focused compatibility tests in `apps/orchestrator_py/tests/test_tool_registry.py` that verify retry behavior for bad JSON/wrong-shape outputs and preserve existing `invoke_plan` fallback semantics.
- Added MCP boundary at `apps/orchestrator_py/src/orchestrator/services/mcp_adapter.py` with `NullMcpAdapter` default and `build_mcp_adapter()` mode switching, so MCP stays inert unless explicitly enabled.
- `ToolRegistry` now supports optional per-tool async handlers plus `register_mcp_tools(...)`, letting MCP tools be registered as `mcp::<name>` specs without touching the existing Codex invoker path.
- Added offline benchmark entrypoint `orchestrator-baseline` that builds synthetic fixtures (200 context sessions + temp repo tree), measures `list_related_session_candidates`, `build_capsule`, and warpgrep inventory build using `time.perf_counter()`, and emits stable JSON evidence without external API/model calls.
- Baseline report artifact integration can safely reuse `ArtifactWriter.write_bounded_text(...)`; with default redaction settings it writes under `RUN_ARTIFACTS_DIR/<run_id>/benchmarks/baseline_report.json` and avoids secret-bearing payload fields.
- `DiscordGateway.handle_message()` now evaluates `evaluate_execution_policy(...)` before `_maybe_model_route_channel(...)`, guaranteeing denied requests return immediately without model-assisted context/routing calls; and `subagent_debug_prompt` payloads are gated by `RUN_DEBUG_PROMPTS` (Discord posting disabled when false, artifacts store `[REDACTED_DEBUG_PROMPT]` placeholder instead of raw prompt text).
- `ArtifactWriter` retention now applies two best-effort passes when a run starts: TTL pruning by `RUN_ARTIFACTS_RETENTION_DAYS` plus count pruning that keeps only the newest N run directories by mtime (oldest removed first, current run directory protected).
