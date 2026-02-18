# opendora

Discord-driven personal agent platform for safe automation:

- Natural-language chat and tasks via Discord DM or mention
- Remote Codex orchestration (plan and code)
- Context offload and retrieval for long conversations
- Deep agent with subagent dispatch

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Discord bot token
- [Codex](https://github.com/openai/codex) CLI

## Quick start

1. Copy env template and set your Discord token:

```bash
cp .env.example .env
# Edit .env: set DISCORD_BOT_TOKEN
```

2. Install dependencies:

```bash
uv sync --directory apps/orchestrator_py
```

3. Run the orchestrator:

```bash
uv run --directory apps/orchestrator_py orchestrator
```

Or via npm scripts:

```bash
npm run start
```

## Scripts

| Script        | Description            |
|---------------|------------------------|
| `npm run start` | Run orchestrator       |
| `npm run dev`   | Run orchestrator (alias) |
| `npm run test`  | Run tests              |
| `npm run check` | Run tests (alias)      |

## Configuration

| Variable                    | Default    | Description                    |
|-----------------------------|------------|--------------------------------|
| `DISCORD_BOT_TOKEN`         | (required) | Discord bot token              |
| `NATURAL_CHAT_ENABLED`      | `true`     | Enable natural-language chat   |
| `HITL_REQUIRED`             | `false`    | Require human-in-the-loop      |
| `HITL_TTL_SEC`              | `600`      | HITL confirmation TTL          |
| `DEFAULT_REPO_PATH`         | `.`        | Default repository path        |
| `CONTEXT_OFFLOAD_ENABLED`   | `true`     | Enable context offload         |
| `CONTEXT_STORE_DIR`         | `.opendora/context` | Context store path      |
| `CONTEXT_MAX_ESTIMATED_TOKENS` | `12000` | Max tokens per context         |
| `CONTEXT_KEEP_RECENT_MESSAGES` | `10`    | Recent messages to keep        |
| `CONTEXT_RETRIEVE_TOP_K`    | `4`        | Top K retrievals               |
| `DEEP_AGENT_ENABLED`        | `true`     | Enable deep agent              |
| `DEEP_AGENT_MAX_SUBAGENTS`  | `3`        | Max subagents per request      |
| `CODEX_BIN`                 | `codex`    | Codex CLI binary               |
| `CODEX_TIMEOUT_MS`          | `900000`   | Codex timeout (ms)             |
| `CODEX_MODEL`               | (empty)    | Optional Codex model override  |
| `CODEX_SANDBOX`             | `workspace-write` | Codex sandbox mode     |

## Project structure

```
opendora/
  apps/orchestrator_py/     # Python orchestrator
    src/orchestrator/
      adapters/             # Discord gateway
      services/             # Codex runtime, context offload, deep agent
    tests/
  .env.example
  package.json              # Npm scripts wrapper
```

## License

MIT. See [LICENSE](LICENSE).
