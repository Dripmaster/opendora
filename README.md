# opendora

**Discord에서 돌리는 개인용 Codex 오케스트레이터.**  
DM이나 멘션으로 말 걸면, 자연어 요청을 TODO로 쪼개고 Codex로 실행한 뒤 결과를 다시 Discord로 돌려준다.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

### Discord 연동

- **DM / 멘션** – 봇에게 DM을 보내거나 채널에서 멘션하면 요청으로 인식한다.
- **세션 단위 처리** – 채널·유저별로 세션을 나누어 대화를 유지한다.
- **채널·카테고리 자동 생성 및 라우팅** – 주제가 바뀌면 모델이 판단해 **새 카테고리·채널을 자동 생성**하고, 해당 채널로 대화를 이관한다. 없으면 카테고리를 만들고, 그 안에 채널을 만들어 라우팅한다.
- **컨텍스트 로테이션** – 오프로드가 많아지면 **active / archive 카테고리**를 사용해 새 채널을 만들고, 기존 채널은 아카이브로 옮긴 뒤 새 대화는 새 채널에서 이어간다.
- **HITL(선택)** – 위험도가 높다고 판단되면 승인 대기 후 실행할 수 있다.
- **서브에이전트 스레드** – TODO 하나마다 Discord 스레드를 만들어 진행 상황을 따로 보고한다.
- **스레드 제어 명령** – `!pause`, `!resume`, `!stop` 명령으로 진행 중인 작업을 제어할 수 있다.
- **메모** – 채널 토픽에 저장된 메모를 조회하는 명령을 지원한다.

### Codex 오케스트레이션

- **원격 실행** – 로컬에 설치된 Codex CLI를 호출해 실제 코드 실행을 맡긴다.
- **샌드박스** – `read-only` / `workspace-write` / `danger-full-access` 중 선택 가능하다.
- **타임아웃** – 실행 시간 상한을 두어 무한 대기를 막는다.
- **자동 재시도** – 일시적 실패 시 지수 백오프로 자동 재시도한다.

### 컨텍스트 오프로드

- **대화 압축** – 긴 대화를 요약·오프로드해 토큰 수를 제한 안쪽으로 유지한다.
- **검색** – 과거 오프로드와 최근 메시지 중에서 질의와 관련된 부분만 골라 컨텍스트로 쓴다.
- **캡슐** – 요청 처리 시 “오프로드된 컨텍스트 + 최근 대화” 조합을 한 번에 만들어 Codex에 넘긴다.

### Deep Agent (LangGraph)

- **모드 라우팅** – 요청이 단순하면 바로 답변, 복잡하면 TODO 플랜을 세운다.
- **TODO 계획 검증** – 생성된 TODO 플랜이 유효한지 검증하고, 잘못된 경우 재생성한다.
- **TODO 계획** – 자연어 요청을 의존 관계가 있는 TODO 리스트로 분해한다.
- **서브에이전트** – TODO 단위로 Codex를 호출하고, 결과를 모아 다음 단계나 최종 답변을 만든다.
- **Warpgrep 도구** – 파일시스템 검색을 위한 Warpgrep 도구 노드를 제공한다.
- **재계획** – 중간 결과를 보고 추가 TODO가 필요하면 다음 라운드를 돌린다.
- **다중 라운드** – 설정한 최대 라운드까지 반복해 목표를 채운다.

---

## Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (권장) 또는 pip
- **Discord 봇 토큰** – [Discord Developer Portal](https://discord.com/developers/applications)에서 발급
- **[Codex](https://github.com/openai/codex)** CLI – 로컬에 설치되어 PATH에 있어야 함

---

## Quick start

**1. 환경 변수**

```bash
cp .env.example .env
# .env 에서 DISCORD_BOT_TOKEN 설정
```

**2. 의존성 설치**

```bash
uv sync --directory apps/orchestrator_py
```

**3. 실행**

```bash
uv run --directory apps/orchestrator_py orchestrator
```

봇이 올라오면 Discord에서 DM 또는 멘션으로 요청을 보내면 된다.

---

## Commands

| 명령 | 설명 |
|------|------|
| `uv run --directory apps/orchestrator_py orchestrator` | 오케스트레이터 실행 |
| `uv run --directory apps/orchestrator_py run --extra dev pytest -q` | 테스트 실행 |

---

## Configuration

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DISCORD_BOT_TOKEN` | (필수) | Discord 봇 토큰 |
| `NATURAL_CHAT_ENABLED` | `true` | 자연어 채팅 처리 사용 여부 |
| `HITL_REQUIRED` | `false` | 위험 작업 시 사람 승인 필수 여부 |
| `HITL_TTL_SEC` | `600` | 승인 대기 유효 시간(초) |
| `DEFAULT_REPO_PATH` | `.` | 기본 작업 디렉터리(레포 경로) |
| `CONTEXT_OFFLOAD_ENABLED` | `true` | 컨텍스트 오프로드 사용 여부 |
| `CONTEXT_STORE_DIR` | `.opendora/context` | 오프로드 저장 디렉터리 |
| `CONTEXT_MAX_ESTIMATED_TOKENS` | `12000` | 컨텍스트당 최대 토큰 수(추정) |
| `CONTEXT_KEEP_RECENT_MESSAGES` | `10` | 유지할 최근 메시지 개수 |
| `CONTEXT_RETRIEVE_TOP_K` | `4` | 검색 시 가져올 오프로드 개수 |
| `CONTEXT_CHANNEL_ROUTER_ENABLED` | `true` | 모델 기반 채널·카테고리 라우팅(자동 생성) 사용 여부 |
| `CONTEXT_CHANNEL_ROTATION_ENABLED` | `false` | 오프로드 많을 때 채널 로테이션(active/archive) 사용 여부 |
| `CONTEXT_CHANNEL_ROTATION_MAX_OFFLOADS` | `24` | 로테이션 트리거 오프로드 개수 |
| `CONTEXT_CHANNEL_ROTATION_MAX_LIVE_MESSAGES` | `80` | 로테이션 트리거 최대 라이브 메시지 수 |
| `CONTEXT_ACTIVE_CATEGORY_NAME` | `opendora-active` | 로테이션 시 새 채널을 둘 카테고리 이름 |
| `CONTEXT_ARCHIVE_CATEGORY_NAME` | `opendora-archive` | 로테이션 시 기존 채널을 옮길 아카이브 카테고리 이름 |
| `DEEP_AGENT_ENABLED` | `true` | Deep Agent(TODO·서브에이전트) 사용 여부 |
| `DEEP_AGENT_MAX_SUBAGENTS` | `3` | 한 번에 둘 수 있는 서브에이전트(라운드당) 상한 |
| `DEEP_AGENT_MAX_ROUNDS` | `3` | Deep Agent 재계획 최대 라운드 수(1 이상) |
| `WARPGREP_MAX_FILES` | `200` | Warpgrep 파일시스템 스캔 최대 파일 수 |
| `WARPGREP_MAX_DEPTH` | `4` | Warpgrep 파일시스템 스캔 최대 깊이 |
| `CODEX_BIN` | `codex` | Codex CLI 실행 파일 이름 |
| `CODEX_TIMEOUT_MS` | `900000` | Codex 실행 타임아웃(ms) |
| `CODEX_MODEL` | (비어 있음) | Codex 모델 오버라이드(선택) |
| `CODEX_SANDBOX` | `workspace-write` | Codex 샌드박스 모드 |
| `CODEX_RETRY_COUNT` | `2` | Codex 일시적 실패 시 재시도 횟수 |
| `CODEX_RETRY_BACKOFF_MS` | `250` | 재시도 백오프 기본 간격(ms) |

---

## Project structure

```
opendora/
  apps/orchestrator_py/
    src/orchestrator/
      adapters/          # Discord 연동 (discord_gateway)
      services/          # Codex 런타임, 컨텍스트 오프로드, Deep Agent·도구
    tests/
  .env.example
```

---

## License

MIT. See [LICENSE](LICENSE).
