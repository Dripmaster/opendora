# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-02-23

### Added
- **Deep Agent (LangGraph) 고도화**
  - 생성된 TODO 플랜의 유효성을 검사하는 `validate_todo_plan` 로직 추가.
  - `DEEP_AGENT_MAX_ROUNDS` 설정을 통한 재계획 라운드 제어 기능.
  - 파일시스템 스캔 성능 최적화를 위한 **Warpgrep 캐시** 및 제한 설정 도입.
- **관측성 및 디버깅 (Observability)**
  - 실행 과정의 모든 이벤트를 기록하는 **Run Artifacts** 시스템 구축 (`.opendora/runs/`).
  - Discord ID, 경로, Prompt 등 민감 정보를 자동으로 마스킹하는 **Redaction** 기능.
  - 작업 종료 후 토큰 사용량 및 성공 여부를 담은 실행 요약 보고 기능.
- **보안 및 정책 (Security & Policy)**
  - 도구 실행 전 권한을 검사하는 **Policy Engine** 도입.
  - Discord 유저 페어링 및 화이트리스트 기반 접근 제어 기능.
  - 실행 결과(Artifacts) 자동 만료 및 삭제 정책 구현.
- **확장성 (Extensibility)**
  - **MCP (Model Context Protocol)** 어댑터 추가로 외부 도구 연동 기반 마련.
  - 에이전트 도구들을 통합 관리하는 **Tool Registry** 서비스 구축.
- **Discord 기능 확장**
  - 작업을 실시간으로 제어할 수 있는 명령어 추가 (`!skip`, `!stop-round`, `!abort`).
  - 개별 TODO 단위의 Discord 스레드 보고 및 진행 상황 스로틀링(Throttling) 적용.
- **CI/CD 및 개발 환경**
  - GitHub Actions 기반의 **CI 워크플로우** 추가 (자동 테스트 및 아티팩트 업로드).
  - `.env.example` 최신화 및 상세 설정 항목 추가.

### Changed
- 컨텍스트 오프로드 시점을 유저 요청 직후 및 서브에이전트 실행 전/후로 세분화하여 메모리 효율 개선.
- Codex CLI 실행 시 지수 백오프(Exponential Backoff) 기반의 자동 재시도 로직 강화.
- 프로젝트 문서(`README.md`)를 최신 기능 위주로 전면 개편.

### Fixed
- 대규모 파일시스템 스캔 시 발생하던 성능 저하 문제 해결.
- JSON 파싱 실패 시의 재시도 진단 로직 개선.
- Discord 메시지 버퍼링 및 전송 속도 최적화.
