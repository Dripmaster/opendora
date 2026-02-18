from orchestrator.services.codex_cli_runtime import parse_codex_json_events


def test_smoke_event_parse_pipeline() -> None:
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"th1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"hello world"}}',
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}',
        ]
    )
    result = parse_codex_json_events(stdout)
    assert result.assistant_message == "hello world"
    assert result.usage.total == 3
