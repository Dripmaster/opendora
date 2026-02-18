from orchestrator.services.codex_cli_runtime import parse_codex_json_events


def test_parse_codex_json_events_extracts_text_and_usage() -> None:
    out = "\n".join(
        [
            "noise",
            '{"type":"thread.started","thread_id":"t-1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}',
            '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":5,"total_tokens":17}}',
        ]
    )
    parsed = parse_codex_json_events(out)
    assert parsed.thread_id == "t-1"
    assert parsed.assistant_message == "hello"
    assert parsed.usage.prompt == 12
    assert parsed.usage.completion == 5
    assert parsed.usage.total == 17


def test_parse_codex_json_events_handles_missing_usage() -> None:
    out = '{"type":"turn.completed"}'
    parsed = parse_codex_json_events(out)
    assert parsed.usage.prompt == 0
    assert parsed.usage.completion == 0
    assert parsed.usage.total == 0
