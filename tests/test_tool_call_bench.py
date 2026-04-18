from pathlib import Path

from helix_proto.tool_call_bench import (
    _extract_tool_call_tag_payload,
    _strip_think_blocks,
    _score_decision,
    _value_matches,
    default_tool_call_suite_path,
    load_tool_call_cases,
    recommend_tool_call_models,
    render_tool_call_markdown,
)


def test_tool_call_suite_loads_cases() -> None:
    cases = load_tool_call_cases(default_tool_call_suite_path())
    assert len(cases) >= 8
    assert any(len(case.steps) > 1 for case in cases)


def test_score_decision_matches_expected_tool_arguments() -> None:
    step = load_tool_call_cases(default_tool_call_suite_path())[0].steps[0]
    score = _score_decision(
        step,
        {
            "kind": "tool",
            "tool_name": "workspace.list_models",
            "arguments": {},
        },
    )
    assert score["json_valid"] is True
    assert score["step_success"] is True


def test_strip_think_blocks_removes_reasoning_wrapper() -> None:
    cleaned = _strip_think_blocks('<think>secret</think>{"kind":"tool","tool_name":"workspace.list_models","arguments":{}}')
    assert cleaned == '{"kind":"tool","tool_name":"workspace.list_models","arguments":{}}'


def test_extract_tool_call_tag_payload_supports_qwen_native_xml() -> None:
    parsed = _extract_tool_call_tag_payload(
        "<tool_call>\n<function=workspace.model_info>\n<parameter=alias>\nlegal-ft\n</parameter>\n</function>\n</tool_call>"
    )
    assert parsed == {
        "kind": "tool",
        "thought": "",
        "tool_name": "workspace.model_info",
        "arguments": {"alias": "legal-ft"},
    }


def test_value_matches_accepts_semantic_string_containment() -> None:
    assert _value_matches("PostgreSQL caído", "problema PostgreSQL caído")


def test_recommend_tool_call_models_prefers_case_success() -> None:
    recommendation = recommend_tool_call_models(
        [
            {
                "model_ref": "model-a",
                "status": "ok",
                "case_success_rate": 0.8,
                "step_success_rate": 0.9,
                "json_valid_rate": 1.0,
                "tokens_per_second": 2.0,
            },
            {
                "model_ref": "model-b",
                "status": "ok",
                "case_success_rate": 0.9,
                "step_success_rate": 0.85,
                "json_valid_rate": 0.9,
                "tokens_per_second": 1.0,
            },
        ]
    )
    assert recommendation is not None
    assert recommendation["model_ref"] == "model-b"


def test_render_tool_call_markdown_mentions_model() -> None:
    markdown = render_tool_call_markdown(
        {
            "generated_at_utc": "2026-03-24T00:00:00Z",
            "case_suite_path": str(Path("benchmarks/tool_call_cases.json")),
            "case_count": 3,
            "local_files_only": True,
            "recommendation": {
                "model_ref": "Qwen/Qwen3.5-4B",
                "case_success_rate": 0.9,
                "step_success_rate": 0.95,
                "json_valid_rate": 1.0,
                "tokens_per_second": 3.2,
            },
            "models": [
                {
                    "model_ref": "Qwen/Qwen3.5-4B",
                    "status": "ok",
                    "case_success_rate": 0.9,
                    "step_success_rate": 0.95,
                    "json_valid_rate": 1.0,
                    "tokens_per_second": 3.2,
                    "rss_peak_mb": 1234.5,
                }
            ],
        }
    )
    assert "Qwen/Qwen3.5-4B" in markdown
