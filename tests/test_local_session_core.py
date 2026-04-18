from __future__ import annotations

from tools import run_local_session_core


def test_profile_contains_budgeted_model_lanes() -> None:
    assert run_local_session_core.PROFILE["gpt2"]["model_ref"] == "gpt2"
    assert run_local_session_core.PROFILE["qwen"]["model_ref"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert run_local_session_core.PROFILE["zamba"]["model_ref"] == "Zyphra/Zamba2-1.2B-Instruct-v2"
    assert run_local_session_core.PROFILE["qwen"]["max_new_tokens"] == 1
    assert run_local_session_core.PROFILE["zamba"]["max_new_tokens"] == 1


def test_artifact_names_are_stable() -> None:
    assert run_local_session_core._artifact_name("gpt2") == "local-session-core-gpt2.json"
    assert run_local_session_core._artifact_name("qwen") == "local-session-core-qwen.json"
    assert run_local_session_core._artifact_name("zamba") == "local-session-core-zamba.json"


def test_percentile_interpolates_microbench_values() -> None:
    assert run_local_session_core._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert run_local_session_core._percentile([], 0.5) is None


def test_parser_accepts_buffered_codec_and_transformer_only() -> None:
    args = run_local_session_core.build_parser().parse_args(
        ["--codec", "rust-hlx-buffered-flat", "--audit-policy", "deferred", "--transformer-only", "--repeats", "3"]
    )

    assert args.codec == "rust-hlx-buffered-flat"
    assert args.audit_policy == "deferred"
    assert args.transformer_only is True
    assert args.repeats == 3
