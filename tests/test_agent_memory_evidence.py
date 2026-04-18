from __future__ import annotations

from tools import run_local_agent_memory_evidence as evidence


def test_percentile_and_speedup_helpers_are_stable() -> None:
    assert evidence._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert evidence._percentile([], 0.5) is None
    assert evidence._safe_ratio(10.0, 2.0) == 5.0
    assert evidence._safe_ratio(10.0, 0.0) is None


def test_model_summary_computes_ttft_speedups() -> None:
    rows = [
        {
            "cold_ttft_ms": 100.0,
            "warm_ttft_including_restore_ms": 25.0,
            "warm_compute_only_ms": 10.0,
            "session_load_time_ms": 15.0,
            "time_to_pending_ms": 5.0,
            "time_to_verified_ms": 30.0,
            "session_total_bytes": 1000,
            "generated_ids_match": True,
            "top1_match": True,
            "max_abs_logit_delta": 0.0,
            "mean_abs_logit_delta": 0.0,
            "finite_cold": True,
            "finite_warm": True,
        },
        {
            "cold_ttft_ms": 120.0,
            "warm_ttft_including_restore_ms": 30.0,
            "warm_compute_only_ms": 12.0,
            "session_load_time_ms": 18.0,
            "time_to_pending_ms": 6.0,
            "time_to_verified_ms": 36.0,
            "session_total_bytes": 1200,
            "generated_ids_match": True,
            "top1_match": True,
            "max_abs_logit_delta": 0.0,
            "mean_abs_logit_delta": 0.0,
            "finite_cold": True,
            "finite_warm": True,
        },
    ]

    summary = evidence._summarize_model_rows(rows)

    assert summary["cold_ttft_ms_p50"] == 110.0
    assert summary["warm_ttft_including_restore_ms_p50"] == 27.5
    assert summary["ttft_speedup_including_restore"] == 4.0
    assert summary["ttft_speedup_compute_only"] == 10.0
    assert summary["generated_ids_match"] is True


def test_capacity_rows_include_measured_and_projected_entries() -> None:
    summary = {
        "models": [
            {
                "status": "completed",
                "model_key": "gpt2",
                "model_ref": "gpt2",
                "mode": "turbo-int8-hadamard",
                "prefix_token_count": 128,
                "session_total_bytes_p50": 1024,
            }
        ]
    }

    rows = evidence.build_capacity_rows(summary, budget_bytes=4096, projected_context_tokens=512)

    assert len(rows) == 2
    assert rows[0]["projection"] is False
    assert rows[0]["agents_fit"] == 4
    assert rows[1]["projection"] is True
    assert rows[1]["projection_method"] == "linear_from_measured_context"
    assert rows[1]["per_agent_cache_bytes"] == 4096
    assert rows[1]["agents_fit"] == 1


def test_parser_defaults_match_evidence_sprint() -> None:
    args = evidence.build_parser().parse_args([])

    assert args.models == "gpt2,qwen"
    assert args.prefix_tokens == 128
    assert args.followup_tokens == 16
    assert args.repeats == 5
    assert args.codec == "rust-hlx-buffered-flat"
    assert args.audit_policy == "deferred"
