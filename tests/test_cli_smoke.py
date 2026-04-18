import json
from pathlib import Path

from helix_proto.cli import (
    _build_gpt_landscape_rows,
    _build_gpt_landscape_variants,
    _cmd_prepare_best_assistants,
    build_parser,
)


def test_cli_parser_accepts_benchmark_local_models() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark-local-models", "--workspace-root", "workspace"])

    assert args.command == "benchmark-local-models"
    assert args.local_files_only is True
    assert args.output == Path("benchmark-output") / "local-models"
    assert args.workspace_root == Path("workspace")


def test_cli_parser_accepts_eval_finetuned_model() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval-finetuned-model",
            "C:/tmp/finetuned-model",
            "--baseline-report",
            "baseline.json",
            "--baseline-model-ref",
            "Qwen/Qwen2.5-1.5B",
        ]
    )

    assert args.command == "eval-finetuned-model"
    assert args.model_refs == ["C:/tmp/finetuned-model"]
    assert args.baseline_model_ref == "Qwen/Qwen2.5-1.5B"


def test_cli_parser_accepts_benchmark_tool_calling() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["benchmark-tool-calling", "Qwen/Qwen2.5-0.5B", "--workspace-root", "workspace", "--limit-cases", "4"]
    )

    assert args.command == "benchmark-tool-calling"
    assert args.model_refs == ["Qwen/Qwen2.5-0.5B"]
    assert args.limit_cases == 4
    assert args.workspace_root == Path("workspace")


def test_cli_parser_accepts_benchmark_gpt_landscape() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-landscape",
            "--prompt-length",
            "96",
            "--session-prompt-lengths",
            "128",
            "512",
            "--session-summary-length",
            "512",
        ]
    )

    assert args.command == "benchmark-gpt-landscape"
    assert args.prompt_length == 96
    assert args.session_prompt_lengths == [128, 512]
    assert args.session_summary_length == 512
    assert args.kv_topk == 8
    assert args.kv_refresh_no_cache == 1
    assert args.kv_refresh_with_cache == 8
    assert args.kv_block_size == 16
    assert args.kv_layer_share_stride == 0


def test_cli_parser_accepts_layer_share_stride_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-kv-modes",
            "--kv-hot-window",
            "4",
            "--kv-topk",
            "8",
            "--kv-block-size",
            "16",
            "--kv-layer-share-stride",
            "4",
        ]
    )

    assert args.command == "benchmark-gpt-kv-modes"
    assert args.kv_hot_window == 4
    assert args.kv_topk == 8
    assert args.kv_block_size == 16
    assert args.kv_layer_share_stride == 4


def test_cli_parser_accepts_benchmark_gpt_kv_policy() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-kv-policy",
            "--kv-hot-window",
            "6",
            "--max-new-tokens",
            "3",
        ]
    )

    assert args.command == "benchmark-gpt-kv-policy"
    assert args.kv_hot_window == 6
    assert args.max_new_tokens == 3


def test_cli_parser_accepts_benchmark_transformers_kv() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-transformers-kv",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--prompt-length",
            "96",
            "--max-new-tokens",
            "4",
        ]
    )

    assert args.command == "benchmark-transformers-kv"
    assert args.model_ref == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.prompt_length == 96
    assert args.max_new_tokens == 4
    assert args.kv_hot_window == 4
    assert args.kv_quant_seed == 7
    assert args.adaptive_medium_kurtosis == 9.0
    assert args.adaptive_high_kurtosis == 20.0
    assert args.variant_set == "stable"


def test_build_gpt_landscape_variants_returns_curated_story_matrix() -> None:
    variants = _build_gpt_landscape_variants(
        kv_hot_window=4,
        kv_topk=8,
        refresh_no_cache=1,
        refresh_with_cache=8,
        kv_block_size=16,
        kv_layer_share_stride=0,
        kv_calibration_tokens=128,
    )

    names = [variant["name"] for variant in variants]

    assert len(variants) == 12
    assert names == [
        "fp32",
        "turbo-int8-qr",
        "turbo-int8-hadamard",
        "turbo-4bit",
        "adaptive",
        "turbo-4bitk-int8v",
        "turbo-int8k-4bitv",
        "turbo-int8-topk8-refresh1",
        "turbo-int8-topk8-refresh8",
        "turbo-int8-topk8-refresh8-block16",
        "turbo-4bitk-int8v-topk8-refresh8-block16",
        "turbo-int8k-4bitv-topk8-refresh8-block16",
    ]


def test_build_gpt_landscape_variants_appends_layer_share_rows_when_enabled() -> None:
    variants = _build_gpt_landscape_variants(
        kv_hot_window=4,
        kv_topk=8,
        refresh_no_cache=1,
        refresh_with_cache=8,
        kv_block_size=16,
        kv_layer_share_stride=4,
        kv_calibration_tokens=128,
    )

    names = [variant["name"] for variant in variants]

    assert len(variants) == 15
    assert names[-3:] == [
        "turbo-int8-topk8-refresh8-block16-share4",
        "turbo-4bitk-int8v-topk8-refresh8-block16-share4",
        "turbo-int8k-4bitv-topk8-refresh8-block16-share4",
    ]


def test_build_gpt_landscape_rows_preserves_policy_and_overlap_fields() -> None:
    runtime_report = {
        "baseline_variant": "fp32",
        "variants": {
            "fp32": {
                "kv_cache_precision": "fp32",
                "kv_key_precision": None,
                "kv_value_precision": None,
                "kv_rotation_mode": "qr",
                "kv_hot_window": 0,
                "kv_topk": 0,
                "kv_index_refresh_interval": 1,
                "kv_block_size": 0,
                "kv_layer_share_stride": 0,
                "prompt_perplexity": 12.5,
                "total_time_s": 10.0,
                "avg_step_ms": 5.0,
                "kv_cache_bytes": 4096,
                "generated_match_vs_baseline": True,
                "logit_comparison_vs_baseline": None,
                "kv_selective_stats": {"full_refreshes": 0},
                "kv_cross_layer_overlap": {
                    "mean_jaccard": 0.81,
                    "adjacent_pairs": [{"pair_index": 0, "mean_jaccard": 0.81, "samples": 4}],
                },
                "current_kv_mode": "fp32",
                "kv_mode_trace": ["fp32"],
                "switch_events": [],
                "policy_baseline_loss": 1.2,
                "policy_recent_loss": 1.1,
                "mode_histogram": {"fp32": 4},
                "kv_kurtosis_profile": [{"layer_index": 0, "selected_mode": "fp32"}],
            }
        },
    }
    session_report = {
        "variants": {
            "128": {
                "fp32": {
                    "session_total_bytes": 1024,
                    "session_size_ratio_vs_fp32": 1.0,
                    "kv_npz_bytes": 512,
                    "npz_size_ratio_vs_fp32": 1.0,
                    "logical_kv_cache_bytes": 4096,
                    "logical_kv_ratio_vs_fp32": 1.0,
                }
            }
        }
    }

    rows = _build_gpt_landscape_rows(runtime_report, session_report, session_summary_length=128)

    assert len(rows) == 1
    row = rows[0]
    assert row["current_kv_mode"] == "fp32"
    assert row["kv_mode_trace"] == ["fp32"]
    assert row["switch_events"] == []
    assert row["policy_baseline_loss"] == 1.2
    assert row["policy_recent_loss"] == 1.1
    assert row["mode_histogram"] == {"fp32": 4}
    assert row["kv_kurtosis_profile"] == [{"layer_index": 0, "selected_mode": "fp32"}]
    assert row["kv_cross_layer_overlap"]["adjacent_pairs"][0]["mean_jaccard"] == 0.81


def test_cli_parser_accepts_prepare_model_cdna() -> None:
    parser = build_parser()
    args = parser.parse_args(["prepare-model", "Qwen/Qwen3.5-4B", "--compress", "cdnav3"])

    assert args.command == "prepare-model"
    assert args.model_ref == "Qwen/Qwen3.5-4B"
    assert args.compress == "cdnav3"


def test_prepare_best_assistants_reuses_same_prepared_alias(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_text(
        json.dumps(
            {
                "recommendations": {
                    "general": {"model_ref": "Qwen/Qwen2.5-0.5B"},
                    "code": {"model_ref": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
                    "legal": {"model_ref": "Qwen/Qwen2.5-0.5B"},
                }
            }
        ),
        encoding="utf-8",
    )

    calls = {"prepare": [], "configure": []}

    class FakeRuntime:
        def __init__(self, *, root=None) -> None:  # noqa: ANN001
            self.root = root

        def list_models(self):  # noqa: ANN201
            return []

        def prepare_model(self, **kwargs):  # noqa: ANN003
            calls["prepare"].append(kwargs)
            return {"alias": kwargs["alias"], "model_ref": kwargs["model_ref"]}

        def configure_assistant(self, assistant_id, **kwargs):  # noqa: ANN003,ANN201
            calls["configure"].append((assistant_id, kwargs))
            return {"assistant_id": assistant_id, **kwargs}

    monkeypatch.setattr("helix_proto.cli.HelixRuntime", FakeRuntime)
    monkeypatch.setattr("helix_proto.cli._print_json", lambda payload: None)

    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare-best-assistants",
            "--report",
            str(report_path),
            "--workspace-root",
            str(tmp_path / "workspace"),
        ]
    )
    result = _cmd_prepare_best_assistants(args)

    assert result == 0
    assert len(calls["prepare"]) == 2
    assert len(calls["configure"]) == 3
