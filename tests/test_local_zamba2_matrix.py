from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "run_local_zamba2_matrix.py"
    spec = importlib.util.spec_from_file_location("run_local_zamba2_matrix", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_prompt_runs_groups_code_and_daily() -> None:
    module = _load_module()
    prompt_runs = [
        {
            "category": "code",
            "report": {
                "rows": [
                    {"name": "native-dense", "hybrid_total_cache_bytes": 100, "speedup_vs_native": 1.0},
                    {
                        "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
                        "speedup_vs_native": 1.2,
                        "prompt_perplexity_delta_pct_vs_native": 0.5,
                        "hybrid_total_runtime_cache_bytes": 40,
                    },
                ]
            },
        },
        {
            "category": "daily",
            "report": {
                "rows": [
                    {"name": "native-dense", "hybrid_total_cache_bytes": 120, "speedup_vs_native": 1.0},
                    {
                        "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
                        "speedup_vs_native": 1.1,
                        "prompt_perplexity_delta_pct_vs_native": -0.25,
                        "hybrid_total_runtime_cache_bytes": 60,
                    },
                ]
            },
        },
    ]

    aggregates = module._aggregate_prompt_runs(prompt_runs)

    assert set(aggregates) == {"agentic", "daily"}
    assert aggregates["agentic"]["prompt_count"] == 1
    assert aggregates["agentic"]["avg_speedup_vs_native"] == 1.2
    assert aggregates["daily"]["avg_hybrid_total_runtime_cache_ratio_vs_native"] == 2.0


def test_build_summary_reports_hxq_blocked_when_no_hxq_smoke() -> None:
    module = _load_module()
    vanilla_smoke = {
        "rows": [
            {"name": "native-dense", "total_time_s": 10.0, "hybrid_total_cache_bytes": 100},
            {
                "name": "turbo-int8-hadamard",
                "speedup_vs_native": 1.1,
                "kv_cache_ratio_vs_native": 2.5,
                "prompt_perplexity_delta_pct_vs_native": 0.0,
                "generated_match_vs_baseline": True,
                "hybrid_total_runtime_cache_bytes": 80,
            },
        ]
    }
    qmamba_smoke = {
        "rows": [
            {"name": "native-dense", "hybrid_total_cache_bytes": 100},
            {
                "name": "q-mamba-dsq-int4",
                "speedup_vs_native": 1.05,
                "kv_cache_ratio_vs_native": 1.0,
                "mamba_state_runtime_ratio_vs_native": 3.0,
                "prompt_perplexity_delta_pct_vs_native": 0.25,
                "generated_match_vs_baseline": True,
                "hybrid_total_runtime_cache_bytes": 50,
            },
            {
                "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
                "speedup_vs_native": 1.2,
                "kv_cache_ratio_vs_native": 2.5,
                "mamba_state_runtime_ratio_vs_native": 3.0,
                "prompt_perplexity_delta_pct_vs_native": 0.5,
                "generated_match_vs_baseline": True,
                "hybrid_total_runtime_cache_bytes": 40,
            },
        ]
    }
    prompt_suite = {"aggregates": {"vanilla": {"code": {"prompt_count": 2}}}}
    hxq_diagnostics = {"logits_finite": False, "load_error": None, "forward_error": None}

    summary = module._build_summary(
        vanilla_smoke=vanilla_smoke,
        qmamba_smoke=qmamba_smoke,
        prompt_suite=prompt_suite,
        hxq_diagnostics=hxq_diagnostics,
        hxq_smoke=None,
    )

    assert summary["KV-only gain"]["kv_cache_ratio_vs_native"] == 2.5
    assert summary["Mamba-state-only gain"]["mamba_state_runtime_ratio_vs_native"] == 3.0
    assert summary["combined hybrid gain"]["hybrid_total_runtime_cache_ratio_vs_native"] == 2.5
    assert summary["vanilla vs HXQ"]["comparison_available"] is False


def test_augment_with_decoded_suffixes_keeps_raw_and_clean_generation_views() -> None:
    module = _load_module()

    class _DummyDecoder:
        def decode(self, token_ids, *, skip_special_tokens: bool = True):  # type: ignore[no-untyped-def]
            prefix = "clean" if skip_special_tokens else "raw"
            return f"{prefix}:{','.join(str(token_id) for token_id in token_ids)}"

    report = {
        "prompt_ids": [10, 11],
        "benchmark_prompt_text": "hello world",
        "rows": [
            {"name": "native-dense", "generated_ids": [10, 11, 12, 13]},
            {"name": "combined", "generated_ids": [10, 11, 21]},
        ],
    }

    compact = module._augment_with_decoded_suffixes(report, decoder=_DummyDecoder())

    assert compact["benchmark_prompt_text"] == "hello world"
    assert compact["generated_ids_by_variant"]["native-dense"] == [10, 11, 12, 13]
    assert compact["generated_suffix_ids_by_variant"]["native-dense"] == [12, 13]
    assert compact["decoded_suffix_by_variant"]["native-dense"] == "clean:12,13"
    assert compact["decoded_suffix_raw_by_variant"]["native-dense"] == "raw:12,13"
    assert compact["decoded_full_text_by_variant"]["combined"] == "clean:10,11,21"
    assert compact["decoded_full_text_raw_by_variant"]["combined"] == "raw:10,11,21"


def test_run_local_zamba2_matrix_budgeted_profile_writes_compact_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    def _fake_benchmark(model_ref, **kwargs):  # type: ignore[no-untyped-def]
        variant_order = [str(item["name"]) for item in kwargs["kv_variants"]]
        receipts_path = None
        for item in kwargs["kv_variants"]:
            if item.get("mamba_receipts_path"):
                receipts_path = Path(str(item["mamba_receipts_path"]))
                receipts_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(receipts_path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps({"run_id": "fake", "receipt_hash": "abc", "prev_hash": "0" * 64}) + "\n")
                break
        rows = [
            {
                "name": "native-dense",
                "speedup_vs_native": 1.0,
                "prompt_perplexity_delta_pct_vs_native": 0.0,
                "hybrid_total_cache_bytes": 100,
                "hybrid_total_runtime_cache_bytes": 100,
                "generated_match_vs_baseline": True,
                "generated_ids": [1, 2, 3],
                "kv_cache_ratio_vs_native": 1.0,
                "mamba_state_runtime_ratio_vs_native": 1.0,
                "mamba_state_fallback_counts": {"int4": 0, "int8": 0, "dense": 6},
                "mamba_receipt_count": 0,
            }
        ]
        if "q-mamba-dsq-int4" in variant_order:
            rows.append(
                {
                    "name": "q-mamba-dsq-int4",
                    "speedup_vs_native": 1.1,
                    "prompt_perplexity_delta_pct_vs_native": 0.0,
                    "hybrid_total_cache_bytes": 100,
                    "hybrid_total_runtime_cache_bytes": 30,
                    "generated_match_vs_baseline": True,
                    "generated_ids": [1, 2, 3],
                    "kv_cache_ratio_vs_native": 1.0,
                    "mamba_state_runtime_ratio_vs_native": 3.3,
                    "mamba_state_fallback_counts": {"int4": 4, "int8": 2, "dense": 0},
                    "mamba_receipt_count": 6,
                }
            )
        if "turbo-int8-hadamard+q-mamba-dsq-int4" in variant_order:
            rows.append(
                {
                    "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
                    "speedup_vs_native": 1.2,
                    "prompt_perplexity_delta_pct_vs_native": 0.5,
                    "hybrid_total_cache_bytes": 100,
                    "hybrid_total_runtime_cache_bytes": 25,
                    "generated_match_vs_baseline": True,
                    "generated_ids": [1, 2, 3],
                    "kv_cache_ratio_vs_native": 1.8,
                    "mamba_state_runtime_ratio_vs_native": 3.3,
                    "mamba_state_fallback_counts": {"int4": 4, "int8": 2, "dense": 0},
                    "mamba_receipt_count": 6,
                }
            )
        if "turbo-int8-hadamard" in variant_order:
            rows.append(
                {
                    "name": "turbo-int8-hadamard",
                    "speedup_vs_native": 1.05,
                    "prompt_perplexity_delta_pct_vs_native": 0.2,
                    "hybrid_total_cache_bytes": 100,
                    "hybrid_total_runtime_cache_bytes": 90,
                    "generated_match_vs_baseline": True,
                    "generated_ids": [1, 2, 3],
                    "kv_cache_ratio_vs_native": 1.5,
                    "mamba_state_runtime_ratio_vs_native": 1.0,
                    "mamba_state_fallback_counts": {"int4": 0, "int8": 0, "dense": 6},
                    "mamba_receipt_count": 0,
                }
            )
        return {
            "model_ref": str(model_ref),
            "device": kwargs.get("device", "cpu"),
            "prompt_ids": [1, 2],
            "prompt_length": 2,
            "max_new_tokens": kwargs.get("max_new_tokens", 1),
            "variant_order": variant_order,
            "rows": rows,
            "variants": {row["name"]: dict(row) for row in rows},
            "benchmark_prompt_text": kwargs.get("prompt_text"),
        }

    monkeypatch.setattr(module, "run_transformers_kv_benchmark", _fake_benchmark)
    monkeypatch.setattr(
        module,
        "run_transformers_model_diagnostics",
        lambda *args, **kwargs: {  # type: ignore[no-untyped-def]
            "requested_model_ref": "EchoLabs33/zamba2-1.2b-hxq",
            "effective_model_ref": "EchoLabs33/zamba2-1.2b-hxq",
            "logits_finite": False,
            "load_error": None,
            "forward_error": None,
            "nan_count": 8,
            "inf_count": 0,
            "model_dtype": "float32",
            "model_device": "cpu",
            "weight_runtime_source": "pypi",
        },
    )
    monkeypatch.setattr(module, "_load_decoder", lambda *args, **kwargs: None)

    outputs = module.run_local_zamba2_matrix(
        argparse.Namespace(
            device="cpu",
            output_dir=tmp_path,
            public_local_profile="laptop-12gb",
            budgeted_local=True,
            vanilla_model_ref=None,
            hxq_model_ref=None,
            prompt_length=None,
            qualitative_max_new_tokens=None,
            kv_quant_seed=None,
            kv_hot_window=None,
            local_files_only=True,
            trust_remote_code=False,
        )
    )

    summary = json.loads((tmp_path / "local-zamba2-stability-summary.json").read_text(encoding="utf-8"))
    aggregates = json.loads((tmp_path / "local-zamba2-agentic-vs-daily.json").read_text(encoding="utf-8"))
    hxq = json.loads((tmp_path / "local-zamba2-hxq-finiteness.json").read_text(encoding="utf-8"))

    assert outputs["state_receipts"].name == "local-zamba2-state-receipts.jsonl.gz"
    assert summary["profile"] == "laptop-12gb"
    assert summary["local_budget"]["budgeted_local"] is True
    assert aggregates["aggregates"]["vanilla"]["agentic"]["prompt_count"] == 2
    assert hxq["logits_finite"] is False
