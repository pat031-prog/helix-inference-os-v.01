from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "run_hybrid_memory_sweep.py"
    spec = importlib.util.spec_from_file_location("run_hybrid_memory_sweep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_context_summary_reads_all_four_hybrid_variants() -> None:
    module = _load_module()
    report = {
        "rows": [
            {"name": "native-dense", "hybrid_total_runtime_cache_bytes": 100},
            {
                "name": "turbo-int8-hadamard",
                "hybrid_total_runtime_cache_ratio_vs_native": 1.05,
                "kv_cache_ratio_vs_native": 1.6,
            },
            {
                "name": "q-mamba-dsq-int4",
                "hybrid_total_runtime_cache_ratio_vs_native": 3.4,
                "mamba_state_runtime_ratio_vs_native": 3.76,
            },
            {
                "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
                "hybrid_total_runtime_cache_ratio_vs_native": 3.57,
                "speedup_vs_native": 1.5,
                "prompt_perplexity_delta_pct_vs_native": 6.4,
                "generated_match_vs_baseline": True,
                "mamba_state_runtime_ratio_vs_native": 3.76,
            },
        ]
    }

    summary = module._extract_context_summary(report)

    assert summary["native_dense_runtime_cache_bytes"] == 100
    assert summary["kv_only_runtime_cache_ratio_vs_native"] == 1.05
    assert summary["state_only_mamba_state_ratio_vs_native"] == 3.76
    assert summary["combined_speedup_vs_native"] == 1.5


def test_aggregate_sweep_runs_picks_best_combined_context_per_model() -> None:
    module = _load_module()
    runs = [
        {
            "model_ref": "Zyphra/Zamba2-1.2B-Instruct-v2",
            "prompt_length": 16,
            "summary": {"combined_runtime_cache_ratio_vs_native": 3.2, "combined_speedup_vs_native": 1.1},
        },
        {
            "model_ref": "Zyphra/Zamba2-1.2B-Instruct-v2",
            "prompt_length": 128,
            "summary": {"combined_runtime_cache_ratio_vs_native": 3.6, "combined_speedup_vs_native": 1.3},
        },
    ]

    aggregate = module._aggregate_sweep_runs(runs)

    assert aggregate["Zyphra/Zamba2-1.2B-Instruct-v2"]["best_combined_context"]["prompt_length"] == 128
    assert aggregate["Zyphra/Zamba2-1.2B-Instruct-v2"]["contexts"][0]["prompt_length"] == 16
