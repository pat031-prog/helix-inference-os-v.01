from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "build_hybrid_memory_figure.py"
    spec = importlib.util.spec_from_file_location("build_hybrid_memory_figure", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_hybrid_memory_summary_extracts_transformer_and_hybrid_panels() -> None:
    module = _load_module()
    transformer_summary = {
        "best_fidelity_default": {"variant": "turbo-int8-hadamard"},
        "best_compression_default": {"variant": "turbo-int8k-4bitv"},
        "models": {
            "Qwen/Qwen2.5-1.5B-Instruct": {
                "turbo_int8_hadamard": {
                    "kv_cache_ratio_vs_native": 1.96,
                    "prompt_perplexity_delta_pct_vs_native": 0.0,
                    "generated_match_vs_baseline": True,
                },
                "turbo_int8k_4bitv": {
                    "kv_cache_ratio_vs_native": 2.6,
                    "prompt_perplexity_delta_pct_vs_native": 0.78,
                    "generated_match_vs_baseline": True,
                },
            }
        },
    }
    hybrid_summary = {
        "KV-only gain": {"hybrid_total_runtime_cache_ratio_vs_native": 1.01},
        "Mamba-state-only gain": {"hybrid_total_runtime_cache_ratio_vs_native": 3.4},
        "combined hybrid gain": {
            "hybrid_total_runtime_cache_ratio_vs_native": 3.57,
            "speedup_vs_native": 1.5,
        },
        "prompt_category_aggregates": {
            "vanilla": {
                "code": {"avg_speedup_vs_native": 1.04},
                "daily": {"avg_speedup_vs_native": 0.89},
            }
        },
        "vanilla vs HXQ": {"comparison_available": False},
    }

    summary = module.build_hybrid_memory_summary(
        transformer_summary=transformer_summary,
        hybrid_summary=hybrid_summary,
    )

    assert summary["benchmark_kind"] == "hybrid-memory-frontier-summary-v1"
    assert summary["transformer_gpu"]["models"][0]["best_compression_kv_ratio_vs_native"] == 2.6
    assert summary["hybrid_local"]["combined_hybrid_gain"]["speedup_vs_native"] == 1.5


def test_build_hybrid_memory_svg_contains_key_labels() -> None:
    module = _load_module()
    summary = {
        "transformer_gpu": {
            "models": [
                {
                    "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
                    "best_fidelity_kv_ratio_vs_native": 1.96,
                    "best_compression_kv_ratio_vs_native": 2.6,
                }
            ]
        },
        "hybrid_local": {
            "kv_only_gain": {"hybrid_total_runtime_cache_ratio_vs_native": 1.01},
            "mamba_state_only_gain": {"hybrid_total_runtime_cache_ratio_vs_native": 3.4},
            "combined_hybrid_gain": {
                "hybrid_total_runtime_cache_ratio_vs_native": 3.57,
                "speedup_vs_native": 1.5,
            },
            "prompt_category_aggregates": {
                "vanilla": {
                    "code": {"avg_speedup_vs_native": 1.04},
                    "daily": {"avg_speedup_vs_native": 0.89},
                }
            },
        },
    }

    svg = module.build_hybrid_memory_svg(summary)

    assert "Helix Memory Frontier Snapshot" in svg
    assert "Transformer GPU: verified KV ratios" in svg
    assert "Hybrid Zamba2 local: total runtime-cache ratios" in svg
    assert "3.57x" in svg
