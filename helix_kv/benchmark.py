from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from helix_kv.cache import CompressedKVCache
from helix_kv.config import KVConfig
from helix_kv.policy import AdaptiveKVPolicy
from helix_kv.transformers_cache import (
    build_gpu_transformers_variants,
    build_transformers_asymmetry_sweep_variants,
    build_transformers_community_variants,
    run_transformers_kv_benchmark,
)


_PUBLISHED_CONTEXT = {
    "kivi": {
        "citation": "Liu et al., KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (arXiv:2402.02750, ICML 2024)",
        "regime": "2-bit asymmetric KV cache quantization on Llama/Falcon/Mistral families",
        "reported_metrics": {
            "peak_memory_reduction_including_weights": "2.6x",
            "throughput_gain_real_workloads": "2.35x-3.47x",
        },
        "comparability_note": (
            "This context comes from the KIVI paper abstract and is not an apples-to-apples comparison with "
            "Helix results unless the model family, context length, and metric definitions match closely."
        ),
        "source": "https://arxiv.org/abs/2402.02750",
    },
    "q_mamba": {
        "citation": "Yang et al., Q-Mamba: On First Exploration of Post-Training Quantization for Mamba (Findings of ACL 2025)",
        "regime": "Decoupled-scale quantization for Mamba state-space models, including H4 recurrent-state compression",
        "reported_metrics": {
            "memory_reduction": "50%",
            "average_accuracy_degradation": "2.13%",
        },
        "comparability_note": (
            "This context informs the hybrid recurrent-state direction. It is not directly comparable with Helix "
            "runtime numbers unless the same architecture family, state representation, and evaluation protocol are used."
        ),
        "source": "https://aclanthology.org/2025.findings-acl.551/",
    },
    "zamba2": {
        "citation": "Hugging Face Transformers Zamba2 docs",
        "regime": "Hybrid Mamba2-Transformer decoder with explicit hybrid cache support and layers_block_type metadata",
        "reported_metrics": {},
        "comparability_note": (
            "This is the architectural reference used for Helix hybrid cache routing rather than a baseline performance paper."
        ),
        "source": "https://huggingface.co/docs/transformers/en/model_doc/zamba2",
    }
}


def _directory_size_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _logit_comparison(current: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float32).reshape(-1)
    diff = np.abs(current - baseline)
    denom = float(np.linalg.norm(current) * np.linalg.norm(baseline))
    cosine = float(np.dot(current, baseline) / denom) if denom > 0.0 else 0.0
    return {
        "cosine_similarity": cosine,
        "max_abs_err": float(np.max(diff)),
        "mean_abs_err": float(np.mean(diff)),
    }


def published_benchmark_context() -> dict[str, Any]:
    return dict(_PUBLISHED_CONTEXT)


def build_transformers_variant_set(
    variant_set: str,
    *,
    kv_quant_seed: int = 7,
    kv_hot_window: int = 4,
    kv_calibration_tokens: int = 128,
    kv_adaptive_medium_kurtosis: float = 9.0,
    kv_adaptive_high_kurtosis: float = 20.0,
) -> list[dict[str, Any]]:
    name = str(variant_set).strip().lower()
    if name == "stable":
        return build_gpu_transformers_variants(
            kv_quant_seed=kv_quant_seed,
            kv_hot_window=kv_hot_window,
            kv_calibration_tokens=kv_calibration_tokens,
            kv_adaptive_medium_kurtosis=kv_adaptive_medium_kurtosis,
            kv_adaptive_high_kurtosis=kv_adaptive_high_kurtosis,
        )
    if name == "asymmetry-sweep":
        return build_transformers_asymmetry_sweep_variants(
            kv_quant_seed=kv_quant_seed,
            kv_hot_window=kv_hot_window,
            kv_calibration_tokens=kv_calibration_tokens,
            kv_adaptive_medium_kurtosis=kv_adaptive_medium_kurtosis,
            kv_adaptive_high_kurtosis=kv_adaptive_high_kurtosis,
        )
    if name == "community":
        return build_transformers_community_variants(
            kv_quant_seed=kv_quant_seed,
            kv_hot_window=kv_hot_window,
            kv_calibration_tokens=kv_calibration_tokens,
            kv_adaptive_medium_kurtosis=kv_adaptive_medium_kurtosis,
            kv_adaptive_high_kurtosis=kv_adaptive_high_kurtosis,
        )
    raise ValueError(f"unsupported transformers variant set: {variant_set}")


def run_kv_landscape(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    kv_variants: list[dict[str, Any]],
    kv_quant_seed: int = 7,
) -> dict[str, Any]:
    from helix_proto.hf import benchmark_gpt2_kv_mode_matrix

    return benchmark_gpt2_kv_mode_matrix(
        Path(export_dir),
        prompt_ids=prompt_ids,
        max_new_tokens=int(max_new_tokens),
        kv_variants=kv_variants,
        kv_quant_seed=int(kv_quant_seed),
    )


def run_adaptive_policy_benchmark(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    kv_quant_seed: int = 7,
    hot_window: int = 4,
    session_root: str | Path | None = None,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    variants = [
        {"name": "static-fp32", "config": KVConfig(mode="fp32", hot_window=hot_window)},
        {
            "name": "static-turbo-int8-hadamard",
            "config": KVConfig(mode="turbo-int8-hadamard", hot_window=hot_window),
        },
        {"name": "static-turbo-4bit", "config": KVConfig(mode="turbo-4bit", hot_window=hot_window)},
        {
            "name": "adaptive-policy",
            "config": KVConfig(mode="turbo-4bit", hot_window=hot_window),
            "policy": AdaptiveKVPolicy(),
            "phase": "benchmark",
            "allowed_modes": ("turbo-4bit", "turbo-int8-hadamard", "fp32"),
        },
    ]

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if session_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="helix-kv-policy-")
        session_root_path = Path(temp_dir.name)
    else:
        session_root_path = Path(session_root)
        session_root_path.mkdir(parents=True, exist_ok=True)

    try:
        results: dict[str, Any] = {}
        baseline_logits: np.ndarray | None = None
        baseline_ids: list[int] | None = None
        baseline_session_size: int | None = None
        baseline_kv_bytes: int | None = None

        for variant in variants:
            cache = CompressedKVCache(
                export_dir,
                variant["config"],
                cache_mode="session",
                kv_quant_seed=int(kv_quant_seed),
            )
            policy = variant.get("policy")
            if isinstance(policy, AdaptiveKVPolicy):
                cache.set_policy(
                    policy,
                    phase=str(variant.get("phase", "benchmark")),
                    allowed_modes=variant.get("allowed_modes"),
                )
            run = cache.generate(prompt_ids, max_new_tokens=int(max_new_tokens))
            session_dir = session_root_path / str(variant["name"])
            cache.save(session_dir)
            session_total_bytes = _directory_size_bytes(session_dir)

            metrics = {
                "current_kv_mode": run.get("current_kv_mode"),
                "kv_mode_trace": list(run.get("kv_mode_trace") or []),
                "switch_events": list(run.get("switch_events") or []),
                "switch_count": len(run.get("switch_events") or []),
                "policy_baseline_loss": run.get("policy_baseline_loss"),
                "policy_recent_loss": run.get("policy_recent_loss"),
                "mode_histogram": dict(run.get("mode_histogram") or {}),
                "total_time_s": float(run["total_time_s"]),
                "avg_step_ms": float(run["avg_step_ms"]),
                "kv_cache_bytes": int(run["kv_cache_bytes"]),
                "session_total_bytes": session_total_bytes,
                "generated_ids": list(run["generated_ids"]),
            }
            current_last_logits = np.asarray(run["last_logits"], dtype=np.float32)
            if baseline_logits is None:
                baseline_logits = current_last_logits
                baseline_ids = list(run["generated_ids"])
                baseline_session_size = session_total_bytes
                baseline_kv_bytes = int(run["kv_cache_bytes"])
                metrics["generated_match_vs_baseline"] = True
                metrics["logit_comparison_vs_baseline"] = None
                metrics["speedup_vs_fp32"] = 1.0
                metrics["kv_cache_ratio_vs_fp32"] = 1.0
                metrics["session_ratio_vs_fp32"] = 1.0
            else:
                assert baseline_ids is not None
                assert baseline_session_size is not None
                assert baseline_kv_bytes is not None
                baseline_time = float(results["static-fp32"]["total_time_s"])
                metrics["generated_match_vs_baseline"] = list(run["generated_ids"]) == baseline_ids
                metrics["logit_comparison_vs_baseline"] = _logit_comparison(current_last_logits, baseline_logits)
                metrics["speedup_vs_fp32"] = (
                    baseline_time / float(run["total_time_s"]) if float(run["total_time_s"]) else float("inf")
                )
                metrics["kv_cache_ratio_vs_fp32"] = (
                    float(baseline_kv_bytes) / float(run["kv_cache_bytes"])
                    if int(run["kv_cache_bytes"])
                    else 0.0
                )
                metrics["session_ratio_vs_fp32"] = (
                    float(baseline_session_size) / float(session_total_bytes) if session_total_bytes else 0.0
                )
            results[str(variant["name"])] = metrics

        return {
            "baseline_variant": "static-fp32",
            "prompt_ids": list(prompt_ids),
            "max_new_tokens": int(max_new_tokens),
            "kv_quant_seed": int(kv_quant_seed),
            "variants": results,
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
