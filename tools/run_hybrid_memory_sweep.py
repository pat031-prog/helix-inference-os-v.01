from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from helix_kv.transformers_cache import run_transformers_kv_benchmark


DEFAULT_SWEEP_PROMPT = (
    "Summarize why hybrid Mamba-Transformer inference needs separate accounting for transformer KV cache "
    "and recurrent state memory."
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _hybrid_memory_variants(*, kv_quant_seed: int, kv_hot_window: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "native-dense",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
            "mamba_state_precision": "native-dense",
        },
        {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "mamba_state_precision": "native-dense",
        },
        {
            "name": "q-mamba-dsq-int4",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
            "mamba_state_precision": "q-mamba-dsq-int4",
        },
        {
            "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "mamba_state_precision": "q-mamba-dsq-int4",
        },
    ]


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_ref": report.get("model_ref"),
        "device": report.get("device"),
        "prompt_length": report.get("prompt_length"),
        "max_new_tokens": report.get("max_new_tokens"),
        "variant_order": report.get("variant_order"),
        "benchmark_prompt_text": report.get("benchmark_prompt_text"),
        "rows": report.get("rows"),
    }


def _row_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    for row in report.get("rows", []):
        if str(row.get("name")) == str(name):
            return dict(row)
    raise KeyError(name)


def _extract_context_summary(report: dict[str, Any]) -> dict[str, Any]:
    native = _row_by_name(report, "native-dense")
    kv_only = _row_by_name(report, "turbo-int8-hadamard")
    state_only = _row_by_name(report, "q-mamba-dsq-int4")
    combined = _row_by_name(report, "turbo-int8-hadamard+q-mamba-dsq-int4")
    return {
        "native_dense_runtime_cache_bytes": native.get("hybrid_total_runtime_cache_bytes") or native.get("hybrid_total_cache_bytes"),
        "kv_only_runtime_cache_ratio_vs_native": kv_only.get("hybrid_total_runtime_cache_ratio_vs_native"),
        "state_only_runtime_cache_ratio_vs_native": state_only.get("hybrid_total_runtime_cache_ratio_vs_native"),
        "combined_runtime_cache_ratio_vs_native": combined.get("hybrid_total_runtime_cache_ratio_vs_native"),
        "combined_speedup_vs_native": combined.get("speedup_vs_native"),
        "combined_prompt_perplexity_delta_pct_vs_native": combined.get("prompt_perplexity_delta_pct_vs_native"),
        "combined_generated_match_vs_baseline": combined.get("generated_match_vs_baseline"),
        "kv_only_kv_ratio_vs_native": kv_only.get("kv_cache_ratio_vs_native"),
        "state_only_mamba_state_ratio_vs_native": state_only.get("mamba_state_runtime_ratio_vs_native"),
        "combined_mamba_state_ratio_vs_native": combined.get("mamba_state_runtime_ratio_vs_native"),
    }


def _aggregate_sweep_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["model_ref"]), []).append(run)
    per_model: dict[str, Any] = {}
    for model_ref, model_runs in grouped.items():
        sorted_runs = sorted(model_runs, key=lambda item: int(item["prompt_length"]))
        best_combined = max(
            sorted_runs,
            key=lambda item: float(item["summary"].get("combined_runtime_cache_ratio_vs_native") or 0.0),
        )
        per_model[model_ref] = {
            "contexts": [
                {
                    "prompt_length": int(item["prompt_length"]),
                    "combined_runtime_cache_ratio_vs_native": item["summary"].get("combined_runtime_cache_ratio_vs_native"),
                    "combined_speedup_vs_native": item["summary"].get("combined_speedup_vs_native"),
                    "combined_prompt_perplexity_delta_pct_vs_native": item["summary"].get(
                        "combined_prompt_perplexity_delta_pct_vs_native"
                    ),
                }
                for item in sorted_runs
            ],
            "best_combined_context": {
                "prompt_length": int(best_combined["prompt_length"]),
                "combined_runtime_cache_ratio_vs_native": best_combined["summary"].get(
                    "combined_runtime_cache_ratio_vs_native"
                ),
                "combined_speedup_vs_native": best_combined["summary"].get("combined_speedup_vs_native"),
            },
        }
    return per_model


def run_hybrid_memory_sweep(args: argparse.Namespace) -> Path:
    output_path = Path(args.output).resolve()
    model_refs = list(args.model_ref or ["Zyphra/Zamba2-1.2B-Instruct-v2", "Zyphra/Zamba2-2.7B-Instruct-v2"])
    variants = _hybrid_memory_variants(
        kv_quant_seed=int(args.kv_quant_seed),
        kv_hot_window=int(args.kv_hot_window),
    )
    runs: list[dict[str, Any]] = []
    for model_ref in model_refs:
        for prompt_length in args.context_lengths:
            report = run_transformers_kv_benchmark(
                model_ref,
                prompt_text=args.prompt_text,
                prompt_length=int(prompt_length),
                max_new_tokens=int(args.max_new_tokens),
                warmup_max_new_tokens=0,
                kv_variants=variants,
                device=args.device,
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
            )
            runs.append(
                {
                    "model_ref": str(model_ref),
                    "prompt_length": int(prompt_length),
                    "summary": _extract_context_summary(report),
                    "report": _compact_report(report),
                }
            )

    payload = {
        "benchmark_kind": "hybrid-memory-context-sweep-v1",
        "device": args.device,
        "model_refs": model_refs,
        "context_lengths": [int(length) for length in args.context_lengths],
        "prompt_text": args.prompt_text,
        "max_new_tokens": int(args.max_new_tokens),
        "variant_order": [variant["name"] for variant in variants],
        "runs": runs,
        "per_model": _aggregate_sweep_runs(runs),
    }
    return _write_json(output_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid-memory context sweeps for Zamba2-style models.")
    parser.add_argument(
        "--model-ref",
        action="append",
        default=None,
        help="Repeat to benchmark multiple hybrid models.",
    )
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[16, 128, 512])
    parser.add_argument("--prompt-text", default=DEFAULT_SWEEP_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("verification") / "hybrid-memory-context-sweep.json")
    parser.add_argument("--kv-quant-seed", type=int, default=7)
    parser.add_argument("--kv-hot-window", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    path = run_hybrid_memory_sweep(args)
    print(f"saved hybrid-memory sweep to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
