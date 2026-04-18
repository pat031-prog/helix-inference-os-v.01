from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from helix_kv.benchmark import build_transformers_variant_set, published_benchmark_context, run_transformers_kv_benchmark


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = dict(value)
        last_logits = cleaned.pop("last_logits", None)
        if isinstance(last_logits, np.ndarray):
            cleaned["last_logits_shape"] = list(last_logits.shape)
        return {key: _json_ready(item) for key, item in cleaned.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cmd_benchmark_transformers_kv(args: argparse.Namespace) -> int:
    output_path = Path(args.output).resolve()
    report_path = output_path if output_path.suffix.lower() == ".json" else output_path / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    variants = build_transformers_variant_set(
        str(args.variant_set),
        kv_quant_seed=int(args.kv_quant_seed),
        kv_hot_window=int(args.kv_hot_window),
        kv_calibration_tokens=int(args.kv_calibration_tokens),
        kv_adaptive_medium_kurtosis=float(args.adaptive_medium_kurtosis),
        kv_adaptive_high_kurtosis=float(args.adaptive_high_kurtosis),
    )

    report = run_transformers_kv_benchmark(
        args.model_ref,
        prompt_ids=None if args.prompt_ids is None else [int(token_id) for token_id in args.prompt_ids],
        prompt_text=args.prompt_text,
        prompt_length=int(args.prompt_length),
        max_new_tokens=int(args.max_new_tokens),
        kv_variants=variants,
        kv_quant_seed=int(args.kv_quant_seed),
        kv_hot_window=int(args.kv_hot_window),
        kv_calibration_tokens=int(args.kv_calibration_tokens),
        kv_adaptive_high_kurtosis=float(args.adaptive_high_kurtosis),
        kv_adaptive_medium_kurtosis=float(args.adaptive_medium_kurtosis),
        local_files_only=bool(args.local_files_only),
        device=args.device,
        trust_remote_code=bool(args.trust_remote_code),
    )
    report["published_context"] = published_benchmark_context()
    report["variant_set"] = str(args.variant_set)
    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")

    print(f"saved transformers KV benchmark to {report_path}")
    for row in report.get("rows", []):
        print(
            f"{row['name']}: "
            f"time_s={row['total_time_s']:.4f} "
            f"speedup_vs_native={row['speedup_vs_native']:.4f} "
            f"kv_ratio_vs_native={row['kv_cache_ratio_vs_native']:.4f} "
            f"session_ratio_vs_native={row['session_size_ratio_vs_native']:.4f} "
            f"match={row['generated_match_vs_baseline']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone helix-kv CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_transformers_kv = subparsers.add_parser(
        "benchmark-transformers-kv",
        help="Benchmark Helix KV compression through Hugging Face Transformers.",
    )
    benchmark_transformers_kv.add_argument("model_ref")
    benchmark_transformers_kv.add_argument("--output", type=Path, default=Path("verification") / "transformers-kv-benchmark")
    benchmark_transformers_kv.add_argument("--prompt-ids", type=int, nargs="+")
    benchmark_transformers_kv.add_argument("--prompt-text")
    benchmark_transformers_kv.add_argument("--prompt-length", type=int, default=512)
    benchmark_transformers_kv.add_argument("--max-new-tokens", type=int, default=32)
    benchmark_transformers_kv.add_argument("--device")
    benchmark_transformers_kv.add_argument("--local-files-only", action="store_true")
    benchmark_transformers_kv.add_argument("--trust-remote-code", action="store_true")
    benchmark_transformers_kv.add_argument("--kv-hot-window", type=int, default=4)
    benchmark_transformers_kv.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_transformers_kv.add_argument("--kv-calibration-tokens", type=int, default=128)
    benchmark_transformers_kv.add_argument("--adaptive-high-kurtosis", type=float, default=20.0)
    benchmark_transformers_kv.add_argument("--adaptive-medium-kurtosis", type=float, default=9.0)
    benchmark_transformers_kv.add_argument("--variant-set", choices=["stable", "asymmetry-sweep", "community"], default="stable")
    benchmark_transformers_kv.set_defaults(func=_cmd_benchmark_transformers_kv)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
