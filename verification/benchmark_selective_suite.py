"""Comprehensive end-to-end benchmark for selective attention.

This script builds a small but realistic GPT2 model and runs head-to-head
comparisons between full KV cache materialization and the new
selective attention architecture (--kv-topk), measuring end-to-end
generation speedup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    sys.path.insert(0, "src")
    from helix_proto.cli import _cmd_build_tiny_gpt2
    from helix_proto.hf import (
        GPT2StreamingEngine,
        benchmark_gpt2_kv_mode_matrix,
        export_huggingface_model,
    )

    # Benchmark parameters
    # Increase model size to better reflect real proportion between attention and MLP
    hidden = 128
    n_heads = 8
    n_layers = 6
    vocab_size = 50257
    max_pos = 1024
    
    prompt_lengths = [64, 128, 256, 512]
    new_tokens = 16
    hot_window = 8

    workdir = Path("verification/benchmark_e2e_suite").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "model-tmp"
    export_dir = workdir / "export"

    print("=" * 80)
    print("END-TO-END SELECTIVE ATTENTION BENCHMARK SUITE")
    print(f"Model: {n_layers} layers, {hidden} hidden, {n_heads} heads")
    print("=" * 80)
    print()

    # Build model using existing CLI utility
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=str(model_dir),
            vocab_size=vocab_size,
            hidden_size=hidden,
            max_position_embeddings=max_pos,
            num_layers=n_layers,
            num_heads=n_heads,
        )
    )
    export_huggingface_model(str(model_dir), str(export_dir), block_rows=16, local_files_only=True)
    print(f"Test model exported to {export_dir}")
    print()

    all_results = {}

    for plen in prompt_lengths:
        # Generate varied prompt
        prompt_ids = [((i * 137) + 511) % vocab_size for i in range(plen)]

        variants = [
            {"name": "int8-full", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard", "kv_hot_window": hot_window, "kv_topk": 0},
            {"name": "int8-topk16", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard", "kv_hot_window": hot_window, "kv_topk": 16},
            {"name": "int8-topk32", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard", "kv_hot_window": hot_window, "kv_topk": 32},
            {"name": "4bit-full", "kv_cache_precision": "turbo-4bit", "kv_rotation_mode": "hadamard", "kv_hot_window": hot_window, "kv_topk": 0},
            {"name": "4bit-topk16", "kv_cache_precision": "turbo-4bit", "kv_rotation_mode": "hadamard", "kv_hot_window": hot_window, "kv_topk": 16},
            {"name": "4bit-topk32", "kv_cache_precision": "turbo-4bit", "kv_rotation_mode": "hadamard", "kv_hot_window": hot_window, "kv_topk": 32},
        ]

        print(f"--- Prompt: {plen} tokens (cold={max(0, plen-hot_window)}), generating {new_tokens} new tokens ---")
        report = benchmark_gpt2_kv_mode_matrix(
            str(export_dir),
            prompt_ids=prompt_ids,
            max_new_tokens=new_tokens,
            kv_variants=variants,
            kv_quant_seed=7,
        )

        for name, metrics in report["variants"].items():
            t = metrics["total_time_s"]
            tok_sec = new_tokens / t if t > 0 else 0
            logit = metrics.get("logit_comparison_vs_baseline")
            cos = logit["cosine_similarity"] if logit else 1.0
            print(f"  {name:>12}: {tok_sec:>6.2f} tok/s | time={t:.3f}s | cosine={cos:.5f}")

        all_results[str(plen)] = report
        print()

    # Build Summary Table
    print()
    print("=" * 80)
    print("SELECTIVE ATTENTION END-TO-END SPEEDUP SUMMARY")
    print("=" * 80)
    fmt = "{:>6} | {:>14} | {:>9} | {:>7} | {:>8} | {:>8} | {:>5}"
    print(fmt.format("Prompt", "Mode", "Time(s)", "Tok/s", "Speedup", "Cosine", "Match"))
    print("-" * 80)

    for plen in prompt_lengths:
        report = all_results[str(plen)]
        variants = report["variants"]

        for prefix in ["int8", "4bit"]:
            full_key = f"{prefix}-full"
            if full_key not in variants:
                continue
            base_t = variants[full_key]["total_time_s"]

            for topk in ["full", "topk16", "topk32"]:
                key = f"{prefix}-{topk}"
                if key not in variants:
                    continue
                
                t = variants[key]["total_time_s"]
                tok_sec = new_tokens / t if t > 0 else 0
                match = variants[key]["generated_match_vs_baseline"]
                logit = variants[key].get("logit_comparison_vs_baseline")
                cos = logit["cosine_similarity"] if logit else 1.0

                if topk == "full":
                    speedup_str = "baseline"
                elif t > 0:
                    speedup_str = f"{base_t / t:.2f}x"
                else:
                    speedup_str = "inf"

                mode = f"{prefix}-{topk}"
                print(fmt.format(plen, mode, f"{t:.3f}", f"{tok_sec:.1f}", speedup_str, f"{cos:.5f}", str(match)))

        print("-" * 80)

    # Save final report cleanly
    report_path = workdir / "e2e_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
