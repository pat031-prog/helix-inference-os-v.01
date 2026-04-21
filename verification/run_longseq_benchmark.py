"""Focused micro-benchmark: selective attention vs full materialization.

Directly measures the core attention computation time difference,
bypassing the full model pipeline overhead. This isolates the
speedup from selective attention itself.
"""
from __future__ import annotations

import time
import numpy as np

import sys
sys.path.insert(0, "src")

from helix_proto.hf import (
    _TurboInt8KVArray as TurboInt8,
    _Turbo4BitKVArray as Turbo4Bit,
    _TurboQJLKVArray as TurboQJL,
    _HotWindowKVArray as HotWindow,
    _HadamardRotation,
    _compute_lloyd_max_codebook,
    _softmax,
)


def benchmark_attention_step(
    num_heads: int,
    head_dim: int,
    cold_length: int,
    hot_length: int,
    topk: int,
    kv_type: str,
    repeats: int = 20,
) -> dict:
    """Benchmark a single attention step: full materialization vs selective."""
    rng = np.random.default_rng(42)

    # Generate random KV cache data
    cold_data = rng.standard_normal((num_heads, cold_length, head_dim)).astype(np.float32) * 0.5
    hot_data = rng.standard_normal((num_heads, hot_length, head_dim)).astype(np.float32) * 0.5
    query = rng.standard_normal((num_heads, head_dim)).astype(np.float32)
    k_new = rng.standard_normal((num_heads, 1, head_dim)).astype(np.float32)
    v_new = rng.standard_normal((num_heads, 1, head_dim)).astype(np.float32)

    rotation = _HadamardRotation(head_dim, seed=7)
    codebook = _compute_lloyd_max_codebook(rotation.rotated_dim, 4) if kv_type != "int8" else None
    qjl_matrix = rng.standard_normal((head_dim, head_dim)).astype(np.float32) if kv_type == "qjl" else None

    # Create compressed cold cache
    if kv_type == "int8":
        cold_k = TurboInt8(cold_data, rotation=rotation)
        cold_v = TurboInt8(cold_data, rotation=rotation)
    elif kv_type == "4bit":
        cold_k = Turbo4Bit(cold_data, rotation=rotation, codebook=codebook)
        cold_v = Turbo4Bit(cold_data, rotation=rotation, codebook=codebook)
    else:  # qjl
        cold_k = TurboQJL(cold_data, rotation=rotation, codebook=codebook, qjl_matrix=qjl_matrix)
        cold_v = TurboQJL(cold_data, rotation=rotation, codebook=codebook, qjl_matrix=qjl_matrix)

    cache_k = HotWindow(cold=cold_k, hot=hot_data.copy())
    cache_v = HotWindow(cold=cold_v, hot=hot_data.copy())

    scale = 1.0 / np.sqrt(float(head_dim))

    # === FULL MATERIALIZATION PATH ===
    full_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        # Step 1: Materialize everything
        full_k = cache_k.to_float32()
        full_v = cache_v.to_float32()
        k_all = np.concatenate([full_k, k_new], axis=1)
        v_all = np.concatenate([full_v, v_new], axis=1)
        # Step 2: Full attention
        scores = np.einsum("hd,hnd->hn", query, k_all, optimize=True) * scale
        probs = _softmax(scores, axis=-1)
        context_full = np.einsum("hn,hnd->hd", probs, v_all, optimize=True)
        full_times.append(time.perf_counter() - t0)

    # === SELECTIVE ATTENTION PATH ===
    effective_topk = min(topk, cold_length)
    selective_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        # Step 1: Approximate scores on cold prefix
        cold_approx = cache_k.cold_approximate_scores(query, head_dim=head_dim)
        # Step 2: Top-K selection
        top_idx = np.argpartition(cold_approx, -effective_topk, axis=-1)[:, -effective_topk:]
        # Step 3: Selective materialization
        sel_cold_k = cache_k.cold_materialize_indices(top_idx)
        sel_cold_v = cache_v.cold_materialize_indices(top_idx)
        # Step 4: Exact attention on subset
        k_exact = np.concatenate([sel_cold_k, cache_k.hot, k_new], axis=1)
        v_exact = np.concatenate([sel_cold_v, cache_v.hot, v_new], axis=1)
        scores = np.einsum("hd,hnd->hn", query, k_exact, optimize=True) * scale
        probs = _softmax(scores, axis=-1)
        context_sel = np.einsum("hn,hnd->hd", probs, v_exact, optimize=True)
        selective_times.append(time.perf_counter() - t0)

    # Fidelity check
    cosine = float(
        np.dot(context_full.ravel(), context_sel.ravel())
        / (np.linalg.norm(context_full) * np.linalg.norm(context_sel) + 1e-8)
    )

    full_avg_ms = np.mean(full_times) * 1000
    sel_avg_ms = np.mean(selective_times) * 1000
    speedup = full_avg_ms / sel_avg_ms if sel_avg_ms > 0 else float("inf")

    return {
        "kv_type": kv_type,
        "cold_length": cold_length,
        "hot_length": hot_length,
        "topk": topk,
        "full_avg_ms": round(full_avg_ms, 3),
        "selective_avg_ms": round(sel_avg_ms, 3),
        "speedup": round(speedup, 2),
        "cosine_fidelity": round(cosine, 6),
    }


def main() -> int:
    num_heads = 8
    head_dim = 64
    hot_length = 4

    print("=" * 85)
    print("SELECTIVE ATTENTION MICRO-BENCHMARK")
    print(f"  heads={num_heads}, head_dim={head_dim}, hot_window={hot_length}, repeats=20")
    print("=" * 85)

    fmt = "{:>6} | {:>8} | {:>6} | {:>12} | {:>12} | {:>8} | {:>8}"
    print(fmt.format("Type", "Cold Len", "Top-K", "Full (ms)", "Selective(ms)", "Speedup", "Cosine"))
    print("-" * 85)

    for kv_type in ["int8", "4bit"]:
        for cold_length in [32, 64, 128, 256, 512]:
            for topk in [4, 8, 16]:
                if topk >= cold_length:
                    continue
                result = benchmark_attention_step(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    cold_length=cold_length,
                    hot_length=hot_length,
                    topk=topk,
                    kv_type=kv_type,
                    repeats=20,
                )
                print(fmt.format(
                    result["kv_type"],
                    result["cold_length"],
                    result["topk"],
                    f"{result['full_avg_ms']:.3f}",
                    f"{result['selective_avg_ms']:.3f}",
                    f"{result['speedup']:.2f}x",
                    f"{result['cosine_fidelity']:.4f}",
                ))
        print("-" * 85)

    print()
    print("Legend:")
    print("  Full (ms)      = average time for full materialization + attention")
    print("  Selective (ms) = average time for approximate score + topK + selective materialization + attention")
    print("  Speedup        = Full / Selective (>1.0x = selective is faster)")
    print("  Cosine         = cosine similarity between full and selective context output (1.0 = identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
