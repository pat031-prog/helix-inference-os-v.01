from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _rust_indexed_dag_class() -> Any:
    try:
        from _helix_merkle_dag import RustIndexedMerkleDAG

        return RustIndexedMerkleDAG
    except Exception:  # noqa: BLE001
        try:
            from helix_kv._helix_merkle_dag import RustIndexedMerkleDAG

            return RustIndexedMerkleDAG
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Rust indexed MerkleDAG extension is not installed: {exc}")


def _build_heavy_dag(depth: int = 5000, bytes_per_node: int = 4000) -> tuple[Any, list[str]]:
    dag = _rust_indexed_dag_class()()
    if not hasattr(dag, "build_context_fast"):
        pytest.skip("Rust extension was not rebuilt with build_context_fast")

    heavy_narrative = "A" * bytes_per_node
    node_hashes: list[str] = []
    parent_hash = None
    for idx in range(depth):
        node = dag.insert_indexed(f"{heavy_narrative}{idx}", parent_hash, None)
        node_hashes.append(node.hash)
        parent_hash = node.hash
    return dag, node_hashes


def _median_ms(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def test_build_context_fast_hard_anchor_shape() -> None:
    dag, node_hashes = _build_heavy_dag(depth=3, bytes_per_node=64)

    anchors = dag.build_context_fast(node_hashes, True)
    legacy = dag.build_context_fast(node_hashes, False)

    assert anchors.count("<hard_anchor>") == 3
    assert "<legacy_memory>" not in anchors
    assert "A" * 64 not in anchors
    assert legacy.count("<legacy_memory>") == 3
    assert "A" * 64 in legacy


def test_build_context_fast_hard_anchor_perf() -> None:
    dag, node_hashes = _build_heavy_dag(depth=5000, bytes_per_node=8192)

    legacy_samples = []
    anchor_samples = []
    for _ in range(7):
        start = time.perf_counter()
        legacy = dag.build_context_fast(node_hashes, False)
        legacy_samples.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        anchors = dag.build_context_fast(node_hashes, True)
        anchor_samples.append((time.perf_counter() - start) * 1000)

    legacy_ms = _median_ms(legacy_samples)
    anchors_ms = _median_ms(anchor_samples)
    speedup = legacy_ms / max(anchors_ms, 0.001)

    assert len(legacy) > len(anchors) * 20
    assert anchors_ms <= 5.0
    assert speedup >= 9.0


if __name__ == "__main__":
    dag, node_hashes = _build_heavy_dag(depth=5000, bytes_per_node=8192)
    legacy_samples = []
    anchor_samples = []
    for _ in range(7):
        start = time.perf_counter()
        dag.build_context_fast(node_hashes, False)
        legacy_samples.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        dag.build_context_fast(node_hashes, True)
        anchor_samples.append((time.perf_counter() - start) * 1000)

    legacy_ms = _median_ms(legacy_samples)
    anchors_ms = _median_ms(anchor_samples)
    speedup = legacy_ms / max(anchors_ms, 0.001)
    print("--- RUST HARD-ANCHORS RESULTS ---")
    print(f"Legacy median latency:       {legacy_ms:.4f} ms")
    print(f"Hard anchors median latency: {anchors_ms:.4f} ms")
    print(f"Speedup:                     {speedup:.4f}x")
