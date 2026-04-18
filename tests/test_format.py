from pathlib import Path

import numpy as np

from helix_proto.format import (
    create_store,
    load_full_tensor,
    load_tensor_rows,
    store_stats,
    streaming_matvec,
    verify_store,
)
from helix_proto.hf import export_local_npz, export_tensor_map


def test_streaming_matvec_matches_dense(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(128, 64)).astype(np.float32)
    vector = rng.normal(size=(64,)).astype(np.float32)
    store = tmp_path / "sample.cdna"

    create_store(matrix, store, block_rows=32)
    verify_store(store)

    dense = matrix @ vector
    streamed = streaming_matvec(store, vector)
    assert np.allclose(dense, streamed)


def test_store_stats_are_present(tmp_path: Path) -> None:
    matrix = np.zeros((64, 16), dtype=np.float32)
    store = tmp_path / "zeros.cdna"

    create_store(matrix, store, block_rows=16)
    stats = store_stats(store)

    assert stats["blocks"] == 4
    assert stats["compressed_bytes"] <= stats["raw_bytes"]


def test_export_tensor_map_builds_manifest(tmp_path: Path) -> None:
    tensors = {
        "encoder.layer1.weight": np.ones((32, 16), dtype=np.float32),
        "encoder.layer1.bias": np.ones((16,), dtype=np.float32),
    }

    manifest = export_tensor_map(tensors, tmp_path / "hf-export", block_rows=8, model_ref="toy/model")

    assert len(manifest["exported"]) == 2
    assert len(manifest["skipped"]) == 0
    exported_path = tmp_path / "hf-export" / manifest["exported"][0]["path"]
    assert exported_path.exists()


def test_export_local_npz(tmp_path: Path) -> None:
    npz_path = tmp_path / "weights.npz"
    np.savez(
        npz_path,
        projection=np.ones((16, 8), dtype=np.float32),
        bias=np.ones((8,), dtype=np.float32),
    )

    manifest = export_local_npz(npz_path, tmp_path / "npz-export", block_rows=4)

    assert len(manifest["exported"]) == 2
    assert manifest["config"]["source"] == "npz"


def test_vector_store_roundtrip(tmp_path: Path) -> None:
    vector = np.arange(10, dtype=np.float32)
    store = tmp_path / "vector.cdna"

    create_store(vector, store, block_rows=4)
    restored = load_full_tensor(store)

    assert np.array_equal(vector, restored)


def test_load_tensor_rows_returns_requested_order(tmp_path: Path) -> None:
    matrix = np.arange(30, dtype=np.float32).reshape(10, 3)
    store = tmp_path / "rows.cdna"

    create_store(matrix, store, block_rows=4)
    rows = load_tensor_rows(store, [7, 1, 7])

    assert np.array_equal(rows[0], matrix[7])
    assert np.array_equal(rows[1], matrix[1])
    assert np.array_equal(rows[2], matrix[7])
