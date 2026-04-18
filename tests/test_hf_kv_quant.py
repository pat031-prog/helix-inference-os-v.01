import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from helix_proto.cli import _cmd_build_tiny_gpt2, build_parser
from helix_proto.hf import (
    GPT2StreamingEngine,
    _HadamardRotation,
    _HotWindowKVArray,
    _KurtosisAccumulator,
    _block_shortlist_needs_expansion,
    _merge_candidate_index_sets,
    _selected_score_threshold,
    _tensor_to_numpy_exportable,
    _TurboInt8KVArray,
    _Turbo4BitKVArray,
    _TurboQJLKVArray,
    _TurboQuantizedKVArray,
    _compute_lloyd_max_codebook,
    _expand_block_indices,
    _merge_selective_candidate_indices,
    _orthogonal_rotation_matrix,
    _selective_candidate_topk,
    _session_meta_matches_engine,
    _should_use_selective_attention,
    export_huggingface_model,
)


def test_turbo_quantized_kv_roundtrip_is_reasonably_close() -> None:
    rotation = _orthogonal_rotation_matrix(8, seed=11)
    values = np.linspace(-1.5, 1.5, num=2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)

    quantized = _TurboQuantizedKVArray(values, rotation=rotation)
    restored = quantized.to_float32()

    assert restored.shape == values.shape
    assert quantized.nbytes < values.nbytes
    assert float(np.max(np.abs(restored - values))) < 0.05


def test_hadamard_rotation_roundtrip_handles_padding() -> None:
    rotation = _HadamardRotation(6, seed=5)
    values = np.linspace(-1.0, 1.0, num=2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)

    restored = rotation.inverse(rotation.forward(values))

    assert restored.shape == values.shape
    assert float(np.max(np.abs(restored - values))) < 1e-5


def test_turbo_4bit_roundtrip_is_reasonably_close_and_smaller_than_int8() -> None:
    values = np.linspace(-1.25, 1.25, num=2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)
    rotation = _HadamardRotation(8, seed=17)
    codebook = _compute_lloyd_max_codebook(rotation.rotated_dim, 4)

    int8_quantized = _TurboQuantizedKVArray(values, rotation=_orthogonal_rotation_matrix(8, seed=17))
    fourbit_quantized = _Turbo4BitKVArray(values, rotation=rotation, codebook=codebook)
    restored = fourbit_quantized.to_float32()

    cosine = float(np.dot(restored.reshape(-1), values.reshape(-1)) / (np.linalg.norm(restored) * np.linalg.norm(values)))

    assert restored.shape == values.shape
    assert fourbit_quantized.nbytes < int8_quantized.nbytes
    assert cosine > 0.97


def test_turbo_qjl_roundtrip_restores_shape_and_adds_residual_payload() -> None:
    values = np.linspace(-0.9, 0.9, num=2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)
    rotation = _HadamardRotation(8, seed=23)
    codebook = _compute_lloyd_max_codebook(rotation.rotated_dim, 4)
    qjl_matrix = np.eye(8, dtype=np.float32)

    fourbit_quantized = _Turbo4BitKVArray(values, rotation=rotation, codebook=codebook)
    qjl_quantized = _TurboQJLKVArray(values, rotation=rotation, codebook=codebook, qjl_matrix=qjl_matrix)
    restored = qjl_quantized.to_float32()

    assert restored.shape == values.shape
    assert qjl_quantized.nbytes > fourbit_quantized.nbytes

    query = values[:, -1, :]
    correction = qjl_quantized.score_correction(query, head_dim=values.shape[-1], score_weight=0.25)

    assert correction.shape == (values.shape[0], values.shape[1])
    assert np.isfinite(correction).all()


def test_hot_window_cache_combines_compressed_prefix_and_exact_tail() -> None:
    values = np.linspace(-1.0, 1.0, num=2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
    cold = _TurboQuantizedKVArray(values[:, :4, :], rotation=_orthogonal_rotation_matrix(8, seed=7))
    hot = values[:, 4:, :]
    cache = _HotWindowKVArray(cold=cold, hot=hot)

    restored = cache.to_float32()

    assert cache.cold_length == 4
    assert cache.hot_length == 2
    assert cache.length == 6
    assert cache.nbytes < values.nbytes
    assert restored.shape == values.shape
    assert np.allclose(restored[:, 4:, :], hot)


def test_turbo_int8_append_preserves_block_summary_cache() -> None:
    values = np.linspace(-1.0, 1.0, num=2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)
    rotation = _HadamardRotation(8, seed=19)
    query = values[:, -1, :]

    cache = _TurboInt8KVArray(values[:, :4, :], rotation=rotation)
    _ = cache.approximate_block_scores(query, head_dim=8, block_size=2)
    appended = cache.append_compressed(values[:, 4:, :])
    rebuilt = _TurboInt8KVArray(values, rotation=rotation)

    assert 2 in appended._block_summary_cache
    assert np.allclose(
        appended.approximate_block_scores(query, head_dim=8, block_size=2),
        rebuilt.approximate_block_scores(query, head_dim=8, block_size=2),
        atol=1e-3,
    )


def test_turbo_4bit_append_preserves_block_summary_cache() -> None:
    values = np.linspace(-1.0, 1.0, num=2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)
    rotation = _HadamardRotation(8, seed=29)
    codebook = _compute_lloyd_max_codebook(rotation.rotated_dim, 4)
    query = values[:, -1, :]

    cache = _Turbo4BitKVArray(values[:, :4, :], rotation=rotation, codebook=codebook)
    _ = cache.approximate_block_scores(query, head_dim=8, block_size=2)
    appended = cache.append_compressed(values[:, 4:, :])
    rebuilt = _Turbo4BitKVArray(values, rotation=rotation, codebook=codebook)

    assert 2 in appended._block_summary_cache
    assert np.allclose(
        appended.approximate_block_scores(query, head_dim=8, block_size=2),
        rebuilt.approximate_block_scores(query, head_dim=8, block_size=2),
        atol=1e-3,
    )


def test_turbo_qjl_append_preserves_block_summary_cache() -> None:
    values = np.linspace(-0.75, 0.75, num=2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8)
    rotation = _HadamardRotation(8, seed=31)
    codebook = _compute_lloyd_max_codebook(rotation.rotated_dim, 4)
    qjl_matrix = np.eye(8, dtype=np.float32)
    query = values[:, -1, :]

    cache = _TurboQJLKVArray(values[:, :4, :], rotation=rotation, codebook=codebook, qjl_matrix=qjl_matrix)
    _ = cache.approximate_block_scores(query, head_dim=8, block_size=2)
    appended = cache.append_compressed(values[:, 4:, :])
    rebuilt = _TurboQJLKVArray(values, rotation=rotation, codebook=codebook, qjl_matrix=qjl_matrix)

    assert 2 in appended._block_summary_cache
    assert np.allclose(
        appended.approximate_block_scores(query, head_dim=8, block_size=2),
        rebuilt.approximate_block_scores(query, head_dim=8, block_size=2),
        atol=1e-3,
    )


def test_cli_parser_accepts_turbo_kv_benchmark_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-cache",
            "--kv-mode",
            "turbo-4bit",
            "--kv-rotation",
            "hadamard",
            "--kv-quant-seed",
            "13",
            "--kv-hot-window",
            "4",
        ]
    )

    assert args.command == "benchmark-gpt-cache"
    assert args.kv_cache_precision == "turbo-4bit"
    assert args.kv_rotation_mode == "hadamard"
    assert args.kv_quant_seed == 13
    assert args.kv_hot_window == 4


def test_cli_parser_defaults_to_hadamard_rotation() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark-gpt-cache"])

    assert args.command == "benchmark-gpt-cache"
    assert args.kv_rotation_mode == "hadamard"
    assert args.kv_hot_window == 0


def test_turbo_int8_approximate_scores_match_exact_scores() -> None:
    """Verify that approximate scores computed in rotated domain are close to
    scores computed via full materialization."""
    from helix_proto.hf import _HadamardRotation

    rotation = _HadamardRotation(8, seed=31)
    values = np.random.default_rng(42).standard_normal((2, 6, 8)).astype(np.float32)
    query = np.random.default_rng(43).standard_normal((2, 8)).astype(np.float32)

    quantized = _TurboQuantizedKVArray(values, rotation=rotation)

    # Exact scores: materialize → dot product
    materialized = quantized.to_float32()
    exact_scores = np.einsum("hd,hnd->hn", query, materialized) / np.sqrt(8.0)

    # Approximate scores: computed in rotated domain
    approx_scores = quantized.approximate_scores(query, head_dim=8)

    # They should be very close (int8 quantization noise is the only difference)
    assert approx_scores.shape == exact_scores.shape
    max_err = float(np.max(np.abs(approx_scores - exact_scores)))
    assert max_err < 0.15, f"approximate scores too far from exact: max_err={max_err}"


def test_turbo_4bit_approximate_scores_match_exact_scores() -> None:
    """Verify that 4-bit approximate scores are close to exact scores."""
    from helix_proto.hf import _HadamardRotation

    rotation = _HadamardRotation(8, seed=37)
    codebook = _compute_lloyd_max_codebook(rotation.rotated_dim, 4)
    values = np.random.default_rng(44).standard_normal((2, 6, 8)).astype(np.float32) * 0.5
    query = np.random.default_rng(45).standard_normal((2, 8)).astype(np.float32)

    quantized = _Turbo4BitKVArray(values, rotation=rotation, codebook=codebook)

    # Exact scores
    materialized = quantized.to_float32()
    exact_scores = np.einsum("hd,hnd->hn", query, materialized) / np.sqrt(8.0)

    # Approximate scores
    approx_scores = quantized.approximate_scores(query, head_dim=8)

    assert approx_scores.shape == exact_scores.shape
    cosine = float(
        np.dot(approx_scores.ravel(), exact_scores.ravel())
        / (np.linalg.norm(approx_scores) * np.linalg.norm(exact_scores) + 1e-8)
    )
    assert cosine > 0.95, f"approximate scores not similar enough to exact: cosine={cosine}"


def test_turbo_int8_approximate_block_scores_shape() -> None:
    rotation = _HadamardRotation(8, seed=39)
    values = np.random.default_rng(47).standard_normal((2, 10, 8)).astype(np.float32)
    query = np.random.default_rng(48).standard_normal((2, 8)).astype(np.float32)
    quantized = _TurboInt8KVArray(values, rotation=rotation)

    scores = quantized.approximate_block_scores(query, head_dim=8, block_size=4)

    assert scores.shape == (2, 3)
    assert np.isfinite(scores).all()


def test_expand_block_indices_expands_each_selected_block() -> None:
    blocks = np.array([[0, 2], [1, 2]], dtype=np.int64)
    expanded = _expand_block_indices(blocks, cold_length=10, block_size=4)

    assert expanded.shape == (2, 6)
    assert expanded[0].tolist() == [0, 1, 2, 3, 8, 9]
    assert expanded[1].tolist() == [4, 5, 6, 7, 8, 9]


def test_turbo_int8_materialize_indices_returns_correct_subset() -> None:
    """Verify that selective materialization returns the correct token subset."""
    from helix_proto.hf import _HadamardRotation

    rotation = _HadamardRotation(8, seed=41)
    values = np.random.default_rng(46).standard_normal((2, 10, 8)).astype(np.float32)

    quantized = _TurboQuantizedKVArray(values, rotation=rotation)
    full = quantized.to_float32()

    # Select indices [1, 5, 8] for head 0, [0, 3, 7] for head 1
    indices = np.array([[1, 5, 8], [0, 3, 7]])
    selected = quantized.materialize_indices(indices)

    assert selected.shape == (2, 3, 8)
    # The selected values should match the corresponding full materialization
    for h in range(2):
        for j, idx in enumerate(indices[h]):
            err = float(np.max(np.abs(selected[h, j] - full[h, idx])))
            assert err < 1e-5, f"materialize_indices mismatch at head={h}, idx={idx}: err={err}"


def test_hot_window_supports_selective_with_compressed_cold() -> None:
    """Verify HotWindowKVArray correctly reports selective support and delegates."""
    rotation = _orthogonal_rotation_matrix(8, seed=7)
    values = np.linspace(-1.0, 1.0, num=2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    cold = _TurboQuantizedKVArray(values[:, :6, :], rotation=rotation)
    hot = values[:, 6:, :]
    cache = _HotWindowKVArray(cold=cold, hot=hot)

    assert cache.supports_selective is True
    assert cache.cold_length == 6
    assert cache.hot_length == 2

    query = np.random.default_rng(50).standard_normal((2, 8)).astype(np.float32)
    scores = cache.cold_approximate_scores(query, head_dim=8)
    assert scores.shape == (2, 6)

    indices = np.array([[0, 2, 4], [1, 3, 5]])
    selected = cache.cold_materialize_indices(indices)
    assert selected.shape == (2, 3, 8)


def test_cli_parser_accepts_kv_topk_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-cache",
            "--kv-mode",
            "turbo-int8",
            "--kv-hot-window",
            "4",
            "--kv-topk",
            "8",
            "--kv-index-refresh-interval",
            "4",
            "--kv-block-size",
            "16",
        ]
    )

    assert args.command == "benchmark-gpt-cache"
    assert args.kv_cache_precision == "turbo-int8"
    assert args.kv_hot_window == 4
    assert args.kv_topk == 8
    assert args.kv_index_refresh_interval == 4
    assert args.kv_block_size == 16


def test_cli_parser_accepts_benchmark_gpt_session_size() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-session-size",
            "--output",
            "verification/session-sizes",
            "--prompt-lengths",
            "128",
            "512",
            "--kv-hot-window",
            "4",
        ]
    )

    assert args.command == "benchmark-gpt-session-size"
    assert args.output == Path("verification/session-sizes")
    assert args.prompt_lengths == [128, 512]
    assert args.kv_hot_window == 4


def test_cli_parser_accepts_benchmark_gpt_kv_modes_adaptive_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-kv-modes",
            "--include-adaptive",
            "--include-asymmetric",
            "--kv-calibration-tokens",
            "32",
        ]
    )

    assert args.command == "benchmark-gpt-kv-modes"
    assert args.include_adaptive is True
    assert args.include_asymmetric is True
    assert args.kv_calibration_tokens == 32


def test_cli_parser_accepts_asymmetric_kv_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-gpt-cache",
            "--kv-mode",
            "turbo-int8",
            "--kv-key-mode",
            "turbo-4bit",
            "--kv-value-mode",
            "turbo-int8",
        ]
    )

    assert args.command == "benchmark-gpt-cache"
    assert args.kv_cache_precision == "turbo-int8"
    assert args.kv_key_precision == "turbo-4bit"
    assert args.kv_value_precision == "turbo-int8"


def test_hot_window_append_token_preserves_cold() -> None:
    rng = np.random.default_rng(60)
    cold_data = rng.standard_normal((2, 4, 8)).astype(np.float32)
    hot_data = rng.standard_normal((2, 2, 8)).astype(np.float32)
    
    rot = _HadamardRotation(8, seed=42)
    cold = _TurboInt8KVArray(cold_data, rotation=rot)
    cache = _HotWindowKVArray(cold=cold, hot=hot_data)
    
    def mock_store(v):
        return _TurboInt8KVArray(v, rotation=rot)
        
    # Append token, hot window fits
    new_token = rng.standard_normal((2, 1, 8)).astype(np.float32)
    cache2 = cache.append_token(new_token, max_hot=4, store_fn=mock_store)
    assert cache2.hot_length == 3
    assert cache2.cold_length == 4
    assert cache2.cold is cache.cold  # pointer equality - zero copy
    
    # Append token, hot window spills
    new_token2 = rng.standard_normal((2, 1, 8)).astype(np.float32)
    cache3 = cache2.append_token(new_token2, max_hot=3, store_fn=mock_store)
    assert cache3.hot_length == 3
    assert cache3.cold_length == 5
    assert isinstance(cache3.cold, _TurboInt8KVArray)


def test_selective_helpers_use_thresholds_and_shortlist_expansion() -> None:
    values = np.random.default_rng(90).standard_normal((2, 32, 8)).astype(np.float32)
    hadamard = _HadamardRotation(8, seed=11)
    int8_cold = _TurboInt8KVArray(values, rotation=hadamard)
    codebook = _compute_lloyd_max_codebook(hadamard.rotated_dim, 4)
    fourbit_cold = _Turbo4BitKVArray(values, rotation=hadamard, codebook=codebook)
    qjl_cold = _TurboQJLKVArray(values, rotation=hadamard, codebook=codebook, qjl_matrix=np.eye(8, dtype=np.float32))

    assert _should_use_selective_attention(int8_cold, cold_length=16, effective_topk=4) is False
    assert _should_use_selective_attention(int8_cold, cold_length=32, effective_topk=4) is True
    assert _should_use_selective_attention(fourbit_cold, cold_length=16, effective_topk=4) is False
    assert _should_use_selective_attention(fourbit_cold, cold_length=32, effective_topk=4) is False
    assert _should_use_selective_attention(fourbit_cold, cold_length=64, effective_topk=4) is True
    assert _should_use_selective_attention(qjl_cold, cold_length=16, effective_topk=4) is False
    assert _should_use_selective_attention(qjl_cold, cold_length=32, effective_topk=4) is True

    assert _selective_candidate_topk(int8_cold, cold_length=32, effective_topk=4) == 4
    assert _selective_candidate_topk(fourbit_cold, cold_length=32, effective_topk=4) == 8


def test_turbo_4bit_optimized_materialize_matches_full() -> None:
    from helix_proto.hf import _Turbo4BitKVArray, _HadamardRotation, _compute_lloyd_max_codebook
    
    rng = np.random.default_rng(70)
    data = rng.standard_normal((2, 16, 8)).astype(np.float32)
    rot = _HadamardRotation(8, seed=42)
    codebook = _compute_lloyd_max_codebook(8, 4)
    
    qcache = _Turbo4BitKVArray(data, rotation=rot, codebook=codebook)
    indices = np.array([[0, 5, 15], [1, 2, 14]])
    
    # Materialize via gathered optimization
    opt = qcache.materialize_indices(indices)
    
    # Materialize full then slice (the baseline behavior)
    full = qcache.to_float32()
    base = np.empty_like(opt)
    for h in range(2):
        base[h] = full[h, indices[h]]
        
    np.testing.assert_allclose(opt, base, rtol=1e-5, atol=1e-6)


def test_turbo_4bit_materialize_indices_unpacks_only_selected_rows(monkeypatch) -> None:
    from helix_proto import hf as hf_module

    rng = np.random.default_rng(71)
    data = rng.standard_normal((2, 16, 8)).astype(np.float32)
    rot = _HadamardRotation(8, seed=43)
    codebook = _compute_lloyd_max_codebook(8, 4)
    qcache = _Turbo4BitKVArray(data, rotation=rot, codebook=codebook)
    indices = np.array([[0, 5, 15], [1, 2, 14]])

    calls: list[tuple[int, ...]] = []
    original_unpack = hf_module._unpack_nibbles

    def tracking_unpack(packed, original_length):  # noqa: ANN001
        calls.append(tuple(np.asarray(packed).shape))
        return original_unpack(packed, original_length)

    monkeypatch.setattr(hf_module, "_unpack_nibbles", tracking_unpack)

    _ = qcache.materialize_indices(indices)

    assert calls, "expected _unpack_nibbles to be called"
    # Only the selected top-K rows should be unpacked, not the full seq_len=16.
    assert calls[-1][1] == indices.shape[1]


def test_session_meta_validation_rejects_mismatched_engine_settings() -> None:
    meta = {
        "export_dir": str((Path.cwd() / "export").resolve()),
        "kv_cache_precision": "turbo-4bit",
        "kv_quant_seed": 7,
        "kv_rotation_mode": "hadamard",
        "kv_hot_window": 4,
        "kv_topk": 8,
    }

    try:
        _session_meta_matches_engine(
            meta,
            export_dir=(Path.cwd() / "export").resolve(),
            kv_cache_precision="turbo-int8",
            kv_key_precision=None,
            kv_value_precision=None,
            kv_quant_seed=7,
            kv_rotation_mode="hadamard",
            kv_hot_window=4,
            kv_topk=8,
            kv_index_refresh_interval=8,
            kv_block_size=0,
            kv_layer_share_stride=0,
        )
    except ValueError as exc:
        assert "kv_cache_precision" in str(exc)
    else:
        raise AssertionError("expected mismatched session metadata to raise")


def test_save_session_persists_compressed_codec_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    session_dir = tmp_path / "session"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=16,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision="turbo-4bit",
        kv_quant_seed=7,
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
    )
    run = engine.generate([3, 5, 8, 13], max_new_tokens=2)
    engine.save_session(session_dir, generated_ids=run["generated_ids"], last_logits=run["last_logits"])

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert meta["session_format_version"] == 2
    assert meta["kv_cache_file"] == "kv_cache.npz"
    assert meta["kv_session_payload"]["format"] == "compressed-kv-v2"
    assert meta["kv_session_payload"]["npz_compressed"] is True
    assert meta["kv_session_payload"]["rotation"]["mode"] == "hadamard"
    assert meta["kv_session_payload"]["codebook"]["bits"] == 4

    with np.load(session_dir / "kv_cache.npz") as data:
        assert "__kv_rotation_signs" in data
        assert "__kv_codebook_centroids" in data
        assert "__kv_codebook_boundaries" in data
        assert "layer_0_k_hot" in data
        assert "layer_0_k_cold_packed" in data

    resumed_engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision="turbo-4bit",
        kv_quant_seed=7,
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
    )
    loaded = resumed_engine.load_session(session_dir)
    assert loaded["generated_ids"] == run["generated_ids"]
    assert isinstance(resumed_engine.caches[0]["k"], _HotWindowKVArray)
    assert isinstance(resumed_engine.caches[0]["k"].cold, _Turbo4BitKVArray)


def test_adaptive_kurtosis_profile_finalizes_and_persists(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    session_dir = tmp_path / "adaptive-session"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=16,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision="adaptive",
        kv_quant_seed=7,
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
        kv_calibration_tokens=2,
        kv_adaptive_high_kurtosis=0.0,
        kv_adaptive_medium_kurtosis=0.0,
    )
    run = engine.generate([3, 5, 8], max_new_tokens=1)

    assert engine._kv_layer_modes == ["fp32"]
    assert run["kv_layer_modes"] == ["fp32"]
    assert run["kv_kurtosis_profile"] is not None
    assert run["kv_kurtosis_profile"][0]["selected_mode"] == "fp32"

    engine.save_session(session_dir, generated_ids=run["generated_ids"], last_logits=run["last_logits"])
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert meta["kv_calibration_tokens"] == 2
    assert meta["kv_layer_modes"] == ["fp32"]
    assert meta["kv_kurtosis_profile"][0]["selected_mode"] == "fp32"
    assert meta["kv_session_payload"]["adaptive"]["ready"] is True

    resumed_engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision="adaptive",
        kv_quant_seed=7,
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
        kv_calibration_tokens=2,
        kv_adaptive_high_kurtosis=0.0,
        kv_adaptive_medium_kurtosis=0.0,
    )
    loaded = resumed_engine.load_session(session_dir)
    assert loaded["kv_layer_modes"] == ["fp32"]
    assert resumed_engine._kv_layer_modes == ["fp32"]


def test_adaptive_profile_buckets_layers_by_kurtosis(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=16,
            num_layers=3,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="adaptive",
        kv_rotation_mode="hadamard",
        kv_calibration_tokens=8,
        kv_adaptive_high_kurtosis=10.0,
        kv_adaptive_medium_kurtosis=3.0,
    )

    high = np.concatenate([np.zeros(2047, dtype=np.float32), np.array([100.0], dtype=np.float32)])
    medium = np.random.default_rng(101).laplace(0.0, 1.0, 10000).astype(np.float32)
    low = np.random.default_rng(102).uniform(-1.0, 1.0, 10000).astype(np.float32)

    engine._kv_kurtosis_state = [
        {"k": _KurtosisAccumulator(), "v": _KurtosisAccumulator()},
        {"k": _KurtosisAccumulator(), "v": _KurtosisAccumulator()},
        {"k": _KurtosisAccumulator(), "v": _KurtosisAccumulator()},
    ]
    for payload in (high, high):
        engine._kv_kurtosis_state[0]["k"].update(payload)
        engine._kv_kurtosis_state[0]["v"].update(payload)
    engine._kv_kurtosis_state[1]["k"].update(medium)
    engine._kv_kurtosis_state[1]["v"].update(medium)
    engine._kv_kurtosis_state[2]["k"].update(low)
    engine._kv_kurtosis_state[2]["v"].update(low)

    modes, profile = engine._profile_to_layer_modes()

    assert modes == ["fp32", "turbo-int8", "turbo-int8"]
    assert [entry["selected_mode"] for entry in profile] == modes
    assert profile[-1]["protected_terminal_band"] is True


def test_asymmetric_kv_modes_store_different_cache_types(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=16,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="turbo-int8",
        kv_key_precision="turbo-4bit",
        kv_value_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=0,
    )
    run = engine.generate([3, 5, 8], max_new_tokens=1)

    assert run["kv_key_precision"] == "turbo-4bit"
    assert run["kv_value_precision"] == "turbo-int8"
    assert isinstance(engine.caches[0]["k"], _Turbo4BitKVArray)
    assert isinstance(engine.caches[0]["v"], _TurboInt8KVArray)


def test_asymmetric_session_round_trip_persists_precisions(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    session_dir = tmp_path / "asymmetric-session"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=16,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision="turbo-int8",
        kv_key_precision="turbo-4bit",
        kv_value_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
    )
    run = engine.generate([3, 5, 8, 13], max_new_tokens=2)
    engine.save_session(session_dir, generated_ids=run["generated_ids"], last_logits=run["last_logits"])

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert meta["kv_key_precision"] == "turbo-4bit"
    assert meta["kv_value_precision"] == "turbo-int8"
    assert meta["kv_session_payload"]["asymmetric"] == {
        "k_mode": "turbo-4bit",
        "v_mode": "turbo-int8",
    }

    resumed_engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision="turbo-int8",
        kv_key_precision="turbo-4bit",
        kv_value_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=2,
    )
    loaded = resumed_engine.load_session(session_dir)

    assert loaded["kv_key_precision"] == "turbo-4bit"
    assert loaded["kv_value_precision"] == "turbo-int8"
    assert isinstance(resumed_engine.caches[0]["k"], _HotWindowKVArray)
    assert isinstance(resumed_engine.caches[0]["k"].cold, _Turbo4BitKVArray)
    assert isinstance(resumed_engine.caches[0]["v"], _HotWindowKVArray)
    assert isinstance(resumed_engine.caches[0]["v"].cold, _TurboInt8KVArray)


def test_merge_selective_candidate_indices_adds_new_cold_tokens() -> None:
    cached = np.array([[1, 4, 7], [0, 3, 5]], dtype=np.int64)
    merged = _merge_selective_candidate_indices(
        cached,
        cold_length=9,
        new_start=8,
        max_candidates=4,
    )

    assert merged.shape == (2, 4)
    assert merged[0, -1] == 8
    assert merged[1, -1] == 8


def test_merge_candidate_index_sets_prioritizes_current_candidates() -> None:
    current = np.array([[8, 9, 10], [7, 8, 9]], dtype=np.int64)
    stale = np.array([[6, 8, 9], [5, 7, 8]], dtype=np.int64)

    merged = _merge_candidate_index_sets(current, stale, max_candidates=4)

    assert merged.shape == (2, 4)
    assert merged[0].tolist() == [8, 9, 10, 6]
    assert merged[1].tolist() == [7, 8, 9, 5]


def test_block_shortlist_needs_expansion_flags_missing_upper_bound() -> None:
    block_scores = np.array([[0.90, 0.55, 0.95]], dtype=np.float32)
    shortlist_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)
    shortlist_scores = np.array([[0.88, 0.86, 0.54, 0.51]], dtype=np.float32)

    assert _block_shortlist_needs_expansion(
        block_scores,
        shortlist_indices,
        shortlist_scores,
        effective_topk=2,
        block_size=4,
        relative_margin=0.02,
    ) is True

    confident_scores = np.array([[0.90, 0.55, 0.60]], dtype=np.float32)
    assert _block_shortlist_needs_expansion(
        confident_scores,
        shortlist_indices,
        shortlist_scores,
        effective_topk=2,
        block_size=4,
        relative_margin=0.02,
    ) is False


def test_selected_score_threshold_returns_kth_largest_per_head() -> None:
    scores = np.array([[0.1, 0.8, 0.6], [0.9, 0.2, 0.3]], dtype=np.float32)

    threshold = _selected_score_threshold(scores, topk=2)

    assert np.allclose(threshold, np.array([0.6, 0.3], dtype=np.float32))


def test_tensor_to_numpy_exportable_converts_bfloat16() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.linspace(-1.0, 1.0, steps=6, dtype=torch.bfloat16).reshape(2, 3)

    array = _tensor_to_numpy_exportable(tensor)

    assert array.shape == (2, 3)
    assert array.dtype == np.float32
    assert np.allclose(array, tensor.to(dtype=torch.float32).numpy(), atol=1e-6)


def test_selective_index_cache_records_reuse_hits(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=4,
        kv_topk=4,
        kv_index_refresh_interval=8,
    )
    prompt_ids = [((idx * 7) + 3) % 64 for idx in range(32)]
    run = engine.generate(prompt_ids, max_new_tokens=2)

    assert run["kv_selective_stats"]["full_refreshes"] > 0
    assert run["kv_selective_stats"]["reuse_hits"] > 0


def test_selective_index_refresh_interval_one_disables_reuse(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=4,
        kv_topk=4,
        kv_index_refresh_interval=1,
    )
    prompt_ids = [((idx * 7) + 3) % 64 for idx in range(32)]
    run = engine.generate(prompt_ids, max_new_tokens=2)

    assert run["kv_selective_stats"]["full_refreshes"] > 0
    assert run["kv_selective_stats"]["reuse_hits"] == 0


def test_selective_block_scoring_records_pruned_steps(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=64,
            num_layers=1,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=4,
        kv_topk=4,
        kv_index_refresh_interval=1,
        kv_block_size=8,
    )
    prompt_ids = [((idx * 7) + 3) % 64 for idx in range(32)]
    run = engine.generate(prompt_ids, max_new_tokens=2)

    assert run["kv_selective_stats"]["block_pruned_steps"] > 0


def test_cross_layer_overlap_records_adjacent_pair_stats(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=96,
            num_layers=3,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=4,
        kv_topk=4,
        kv_index_refresh_interval=8,
    )
    prompt_ids = [((idx * 7) + 3) % 64 for idx in range(40)]
    run = engine.generate(prompt_ids, max_new_tokens=2)

    overlap = run["kv_cross_layer_overlap"]

    assert overlap["global_samples"] > 0
    assert len(overlap["adjacent_pairs"]) == 2
    assert any(pair["samples"] > 0 for pair in overlap["adjacent_pairs"])


def test_cross_layer_share_records_share_hits(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-gpt2"
    export_dir = tmp_path / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=64,
            hidden_size=32,
            max_position_embeddings=96,
            num_layers=3,
            num_heads=2,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=8, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        kv_cache_precision="turbo-int8",
        kv_rotation_mode="hadamard",
        kv_hot_window=4,
        kv_topk=4,
        kv_index_refresh_interval=8,
        kv_layer_share_stride=2,
    )
    prompt_ids = [((idx * 7) + 3) % 64 for idx in range(40)]
    run = engine.generate(prompt_ids, max_new_tokens=2)

    assert run["kv_selective_stats"]["cross_layer_share_hits"] > 0
    assert run["kv_selective_stats"]["cross_layer_share_candidate_rows"] > 0
