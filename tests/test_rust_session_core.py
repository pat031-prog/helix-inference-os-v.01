from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from helix_kv import rust_session


def test_rust_hlx_session_roundtrip_preserves_arrays(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    arrays = {
        "float32_values": np.arange(6, dtype=np.float32).reshape(2, 3),
        "float16_values": np.arange(4, dtype=np.float16),
        "int8_values": np.array([-1, 0, 1], dtype=np.int8),
        "uint8_values": np.array([0, 127, 255], dtype=np.uint8),
        "int64_values": np.array([42, 99], dtype=np.int64),
    }

    receipt = rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays=arrays,
        session_codec="rust-hlx",
    )
    meta, restored, verify = rust_session.load_session_bundle(session_dir)

    assert meta["format"] == "test-session"
    assert receipt["session_codec"] == "rust-hlx"
    assert verify["ok"] is True
    for name, array in arrays.items():
        assert restored[name].dtype == array.dtype
        assert restored[name].shape == array.shape
        np.testing.assert_array_equal(restored[name], array)


def test_python_npz_session_roundtrip_stays_default(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    arrays = {"values": np.arange(3, dtype=np.float32)}

    rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays=arrays,
        session_codec="python-npz",
    )
    meta, restored, receipt = rust_session.load_session_bundle(session_dir)

    assert (session_dir / "kv_cache.npz").exists()
    assert not (session_dir / "kv_cache.hlx").exists()
    assert meta["session_codec"] == "python-npz"
    assert receipt["session_codec"] == "python-npz"
    np.testing.assert_array_equal(restored["values"], arrays["values"])


def test_rust_hlx_buffered_roundtrip_and_copy_count(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    arrays = {
        "contiguous_values": base.copy(),
        "non_contiguous_values": base[:, ::2],
        "uint8_values": np.array([0, 127, 255], dtype=np.uint8),
    }

    receipt = rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays=arrays,
        session_codec="rust-hlx-buffered",
    )
    meta, restored, verify = rust_session.load_session_bundle(session_dir, verify_policy="receipt-only")

    assert meta["session_codec"] == "rust-hlx-buffered"
    assert receipt["session_codec"] == "rust-hlx-buffered"
    assert receipt["buffered_array_count"] == 3
    assert receipt["copied_array_count"] == 1
    assert verify["session_codec"] == "rust-hlx-buffered"
    for name, array in arrays.items():
        assert restored[name].dtype == array.dtype
        assert restored[name].shape == array.shape
        np.testing.assert_array_equal(restored[name], array)


def test_rust_hlx_buffered_flat_roundtrip_and_group_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    arrays = {
        "float32_values": base.copy(),
        "float32_non_contiguous": base[:, ::2],
        "float16_values": np.arange(4, dtype=np.float16),
        "int8_values": np.array([-1, 0, 1], dtype=np.int8),
        "uint8_values": np.array([0, 127, 255], dtype=np.uint8),
        "int64_values": np.array([42, 99], dtype=np.int64),
    }

    receipt = rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays=arrays,
        session_codec="rust-hlx-buffered-flat",
    )
    meta, restored, verify = rust_session.load_session_bundle(session_dir, verify_policy="receipt-only")

    assert meta["session_codec"] == "rust-hlx-buffered-flat"
    assert receipt["session_codec"] == "rust-hlx-buffered-flat"
    assert receipt["original_array_count"] == len(arrays)
    assert receipt["flat_group_count"] < len(arrays)
    assert receipt["buffer_spec_count"] == receipt["flat_group_count"]
    assert receipt["flatten_input_copied_array_count"] == 1
    assert verify["session_codec"] == "rust-hlx-buffered-flat"
    assert verify["flattened_arrays"]["original_array_count"] == len(arrays)
    for name, array in arrays.items():
        assert restored[name].dtype == array.dtype
        assert restored[name].shape == array.shape
        np.testing.assert_array_equal(restored[name], array)


def test_hlx_layer_slice_loads_only_requested_layer_arrays(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    arrays = {
        "layer_0_k": np.arange(4, dtype=np.float32),
        "layer_0_v": np.arange(4, 8, dtype=np.float32),
        "layer_1_k": np.arange(8, 12, dtype=np.float32),
        "layer_1_v": np.arange(12, 16, dtype=np.float32),
        "global": np.array([99], dtype=np.int64),
    }
    layer_meta = {
        "format": "helix-layer-slices-v0",
        "layers": [
            {
                "layer_index": 1,
                "layer_name": "transformer.h.1",
                "arrays": [
                    {"name": "layer_1_k", "layer_index": 1, "cache_kind": "key_cache"},
                    {"name": "layer_1_v", "layer_index": 1, "cache_kind": "value_cache"},
                ],
            }
        ],
    }

    rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session", "helix_layer_slices": layer_meta},
        arrays=arrays,
        session_codec="rust-hlx-buffered",
    )
    _, restored, receipt = rust_session.load_hlx_layer_slice(session_dir, 1, verify_policy="receipt-only")
    _, missing, missing_receipt = rust_session.load_hlx_layer_slice(session_dir, 99, verify_policy="receipt-only")

    assert sorted(restored) == ["layer_1_k", "layer_1_v"]
    np.testing.assert_array_equal(restored["layer_1_k"], arrays["layer_1_k"])
    assert receipt["layer_slice"]["status"] == "hit"
    assert receipt["layer_slice"]["read_mode"] == "direct-selected"
    assert missing == {}
    assert missing_receipt["status"] == "miss"


def test_rust_hlx_deferred_audit_roundtrip_and_verify(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    arrays = {
        "values": np.arange(16, dtype=np.float32).reshape(4, 4),
        "ids": np.arange(4, dtype=np.int64),
    }

    receipt = rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays=arrays,
        session_codec="rust-hlx-buffered-flat",
        audit_policy="deferred",
    )
    if receipt.get("audit_policy_effective") != "deferred":
        pytest.skip("pending PyO3 writer is unavailable in this environment")
    meta, restored, pending = rust_session.load_session_bundle(session_dir, verify_policy="receipt-only")

    assert meta["audit_status"] == "pending"
    assert pending["audit_status"] == "pending"
    assert pending["session_codec"] == "rust-hlx-buffered-flat"
    assert "session_hash" not in pending
    for name, array in arrays.items():
        np.testing.assert_array_equal(restored[name], array)

    verified = rust_session.verify_deferred_session(session_dir)
    assert verified["audit_status"] == "verified"
    assert verified["ok"] is True
    assert verified["session_hash"]
    _, _, full = rust_session.load_session_bundle(session_dir, verify_policy="full")
    assert full["audit_status"] == "verified"
    assert full["session_hash"] == verified["session_hash"]


def test_rust_hlx_deferred_audit_detects_tamper_after_verification(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    receipt = rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays={"values": np.array([1, 2, 3], dtype=np.uint8)},
        session_codec="rust-hlx-buffered-flat",
        audit_policy="deferred",
    )
    if receipt.get("audit_policy_effective") != "deferred":
        pytest.skip("pending PyO3 writer is unavailable in this environment")
    rust_session.verify_deferred_session(session_dir)
    bundle = session_dir / "kv_cache.hlx"
    payload = bytearray(bundle.read_bytes())
    payload[-1] ^= 0xFF
    bundle.write_bytes(payload)

    with pytest.raises(RuntimeError):
        rust_session.verify_deferred_session(session_dir)
    failed = json.loads((session_dir / "session-hlx-receipt.json").read_text(encoding="utf-8"))
    assert failed["audit_status"] == "failed"


def test_rust_hlx_verify_detects_tamper(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    rust_session.save_session_bundle(
        session_dir,
        meta={"format": "test-session"},
        arrays={"values": np.array([1, 2, 3], dtype=np.uint8)},
        session_codec="rust-hlx",
    )
    bundle = session_dir / "kv_cache.hlx"
    payload = bytearray(bundle.read_bytes())
    payload[-1] ^= 0xFF
    bundle.write_bytes(payload)

    with pytest.raises(RuntimeError):
        rust_session.verify_hlx_session(session_dir)


def test_toolchain_report_has_cli_or_module_status() -> None:
    report = rust_session.toolchain_report()

    assert "pyo3_module_available" in report
    assert report["cargo_manifest"].endswith("Cargo.toml")
