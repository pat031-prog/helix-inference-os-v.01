from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix_kv import rust_session  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _layer_meta() -> dict[str, Any]:
    return {
        "format": "helix-layer-slices-v0",
        "architecture": "transformer",
        "layers": [
            {
                "layer_index": 0,
                "layer_name": "transformer.h.0",
                "block_type": "transformer",
                "token_start": 0,
                "token_count": 8,
                "arrays": [
                    {"name": "layer_0_k", "layer_index": 0, "layer_name": "transformer.h.0", "cache_kind": "key_cache"},
                    {"name": "layer_0_v", "layer_index": 0, "layer_name": "transformer.h.0", "cache_kind": "value_cache"},
                ],
            },
            {
                "layer_index": 1,
                "layer_name": "transformer.h.1",
                "block_type": "transformer",
                "token_start": 0,
                "token_count": 8,
                "arrays": [
                    {"name": "layer_1_k", "layer_index": 1, "layer_name": "transformer.h.1", "cache_kind": "key_cache"},
                    {"name": "layer_1_v", "layer_index": 1, "layer_name": "transformer.h.1", "cache_kind": "value_cache"},
                ],
            },
        ],
    }


def run_smoke(output_dir: Path, *, codec: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-layer-slice-") as temp:
        root = Path(temp)
        session_dir = root / "session"
        arrays = {
            "layer_0_k": np.arange(16, dtype=np.float32).reshape(2, 8),
            "layer_0_v": np.arange(16, 32, dtype=np.float32).reshape(2, 8),
            "layer_1_k": np.arange(32, 48, dtype=np.float32).reshape(2, 8),
            "layer_1_v": np.arange(48, 64, dtype=np.float32).reshape(2, 8),
            "unrelated_global": np.arange(4, dtype=np.int64),
        }
        receipt = rust_session.save_session_bundle(
            session_dir,
            meta={
                "format": "helix-layer-slice-smoke",
                "session_codec": codec,
                "helix_layer_slices": _layer_meta(),
            },
            arrays=arrays,
            session_codec=codec,
        )
        _, layer_arrays, layer_receipt = rust_session.load_hlx_layer_slice(session_dir, 1, verify_policy="receipt-only")
        _, missing_arrays, missing_receipt = rust_session.load_hlx_layer_slice(session_dir, 99, verify_policy="receipt-only")
        tamper_detected = False
        tamper_message = None
        if (session_dir / "kv_cache.hlx").exists():
            payload = bytearray((session_dir / "kv_cache.hlx").read_bytes())
            payload[-1] ^= 0xFF
            (session_dir / "kv_cache.hlx").write_bytes(payload)
            try:
                rust_session.verify_hlx_session(session_dir)
            except Exception as exc:  # noqa: BLE001
                tamper_detected = True
                tamper_message = str(exc)
        selected = layer_receipt.get("layer_slice", {})
        payload = {
            "title": "HeliX .hlx Layer Slice Smoke",
            "benchmark_kind": "session-os-hlx-layer-slice-v1",
            "status": "completed",
            "codec": codec,
            "session_codec_effective": receipt.get("session_codec"),
            "layer_index": 1,
            "selected_array_names": selected.get("selected_array_names", []),
            "declared_array_names": selected.get("declared_array_names", []),
            "selected_array_count": len(layer_arrays),
            "wrong_layer_status": missing_receipt.get("status") or missing_receipt.get("layer_slice", {}).get("status"),
            "wrong_layer_array_count": len(missing_arrays),
            "bytes_read": selected.get("bytes_read"),
            "read_mode": selected.get("read_mode"),
            "unrelated_array_loaded": "unrelated_global" in layer_arrays,
            "tamper_detected": tamper_detected,
            "tamper_message": tamper_message,
            "claim_boundary": "Layer-slice v1 validates selective cache-addressing metadata and CPU array reads; it does not use raw pointer injection.",
        }
        _write_json(output_dir / "local-hlx-layer-slice-smoke.json", payload)
        shutil.rmtree(root, ignore_errors=True)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny .hlx layer-slice smoke.")
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--codec", default="rust-hlx-buffered", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"])
    args = parser.parse_args()
    payload = run_smoke(Path(args.output_dir), codec=str(args.codec))
    print(json.dumps({"status": payload["status"], "artifact": str(Path(args.output_dir) / "local-hlx-layer-slice-smoke.json")}, indent=2))


if __name__ == "__main__":
    main()
