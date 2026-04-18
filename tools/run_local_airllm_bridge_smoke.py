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
from helix_kv.layer_bridge import run_mock_airllm_loop  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_smoke(output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-airllm-bridge-") as temp:
        root = Path(temp)
        session_dir = root / "session"
        arrays = {
            "layer_0_k": np.ones((2, 4), dtype=np.float32),
            "layer_0_v": np.ones((2, 4), dtype=np.float32) * 2,
            "layer_1_k": np.ones((2, 4), dtype=np.float32) * 3,
            "layer_1_v": np.ones((2, 4), dtype=np.float32) * 4,
            "layer_2_k": np.ones((2, 4), dtype=np.float32) * 5,
            "layer_2_v": np.ones((2, 4), dtype=np.float32) * 6,
        }
        layer_meta = {
            "format": "helix-layer-slices-v0",
            "architecture": "transformer",
            "layers": [
                {
                    "layer_index": index,
                    "layer_name": f"mock.layers.{index}",
                    "block_type": "transformer",
                    "arrays": [
                        {"name": f"layer_{index}_k", "layer_index": index, "layer_name": f"mock.layers.{index}", "cache_kind": "key_cache"},
                        {"name": f"layer_{index}_v", "layer_index": index, "layer_name": f"mock.layers.{index}", "cache_kind": "value_cache"},
                    ],
                }
                for index in range(3)
            ],
        }
        rust_session.save_session_bundle(
            session_dir,
            meta={"format": "helix-airllm-bridge-smoke", "helix_layer_slices": layer_meta},
            arrays=arrays,
            session_codec="rust-hlx-buffered",
        )
        result = run_mock_airllm_loop(session_dir=session_dir, layer_indices=[0, 1, 2], verify_policy="receipt-only")
        injected = [event for event in result["timeline"] if event.get("event") == "inject_layer_cache"]
        payload = {
            "title": "HeliX AirLLM Bridge Smoke",
            "benchmark_kind": "session-os-airllm-bridge-v1",
            "status": "completed",
            "airllm_dependency_required": False,
            "bridge_mode": result["bridge_mode"],
            "layer_indices": result["layer_indices"],
            "timeline": result["timeline"],
            "all_layer_injections_hit": result["all_layer_injections_hit"],
            "total_injected_arrays": result["total_injected_arrays"],
            "total_bytes_read": result["total_bytes_read"],
            "unrelated_layer_loaded_per_step": any(int(event.get("array_count") or 0) > 2 for event in injected),
            "future_fork_target": "Inject HeliX calls around AirLLM load_layer_to_cpu, move_layer_to_device, and per-layer forward loop after this seam is stable.",
            "claim_boundary": "This is a dependency-free adapter seam smoke, not a real AirLLM giant-model run.",
        }
        _write_json(output_dir / "local-airllm-bridge-smoke.json", payload)
        shutil.rmtree(root, ignore_errors=True)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dependency-free mock AirLLM bridge smoke.")
    parser.add_argument("--output-dir", default="verification")
    args = parser.parse_args()
    payload = run_smoke(Path(args.output_dir))
    print(json.dumps({"status": payload["status"], "artifact": str(Path(args.output_dir) / "local-airllm-bridge-smoke.json")}, indent=2))


if __name__ == "__main__":
    main()
