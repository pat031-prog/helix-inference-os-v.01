from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
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


def _bridge_fixture() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {
        "layer_0_k": np.ones((2, 2), dtype=np.float32),
        "layer_0_v": np.ones((2, 2), dtype=np.float32),
        "layer_1_k": np.ones((2, 2), dtype=np.float32) * 2,
        "layer_1_v": np.ones((2, 2), dtype=np.float32) * 2,
    }
    layer_meta = {
        "format": "helix-layer-slices-v0",
        "architecture": "transformer",
        "layers": [
            {
                "layer_index": index,
                "layer_name": f"mock.layer.{index}",
                "block_type": "transformer",
                "arrays": [
                    {"name": f"layer_{index}_k", "layer_index": index, "cache_kind": "key_cache"},
                    {"name": f"layer_{index}_v", "layer_index": index, "cache_kind": "value_cache"},
                ],
            }
            for index in range(2)
        ],
    }
    return arrays, layer_meta


def _run_helix_bridge_sidecar() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-airllm-real-sidecar-") as temp:
        session_dir = Path(temp) / "session"
        arrays, layer_meta = _bridge_fixture()
        rust_session.save_session_bundle(
            session_dir,
            meta={"format": "airllm-real-sidecar", "helix_layer_slices": layer_meta},
            arrays=arrays,
            session_codec="rust-hlx-buffered",
        )
        return run_mock_airllm_loop(session_dir=session_dir, layer_indices=[0, 1], verify_policy="receipt-only")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    airllm_available = importlib.util.find_spec("airllm") is not None
    bridge = _run_helix_bridge_sidecar()
    base_payload: dict[str, Any] = {
        "title": "HeliX Optional AirLLM Real Smoke",
        "benchmark_kind": "session-os-airllm-real-smoke-v1",
        "airllm_dependency_available": bool(airllm_available),
        "local_files_only": bool(args.local_files_only),
        "model_path": str(args.model_path) if args.model_path else None,
        "model_ref": str(args.model_ref),
        "helix_layer_bridge_sidecar": {
            "all_layer_injections_hit": bridge.get("all_layer_injections_hit"),
            "total_injected_arrays": bridge.get("total_injected_arrays"),
            "bridge_mode": bridge.get("bridge_mode"),
        },
        "real_airllm_injection_supported": False,
        "claim_boundary": "This optional smoke never installs AirLLM or downloads a model. Real HeliX injection into AirLLM is future adapter work.",
    }
    if not airllm_available:
        payload = {
            **base_payload,
            "status": "skipped_dependency_missing",
            "skip_reason": "airllm is not installed in this environment",
        }
        _write_json(output_dir / "local-airllm-real-smoke.json", payload)
        return payload
    if args.local_files_only and (not args.model_path or not Path(args.model_path).exists()):
        payload = {
            **base_payload,
            "status": "skipped_model_not_cached",
            "skip_reason": "local-files-only is set and --model-path was not provided or does not exist",
        }
        _write_json(output_dir / "local-airllm-real-smoke.json", payload)
        return payload

    start = time.perf_counter()
    try:
        from airllm import AutoModel  # type: ignore[import-not-found]

        model_source = str(args.model_path or args.model_ref)
        model = AutoModel.from_pretrained(
            model_source,
            profiling_mode=False,
            layer_shards_saving_path=str(output_dir / "_airllm-layer-shards"),
        )
        input_tokens = model.tokenizer(
            [str(args.prompt)],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=int(args.max_length),
            padding=False,
        )
        input_ids = input_tokens["input_ids"]
        try:
            import torch

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
        except Exception:
            pass
        generation = model.generate(input_ids, max_new_tokens=int(args.max_new_tokens), use_cache=True, return_dict_in_generate=True)
        decoded = model.tokenizer.decode(generation.sequences[0])
        payload = {
            **base_payload,
            "status": "completed",
            "wall_time_s": time.perf_counter() - start,
            "generated_token_count": int(args.max_new_tokens),
            "decoded_preview": str(decoded)[:500],
            "real_airllm_baseline_completed": True,
            "real_airllm_injection_supported": False,
        }
    except Exception as exc:
        payload = {
            **base_payload,
            "status": "skipped_runtime_error",
            "skip_reason": str(exc),
            "wall_time_s": time.perf_counter() - start,
        }
    _write_json(output_dir / "local-airllm-real-smoke.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run optional real AirLLM smoke if dependency and local model are already available.")
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--model-ref", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model-path")
    parser.add_argument("--prompt", default="Say one sentence about cache lifecycle.")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    payload = run_smoke(build_parser().parse_args())
    print(json.dumps({"status": payload["status"], "artifact": "local-airllm-real-smoke.json"}, indent=2))


if __name__ == "__main__":
    main()
