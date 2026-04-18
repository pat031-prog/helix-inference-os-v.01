from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(max(1, min(2, int(torch.get_num_threads()))))

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_kv import rust_session  # noqa: E402
from helix_kv.transformers_cache import (  # noqa: E402
    _load_benchmark_cache,
    _load_causal_model,
    _load_text_adapter,
    _save_benchmark_cache,
)
from tools.run_local_hybrid_stress import (  # noqa: E402
    _compare_restore_equivalence,
    _encode_prompt_text,
    _json_ready,
    _run_generation_trace,
    _sync_device,
    _write_json,
)


PROFILE = {
    "gpt2": {
        "model_ref": "gpt2",
        "prompt_tokens": 128,
        "probe_tokens": 32,
        "max_new_tokens": 4,
        "variant": {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": 4,
            "kv_quant_seed": 7,
            "mamba_state_precision": "native-dense",
        },
    },
    "qwen": {
        "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_tokens": 64,
        "probe_tokens": 16,
        "max_new_tokens": 1,
        "variant": {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": 4,
            "kv_quant_seed": 7,
            "mamba_state_precision": "native-dense",
        },
    },
    "zamba": {
        "model_ref": "Zyphra/Zamba2-1.2B-Instruct-v2",
        "prompt_tokens": 64,
        "probe_tokens": 16,
        "max_new_tokens": 1,
        "variant": {
            "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": 4,
            "kv_quant_seed": 7,
            "mamba_state_precision": "q-mamba-dsq-int4",
            "mamba_state_block_size": 64,
            "mamba_state_scale_floor": 1e-8,
            "mamba_state_clip_threshold_pct": 2.0,
            "mamba_state_rel_rmse_threshold": 0.2,
            "mamba_state_auto_promote": False,
        },
    },
}

PROMPT = (
    "HeliX is testing local inference session snapshots. Explain briefly why deterministic save/load "
    "matters for a developer running agents on a small laptop."
)
PROBE = " Continue with one concrete engineering implication."


def _hf_model_cached(model_ref: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_ref.replace('/', '--')}"
    return cache_dir.exists()


def _artifact_name(model_key: str) -> str:
    return f"local-session-core-{model_key}.json"


def _write_toolchain(output_dir: Path) -> Path:
    return _write_json(output_dir / "local-session-core-toolchain.json", rust_session.toolchain_report())


def _load_model_bundle(model_ref: str, *, device: torch.device, local_files_only: bool) -> tuple[Any, Any, str]:
    model = _load_causal_model(model_ref, local_files_only=local_files_only, trust_remote_code=False)
    adapter, input_adapter, _, _ = _load_text_adapter(model_ref, local_files_only=local_files_only, trust_remote_code=False)
    model = model.to(device)
    model.eval()
    return model, adapter, input_adapter


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * float(pct)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _save_and_load_cache(
    cache: Any,
    *,
    model_config: Any,
    session_dir: Path,
    codec: str,
    device: torch.device,
    verify_policy: str = "full",
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    if session_dir.exists():
        import shutil

        shutil.rmtree(session_dir)
    start = time.perf_counter()
    _save_benchmark_cache(cache, model_config=model_config, path=session_dir, session_codec=codec, audit_policy=audit_policy)
    save_time_ms = (time.perf_counter() - start) * 1000.0
    verify_time_ms = None
    receipt: dict[str, Any] = {}
    receipt_path = session_dir / "session-hlx-receipt.json"
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            receipt = {}
    loaded_verify_policy = "receipt-only" if str(audit_policy) == "deferred" else verify_policy
    start = time.perf_counter()
    loaded = _load_benchmark_cache(session_dir, model_config=model_config, device=device, verify_policy=loaded_verify_policy)
    load_time_ms = (time.perf_counter() - start) * 1000.0
    if codec in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"} and str(audit_policy) == "deferred":
        verify_start = time.perf_counter()
        verify_receipt = rust_session.verify_deferred_session(session_dir)
        verify_time_ms = (time.perf_counter() - verify_start) * 1000.0
        receipt.update(verify_receipt)
        receipt["verify_receipt"] = verify_receipt
    elif codec in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
        verify_start = time.perf_counter()
        verify_receipt = rust_session.verify_hlx_session(session_dir)
        verify_time_ms = (time.perf_counter() - verify_start) * 1000.0
        receipt["verify_receipt"] = verify_receipt
        if "session_hash" not in receipt:
            inner = verify_receipt.get("receipt") if isinstance(verify_receipt.get("receipt"), dict) else verify_receipt
            receipt.update(inner)
    total_bytes = int(sum(item.stat().st_size for item in session_dir.iterdir() if item.is_file()))
    return {
        "cache": loaded,
        "save_time_ms": float(save_time_ms),
        "load_time_ms": float(load_time_ms),
        "verify_time_ms": verify_time_ms,
        "time_to_pending_ms": float(save_time_ms) if str(audit_policy) == "deferred" else None,
        "time_to_verified_ms": float(save_time_ms + (verify_time_ms or 0.0)) if str(audit_policy) == "deferred" else float(save_time_ms),
        "session_total_bytes": total_bytes,
        "receipt": receipt,
    }


def _repeat_codec_benchmark(
    cache: Any,
    *,
    model_config: Any,
    temp_root: Path,
    codec: str,
    device: torch.device,
    repeats: int,
    audit_policy: str = "blocking",
) -> dict[str, Any] | None:
    if int(repeats) <= 1 or codec not in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
        return None
    rows: list[dict[str, Any]] = []
    for repeat_index in range(int(repeats)):
        row = _save_and_load_cache(
            cache,
            model_config=model_config,
            session_dir=temp_root / f"repeat_{repeat_index:03d}",
            codec=codec,
            device=device,
            verify_policy="receipt-only",
            audit_policy=audit_policy,
        )
        rows.append(
            {
                "save_time_ms": row["save_time_ms"],
                "load_time_ms": row["load_time_ms"],
                "verify_time_ms": row["verify_time_ms"],
                "time_to_pending_ms": row["time_to_pending_ms"],
                "time_to_verified_ms": row["time_to_verified_ms"],
                "session_total_bytes": row["session_total_bytes"],
                "receipt": row["receipt"],
            }
        )
    save_times = [float(row["save_time_ms"]) for row in rows]
    load_times = [float(row["load_time_ms"]) for row in rows]
    verify_times = [float(row["verify_time_ms"] or 0.0) for row in rows]
    pending_times = [float(row["time_to_pending_ms"] or row["save_time_ms"]) for row in rows]
    verified_times = [float(row["time_to_verified_ms"] or row["save_time_ms"]) for row in rows]
    buffer_export_times = [float((row["receipt"] or {}).get("buffer_export_time_ms", 0.0)) for row in rows]
    rust_hash_times = [float((row["receipt"] or {}).get("hash_time_ms", 0.0)) for row in rows]
    rust_write_times = [float((row["receipt"] or {}).get("write_time_ms", 0.0)) for row in rows]
    flatten_copy_times = [float((row["receipt"] or {}).get("flatten_copy_time_ms", 0.0)) for row in rows]
    copied_counts = [int((row["receipt"] or {}).get("copied_array_count", 0)) for row in rows]
    original_array_counts = [int((row["receipt"] or {}).get("original_array_count", 0)) for row in rows]
    flat_group_counts = [int((row["receipt"] or {}).get("flat_group_count", 0)) for row in rows]
    buffer_spec_counts = [int((row["receipt"] or {}).get("buffer_spec_count", 0)) for row in rows]
    return {
        "repeat_count": len(rows),
        "save_time_ms_p50": _percentile(save_times, 0.50),
        "save_time_ms_p95": _percentile(save_times, 0.95),
        "load_time_ms_p50": _percentile(load_times, 0.50),
        "load_time_ms_p95": _percentile(load_times, 0.95),
        "verify_time_ms_p50": _percentile(verify_times, 0.50),
        "verify_time_ms_p95": _percentile(verify_times, 0.95),
        "time_to_pending_ms_p50": _percentile(pending_times, 0.50),
        "time_to_pending_ms_p95": _percentile(pending_times, 0.95),
        "time_to_verified_ms_p50": _percentile(verified_times, 0.50),
        "time_to_verified_ms_p95": _percentile(verified_times, 0.95),
        "buffer_export_time_ms_p50": _percentile(buffer_export_times, 0.50),
        "rust_hash_time_ms_p50": _percentile(rust_hash_times, 0.50),
        "rust_write_time_ms_p50": _percentile(rust_write_times, 0.50),
        "flatten_copy_time_ms_p50": _percentile(flatten_copy_times, 0.50),
        "copied_array_count_max": max(copied_counts) if copied_counts else 0,
        "original_array_count_median": int(statistics.median(original_array_counts)) if original_array_counts else 0,
        "flat_group_count_median": int(statistics.median(flat_group_counts)) if flat_group_counts else 0,
        "buffer_spec_count_median": int(statistics.median(buffer_spec_counts)) if buffer_spec_counts else 0,
        "session_total_bytes_median": int(statistics.median([int(row["session_total_bytes"]) for row in rows])),
    }


def _run_model_case(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    config = PROFILE[model_key]
    model_ref = str(config["model_ref"])
    output_dir = Path(args.output_dir)
    if args.local_files_only and not _hf_model_cached(model_ref):
        payload = {
            "model_key": model_key,
            "model_ref": model_ref,
            "status": "skipped",
            "skip_reason": "skipped_not_cached",
            "local_files_only": True,
        }
        _write_json(output_dir / _artifact_name(model_key), payload)
        return payload

    device = torch.device(args.device)
    model, adapter, input_adapter = _load_model_bundle(model_ref, device=device, local_files_only=bool(args.local_files_only))
    prompt_inputs, prompt_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=PROMPT,
        max_tokens=int(config["prompt_tokens"]),
    )
    probe_inputs, probe_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=PROBE,
        max_tokens=int(config["probe_tokens"]),
    )
    variant = dict(config["variant"])
    base = _run_generation_trace(
        model,
        prompt_inputs=prompt_inputs,
        prompt_ids=prompt_ids,
        variant=variant,
        max_new_tokens=0,
        adapter=adapter,
        device=device,
        return_cache=True,
    )
    base_cache = base.pop("_cache")
    with tempfile.TemporaryDirectory(prefix=f"helix-session-core-{model_key}-") as temp:
        temp_root = Path(temp)
        python_roundtrip = _save_and_load_cache(
            base_cache,
            model_config=model.config,
            session_dir=temp_root / "python",
            codec="python-npz",
            device=device,
        )
        rust_roundtrip = _save_and_load_cache(
            base_cache,
            model_config=model.config,
            session_dir=temp_root / "rust",
            codec=str(args.codec),
            device=device,
            verify_policy="full",
            audit_policy=str(args.audit_policy),
        )
        repeat_metrics = _repeat_codec_benchmark(
            base_cache,
            model_config=model.config,
            temp_root=temp_root,
            codec=str(args.codec),
            device=device,
            repeats=int(getattr(args, "repeats", 1)),
            audit_policy=str(args.audit_policy),
        )
        pre = _run_generation_trace(
            model,
            prompt_inputs=probe_inputs,
            prompt_ids=probe_ids,
            variant=variant,
            max_new_tokens=int(config["max_new_tokens"]),
            adapter=adapter,
            device=device,
            initial_cache=base_cache,
            resume_prompt_token_by_token=(model_key == "zamba"),
            capture_selection_logits=True,
        )
        post = _run_generation_trace(
            model,
            prompt_inputs=probe_inputs,
            prompt_ids=probe_ids,
            variant=variant,
            max_new_tokens=int(config["max_new_tokens"]),
            adapter=adapter,
            device=device,
            initial_cache=rust_roundtrip["cache"],
            resume_prompt_token_by_token=(model_key == "zamba"),
            capture_selection_logits=True,
        )
    comparison = _compare_restore_equivalence(pre, post)
    receipt = rust_roundtrip["receipt"].get("receipt") or rust_roundtrip["receipt"]
    payload = {
        "model_key": model_key,
        "model_ref": model_ref,
        "cache_kind": "hybrid" if model_key == "zamba" else "transformer",
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "prompt_token_count": len(prompt_ids),
        "max_new_tokens": int(config["max_new_tokens"]),
        "python_npz_save_time_ms": python_roundtrip["save_time_ms"],
        "python_npz_load_time_ms": python_roundtrip["load_time_ms"],
        "rust_hlx_save_time_ms": rust_roundtrip["save_time_ms"],
        "rust_hlx_load_time_ms": rust_roundtrip["load_time_ms"],
        "rust_hlx_verify_time_ms": rust_roundtrip["verify_time_ms"],
        "time_to_pending_ms": rust_roundtrip["time_to_pending_ms"],
        "time_to_verified_ms": rust_roundtrip["time_to_verified_ms"],
        "rust_hlx_buffered_save_time_ms": rust_roundtrip["save_time_ms"]
        if str(args.codec) == "rust-hlx-buffered"
        else None,
        "rust_hlx_buffered_load_time_ms": rust_roundtrip["load_time_ms"]
        if str(args.codec) == "rust-hlx-buffered"
        else None,
        "rust_hlx_buffered_verify_time_ms": rust_roundtrip["verify_time_ms"]
        if str(args.codec) == "rust-hlx-buffered"
        else None,
        "python_npz_session_bytes": python_roundtrip["session_total_bytes"],
        "rust_hlx_session_bytes": rust_roundtrip["session_total_bytes"],
        "rust_hlx_buffered_session_bytes": rust_roundtrip["session_total_bytes"]
        if str(args.codec) == "rust-hlx-buffered"
        else None,
        "rust_hlx_buffered_flat_save_time_ms": rust_roundtrip["save_time_ms"]
        if str(args.codec) == "rust-hlx-buffered-flat"
        else None,
        "rust_hlx_buffered_flat_load_time_ms": rust_roundtrip["load_time_ms"]
        if str(args.codec) == "rust-hlx-buffered-flat"
        else None,
        "rust_hlx_buffered_flat_verify_time_ms": rust_roundtrip["verify_time_ms"]
        if str(args.codec) == "rust-hlx-buffered-flat"
        else None,
        "rust_hlx_buffered_flat_session_bytes": rust_roundtrip["session_total_bytes"]
        if str(args.codec) == "rust-hlx-buffered-flat"
        else None,
        "rust_hlx_receipt": receipt,
        "repeat_benchmark": repeat_metrics,
        "audit_status": receipt.get("audit_status"),
        "session_hash": receipt.get("session_hash"),
        "merkle_root": receipt.get("merkle_root"),
        "hash_match": bool(rust_roundtrip["receipt"].get("ok", True)),
        "generated_ids_match": comparison["generated_ids_match"],
        "top1_match_all": comparison["top1_match_all"],
        "max_abs_logit_delta": comparison["max_abs_logit_delta"],
        "mean_abs_logit_delta": comparison["mean_abs_logit_delta"],
        "finite_before": comparison["finite_before"],
        "finite_after": comparison["finite_after"],
        "pre_answer_preview": pre.get("answer_preview"),
        "post_answer_preview": post.get("answer_preview"),
        "status": "completed",
        "skip_reason": None,
    }
    _sync_device(device)
    _write_json(output_dir / _artifact_name(model_key), payload)
    return payload


def _run_internal(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_toolchain(output_dir)
    payload = _run_model_case(str(args.internal_model), args)
    print(json.dumps(_json_ready(payload), indent=2))


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_toolchain(output_dir)
    selected = [item.strip() for item in str(args.models).split(",") if item.strip()]
    if bool(getattr(args, "transformer_only", False)):
        selected = [item for item in selected if item != "zamba"]
    results = []
    for model_key in selected:
        if model_key not in PROFILE:
            results.append({"model_key": model_key, "status": "skipped", "skip_reason": "unknown_model_key"})
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-model",
            model_key,
            "--profile",
            str(args.profile),
            "--codec",
            str(args.codec),
            "--output-dir",
            str(output_dir),
            "--device",
            str(args.device),
            "--repeats",
            str(args.repeats),
            "--audit-policy",
            str(args.audit_policy),
        ]
        if args.quick:
            command.append("--quick")
        if args.local_files_only:
            command.append("--local-files-only")
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(args.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            payload = {
                "model_key": model_key,
                "model_ref": PROFILE[model_key]["model_ref"],
                "status": "timeout",
                "skip_reason": f"timeout_{int(args.timeout_seconds)}s",
            }
            _write_json(output_dir / _artifact_name(model_key), payload)
            results.append(payload)
            continue
        if completed.returncode != 0:
            payload = {
                "model_key": model_key,
                "model_ref": PROFILE[model_key]["model_ref"],
                "status": "failed",
                "skip_reason": completed.stderr[-2000:] or completed.stdout[-2000:],
            }
            _write_json(output_dir / _artifact_name(model_key), payload)
            results.append(payload)
            continue
        artifact_path = output_dir / _artifact_name(model_key)
        results.append(json.loads(artifact_path.read_text(encoding="utf-8")))
    if str(args.audit_policy) == "deferred":
        summary_title = "HeliX Session Core v3 Deferred Audit"
    elif str(args.codec) == "rust-hlx-buffered-flat":
        summary_title = "HeliX Session Core v2 Tensor Flattening"
    elif str(args.codec) == "rust-hlx-buffered":
        summary_title = "HeliX Session Core v1 Zero-Copy"
    else:
        summary_title = "HeliX Local Session Core v0"
    baseline_summary: dict[str, Any] | None = None
    if str(args.codec) == "rust-hlx-buffered-flat":
        baseline_path = output_dir / "local-session-core-zero-copy-summary.json"
        if baseline_path.exists():
            try:
                baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                baseline_summary = None
    summary = {
        "title": summary_title,
        "profile": str(args.profile),
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "transformer_only": bool(getattr(args, "transformer_only", False)),
        "repeats": int(args.repeats),
        "models": results,
        "completed_count": sum(1 for item in results if item.get("status") == "completed"),
        "timeout_count": sum(1 for item in results if item.get("status") == "timeout"),
        "failed_count": sum(1 for item in results if item.get("status") == "failed"),
        "skipped_count": sum(1 for item in results if item.get("status") == "skipped"),
        "comparison_baseline": "local-session-core-zero-copy-summary.json" if baseline_summary else None,
    }
    if baseline_summary:
        summary["baseline_codec"] = baseline_summary.get("codec")
        summary["baseline_models_completed"] = baseline_summary.get("completed_count")
    _write_json(output_dir / "local-session-core-summary.json", summary)
    if str(args.codec) == "rust-hlx-buffered":
        _write_json(output_dir / "local-session-core-zero-copy-summary.json", summary)
    if str(args.codec) == "rust-hlx-buffered-flat" and str(args.audit_policy) != "deferred":
        _write_json(output_dir / "local-session-core-flattened-summary.json", summary)
    if str(args.audit_policy) == "deferred":
        _write_json(output_dir / "local-session-core-deferred-audit-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quick local HeliX session-core benchmarks.")
    parser.add_argument("--models", default="gpt2", help="Comma-separated model keys: gpt2,qwen,zamba")
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--codec", default="rust-hlx", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="blocking", choices=["blocking", "deferred"])
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--transformer-only", action="store_true")
    parser.add_argument("--internal-model", choices=sorted(PROFILE), default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.internal_model:
        _run_internal(args)
        return
    summary = _run_parent(args)
    print(json.dumps(_json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
