from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

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
from helix_kv.session_os import PrefixResolver, SessionCatalog  # noqa: E402
from helix_kv.transformers_cache import _layers_block_type, _load_benchmark_cache, _save_benchmark_cache  # noqa: E402
from tools.run_local_agent_memory_evidence import _make_prompt_text, _patch_session_metadata, _safe_ratio  # noqa: E402
from tools.run_local_hybrid_stress import _encode_prompt_text, _json_ready, _run_generation_trace, _write_json  # noqa: E402
from tools.run_local_session_core import PROFILE, _hf_model_cached, _load_model_bundle  # noqa: E402


def _hybrid_counts(model_config: Any) -> tuple[list[str], int, int]:
    block_types = _layers_block_type(getattr(model_config, "decoder_config", model_config))
    kv_count = sum(1 for item in block_types if str(item) == "hybrid")
    mamba_count = sum(1 for item in block_types if str(item) == "mamba")
    return block_types, kv_count, mamba_count


def _inputs_from_ids(token_ids: list[int]) -> dict[str, torch.Tensor]:
    ids = [int(item) for item in token_ids]
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
    }


def run_hybrid_prefix_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_key = str(args.model)
    if model_key != "zamba":
        payload = {"status": "skipped", "skip_reason": "only_zamba_supported_in_v0", "model_key": model_key}
        _write_json(output_dir / "local-hybrid-prefix-checkpoint-summary.json", payload)
        return payload
    config = PROFILE["zamba"]
    model_ref = str(config["model_ref"])
    if args.local_files_only and not _hf_model_cached(model_ref):
        payload = {
            "title": "HeliX Hybrid Prefix Checkpoint Summary",
            "benchmark_kind": "session-os-hybrid-prefix-checkpoint-v0",
            "status": "skipped",
            "skip_reason": "skipped_not_cached",
            "model_ref": model_ref,
            "local_files_only": True,
        }
        _write_json(output_dir / "local-hybrid-prefix-checkpoint-summary.json", payload)
        return payload
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    try:
        model, adapter, input_adapter = _load_model_bundle(model_ref, device=device, local_files_only=bool(args.local_files_only))
        needed_tokens = int(args.prefix_tokens) + int(args.followup_tokens)
        _, all_token_ids = _encode_prompt_text(
            model_ref=model_ref,
            adapter=adapter,
            input_adapter=input_adapter,
            prompt_text=_make_prompt_text(needed_tokens),
            max_tokens=needed_tokens,
        )
        prefix_ids = all_token_ids[: int(args.prefix_tokens)]
        followup_ids = all_token_ids[int(args.prefix_tokens) : int(args.prefix_tokens) + int(args.followup_tokens)]
        full_ids = prefix_ids + followup_ids
        variant = dict(config["variant"])
        block_types, kv_layer_count, mamba_layer_count = _hybrid_counts(model.config)
        with tempfile.TemporaryDirectory(prefix="helix-hybrid-prefix-checkpoint-") as temp:
            temp_root = Path(temp)
            catalog: SessionCatalog | None = None
            try:
                catalog = SessionCatalog.open(temp_root / "catalog.sqlite")
                for repeat_index in range(int(args.repeats)):
                    cold_result = _run_generation_trace(
                        model,
                        prompt_inputs=_inputs_from_ids(full_ids),
                        prompt_ids=full_ids,
                        variant=variant,
                        max_new_tokens=1,
                        adapter=adapter,
                        device=device,
                        capture_selection_logits=True,
                    )
                    cold_logits = torch.tensor(cold_result["selection_logits"][0], dtype=torch.float32)
                    cold_ttft_ms = float(cold_result["total_time_s"]) * 1000.0
                    prefix_result = _run_generation_trace(
                        model,
                        prompt_inputs=_inputs_from_ids(prefix_ids),
                        prompt_ids=prefix_ids,
                        variant=variant,
                        max_new_tokens=0,
                        adapter=adapter,
                        device=device,
                        return_cache=True,
                    )
                    prefix_cache = prefix_result.pop("_cache")
                    prefix_ms = float(prefix_result["total_time_s"]) * 1000.0
                    session_dir = temp_root / "sessions" / f"zamba_hybrid_prefix_{repeat_index:03d}"
                    if session_dir.exists():
                        shutil.rmtree(session_dir)
                    save_start = time.perf_counter()
                    _save_benchmark_cache(
                        prefix_cache,
                        model_config=model.config,
                        path=session_dir,
                        session_codec=str(args.codec),
                        audit_policy=str(args.audit_policy),
                    )
                    save_ms = (time.perf_counter() - save_start) * 1000.0
                    _patch_session_metadata(
                        session_dir,
                        {
                            "prefix_kind": "hybrid_checkpoint_v0",
                            "prefix_token_count": len(prefix_ids),
                            "session_token_ids": prefix_ids,
                            "layers_block_type": block_types,
                            "cache_position": len(prefix_ids),
                            "kv_layer_count": kv_layer_count,
                            "mamba_layer_count": mamba_layer_count,
                            "mamba_state_precision": str(variant.get("mamba_state_precision", "native-dense")),
                            "hybrid_prefix_note": "Exact checkpoint restore only; arbitrary Mamba prefix slicing is not used.",
                        },
                    )
                    receipt = {}
                    receipt_path = session_dir / "session-hlx-receipt.json"
                    if receipt_path.exists():
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    session_bytes = int(sum(item.stat().st_size for item in session_dir.iterdir() if item.is_file()))
                    catalog.record_session(
                        session_id=f"zamba-hybrid-prefix-{repeat_index:03d}",
                        model_id="zamba",
                        agent_id="hybrid-prefix-agent",
                        model_ref=model_ref,
                        arch="hybrid-mamba-transformer",
                        path=session_dir,
                        token_ids=prefix_ids,
                        session_bytes=session_bytes,
                        codec=str(args.codec),
                        audit_status=receipt.get("audit_status"),
                        session_hash=receipt.get("session_hash") or receipt.get("fast_payload_checksum"),
                        merkle_root=receipt.get("merkle_root"),
                    )
                    match = PrefixResolver(catalog).find_best_prefix(
                        model_id="zamba",
                        agent_id="hybrid-prefix-agent",
                        token_ids=full_ids,
                        arch="hybrid-mamba-transformer",
                    )
                    load_start = time.perf_counter()
                    loaded_cache = _load_benchmark_cache(
                        match.session.path if match.session else session_dir,
                        model_config=model.config,
                        device=device,
                        verify_policy="receipt-only" if str(args.audit_policy) == "deferred" else "full",
                    )
                    restore_ms = (time.perf_counter() - load_start) * 1000.0
                    warm_result = _run_generation_trace(
                        model,
                        prompt_inputs=_inputs_from_ids(full_ids[match.prefix_match_tokens :]),
                        prompt_ids=full_ids[match.prefix_match_tokens :],
                        variant=variant,
                        max_new_tokens=1,
                        adapter=adapter,
                        device=device,
                        initial_cache=loaded_cache,
                        resume_prompt_token_by_token=True,
                        capture_selection_logits=True,
                    )
                    warm_logits = torch.tensor(warm_result["selection_logits"][0], dtype=torch.float32)
                    warm_compute_ms = float(warm_result["total_time_s"]) * 1000.0
                    verify_start = time.perf_counter()
                    if str(args.audit_policy) == "deferred":
                        verified = rust_session.verify_deferred_session(session_dir)
                    elif str(args.codec) in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
                        verified = rust_session.verify_hlx_session(session_dir)
                    else:
                        verified = {}
                    verify_ms = (time.perf_counter() - verify_start) * 1000.0
                    delta = (cold_logits - warm_logits).abs()
                    top1_match = int(torch.argmax(cold_logits, dim=-1).item()) == int(torch.argmax(warm_logits, dim=-1).item())
                    generated_ids_match = list(cold_result.get("answer_token_ids") or []) == list(
                        warm_result.get("answer_token_ids") or []
                    )
                    rows.append(
                        {
                            "repeat_index": repeat_index,
                            "prefix_kind": "hybrid_checkpoint_v0",
                            "prefix_reuse_status": match.status,
                            "prefix_match_tokens": match.prefix_match_tokens,
                            "new_tokens_computed": match.new_tokens_computed,
                            "full_prompt_tokens": len(full_ids),
                            "prefill_saved_tokens": max(0, len(full_ids) - match.new_tokens_computed),
                            "cold_ttft_ms": float(cold_ttft_ms),
                            "prefix_compute_ms": float(prefix_ms),
                            "warm_prefix_ttft_ms": float(restore_ms + warm_compute_ms),
                            "warm_compute_only_ms": float(warm_compute_ms),
                            "restore_ms": float(restore_ms),
                            "time_to_pending_ms": float(save_ms),
                            "time_to_verified_ms": float(save_ms + verify_ms),
                            "session_total_bytes": session_bytes,
                            "generated_ids_match": generated_ids_match,
                            "top1_match": top1_match,
                            "max_abs_logit_delta": float(delta.max().item()),
                            "mean_abs_logit_delta": float(delta.mean().item()),
                            "finite_cold": bool(torch.isfinite(cold_logits).all().item()),
                            "finite_warm": bool(torch.isfinite(warm_logits).all().item()),
                            "audit_status": verified.get("audit_status"),
                        }
                    )
                    del prefix_cache, loaded_cache
                    gc.collect()
            finally:
                if catalog is not None:
                    catalog.close()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "title": "HeliX Hybrid Prefix Checkpoint Summary",
            "benchmark_kind": "session-os-hybrid-prefix-checkpoint-v0",
            "status": "failed",
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "model_ref": model_ref,
            "claim_boundary": "Hybrid prefix checkpoint failures do not affect Transformer prefix reuse claims.",
        }
        _write_json(output_dir / "local-hybrid-prefix-checkpoint-summary.json", payload)
        return payload
    speedups = [_safe_ratio(float(row["cold_ttft_ms"]), float(row["warm_prefix_ttft_ms"])) for row in rows]
    payload = {
        "title": "HeliX Hybrid Prefix Checkpoint Summary",
        "benchmark_kind": "session-os-hybrid-prefix-checkpoint-v0",
        "status": "completed",
        "model_key": "zamba",
        "model_ref": model_ref,
        "profile": str(args.profile),
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "prefix_kind": "hybrid_checkpoint_v0",
        "prefix_token_count": int(args.prefix_tokens),
        "followup_token_count": int(args.followup_tokens),
        "kv_layer_count": rows and kv_layer_count,
        "mamba_layer_count": rows and mamba_layer_count,
        "mamba_state_precision": str(PROFILE["zamba"]["variant"].get("mamba_state_precision")),
        "samples": rows,
        "best_speedup": max((value for value in speedups if value is not None), default=None),
        "top1_match_all": all(row.get("top1_match") is True for row in rows) if rows else None,
        "finite_all": all(row.get("finite_cold") is True and row.get("finite_warm") is True for row in rows) if rows else None,
        "claim_boundary": "Hybrid prefix v0 restores exact saved checkpoints only; it does not slice Mamba recurrent state at arbitrary token positions.",
    }
    _write_json(output_dir / "local-hybrid-prefix-checkpoint-summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local exact-prefix checkpoint restore for Zamba2 hybrid cache.")
    parser.add_argument("--model", default="zamba", choices=["zamba"])
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--followup-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--codec", default="rust-hlx-buffered-flat", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_hybrid_prefix_checkpoint(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
