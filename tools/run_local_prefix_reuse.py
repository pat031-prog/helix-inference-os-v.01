from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
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
from helix_kv.transformers_cache import _load_benchmark_cache, _save_benchmark_cache  # noqa: E402
from tools.run_local_agent_memory_evidence import (  # noqa: E402
    _forward_to_last_logits,
    _make_prompt_text,
    _patch_session_metadata,
    _safe_ratio,
    _summarize_model_rows,
)
from tools.run_local_hybrid_stress import _build_cache_for_variant, _encode_prompt_text, _json_ready, _write_json  # noqa: E402
from tools.run_local_session_core import PROFILE, _hf_model_cached, _load_model_bundle  # noqa: E402


def _native_dense_variant() -> dict[str, Any]:
    return {
        "name": "native-dense",
        "kv_cache_precision": "native-dense",
        "kv_rotation_mode": "qr",
        "kv_hot_window": 0,
        "mamba_state_precision": "native-dense",
    }


def _run_model_prefix_case(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    config = PROFILE[model_key]
    model_ref = str(config["model_ref"])
    arch = "hybrid-mamba-transformer" if model_key == "zamba" else "transformer"
    if args.local_files_only and not _hf_model_cached(model_ref):
        return {
            "model_key": model_key,
            "model_ref": model_ref,
            "status": "skipped",
            "skip_reason": "skipped_not_cached",
            "local_files_only": True,
        }
    if arch != "transformer":
        return {
            "model_key": model_key,
            "model_ref": model_ref,
            "status": "skipped",
            "skip_reason": "unsupported_hybrid_v0",
            "prefix_reuse_status": "unsupported_hybrid_v0",
        }
    device = torch.device(args.device)
    model, adapter, input_adapter = _load_model_bundle(model_ref, device=device, local_files_only=bool(args.local_files_only))
    needed_tokens = int(args.prefix_tokens) + int(args.followup_tokens)
    prompt_inputs, all_token_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=_make_prompt_text(needed_tokens),
        max_tokens=needed_tokens,
    )
    del prompt_inputs
    prefix_ids = all_token_ids[: int(args.prefix_tokens)]
    followup_ids = all_token_ids[int(args.prefix_tokens) : int(args.prefix_tokens) + int(args.followup_tokens)]
    full_ids = prefix_ids + followup_ids
    variants: list[dict[str, Any]] = []
    if bool(args.include_native_dense):
        variants.append(_native_dense_variant())
    variants.append(dict(config["variant"]))
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"helix-prefix-reuse-{model_key}-") as temp:
        temp_root = Path(temp)
        catalog = SessionCatalog.open(temp_root / "catalog.sqlite")
        for cache_variant in variants:
            variant_name = str(cache_variant.get("name") or "variant")
            for repeat_index in range(int(args.repeats)):
                cold_cache, _ = _build_cache_for_variant(model, cache_variant, device=device)
                cold_logits, _, cold_ttft_ms = _forward_to_last_logits(
                    model,
                    input_ids=full_ids,
                    cache=cold_cache,
                    device=device,
                    base_seq_length=0,
                )
                prefix_cache, _ = _build_cache_for_variant(model, cache_variant, device=device)
                _, prefix_cache, _ = _forward_to_last_logits(
                    model,
                    input_ids=prefix_ids,
                    cache=prefix_cache,
                    device=device,
                    base_seq_length=0,
                )
                session_dir = temp_root / "sessions" / f"{model_key}_{variant_name}_{repeat_index:03d}"
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
                save_time_ms = (time.perf_counter() - save_start) * 1000.0
                _patch_session_metadata(
                    session_dir,
                    {
                        "session_token_ids": prefix_ids,
                        "prefix_token_count": len(prefix_ids),
                        "prefix_matching_implemented": True,
                        "prefix_reuse_status": "transformer_exact_prefix_v0",
                        "cache_reuse_mode": variant_name,
                    },
                )
                receipt = {}
                receipt_path = session_dir / "session-hlx-receipt.json"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                session_bytes = int(sum(item.stat().st_size for item in session_dir.iterdir() if item.is_file()))
                catalog.record_session(
                    session_id=f"{model_key}-{variant_name}-{repeat_index:03d}",
                    model_id=f"{model_key}:{variant_name}",
                    agent_id="prefix-agent",
                    model_ref=model_ref,
                    arch=arch,
                    path=session_dir,
                    token_ids=prefix_ids,
                    session_bytes=session_bytes,
                    codec=str(args.codec),
                    audit_status=receipt.get("audit_status"),
                    session_hash=receipt.get("session_hash") or receipt.get("fast_payload_checksum"),
                    merkle_root=receipt.get("merkle_root"),
                )
                match = PrefixResolver(catalog).find_best_prefix(
                    model_id=f"{model_key}:{variant_name}",
                    agent_id="prefix-agent",
                    token_ids=full_ids,
                    arch=arch,
                )
                load_start = time.perf_counter()
                loaded_cache = _load_benchmark_cache(
                    match.session.path if match.session else session_dir,
                    model_config=model.config,
                    device=device,
                    verify_policy="receipt-only" if str(args.audit_policy) == "deferred" else "full",
                )
                restore_ms = (time.perf_counter() - load_start) * 1000.0
                warm_logits, _, warm_compute_ms = _forward_to_last_logits(
                    model,
                    input_ids=full_ids[match.prefix_match_tokens :],
                    cache=loaded_cache,
                    device=device,
                    base_seq_length=int(loaded_cache.get_seq_length()),
                )
                verify_start = time.perf_counter()
                if str(args.audit_policy) == "deferred":
                    verified = rust_session.verify_deferred_session(session_dir)
                elif str(args.codec) in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
                    verified = rust_session.verify_hlx_session(session_dir)
                else:
                    verified = {}
                verify_ms = (time.perf_counter() - verify_start) * 1000.0
                delta = (cold_logits - warm_logits).abs()
                rows.append(
                    {
                        "repeat_index": repeat_index,
                        "cache_reuse_mode": variant_name,
                        "cache_reuse_scope": "native_dense_exact_prefix" if variant_name == "native-dense" else "compressed_exact_prefix",
                        "prefix_reuse_status": match.status,
                        "prefix_match_tokens": match.prefix_match_tokens,
                        "new_tokens_computed": match.new_tokens_computed,
                        "full_prompt_tokens": len(full_ids),
                        "prefill_saved_tokens": max(0, len(full_ids) - match.new_tokens_computed),
                        "prefix_hit": match.status == "hit",
                        "cold_ttft_ms": float(cold_ttft_ms),
                        "warm_prefix_ttft_ms": float(restore_ms + warm_compute_ms),
                        "warm_compute_only_ms": float(warm_compute_ms),
                        "restore_ms": float(restore_ms),
                        "time_to_pending_ms": float(save_time_ms),
                        "time_to_verified_ms": float(save_time_ms + verify_ms),
                        "session_total_bytes": session_bytes,
                        "generated_ids_match": int(torch.argmax(cold_logits, dim=-1).item()) == int(torch.argmax(warm_logits, dim=-1).item()),
                        "top1_match": int(torch.argmax(cold_logits, dim=-1).item()) == int(torch.argmax(warm_logits, dim=-1).item()),
                        "max_abs_logit_delta": float(delta.max().item()),
                        "mean_abs_logit_delta": float(delta.mean().item()),
                        "finite_cold": bool(torch.isfinite(cold_logits).all().item()),
                        "finite_warm": bool(torch.isfinite(warm_logits).all().item()),
                        "audit_status": verified.get("audit_status"),
                    }
                )
                del cold_cache, prefix_cache, loaded_cache
                gc.collect()
        catalog.close()
    summary = _summarize_model_rows(
        [
            {
                **row,
                "warm_ttft_including_restore_ms": row["warm_prefix_ttft_ms"],
                "session_load_time_ms": row["restore_ms"],
            }
            for row in rows
        ]
    )
    cold_p50 = summary.get("cold_ttft_ms_p50")
    warm_p50 = summary.get("warm_ttft_including_restore_ms_p50")
    variant_summaries: dict[str, Any] = {}
    for variant_name in sorted({str(row.get("cache_reuse_mode")) for row in rows}):
        variant_rows = [row for row in rows if row.get("cache_reuse_mode") == variant_name]
        variant_summary = _summarize_model_rows(
            [
                {
                    **row,
                    "warm_ttft_including_restore_ms": row["warm_prefix_ttft_ms"],
                    "session_load_time_ms": row["restore_ms"],
                }
                for row in variant_rows
            ]
        )
        variant_summary["prefix_hit_rate"] = (
            sum(1 for row in variant_rows if row.get("prefix_hit")) / len(variant_rows) if variant_rows else None
        )
        variant_summary["prefill_saved_tokens_p50"] = (
            variant_rows[len(variant_rows) // 2]["prefill_saved_tokens"] if variant_rows else None
        )
        variant_summary["cache_reuse_mode"] = variant_name
        variant_summary["cache_reuse_scope"] = (
            variant_rows[0].get("cache_reuse_scope") if variant_rows else None
        )
        variant_summaries[variant_name] = variant_summary
    dense_summary = variant_summaries.get("native-dense")
    if dense_summary is None:
        dense_equivalence_status = "not_run"
    elif dense_summary.get("max_abs_logit_delta") == 0.0 and dense_summary.get("top1_match_all") is True:
        dense_equivalence_status = "bit_exact"
    elif dense_summary.get("top1_match_all") is True:
        dense_equivalence_status = "top1_equivalent"
    else:
        dense_equivalence_status = "diagnostic_mismatch"
    claim_candidates = [
        (name, data)
        for name, data in variant_summaries.items()
        if data.get("top1_match_all") is True
        and data.get("generated_ids_match") is True
        and data.get("finite_before") is True
        and data.get("finite_after") is True
    ]
    claim_variant = None
    claim_summary: dict[str, Any] = {}
    if claim_candidates:
        claim_variant, claim_summary = max(
            claim_candidates,
            key=lambda item: float(item[1].get("ttft_speedup_including_restore") or 0.0),
        )
    return {
        "model_key": model_key,
        "model_ref": model_ref,
        "status": "completed",
        "arch": arch,
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "prefix_matching_implemented": True,
        "prefix_reuse_scope": "transformer_exact_prefix_v0",
        "cache_reuse_modes": [str(variant.get("name")) for variant in variants],
        "prefix_token_count": len(prefix_ids),
        "followup_token_count": len(followup_ids),
        "samples": rows,
        "variant_summaries": variant_summaries,
        "claim_variant": claim_variant,
        "claim_speedup_including_restore": claim_summary.get("ttft_speedup_including_restore"),
        "claim_top1_match_all": claim_summary.get("top1_match_all"),
        "claim_generated_ids_match": claim_summary.get("generated_ids_match"),
        "claim_max_abs_logit_delta": claim_summary.get("max_abs_logit_delta"),
        "dense_equivalence_status": dense_equivalence_status,
        "dense_max_abs_logit_delta": dense_summary.get("max_abs_logit_delta") if dense_summary else None,
        "prefix_hit_rate": sum(1 for row in rows if row.get("prefix_hit")) / len(rows) if rows else None,
        "prefix_match_tokens_p50": summary.get("prefix_match_tokens_p50") or rows[len(rows) // 2]["prefix_match_tokens"],
        "new_tokens_computed_p50": rows[len(rows) // 2]["new_tokens_computed"] if rows else None,
        "prefill_saved_tokens_p50": rows[len(rows) // 2]["prefill_saved_tokens"] if rows else None,
        "speedup": _safe_ratio(cold_p50, warm_p50),
        **summary,
    }


def run_prefix_reuse(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [item.strip() for item in str(args.models).split(",") if item.strip()]
    results = []
    for model_key in selected:
        if model_key not in PROFILE:
            results.append({"model_key": model_key, "status": "skipped", "skip_reason": "unknown_model_key"})
            continue
        results.append(_run_model_prefix_case(model_key, args))
    payload = {
        "title": "HeliX Local Prefix Reuse TTFT Summary",
        "benchmark_kind": "session-os-prefix-reuse-v0",
        "profile": str(args.profile),
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "models": results,
        "claim_boundary": "Prefix reuse v0 is exact-prefix and Transformer-only; hybrid Mamba/Transformer sessions use full restore.",
    }
    _write_json(output_dir / "local-prefix-reuse-ttft-summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Transformer-only prefix reuse smoke for HeliX Session OS.")
    parser.add_argument("--models", default="gpt2,qwen")
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--followup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-native-dense", dest="include_native_dense", action="store_false")
    parser.set_defaults(include_native_dense=True)
    parser.add_argument("--codec", default="rust-hlx-buffered-flat", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_prefix_reuse(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
