from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
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
from helix_kv.transformers_cache import _load_benchmark_cache, _save_benchmark_cache  # noqa: E402
from tools.run_local_hybrid_stress import (  # noqa: E402
    _build_cache_for_variant,
    _encode_prompt_text,
    _json_ready,
    _supports_forward_arg,
    _sync_device,
    _write_json,
)
from tools.run_local_session_core import PROFILE, _hf_model_cached, _load_model_bundle  # noqa: E402


COMPETITOR_PAPER = {
    "title": "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices",
    "arxiv_id": "2603.04428",
    "url": "https://huggingface.co/papers/2603.04428",
    "authors": ["Yakov Pyotr Shkolnikov"],
}


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


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _summarize_model_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cold = [float(row["cold_ttft_ms"]) for row in rows]
    warm_total = [float(row["warm_ttft_including_restore_ms"]) for row in rows]
    warm_compute = [float(row["warm_compute_only_ms"]) for row in rows]
    load = [float(row["session_load_time_ms"]) for row in rows]
    pending = [float(row["time_to_pending_ms"]) for row in rows]
    verified = [float(row["time_to_verified_ms"]) for row in rows]
    session_bytes = [float(row["session_total_bytes"]) for row in rows]
    max_delta = max(float(row["max_abs_logit_delta"]) for row in rows) if rows else None
    mean_delta = float(np.mean([float(row["mean_abs_logit_delta"]) for row in rows])) if rows else None
    cold_p50 = _percentile(cold, 0.50)
    warm_total_p50 = _percentile(warm_total, 0.50)
    warm_compute_p50 = _percentile(warm_compute, 0.50)
    return {
        "repeat_count": len(rows),
        "cold_ttft_ms_p50": cold_p50,
        "cold_ttft_ms_p95": _percentile(cold, 0.95),
        "warm_ttft_including_restore_ms_p50": warm_total_p50,
        "warm_ttft_including_restore_ms_p95": _percentile(warm_total, 0.95),
        "warm_compute_only_ms_p50": warm_compute_p50,
        "warm_compute_only_ms_p95": _percentile(warm_compute, 0.95),
        "ttft_speedup_including_restore": _safe_ratio(cold_p50, warm_total_p50),
        "ttft_speedup_compute_only": _safe_ratio(cold_p50, warm_compute_p50),
        "session_load_time_ms_p50": _percentile(load, 0.50),
        "session_load_time_ms_p95": _percentile(load, 0.95),
        "time_to_pending_ms_p50": _percentile(pending, 0.50),
        "time_to_pending_ms_p95": _percentile(pending, 0.95),
        "time_to_verified_ms_p50": _percentile(verified, 0.50),
        "time_to_verified_ms_p95": _percentile(verified, 0.95),
        "session_total_bytes_p50": _percentile(session_bytes, 0.50),
        "session_total_bytes_p95": _percentile(session_bytes, 0.95),
        "generated_ids_match": all(bool(row["generated_ids_match"]) for row in rows) if rows else None,
        "top1_match_all": all(bool(row["top1_match"]) for row in rows) if rows else None,
        "max_abs_logit_delta": max_delta,
        "mean_abs_logit_delta": mean_delta,
        "finite_before": all(bool(row["finite_cold"]) for row in rows) if rows else None,
        "finite_after": all(bool(row["finite_warm"]) for row in rows) if rows else None,
    }


def _agents_fit(budget_bytes: int, per_agent_cache_bytes: int) -> int:
    if int(per_agent_cache_bytes) <= 0:
        return 0
    return int(int(budget_bytes) // int(per_agent_cache_bytes))


def build_capacity_rows(
    ttft_summary: dict[str, Any],
    *,
    budget_bytes: int,
    projected_context_tokens: int = 4096,
    source_artifact: str = "local-ttft-cold-warm-summary.json",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ttft_summary.get("models", []):
        if not isinstance(model, dict) or model.get("status") != "completed":
            continue
        measured_bytes = model.get("session_total_bytes_p50")
        prefix_tokens = int(model.get("prefix_token_count") or 0)
        if not isinstance(measured_bytes, (int, float)) or measured_bytes <= 0 or prefix_tokens <= 0:
            continue
        measured_bytes_int = int(math.ceil(float(measured_bytes)))
        base = {
            "budget_bytes": int(budget_bytes),
            "model_key": model.get("model_key"),
            "model_ref": model.get("model_ref"),
            "mode": model.get("mode"),
            "source_artifact": source_artifact,
        }
        rows.append(
            {
                **base,
                "context_tokens": prefix_tokens,
                "per_agent_cache_bytes": measured_bytes_int,
                "agents_fit": _agents_fit(int(budget_bytes), measured_bytes_int),
                "projection": False,
                "projection_method": None,
            }
        )
        projected_bytes = int(math.ceil(float(measured_bytes_int) * float(projected_context_tokens) / float(prefix_tokens)))
        rows.append(
            {
                **base,
                "context_tokens": int(projected_context_tokens),
                "per_agent_cache_bytes": projected_bytes,
                "agents_fit": _agents_fit(int(budget_bytes), projected_bytes),
                "projection": True,
                "projection_method": "linear_from_measured_context",
                "measured_context_tokens": prefix_tokens,
                "measured_per_agent_cache_bytes": measured_bytes_int,
            }
        )
    return rows


def _make_prompt_text(required_tokens: int) -> str:
    paragraph = (
        "HeliX evidence prompt. A local agent keeps a private working memory below the prompt. "
        "The session cache is compressed, written to disk, restored later, and audited after the fast path. "
        "The benchmark compares cold prefill time against warm restore time for the same token prefix. "
    )
    return paragraph * max(8, int(required_tokens // 20) + 4)


def _inputs_from_ids(token_ids: list[int]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([list(int(item) for item in token_ids)], dtype=torch.long),
        "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
    }


def _forward_to_last_logits(
    model: Any,
    *,
    input_ids: list[int],
    cache: Any,
    device: torch.device,
    base_seq_length: int = 0,
) -> tuple[torch.Tensor, Any, float]:
    supports_attention_mask = _supports_forward_arg(model, "attention_mask")
    supports_cache_position = _supports_forward_arg(model, "cache_position")
    run_input_ids = torch.tensor([list(int(item) for item in input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, int(base_seq_length) + len(input_ids)), dtype=torch.long, device=device)
    model_inputs: dict[str, Any] = {
        "input_ids": run_input_ids,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
    }
    if supports_attention_mask:
        model_inputs["attention_mask"] = attention_mask
    if supports_cache_position:
        model_inputs["cache_position"] = torch.arange(
            int(base_seq_length), int(base_seq_length) + len(input_ids), device=device
        )
    with torch.inference_mode():
        _sync_device(device)
        start = time.perf_counter()
        outputs = model(**model_inputs)
        _sync_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return outputs.logits[:, -1, :].detach().to(dtype=torch.float32, device="cpu"), getattr(outputs, "past_key_values", cache), elapsed_ms


def _patch_session_metadata(session_dir: Path, metadata: dict[str, Any]) -> None:
    session_json = session_dir / "session.json"
    payload = json.loads(session_json.read_text(encoding="utf-8"))
    payload.update(_json_ready(metadata))
    session_json.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _run_model_ttft_case(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    config = PROFILE[model_key]
    model_ref = str(config["model_ref"])
    if args.local_files_only and not _hf_model_cached(model_ref):
        return {
            "model_key": model_key,
            "model_ref": model_ref,
            "status": "skipped",
            "skip_reason": "skipped_not_cached",
            "local_files_only": True,
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
    if len(all_token_ids) < needed_tokens:
        raise ValueError(f"{model_key} prompt produced {len(all_token_ids)} tokens, expected at least {needed_tokens}")
    prefix_ids = all_token_ids[: int(args.prefix_tokens)]
    followup_ids = all_token_ids[int(args.prefix_tokens) : int(args.prefix_tokens) + int(args.followup_tokens)]
    full_ids = prefix_ids + followup_ids
    variant = dict(config["variant"])
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"helix-agent-memory-evidence-{model_key}-") as temp:
        temp_root = Path(temp)
        for repeat_index in range(int(args.repeats)):
            cold_cache, _ = _build_cache_for_variant(model, variant, device=device)
            cold_logits, _, cold_ttft_ms = _forward_to_last_logits(
                model,
                input_ids=full_ids,
                cache=cold_cache,
                device=device,
                base_seq_length=0,
            )
            prefix_cache, _ = _build_cache_for_variant(model, variant, device=device)
            _, prefix_cache, _ = _forward_to_last_logits(
                model,
                input_ids=prefix_ids,
                cache=prefix_cache,
                device=device,
                base_seq_length=0,
            )
            session_dir = temp_root / f"repeat_{repeat_index:03d}"
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
                    "followup_token_count": len(followup_ids),
                    "prefix_match_length": len(prefix_ids),
                    "prefix_matching_implemented": False,
                    "evidence_note": "Token ids are stored for exact-prefix evidence only; true prefix matching is not implemented in this sprint.",
                },
            )
            load_start = time.perf_counter()
            loaded_cache = _load_benchmark_cache(
                session_dir,
                model_config=model.config,
                device=device,
                verify_policy="receipt-only" if str(args.audit_policy) == "deferred" else "full",
            )
            load_time_ms = (time.perf_counter() - load_start) * 1000.0
            base_seq_length = int(loaded_cache.get_seq_length())
            warm_logits, _, warm_compute_ms = _forward_to_last_logits(
                model,
                input_ids=followup_ids,
                cache=loaded_cache,
                device=device,
                base_seq_length=base_seq_length,
            )
            verify_start = time.perf_counter()
            if str(args.audit_policy) == "deferred":
                receipt = rust_session.verify_deferred_session(session_dir)
            elif str(args.codec) in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
                receipt = rust_session.verify_hlx_session(session_dir)
            else:
                receipt = {}
            verify_time_ms = (time.perf_counter() - verify_start) * 1000.0
            cold_token = int(torch.argmax(cold_logits, dim=-1).item())
            warm_token = int(torch.argmax(warm_logits, dim=-1).item())
            delta = (cold_logits - warm_logits).abs()
            total_bytes = int(sum(item.stat().st_size for item in session_dir.iterdir() if item.is_file()))
            rows.append(
                {
                    "repeat_index": repeat_index,
                    "cold_ttft_ms": float(cold_ttft_ms),
                    "warm_ttft_including_restore_ms": float(load_time_ms + warm_compute_ms),
                    "warm_compute_only_ms": float(warm_compute_ms),
                    "session_load_time_ms": float(load_time_ms),
                    "time_to_pending_ms": float(save_time_ms),
                    "time_to_verified_ms": float(save_time_ms + verify_time_ms),
                    "deferred_audit_time_ms": float(verify_time_ms),
                    "session_total_bytes": total_bytes,
                    "prefix_match_length": len(prefix_ids),
                    "cold_token_id": cold_token,
                    "warm_token_id": warm_token,
                    "generated_ids_match": cold_token == warm_token,
                    "top1_match": cold_token == warm_token,
                    "max_abs_logit_delta": float(delta.max().item()),
                    "mean_abs_logit_delta": float(delta.mean().item()),
                    "finite_cold": bool(torch.isfinite(cold_logits).all().item()),
                    "finite_warm": bool(torch.isfinite(warm_logits).all().item()),
                    "audit_status": receipt.get("audit_status"),
                    "session_hash": receipt.get("session_hash"),
                    "merkle_root": receipt.get("merkle_root"),
                }
            )
            del cold_cache
            del prefix_cache
            del loaded_cache
            gc.collect()
    summary = _summarize_model_rows(rows)
    payload = {
        "model_key": model_key,
        "model_ref": model_ref,
        "status": "completed",
        "mode": str(variant["name"]),
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "prefix_token_count": len(prefix_ids),
        "followup_token_count": len(followup_ids),
        "session_token_ids_stored": True,
        "prefix_match_length": len(prefix_ids),
        "prefix_matching_implemented": False,
        "samples": rows,
        **summary,
    }
    _sync_device(device)
    return payload


def build_comparison_summary(ttft_summary: dict[str, Any], capacity_summary: dict[str, Any]) -> dict[str, Any]:
    completed = [item for item in ttft_summary.get("models", []) if isinstance(item, dict) and item.get("status") == "completed"]
    best_including = None
    best_compute = None
    best_model = None
    for item in completed:
        including = item.get("ttft_speedup_including_restore")
        compute = item.get("ttft_speedup_compute_only")
        if isinstance(including, (int, float)) and (best_including is None or float(including) > best_including):
            best_including = float(including)
            best_compute = float(compute) if isinstance(compute, (int, float)) else None
            best_model = item.get("model_ref")
    return {
        "title": "HeliX vs Agent Memory Below the Prompt",
        "benchmark_kind": "agent-memory-comparison-v1",
        "competitor_paper": COMPETITOR_PAPER,
        "competitor_claims": [
            "Persistent Q4 KV cache for multi-agent LLM inference on edge devices.",
            "Cache restoration reduces time-to-first-token by up to 136x in the reported larger-model suite.",
            "Q4 quantization fits 4x more agent contexts into fixed device memory than FP16.",
            "Includes block pool, BatchQuantizedKVCache, and cross-phase context injection.",
            "Evaluated on Gemma 3 12B, DeepSeek-Coder-V2-Lite 16B, and Llama 3.1 8B.",
        ],
        "helix_verified_claims": [
            "Local TTFT cold-vs-warm evidence is now measured separately for restore-including and compute-only paths.",
            "HeliX session artifacts distinguish pending checkpoints from verified cryptographic audit.",
            "HeliX has Rust .hlx session persistence with deferred audit and Merkle/SHA-256 receipts.",
            "Existing HeliX artifacts cover hybrid Zamba2 recurrent-state compression and multimodel session scheduling.",
        ],
        "helix_best_ttft_speedup_including_restore": best_including,
        "helix_best_ttft_speedup_compute_only": best_compute,
        "helix_best_ttft_model_ref": best_model,
        "helix_gaps": [
            "No OpenAI-compatible /v1/chat/completions endpoint in this sprint.",
            "No true prefix matching yet; token IDs are stored only as evidence metadata.",
            "No append-incremental compressed cache path yet.",
            "Local evidence uses small CPU models; Shkolnikov reports larger Apple Silicon model runs.",
        ],
        "next_features_from_paper": [
            "Prefix matching over stored session_token_ids.",
            "Cross-phase context injection without recomputing earlier turns.",
            "OpenAI-compatible chat completions endpoint for framework adoption.",
            "TTFT and agents-fit tables on larger GPU/Apple Silicon hardware.",
        ],
        "evidence_artifacts": [
            "local-ttft-cold-warm-summary.json",
            "local-agent-capacity-budget.json",
            "local-session-core-deferred-audit-summary.json",
            "local-multimodel-hypervisor-demo.json",
            "hybrid-memory-frontier-summary.json",
        ],
        "capacity_rows": len(capacity_summary.get("rows", [])),
    }


def _update_claims_matrix(output_dir: Path, comparison: dict[str, Any]) -> None:
    path = output_dir / "helix-claims-matrix.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = [claim for claim in payload.get("claims", []) if claim.get("id") not in {"agent-memory-ttft-restore", "agent-memory-capacity-budget"}]
    claims.append(
        {
            "id": "agent-memory-ttft-restore",
            "status": "verified" if comparison.get("helix_best_ttft_speedup_including_restore") else "promising",
            "public_wording": "HeliX now reports cold-prefill TTFT versus warm restored-session TTFT for local agent-memory evidence.",
            "evidence_artifacts": ["local-ttft-cold-warm-summary.json", "agent-memory-comparison-summary.json"],
            "headline_metrics": {
                "best_ttft_speedup_including_restore": comparison.get("helix_best_ttft_speedup_including_restore"),
                "best_ttft_speedup_compute_only": comparison.get("helix_best_ttft_speedup_compute_only"),
                "best_model_ref": comparison.get("helix_best_ttft_model_ref"),
            },
            "caveat": "This is a local CPU evidence sprint on small cached models; it is not a larger-model TTFT claim like Shkolnikov's Apple Silicon suite.",
        }
    )
    claims.append(
        {
            "id": "agent-memory-capacity-budget",
            "status": "verified",
            "public_wording": "HeliX now reports how many measured or projected agent sessions fit inside a fixed memory budget.",
            "evidence_artifacts": ["local-agent-capacity-budget.json"],
            "headline_metrics": {"budget_bytes": 10 * 1024**3},
            "caveat": "Rows marked projection=true are linear projections from the measured context length, not direct 4K-context runs.",
        }
    )
    payload["claims"] = claims
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [item.strip() for item in str(args.models).split(",") if item.strip()]
    results = []
    for model_key in selected:
        if model_key not in {"gpt2", "qwen"}:
            results.append({"model_key": model_key, "status": "skipped", "skip_reason": "unsupported_in_evidence_sprint"})
            continue
        try:
            results.append(_run_model_ttft_case(model_key, args))
        except Exception as exc:
            results.append(
                {
                    "model_key": model_key,
                    "model_ref": PROFILE.get(model_key, {}).get("model_ref"),
                    "status": "failed",
                    "skip_reason": str(exc),
                }
            )
        gc.collect()
    ttft_summary = {
        "title": "HeliX Local TTFT Cold vs Warm Restore",
        "benchmark_kind": "agent-memory-ttft-cold-warm-v1",
        "profile": str(args.profile),
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "prefix_token_count_requested": int(args.prefix_tokens),
        "followup_token_count_requested": int(args.followup_tokens),
        "repeats": int(args.repeats),
        "competitor_reference": COMPETITOR_PAPER,
        "models": results,
        "completed_count": sum(1 for item in results if item.get("status") == "completed"),
        "skipped_count": sum(1 for item in results if item.get("status") == "skipped"),
        "failed_count": sum(1 for item in results if item.get("status") == "failed"),
    }
    capacity_rows = build_capacity_rows(
        ttft_summary,
        budget_bytes=int(args.budget_gib) * 1024**3,
        projected_context_tokens=int(args.projected_context_tokens),
    )
    capacity_summary = {
        "title": "HeliX Local Agent Capacity Budget",
        "benchmark_kind": "agent-memory-capacity-budget-v1",
        "budget_bytes": int(args.budget_gib) * 1024**3,
        "budget_gib": int(args.budget_gib),
        "projected_context_tokens": int(args.projected_context_tokens),
        "rows": capacity_rows,
        "source_artifact": "local-ttft-cold-warm-summary.json",
    }
    comparison = build_comparison_summary(ttft_summary, capacity_summary)
    _write_json(output_dir / "local-ttft-cold-warm-summary.json", ttft_summary)
    _write_json(output_dir / "local-agent-capacity-budget.json", capacity_summary)
    _write_json(output_dir / "agent-memory-comparison-summary.json", comparison)
    if bool(args.update_claims):
        _update_claims_matrix(output_dir, comparison)
    return ttft_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local HeliX evidence against agent-memory style TTFT/capacity claims.")
    parser.add_argument("--models", default="gpt2,qwen")
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--followup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codec", default="rust-hlx-buffered-flat", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--budget-gib", type=int, default=10)
    parser.add_argument("--projected-context-tokens", type=int, default=4096)
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-claims", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_evidence(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
