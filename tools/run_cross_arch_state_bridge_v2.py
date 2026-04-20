"""
run_cross_arch_state_bridge_v2.py
==================================
Cross-Architecture State Bridge v2 — mixes:
    * Local GPT-2 (transformer, 124M) + Zamba2 (hybrid-ssm, 1.2B) for
      bit-identity .hlx state proofs (their outputs are NOT used for task
      content).
    * DeepInfra cloud models for the actual task generation across the
      bridge (instruction-tuned models produce coherent continuations;
      the local base/tiny models in v1 degenerated into loops).

Claim split (explicit in the artifact):
    Claim-A  per_arch_bit_identity
        Local GPT-2 and local Zamba2 each serialise their internal state
        to .hlx and re-read it bit-identically within their own family.

    Claim-B  cross_model_task_continuity
        Two heterogeneous cloud models (default: Llama-3.2-3B vs
        Qwen2.5-7B) exchange a signed token+hmem bridge across four
        rounds, with a fifth round replaying Round-1 prompt against the
        same cloud model to measure regression.

No claim is made about bijective KV<->SSM transfer: the bridge is
tokenised + signed hmem, never a numerical state projection.

Usage
-----
DEEPINFRA_API_TOKEN must be set (the secure wrapper injects it).
    python tools/run_cross_arch_state_bridge_v2.py \
        --output-dir verification \
        --tokens-per-round 192 \
        --analyst-model meta-llama/Llama-3.2-3B-Instruct \
        --continuist-model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402
from tools.run_cross_arch_state_bridge_v1 import (  # noqa: E402
    GPT2_REF,
    ZAMBA_REF,
    _digest_token_ids,
    _free_model,
    _generate_and_capture_kv,
    _keyword_overlap,
    _load_model,
    _ref_cached,
    _sha256_bytes,
    _sha256_path,
    _sign_bridge_memory,
    _text_edit_distance,
    _utc_now,
    _write_hlx,
    _write_json,
)


DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"

DEFAULT_ANALYST_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_CONTINUIST_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TASK_PROMPT = (
    "Explain step by step how HeliX persists agentic state across "
    "architecturally heterogeneous local models (GPT-2 transformer and "
    "Zamba2 hybrid-SSM). Enumerate: (1) what each architecture stores "
    "internally, (2) why KV-cache and SSM hidden state cannot be "
    "numerically mapped, and (3) how Helix uses signed memory plus "
    "shared tokens to preserve task continuity across the swap."
)


async def _deepinfra_chat(
    *,
    model: str,
    system: str,
    user: str,
    token: str,
    max_tokens: int,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> dict[str, Any]:
    import httpx

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    retry_count = 0
    last_error: str | None = None
    while True:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{DEEPINFRA_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
        if resp.status_code in (429, 500, 502, 503, 504) and retry_count < 3:
            retry_count += 1
            last_error = f"{resp.status_code}"
            await asyncio.sleep(2 ** retry_count)
            continue
        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "requested_model": model,
                "actual_model": None,
                "text": "",
                "tokens_used": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "retry_count": retry_count,
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
            }
        data = resp.json()
        break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = str(data["choices"][0]["message"].get("content") or "").strip()
    actual_model = str(data.get("model") or model)
    tokens_used = int(data.get("usage", {}).get("total_tokens") or 0)
    return {
        "status": "ok",
        "requested_model": model,
        "actual_model": actual_model,
        "text": text,
        "tokens_used": tokens_used,
        "latency_ms": round(latency_ms, 3),
        "retry_count": retry_count,
        "last_retryable_error": last_error,
    }


def _local_state_proof(
    model_ref: str,
    *,
    trust_remote_code: bool,
    prompt_text: str,
    hlx_path: Path,
    tokens_to_capture: int,
) -> dict[str, Any]:
    """Generate a short trace from a local model JUST to serialize its
    KV/SSM state and prove bit-identity roundtrip on .hlx. The text is
    not used for task content."""
    model, tok, load_ms = _load_model(model_ref, trust_remote_code=trust_remote_code)
    try:
        gen = _generate_and_capture_kv(
            model, tok, prompt_text, max_new_tokens=tokens_to_capture
        )
        hlx = _write_hlx(hlx_path, gen["kv_blob"])
        return {
            "model_ref": model_ref,
            "load_time_ms": round(load_ms, 3),
            "generation_time_ms": round(gen["generation_time_ms"], 3),
            "captured_tokens": gen["generated_token_count"],
            "kv_repr": gen["kv_repr"],
            "hlx": hlx,
        }
    finally:
        _free_model([model, tok])


def run_cross_arch_state_bridge_v2(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "DEEPINFRA_API_TOKEN is not set. Run via the secure wrapper "
            "(run_cross_arch_state_bridge_v2_secure.ps1)."
        )
    if not _ref_cached(GPT2_REF):
        raise RuntimeError(f"GPT-2 not cached locally: {GPT2_REF}")
    if not _ref_cached(ZAMBA_REF):
        raise RuntimeError(f"Zamba2 not cached locally: {ZAMBA_REF}")

    run_id = args.run_id or f"cross-arch-state-bridge-v2-{uuid.uuid4().hex[:12]}"
    run_started = _utc_now()
    output_dir = Path(args.output_dir)
    workspace = output_dir / f"_cross-arch-state-bridge-v2-{run_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    hlx_dir = workspace / "hlx"
    hlx_dir.mkdir(parents=True, exist_ok=True)

    _prev_signing_mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE")
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = "ephemeral_preregistered"

    catalog = MemoryCatalog.open(workspace / "memory.sqlite")
    project = "cross-arch-state-bridge-v2"
    agents = {
        "analyst": "cloud-analyst",
        "continuist": "cloud-continuist",
        "scheduler": "scheduler-bridger",
    }

    rounds: list[dict[str, Any]] = []
    memory_chain: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []

    analyst_model = args.analyst_model
    continuist_model = args.continuist_model

    # --------------------------------------------------------------
    # R1 - analyst (cloud) generates + GPT-2 local serializes state
    # --------------------------------------------------------------
    system_analyst = (
        f"You are a technical analyst. Produce a concise, structured "
        f"answer. Be specific and avoid repetition."
    )
    r1_cloud = asyncio.run(_deepinfra_chat(
        model=analyst_model,
        system=system_analyst,
        user=TASK_PROMPT,
        token=token,
        max_tokens=args.tokens_per_round,
    ))
    r1_text = r1_cloud["text"] if r1_cloud["status"] == "ok" else ""

    lifecycle.append({"event": "local_state_proof_begin", "model": "gpt2", "at": _utc_now()})
    r1_local = _local_state_proof(
        GPT2_REF,
        trust_remote_code=False,
        prompt_text=TASK_PROMPT,
        hlx_path=hlx_dir / "r1_gpt2.hlx",
        tokens_to_capture=args.local_tokens,
    )
    lifecycle.append({"event": "local_state_proof_end", "model": "gpt2", "at": _utc_now()})

    r1_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["analyst"],
        run_id=run_id,
        memory_id_suffix="r1-analyst-output",
        summary=f"R1 analyst output ({analyst_model})",
        content=r1_text,
        tags=["r1", "analyst", "cloud"],
    )
    memory_chain.append({"round": 1, **r1_mem})
    rounds.append({
        "round": 1,
        "role": "analyst",
        "cloud": {
            "requested_model": analyst_model,
            "actual_model": r1_cloud.get("actual_model"),
            "status": r1_cloud["status"],
            "error": r1_cloud.get("error"),
            "latency_ms": r1_cloud.get("latency_ms"),
            "tokens_used": r1_cloud.get("tokens_used"),
            "output_digest": _digest_token_ids(list(r1_text.encode("utf-8"))),
            "output_preview": r1_text[:400],
        },
        "local_state_proof": {
            "arch": "transformer",
            "model_ref": GPT2_REF,
            **r1_local,
        },
        "memory": r1_mem,
    })

    # --------------------------------------------------------------
    # R2 - signed hmem bridge (cloud output + concept marker)
    # --------------------------------------------------------------
    bridge_content = (
        "SEMANTIC+TOKEN MIGRATION PACKET v2\n"
        f"Analyst model (R1): {analyst_model}\n"
        f"Continuist model (R3): {continuist_model}\n"
        f"Local state proofs: GPT-2 (transformer) at R1, "
        f"Zamba2 (hybrid-ssm) at R3.\n"
        f"Bridge kind: tokens+signed_hmem (NOT bijective KV->SSM).\n"
        f"R1 cloud output (first 400 chars):\n{r1_text[:400]}"
    )
    r2_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["scheduler"],
        run_id=run_id,
        memory_id_suffix="r2-bridge",
        summary="Cross-model bridge (cloud tokens + local state proofs)",
        content=bridge_content,
        tags=["r2", "bridge", "migration"],
    )
    memory_chain.append({"round": 2, **r2_mem})
    rounds.append({
        "round": 2,
        "role": "scheduler",
        "bridge_kind": "tokens+signed_hmem",
        "bridge_preview": bridge_content[:400],
        "memory": r2_mem,
    })

    # --------------------------------------------------------------
    # R3 - continuist (cloud) continues + Zamba2 local serializes state
    # --------------------------------------------------------------
    system_continuist = (
        "You are continuing a technical explanation written by a previous "
        "analyst. Do not repeat what they wrote. Focus on points (2) and "
        "(3). Be concise and avoid loops."
    )
    continuist_user = (
        f"{TASK_PROMPT}\n\n"
        f"[PRIOR ANALYST OUTPUT]\n{r1_text}\n[/PRIOR]\n\n"
        "Continue from here. Add substantively new content."
    )
    r3_cloud = asyncio.run(_deepinfra_chat(
        model=continuist_model,
        system=system_continuist,
        user=continuist_user,
        token=token,
        max_tokens=args.tokens_per_round,
    ))
    r3_text = r3_cloud["text"] if r3_cloud["status"] == "ok" else ""

    lifecycle.append({"event": "local_state_proof_begin", "model": "zamba", "at": _utc_now()})
    r3_local = _local_state_proof(
        ZAMBA_REF,
        trust_remote_code=True,
        prompt_text=TASK_PROMPT,
        hlx_path=hlx_dir / "r3_zamba.hlx",
        tokens_to_capture=args.local_tokens,
    )
    lifecycle.append({"event": "local_state_proof_end", "model": "zamba", "at": _utc_now()})

    r3_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["continuist"],
        run_id=run_id,
        memory_id_suffix="r3-continuist-output",
        summary=f"R3 continuist output ({continuist_model})",
        content=r3_text,
        tags=["r3", "continuist", "cloud"],
    )
    memory_chain.append({"round": 3, **r3_mem})
    rounds.append({
        "round": 3,
        "role": "continuist",
        "cloud": {
            "requested_model": continuist_model,
            "actual_model": r3_cloud.get("actual_model"),
            "status": r3_cloud["status"],
            "error": r3_cloud.get("error"),
            "latency_ms": r3_cloud.get("latency_ms"),
            "tokens_used": r3_cloud.get("tokens_used"),
            "output_digest": _digest_token_ids(list(r3_text.encode("utf-8"))),
            "output_preview": r3_text[:400],
        },
        "local_state_proof": {
            "arch": "hybrid-ssm-attention",
            "model_ref": ZAMBA_REF,
            **r3_local,
        },
        "memory": r3_mem,
    })

    # --------------------------------------------------------------
    # R4 - analyst closes + GPT-2 local restore-check of R1 .hlx
    # --------------------------------------------------------------
    r4_user = (
        f"{TASK_PROMPT}\n\n"
        f"[YOUR EARLIER OUTPUT]\n{r1_text}\n[/YOURS]\n\n"
        f"[CONTINUIST OUTPUT]\n{r3_text}\n[/CONTINUIST]\n\n"
        "Write a final synthesis paragraph that closes the explanation."
    )
    r4_cloud = asyncio.run(_deepinfra_chat(
        model=analyst_model,
        system=system_analyst,
        user=r4_user,
        token=token,
        max_tokens=args.tokens_per_round,
    ))
    r4_text = r4_cloud["text"] if r4_cloud["status"] == "ok" else ""

    r1_hlx_path = Path(r1_local["hlx"]["hlx_path"])
    r1_hlx_reread = _sha256_path(r1_hlx_path)
    r4_restore_check = {
        "restored_from": str(r1_hlx_path),
        "digest_original": r1_local["hlx"]["digest_pre_serialize"],
        "digest_reread": r1_hlx_reread,
        "bit_identity_post_restore": r1_hlx_reread == r1_local["hlx"]["digest_pre_serialize"],
    }

    r4_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["analyst"],
        run_id=run_id,
        memory_id_suffix="r4-analyst-closing",
        summary=f"R4 analyst closing ({analyst_model})",
        content=r4_text,
        tags=["r4", "analyst", "closing"],
    )
    memory_chain.append({"round": 4, **r4_mem})
    rounds.append({
        "round": 4,
        "role": "analyst-closing",
        "cloud": {
            "requested_model": analyst_model,
            "actual_model": r4_cloud.get("actual_model"),
            "status": r4_cloud["status"],
            "error": r4_cloud.get("error"),
            "latency_ms": r4_cloud.get("latency_ms"),
            "tokens_used": r4_cloud.get("tokens_used"),
            "output_digest": _digest_token_ids(list(r4_text.encode("utf-8"))),
            "output_preview": r4_text[:400],
        },
        "r1_hlx_restore_check": r4_restore_check,
        "memory": r4_mem,
    })

    # --------------------------------------------------------------
    # R5 - regression probe: replay R1 prompt against analyst model
    # --------------------------------------------------------------
    r5_cloud = asyncio.run(_deepinfra_chat(
        model=analyst_model,
        system=system_analyst,
        user=TASK_PROMPT,
        token=token,
        max_tokens=args.tokens_per_round,
    ))
    r5_text = r5_cloud["text"] if r5_cloud["status"] == "ok" else ""

    edit_dist = _text_edit_distance(r1_text, r5_text)
    overlap = _keyword_overlap(r1_text, r5_text)

    severity = "negligible"
    if overlap["jaccard"] < 0.4:
        severity = "severe"
    elif overlap["jaccard"] < 0.7:
        severity = "moderate"
    elif edit_dist > 0:
        severity = "minor"

    rounds.append({
        "round": 5,
        "role": "regression-probe",
        "cloud": {
            "requested_model": analyst_model,
            "actual_model": r5_cloud.get("actual_model"),
            "status": r5_cloud["status"],
            "error": r5_cloud.get("error"),
            "latency_ms": r5_cloud.get("latency_ms"),
            "tokens_used": r5_cloud.get("tokens_used"),
            "output_digest": _digest_token_ids(list(r5_text.encode("utf-8"))),
            "output_preview": r5_text[:400],
        },
        "r1_prompt_replay": True,
        "output_edit_distance_vs_r1": edit_dist,
        "keyword_overlap_vs_r1": overlap,
        "regression_severity": severity,
    })

    # --------------------------------------------------------------
    # Finalize artifact
    # --------------------------------------------------------------
    per_arch_bit_identity_ok = bool(
        r1_local["hlx"]["bit_identity"] and r3_local["hlx"]["bit_identity"]
    )
    cloud_statuses = [r.get("cloud", {}).get("status") for r in rounds if "cloud" in r]
    cloud_all_ok = all(s == "ok" for s in cloud_statuses)

    try:
        catalog.close()
    except Exception:
        pass
    if _prev_signing_mode is None:
        os.environ.pop("HELIX_RECEIPT_SIGNING_MODE", None)
    else:
        os.environ["HELIX_RECEIPT_SIGNING_MODE"] = _prev_signing_mode

    artifact = {
        "artifact": "local-cross-arch-state-bridge-v2",
        "schema_version": 2,
        "run_id": run_id,
        "run_started_utc": run_started,
        "run_ended_utc": _utc_now(),
        "status": "completed" if (per_arch_bit_identity_ok and cloud_all_ok) else "partial",
        "claim_boundary": (
            "Claim-A (per_arch_bit_identity): local GPT-2 (transformer) "
            "and local Zamba2 (hybrid-ssm) each serialise their internal "
            "state to .hlx and re-read it bit-identically within their "
            "own family. Claim-B (cross_model_task_continuity): two "
            "heterogeneous cloud models exchange a signed token+hmem "
            "bridge across four rounds with a regression probe on round "
            "five. No claim is made about bijective KV<->SSM transfer; "
            "that bridge is semantic/tokenised, not bijective."
        ),
        "cross_arch_bridge_kind": "tokens+signed_hmem",
        "per_arch_bit_identity_ok": per_arch_bit_identity_ok,
        "cloud_all_ok": cloud_all_ok,
        "local_models": [
            {"ref": GPT2_REF, "arch": "transformer", "role": "state_proof_r1"},
            {"ref": ZAMBA_REF, "arch": "hybrid-ssm-attention", "role": "state_proof_r3",
             "trust_remote_code": True},
        ],
        "cloud_models": {
            "analyst_requested": analyst_model,
            "continuist_requested": continuist_model,
        },
        "rounds": rounds,
        "model_lifecycle_events": lifecycle,
        "signed_memory_chain": memory_chain,
        "task_prompt_digest": _sha256_bytes(TASK_PROMPT.encode("utf-8")),
        "workspace": str(workspace),
        "receipt_signing_mode": "ephemeral_preregistered",
        "claims_not_allowed": [
            "This run does not claim bijective KV-cache to SSM hidden-state transfer.",
            "The cross-family continuity measured is task-level (tokens + "
            "signed memory) between cloud Transformer models of different "
            "families (Llama vs Qwen by default). The local GPT-2/Zamba2 "
            "pair demonstrates bit-identity state persistence only; their "
            "outputs are not part of the task content.",
        ],
    }

    artifact_path = output_dir / f"local-cross-arch-state-bridge-v2-{run_id}.json"
    _write_json(artifact_path, artifact)
    artifact["artifact_path"] = str(artifact_path)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-architecture state-bridge runner v2 (local state proofs + cloud task generation)."
    )
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--tokens-per-round", type=int, default=192,
                        help="max_tokens per DeepInfra call")
    parser.add_argument("--local-tokens", type=int, default=16,
                        help="Tokens generated locally just to capture KV/SSM state")
    parser.add_argument("--analyst-model", default=DEFAULT_ANALYST_MODEL)
    parser.add_argument("--continuist-model", default=DEFAULT_CONTINUIST_MODEL)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_cross_arch_state_bridge_v2(args)
    summary = {
        "artifact_path": artifact.get("artifact_path"),
        "status": artifact["status"],
        "per_arch_bit_identity_ok": artifact["per_arch_bit_identity_ok"],
        "cloud_all_ok": artifact["cloud_all_ok"],
        "rounds": len(artifact["rounds"]),
    }
    print(json.dumps(summary, indent=2))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
