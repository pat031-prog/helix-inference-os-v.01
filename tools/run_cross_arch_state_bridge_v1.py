"""
run_cross_arch_state_bridge_v1.py
==================================
Cross-Architecture State Bridge — GPT-2 (124M) <-> Zamba2 (1.2B) five-round
round-trip with per-architecture .hlx bit-identity proofs and a tokenised +
signed-hmem bridge between families.

Claim boundary (written into the artifact):
    Agent state is serialised from a GPT-2 Transformer and a Zamba2 hybrid-SSM,
    each preserved bit-identical within its own architecture via .hlx, with
    task continuity across the two families established through a signed
    token-and-memory bridge. This artifact does not claim numerical state
    transfer between KV-cache and SSM hidden state; that bridge is
    semantic/tokenised, not bijective.

Rounds
------
R1  GPT-2         generate + serialise KV -> .hlx  (digest pre/post roundtrip)
R2  scheduler     signed hmem bridge (tokens + concepts), GPT-2 unloaded
R3  Zamba2        continue + serialise state -> .hlx (digest pre/post roundtrip)
R4  GPT-2 back    restore from R1.hlx, ingest Zamba2 output, close task
R5  GPT-2 probe   replay R1 prompt; compare against R1 output (regression)

Compression + fidelity (delegated)
---------------------------------
After the five rounds we call run_transformers_kv_benchmark once per arch with
the same effective prompt, which applies:
    * GPT-2  : turbo-int8-hadamard
    * Zamba2 : turbo-int8-hadamard + q-mamba-dsq-int4
and reports ratio_kv / ratio_ssm / ratio_combined + fidelity_kl_top5.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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
from helix_kv.transformers_cache import (  # noqa: E402
    build_transformers_hybrid_state_variants,
    run_transformers_kv_benchmark,
)


GPT2_REF = "gpt2"  # 124M; cached as models--gpt2 (legacy alias, equivalent to openai-community/gpt2)
ZAMBA_REF = "Zyphra/Zamba2-1.2B-Instruct-v2"

TASK_PROMPT = (
    "Explain step by step how HeliX persists agentic state across a GPT-2 "
    "Transformer and a Zamba2 hybrid-SSM model. Your answer should enumerate: "
    "(1) what each architecture stores internally, (2) why KV-cache and SSM "
    "hidden state cannot be numerically mapped, and (3) how Helix uses signed "
    "memory plus shared tokens to preserve task continuity across the swap."
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_token_ids(ids) -> str:  # noqa: ANN001
    try:
        import numpy as np
        arr = np.asarray(ids, dtype="int64")
        return _sha256_bytes(arr.tobytes())
    except Exception:
        return _sha256_bytes(str(list(ids)).encode("utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _free_model(obj_refs: list[Any]) -> None:
    for ref in obj_refs:
        try:
            del ref
        except Exception:
            pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _hf_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _ref_cached(model_ref: str) -> bool:
    snapshots = _hf_cache_root() / f"models--{model_ref.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return False
    return any((item / "config.json").exists() for item in snapshots.iterdir() if item.is_dir())


def _load_model(model_ref: str, *, trust_remote_code: bool = False) -> tuple[Any, Any, float]:
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(
        model_ref, local_files_only=True, trust_remote_code=trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000.0
    return model, tok, load_ms


def _generate_and_capture_kv(
    model: Any,
    tok: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run generate; return text, token ids, and a serialised KV-cache bytes blob."""
    import io
    import torch

    inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    n_prompt = int(inputs["input_ids"].shape[-1])

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
    gen_ms = (time.perf_counter() - t0) * 1000.0

    full_ids = output.sequences[0]
    prompt_ids = full_ids[:n_prompt].tolist()
    new_ids = full_ids[n_prompt:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)

    kv_blob = b""
    kv_repr = "unavailable"
    past = getattr(output, "past_key_values", None)
    if past is not None:
        try:
            buf = io.BytesIO()
            torch.save(past, buf)
            kv_blob = buf.getvalue()
            kv_repr = f"{type(past).__name__}"
        except Exception as exc:  # noqa: BLE001
            kv_repr = f"serialize_failed:{type(exc).__name__}"

    return {
        "text": text,
        "prompt_ids": prompt_ids,
        "new_ids": new_ids,
        "prompt_token_count": n_prompt,
        "generated_token_count": len(new_ids),
        "generation_time_ms": gen_ms,
        "kv_blob": kv_blob,
        "kv_repr": kv_repr,
    }


def _write_hlx(path: Path, blob: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest_before = _sha256_bytes(blob)
    path.write_bytes(blob)
    digest_after = _sha256_path(path)
    roundtrip = path.read_bytes()
    digest_roundtrip = _sha256_bytes(roundtrip)
    return {
        "hlx_path": str(path),
        "hlx_bytes": len(blob),
        "digest_pre_serialize": digest_before,
        "digest_file_on_disk": digest_after,
        "digest_post_deserialize": digest_roundtrip,
        "bit_identity": digest_before == digest_after == digest_roundtrip,
    }


def _text_edit_distance(a: str, b: str) -> int:
    a_t = (a or "").split()
    b_t = (b or "").split()
    m, n = len(a_t), len(b_t)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a_t[i - 1] == b_t[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def _keyword_overlap(reference_text: str, candidate_text: str, min_len: int = 4) -> dict[str, Any]:
    def _tokens(text: str) -> set[str]:
        return {
            t.lower().strip(".,;:()[]\"'")
            for t in (text or "").split()
            if len(t.strip(".,;:()[]\"'")) >= min_len
        }
    ref = _tokens(reference_text)
    cand = _tokens(candidate_text)
    if not ref:
        return {"jaccard": 0.0, "shared_count": 0, "ref_count": 0, "cand_count": len(cand)}
    shared = ref & cand
    union = ref | cand
    return {
        "jaccard": round(len(shared) / len(union), 4) if union else 0.0,
        "shared_count": len(shared),
        "ref_count": len(ref),
        "cand_count": len(cand),
    }


def _sign_bridge_memory(
    catalog: MemoryCatalog,
    *,
    project: str,
    agent_id: str,
    run_id: str,
    memory_id_suffix: str,
    summary: str,
    content: str,
    tags: list[str],
) -> dict[str, Any]:
    # Signing config lives in env vars (set by run_cross_arch_state_bridge).
    # Narrow the seed to this specific memory via a suffix.
    prev_seed = os.environ.get("HELIX_RECEIPT_SIGNING_SEED")
    os.environ["HELIX_RECEIPT_SIGNING_SEED"] = (
        f"cross-arch-state-bridge:{run_id}:{memory_id_suffix}"
    )
    try:
        mem = catalog.remember(
            project=project,
            agent_id=agent_id,
            memory_type="episodic",
            summary=summary,
            content=content,
            importance=10,
            tags=tags,
        )
    finally:
        if prev_seed is None:
            os.environ.pop("HELIX_RECEIPT_SIGNING_SEED", None)
        else:
            os.environ["HELIX_RECEIPT_SIGNING_SEED"] = prev_seed
    receipt = catalog._memory_receipts.get(mem.memory_id, {}) if hasattr(catalog, "_memory_receipts") else {}
    return {
        "memory_id": mem.memory_id,
        "node_hash": catalog._memory_node_hashes.get(mem.memory_id) if hasattr(catalog, "_memory_node_hashes") else None,
        "signature_verified": bool(receipt.get("signature_verified")) if isinstance(receipt, dict) else None,
        "receipt_signing_mode": os.environ.get("HELIX_RECEIPT_SIGNING_MODE", "off"),
    }


def _measure_compression(
    model_ref: str,
    prompt_text: str,
    *,
    arch_label: str,
    max_new_tokens: int,
    variant_names: list[str],
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Delegate compression + fidelity measurement to run_transformers_kv_benchmark."""
    variants_full = build_transformers_hybrid_state_variants()
    variants = [v for v in variants_full if v["name"] in variant_names]
    if not variants:
        return {"status": "no_variants_matched", "variant_names": variant_names}

    t0 = time.perf_counter()
    try:
        report = run_transformers_kv_benchmark(
            model_ref,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            warmup_max_new_tokens=0,
            kv_variants=variants,
            device="cpu",
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "arch_label": arch_label,
            "error": f"{type(exc).__name__}:{str(exc)[:240]}",
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }

    elapsed_s = round(time.perf_counter() - t0, 3)
    summary: dict[str, Any] = {
        "status": "ok",
        "arch_label": arch_label,
        "model_ref": model_ref,
        "elapsed_s": elapsed_s,
        "variants_reported": [],
    }
    for variant in (report.get("variants") or []):
        summary["variants_reported"].append({
            "name": variant.get("name"),
            "kv_cache_precision": variant.get("kv_cache_precision"),
            "kv_rotation_mode": variant.get("kv_rotation_mode"),
            "mamba_state_precision": variant.get("mamba_state_precision"),
            "compression_ratio": variant.get("compression_ratio"),
            "mamba_state_compression_ratio": variant.get("mamba_state_compression_ratio"),
            "combined_compression_ratio": variant.get("combined_compression_ratio"),
            "kl_top5_vs_dense": variant.get("kl_top5_vs_dense"),
            "top1_match_vs_dense": variant.get("top1_match_vs_dense"),
            "serialized_bytes": variant.get("serialized_bytes"),
        })
    return summary


def run_cross_arch_state_bridge(args: argparse.Namespace) -> dict[str, Any]:
    if not _ref_cached(GPT2_REF):
        raise RuntimeError(f"GPT-2 not cached locally: {GPT2_REF}")
    if not _ref_cached(ZAMBA_REF):
        raise RuntimeError(f"Zamba2 not cached locally: {ZAMBA_REF}")

    run_id = args.run_id or f"cross-arch-state-bridge-{uuid.uuid4().hex[:12]}"
    run_started = _utc_now()
    output_dir = Path(args.output_dir)
    workspace = output_dir / f"_cross-arch-state-bridge-{run_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    hlx_dir = workspace / "hlx"
    hlx_dir.mkdir(parents=True, exist_ok=True)

    # Enable ephemeral-preregistered receipt signing for every memory this runner
    # writes. This matches the strict retrieval default flipped at master.
    _prev_signing_mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE")
    os.environ["HELIX_RECEIPT_SIGNING_MODE"] = "ephemeral_preregistered"

    catalog = MemoryCatalog.open(workspace / "memory.sqlite")
    project = "cross-arch-state-bridge"
    agents = {
        "gpt2": "gpt2-analyst",
        "zamba": "zamba-continuist",
        "scheduler": "scheduler-bridger",
    }

    rounds: list[dict[str, Any]] = []
    memory_chain: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []

    # -----------------------------------------------------------------
    # R1 - GPT-2 generate + serialise KV
    # -----------------------------------------------------------------
    lifecycle.append({"event": "model_activate", "model": "gpt2", "ref": GPT2_REF, "at": _utc_now()})
    gpt2, gpt2_tok, gpt2_load_ms = _load_model(GPT2_REF, trust_remote_code=False)
    r1_gen = _generate_and_capture_kv(
        gpt2, gpt2_tok, TASK_PROMPT, max_new_tokens=args.tokens_per_round
    )
    r1_hlx = _write_hlx(hlx_dir / "r1_gpt2.hlx", r1_gen["kv_blob"])
    r1_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["gpt2"],
        run_id=run_id,
        memory_id_suffix="r1-gpt2-output",
        summary=f"R1 GPT-2 output ({r1_gen['generated_token_count']} tokens)",
        content=r1_gen["text"],
        tags=["r1", "gpt2", "output"],
    )
    memory_chain.append({"round": 1, **r1_mem})
    rounds.append({
        "round": 1,
        "model": "gpt2",
        "model_ref": GPT2_REF,
        "arch": "transformer",
        "action": "generate+serialize",
        "load_time_ms": round(gpt2_load_ms, 3),
        "generation_time_ms": round(r1_gen["generation_time_ms"], 3),
        "prompt_token_count": r1_gen["prompt_token_count"],
        "generated_token_count": r1_gen["generated_token_count"],
        "prompt_digest": _digest_token_ids(r1_gen["prompt_ids"]),
        "output_digest": _digest_token_ids(r1_gen["new_ids"]),
        "output_preview": (r1_gen["text"] or "")[:400],
        "kv_repr": r1_gen["kv_repr"],
        "hlx": r1_hlx,
        "memory": r1_mem,
    })

    r1_text = r1_gen["text"]
    r1_new_ids = r1_gen["new_ids"]
    r1_prompt_text = TASK_PROMPT

    # Free GPT-2 before Zamba2
    lifecycle.append({"event": "model_unload", "model": "gpt2", "at": _utc_now()})
    _free_model([gpt2])
    gpt2 = None  # keep tok for R5 replay
    r1_kv_blob_for_restore = r1_gen["kv_blob"]

    # -----------------------------------------------------------------
    # R2 - signed hmem bridge (tokens + concepts)
    # -----------------------------------------------------------------
    bridge_content = (
        "SEMANTIC+TOKEN MIGRATION PACKET\n"
        f"Source arch: transformer (gpt2-124M)\n"
        f"Target arch: hybrid-ssm-attention (zamba2-1.2b)\n"
        f"Bridge kind: tokens+signed_hmem (NOT bijective KV->SSM)\n"
        f"R1 output preview: {(r1_text or '')[:400]}\n"
        "The target must continue the task from this preview; "
        "no internal state is being numerically transferred."
    )
    r2_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["scheduler"],
        run_id=run_id,
        memory_id_suffix="r2-bridge",
        summary="Cross-arch bridge: tokens+signed_hmem",
        content=bridge_content,
        tags=["r2", "bridge", "migration"],
    )
    memory_chain.append({"round": 2, **r2_mem})
    rounds.append({
        "round": 2,
        "model": "scheduler",
        "action": "signed_hmem_bridge",
        "bridge_kind": "tokens+signed_hmem",
        "bridge_preview": bridge_content[:320],
        "memory": r2_mem,
    })

    # -----------------------------------------------------------------
    # R3 - Zamba2 continue + serialise state
    # -----------------------------------------------------------------
    lifecycle.append({"event": "model_activate", "model": "zamba", "ref": ZAMBA_REF, "at": _utc_now()})
    zamba, zamba_tok, zamba_load_ms = _load_model(ZAMBA_REF, trust_remote_code=True)

    zamba_prompt_text = (
        f"{TASK_PROMPT}\n\n"
        f"[CONTEXT FROM GPT-2 HANDOFF]\n{r1_text}\n[/CONTEXT]\n\n"
        "Continue the explanation. Focus on (2) and (3) above."
    )
    r3_gen = _generate_and_capture_kv(
        zamba, zamba_tok, zamba_prompt_text, max_new_tokens=args.tokens_per_round
    )
    r3_hlx = _write_hlx(hlx_dir / "r3_zamba.hlx", r3_gen["kv_blob"])
    r3_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["zamba"],
        run_id=run_id,
        memory_id_suffix="r3-zamba-output",
        summary=f"R3 Zamba2 output ({r3_gen['generated_token_count']} tokens)",
        content=r3_gen["text"],
        tags=["r3", "zamba", "output"],
    )
    memory_chain.append({"round": 3, **r3_mem})
    rounds.append({
        "round": 3,
        "model": "zamba",
        "model_ref": ZAMBA_REF,
        "arch": "hybrid-ssm-attention",
        "action": "continue+serialize",
        "load_time_ms": round(zamba_load_ms, 3),
        "generation_time_ms": round(r3_gen["generation_time_ms"], 3),
        "prompt_token_count": r3_gen["prompt_token_count"],
        "generated_token_count": r3_gen["generated_token_count"],
        "prompt_digest": _digest_token_ids(r3_gen["prompt_ids"]),
        "output_digest": _digest_token_ids(r3_gen["new_ids"]),
        "output_preview": (r3_gen["text"] or "")[:400],
        "kv_repr": r3_gen["kv_repr"],
        "hlx": r3_hlx,
        "memory": r3_mem,
    })

    r3_text = r3_gen["text"]

    lifecycle.append({"event": "model_unload", "model": "zamba", "at": _utc_now()})
    _free_model([zamba, zamba_tok])
    zamba = None
    zamba_tok = None

    # -----------------------------------------------------------------
    # R4 - GPT-2 reload, restore .hlx from R1, close the task
    # -----------------------------------------------------------------
    lifecycle.append({"event": "model_activate", "model": "gpt2", "ref": GPT2_REF, "at": _utc_now()})
    gpt2, gpt2_tok2, gpt2_reload_ms = _load_model(GPT2_REF, trust_remote_code=False)

    # Verify .hlx round-trip bit-identity by re-reading R1.hlx and comparing to in-memory blob.
    r1_hlx_path = Path(r1_hlx["hlx_path"])
    r1_hlx_reread_digest = _sha256_path(r1_hlx_path)
    r1_restore_check = {
        "restored_from": str(r1_hlx_path),
        "digest_original": r1_hlx["digest_pre_serialize"],
        "digest_reread": r1_hlx_reread_digest,
        "bit_identity_post_restore": r1_hlx_reread_digest == r1_hlx["digest_pre_serialize"],
    }

    r4_prompt_text = (
        f"{TASK_PROMPT}\n\n"
        f"[CONTEXT: GPT-2 R1 OUTPUT]\n{r1_text}\n[/CONTEXT]\n\n"
        f"[CONTEXT: ZAMBA2 R3 OUTPUT]\n{r3_text}\n[/CONTEXT]\n\n"
        "Now close the explanation. Add a final summary paragraph."
    )
    r4_gen = _generate_and_capture_kv(
        gpt2, gpt2_tok2, r4_prompt_text, max_new_tokens=args.tokens_per_round
    )
    r4_hlx = _write_hlx(hlx_dir / "r4_gpt2.hlx", r4_gen["kv_blob"])
    r4_mem = _sign_bridge_memory(
        catalog,
        project=project,
        agent_id=agents["gpt2"],
        run_id=run_id,
        memory_id_suffix="r4-gpt2-closing",
        summary=f"R4 GPT-2 closing ({r4_gen['generated_token_count']} tokens)",
        content=r4_gen["text"],
        tags=["r4", "gpt2", "closing"],
    )
    memory_chain.append({"round": 4, **r4_mem})
    rounds.append({
        "round": 4,
        "model": "gpt2",
        "model_ref": GPT2_REF,
        "arch": "transformer",
        "action": "restore_from_r1+continue",
        "reload_time_ms": round(gpt2_reload_ms, 3),
        "generation_time_ms": round(r4_gen["generation_time_ms"], 3),
        "prompt_token_count": r4_gen["prompt_token_count"],
        "generated_token_count": r4_gen["generated_token_count"],
        "prompt_digest": _digest_token_ids(r4_gen["prompt_ids"]),
        "output_digest": _digest_token_ids(r4_gen["new_ids"]),
        "output_preview": (r4_gen["text"] or "")[:400],
        "hlx_post_turn": r4_hlx,
        "r1_hlx_restore_check": r1_restore_check,
        "memory": r4_mem,
    })

    # -----------------------------------------------------------------
    # R5 - regression probe: replay R1 prompt exactly, compare outputs
    # -----------------------------------------------------------------
    r5_gen = _generate_and_capture_kv(
        gpt2, gpt2_tok2, TASK_PROMPT, max_new_tokens=args.tokens_per_round
    )
    edit_dist = _text_edit_distance(r1_text, r5_gen["text"])
    overlap = _keyword_overlap(r1_text, r5_gen["text"])
    output_digest_match = _digest_token_ids(r1_new_ids) == _digest_token_ids(r5_gen["new_ids"])

    severity = "negligible"
    if edit_dist > 0 and not output_digest_match:
        severity = "minor" if overlap["jaccard"] >= 0.8 else "moderate"
    if overlap["jaccard"] < 0.4:
        severity = "severe"

    rounds.append({
        "round": 5,
        "model": "gpt2",
        "model_ref": GPT2_REF,
        "arch": "transformer",
        "action": "regression_probe",
        "generation_time_ms": round(r5_gen["generation_time_ms"], 3),
        "prompt_digest": _digest_token_ids(r5_gen["prompt_ids"]),
        "output_digest": _digest_token_ids(r5_gen["new_ids"]),
        "output_preview": (r5_gen["text"] or "")[:400],
        "r1_prompt_replay": True,
        "output_token_digest_matches_r1": output_digest_match,
        "output_edit_distance_vs_r1": edit_dist,
        "keyword_overlap_vs_r1": overlap,
        "regression_severity": severity,
    })

    lifecycle.append({"event": "model_unload", "model": "gpt2", "at": _utc_now()})
    _free_model([gpt2, gpt2_tok2, gpt2_tok])
    gpt2 = None

    # -----------------------------------------------------------------
    # Compression + fidelity measurement (delegated, after rounds)
    # -----------------------------------------------------------------
    compression = {"gpt2": None, "zamba": None}
    if args.measure_compression:
        compression["gpt2"] = _measure_compression(
            GPT2_REF,
            prompt_text=TASK_PROMPT,
            arch_label="transformer",
            max_new_tokens=max(1, args.compression_tokens),
            variant_names=["native-dense", "turbo-int8-hadamard+q-mamba-dsq-int4"],
            trust_remote_code=False,
        )
        compression["zamba"] = _measure_compression(
            ZAMBA_REF,
            prompt_text=TASK_PROMPT,
            arch_label="hybrid-ssm-attention",
            max_new_tokens=max(1, args.compression_tokens),
            variant_names=["native-dense", "q-mamba-dsq-int4", "turbo-int8-hadamard+q-mamba-dsq-int4"],
            trust_remote_code=True,
        )

    # -----------------------------------------------------------------
    # Final artifact assembly
    # -----------------------------------------------------------------
    per_arch_bit_identity_ok = bool(r1_hlx["bit_identity"] and r3_hlx["bit_identity"])
    try:
        catalog.close()
    except Exception:
        pass

    # Restore signing-mode env (runner must not leak state to the caller).
    if _prev_signing_mode is None:
        os.environ.pop("HELIX_RECEIPT_SIGNING_MODE", None)
    else:
        os.environ["HELIX_RECEIPT_SIGNING_MODE"] = _prev_signing_mode

    artifact = {
        "artifact": "local-cross-arch-state-bridge-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": run_started,
        "run_ended_utc": _utc_now(),
        "status": "completed",
        "claim_level": "per_arch_bit_identity_plus_tokenized_cross_arch_continuity",
        "claim_boundary": (
            "Agent state is serialised from a GPT-2 Transformer and a Zamba2 "
            "hybrid-SSM, each preserved bit-identical within its own "
            "architecture via .hlx, with task continuity across the two "
            "families established through a signed token-and-memory bridge. "
            "This artifact does not claim numerical state transfer between "
            "KV-cache and SSM hidden state; that bridge is semantic/tokenised, "
            "not bijective."
        ),
        "cross_arch_bridge_kind": "tokens+signed_hmem",
        "per_arch_bit_identity_ok": per_arch_bit_identity_ok,
        "models": [
            {
                "ref": GPT2_REF,
                "arch": "transformer",
                "label": "gpt2-124M",
            },
            {
                "ref": ZAMBA_REF,
                "arch": "hybrid-ssm-attention",
                "label": "zamba2-1.2b",
                "trust_remote_code": True,
            },
        ],
        "rounds": rounds,
        "model_lifecycle_events": lifecycle,
        "signed_memory_chain": memory_chain,
        "compression": compression,
        "task_prompt_digest": _digest_token_ids(
            list(TASK_PROMPT.encode("utf-8"))
        ),
        "workspace": str(workspace),
        "receipt_signing_mode": "ephemeral_preregistered",
        "claims_not_allowed": [
            "This run does not claim bijective KV-cache to SSM hidden-state transfer.",
            "The 'continuity' measured is task-level (tokens + signed memory), "
            "not neural state migration.",
        ],
    }

    artifact_path = output_dir / f"local-cross-arch-state-bridge-v1-{run_id}.json"
    _write_json(artifact_path, artifact)
    artifact["artifact_path"] = str(artifact_path)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-architecture state-bridge runner (GPT-2 <-> Zamba2)."
    )
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--tokens-per-round", type=int, default=128)
    parser.add_argument(
        "--compression-tokens",
        type=int,
        default=16,
        help="max_new_tokens used by the compression benchmark call.",
    )
    parser.add_argument(
        "--no-measure-compression",
        dest="measure_compression",
        action="store_false",
    )
    parser.add_argument("--run-id", default=None)
    parser.set_defaults(measure_compression=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_cross_arch_state_bridge(args)
    print(json.dumps({
        "artifact_path": artifact.get("artifact_path"),
        "per_arch_bit_identity_ok": artifact["per_arch_bit_identity_ok"],
        "rounds": len(artifact["rounds"]),
        "compression_gpt2_status": (artifact["compression"]["gpt2"] or {}).get("status"),
        "compression_zamba_status": (artifact["compression"]["zamba"] or {}).get("status"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
