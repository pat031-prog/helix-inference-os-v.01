"""
run_cloud_provider_substitution_longitudinal_v1.py
==================================================

Cloud provider substitution and drift suite for HeliX.

This runner repeatedly calls a panel of DeepInfra-hosted models with the same
probe, records requested/actual model identity, output digests, latency, signed
memory receipts, and CT-style transparency proofs. Provider substitutions are
not treated as automatic failures; they are evidence to be made auditable.

Claim boundary:
    This proves local attestability of observed cloud calls and append-only
    evidence handling. It does not prove model semantic truth, provider
    honesty outside the observed API response, or global non-equivocation.
"""
from __future__ import annotations

import argparse
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

from helix_kv.merkle_dag import DAG_HASH_PROFILE_V2  # noqa: E402
from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402
from helix_proto.signed_receipts import canonical_json  # noqa: E402
from tools.run_agent_run_transparency_gauntlet_v1 import (  # noqa: E402
    AgentRunTransparencyLog,
    TREE_HASH_PROFILE,
    _deepinfra_chat_sync,
    _git_diff_digest,
    _sha256_text,
    build_agent_run_attestation,
    derive_ephemeral_keypair,
    verify_consistency_proof,
    verify_inclusion_proof,
    verify_standalone_bundle,
)


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/cloud-provider-substitution-longitudinal"
DEFAULT_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "anthropic/claude-sonnet-4-6",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
]
SUITE_VERSION = "helix-cloud-provider-substitution-longitudinal-v1"
PROJECT = "cloud-provider-substitution-longitudinal-v1"
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_models(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").split(",")
    models = []
    for item in raw:
        model = str(item).strip()
        if model and model not in models:
            models.append(model)
    if len(models) < 2:
        raise ValueError("--models must contain at least 2 distinct DeepInfra model refs")
    return models


def _probe_prompts() -> tuple[str, str]:
    system = (
        "You are participating in a HeliX provider-substitution audit. "
        "Return compact JSON only. Distinguish observed provenance from semantic truth. "
        "Do not claim global non-equivocation without external witnesses."
    )
    user = (
        "Probe id: helix-cloud-provider-substitution-longitudinal-v1\n"
        "Task: assess HeliX as an agentic transparency system. "
        "Return JSON with exactly these keys: "
        "strongest_signal, provider_identity_risk, output_drift_risk, missing_witness, next_test. "
        "Keep each value under 24 words."
    )
    return system, user


def _compact_text(value: str, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}...[+{len(text) - limit} chars]"


def _summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_requested: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        by_requested.setdefault(str(call["requested_model"]), []).append(call)

    summaries: dict[str, Any] = {}
    for requested, items in by_requested.items():
        ok_items = [item for item in items if item.get("status") == "ok"]
        actual_models = sorted({str(item.get("actual_model")) for item in ok_items if item.get("actual_model")})
        output_digests = sorted({str(item.get("text_digest")) for item in ok_items if item.get("text_digest")})
        latencies = [int(item.get("latency_ms") or 0) for item in ok_items]
        summaries[requested] = {
            "total_calls": len(items),
            "ok_calls": len(ok_items),
            "actual_models": actual_models,
            "actual_model_count": len(actual_models),
            "substitution_count": sum(1 for item in items if bool(item.get("provider_mismatch"))),
            "output_digest_count": len(output_digests),
            "output_stable": len(output_digests) <= 1,
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "errors": [item.get("error") for item in items if item.get("status") != "ok"],
        }

    substitutions = [
        {
            "round_index": call["round_index"],
            "requested_model": call["requested_model"],
            "actual_model": call.get("actual_model"),
            "text_digest": call.get("text_digest"),
        }
        for call in calls
        if bool(call.get("provider_mismatch"))
    ]
    return {
        "model_summaries": summaries,
        "provider_substitution_detected": bool(substitutions),
        "substitutions": substitutions,
        "actual_model_drift_detected": any(item["actual_model_count"] > 1 for item in summaries.values()),
        "output_drift_detected": any(item["output_digest_count"] > 1 for item in summaries.values()),
        "availability_errors": [
            {
                "round_index": call["round_index"],
                "requested_model": call["requested_model"],
                "error": call.get("error"),
            }
            for call in calls
            if call.get("status") != "ok"
        ],
    }


def _verify_adjacent_checkpoints(
    *,
    log: AgentRunTransparencyLog,
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for previous, current in zip(checkpoints, checkpoints[1:]):
        proof = log.consistency_proof(
            old_size=int(previous["sth"]["tree_size"]),
            new_size=int(current["sth"]["tree_size"]),
        )
        result = verify_consistency_proof(previous["sth"], current["sth"], proof)
        results.append(
            {
                "from_label": previous["label"],
                "to_label": current["label"],
                "from_size": previous["sth"]["tree_size"],
                "to_size": current["sth"]["tree_size"],
                "ok": bool(result.get("ok")),
                "reason": result.get("reason"),
                "proof": proof,
                "result": result,
            }
        )
    return results


def run_longitudinal_suite(
    *,
    run_id: str,
    models: list[str],
    rounds: int = 3,
    output_dir: Path | None = None,
    max_tokens: int = 420,
    temperature: float = 0.15,
    timeout: float = 240.0,
) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is required")
    model_refs = _parse_models(models)
    if rounds < 1:
        raise ValueError("--rounds must be at least 1")

    run_dir = (output_dir or (REPO_ROOT / DEFAULT_OUTPUT_DIR)) / "_cloud-provider-longitudinal" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(run_dir / "memory.sqlite")
    session_id = f"cloud-provider-substitution-longitudinal:{run_id}"
    agent_id = "helix-cloud-provider-longitudinal-runner"
    temperature_milli = int(round(float(temperature) * 1000))
    system_prompt, user_prompt = _probe_prompts()
    changed_files = [
        "tools/run_cloud_provider_substitution_longitudinal_v1.py",
        "tools/run_cloud_provider_substitution_longitudinal_secure.ps1",
        "tools/run_agent_run_transparency_gauntlet_v1.py",
        "tools/verify_agent_run_bundle.py",
        "helix_kv/memory_catalog.py",
        "helix_kv/merkle_dag.py",
    ]
    patch_info = _git_diff_digest(changed_files)

    try:
        task_payload = {
            "run_id": run_id,
            "suite_version": SUITE_VERSION,
            "goal": "Detect provider substitution, actual-model drift, and output digest drift across repeated cloud model calls.",
            "models_requested": model_refs,
            "rounds": int(rounds),
            "max_tokens": int(max_tokens),
            "temperature_milli": temperature_milli,
            "timeout_s": int(round(float(timeout))),
            "claim_boundary": "Observed DeepInfra API evidence only; no semantic truth or global non-equivocation.",
        }
        task_obs = catalog.observe(
            project=PROJECT,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="task_capsule",
            summary="Cloud provider substitution longitudinal task capsule",
            content=canonical_json(task_payload),
            tags=["transparency", "cloud", "deepinfra", "longitudinal", "task-capsule"],
        )

        keypair = derive_ephemeral_keypair(f"{PROJECT}:{run_id}:log-key")
        log = AgentRunTransparencyLog(tree_id=f"helix-cloud-provider-longitudinal:{run_id}", run_id=run_id, keypair=keypair)
        log.append("task_capsule", {**task_payload, "catalog_node_hash": task_obs.get("node_hash")})
        checkpoints: list[dict[str, Any]] = [
            {"label": "after-task", "sth": log.signed_tree_head(label="after-task")}
        ]

        calls: list[dict[str, Any]] = []
        transcripts: list[dict[str, Any]] = []
        for round_index in range(1, int(rounds) + 1):
            round_call_ids: list[str] = []
            for model_index, model in enumerate(model_refs, start=1):
                call = _deepinfra_chat_sync(
                    model=model,
                    system=system_prompt,
                    user=user_prompt,
                    token=token,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    timeout=float(timeout),
                )
                text = str(call.get("text") or "")
                safe_call = {
                    "call_id": f"r{round_index:02d}-m{model_index:02d}",
                    "round_index": round_index,
                    "model_index": model_index,
                    "requested_model": model,
                    "actual_model": call.get("actual_model"),
                    "provider_mismatch": bool(call.get("provider_mismatch")),
                    "status": call.get("status"),
                    "finish_reason": call.get("finish_reason"),
                    "tokens_used": int(call.get("tokens_used") or 0),
                    "latency_ms": int(call.get("latency_ms") or 0),
                    "retry_count": int(call.get("retry_count") or 0),
                    "text_digest": call.get("text_digest"),
                    "output_chars": len(text),
                    "error": call.get("error"),
                }
                call_obs = catalog.observe(
                    project=PROJECT,
                    agent_id=agent_id,
                    session_id=session_id,
                    observation_type="cloud_model_call",
                    summary=f"Round {round_index} DeepInfra metadata for {model}",
                    content=canonical_json({**safe_call, "text_preview": _compact_text(text)}),
                    tags=["transparency", "cloud", "deepinfra", "model-call", f"round-{round_index}"],
                )
                memory = catalog.remember(
                    project=PROJECT,
                    agent_id=agent_id,
                    session_id=session_id,
                    memory_type="episodic",
                    summary=f"Round {round_index} cloud output from {model}",
                    content=text or f"ERROR: {call.get('error') or 'empty_response'}",
                    tags=["transparency", "cloud", "deepinfra", "longitudinal", "admitted-memory"],
                    importance=8 if call.get("status") == "ok" else 3,
                    llm_call_id=f"deepinfra:{run_id}:r{round_index}:m{model_index}",
                )
                memory_hash = str(catalog.get_memory_node_hash(memory.memory_id) or "")
                receipt = catalog.get_memory_receipt(memory.memory_id) or {}
                chain = catalog.verify_chain(memory_hash) if memory_hash else {"status": "missing"}
                node = catalog.dag.lookup(memory_hash) if memory_hash else None
                public_call = {
                    **safe_call,
                    "catalog_node_hash": call_obs.get("node_hash"),
                    "memory_id": memory.memory_id,
                    "memory_node_hash": memory_hash,
                    "node_hash_profile": getattr(node, "hash_profile", None),
                    "receipt_signature_verified": bool(receipt.get("signature_verified")),
                    "receipt_digest": f"sha256:{_sha256_text(canonical_json(receipt))}",
                    "chain_status": chain.get("status"),
                }
                calls.append(public_call)
                transcripts.append(
                    {
                        "call_id": safe_call["call_id"],
                        "round_index": round_index,
                        "requested_model": model,
                        "actual_model": call.get("actual_model"),
                        "status": call.get("status"),
                        "text_digest": call.get("text_digest"),
                        "text": text,
                        "error": call.get("error"),
                    }
                )
                event = log.append(
                    "model_call",
                    {
                        "call_id": public_call["call_id"],
                        "round_index": public_call["round_index"],
                        "model_index": public_call["model_index"],
                        "requested_model": public_call["requested_model"],
                        "actual_model": public_call["actual_model"],
                        "provider_mismatch": public_call["provider_mismatch"],
                        "status": public_call["status"],
                        "finish_reason": public_call["finish_reason"],
                        "tokens_used": public_call["tokens_used"],
                        "latency_ms": public_call["latency_ms"],
                        "retry_count": public_call["retry_count"],
                        "text_digest": public_call["text_digest"],
                        "output_chars": public_call["output_chars"],
                        "memory_id": public_call["memory_id"],
                        "memory_node_hash": public_call["memory_node_hash"],
                        "node_hash_profile": public_call["node_hash_profile"],
                        "receipt_signature_verified": public_call["receipt_signature_verified"],
                        "chain_status": public_call["chain_status"],
                        "catalog_node_hash": public_call["catalog_node_hash"],
                        "semantic_truth_status": "unproven",
                    },
                )
                round_call_ids.append(event.event_id)

            log.append(
                "round_summary",
                {
                    "round_index": round_index,
                    "model_call_event_ids": round_call_ids,
                    "ok_count": sum(1 for call in calls if call["round_index"] == round_index and call["status"] == "ok"),
                    "substitution_count": sum(
                        1 for call in calls if call["round_index"] == round_index and call["provider_mismatch"]
                    ),
                    "output_digests": [
                        call["text_digest"] for call in calls if call["round_index"] == round_index
                    ],
                },
            )
            checkpoints.append(
                {
                    "label": f"after-round-{round_index}",
                    "sth": log.signed_tree_head(label=f"after-round-{round_index}"),
                }
            )

        poison = catalog.remember_quarantined(
            project=PROJECT,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="semantic",
            summary="Signed poison control for cloud provider longitudinal suite",
            content="SIGNED_POISON_CONTROL: a stable provider response must not grant semantic authority.",
            tags=["transparency", "cloud", "poison-control"],
            importance=1,
            record_kind="signed_poison_control",
            quarantine_reason="semantic_authority_control",
            quarantine_class="test_control",
            disposition="quarantined_control",
            llm_call_id=f"deepinfra:{run_id}:signed-poison",
        )
        poison_hash = str(poison.get("node_hash") or "")
        poison_receipt = dict(poison.get("signed_receipt") or poison.get("receipt") or {})
        poison_node = catalog.dag.lookup(poison_hash) if poison_hash else None

        summary = _summarize_calls(calls)
        subject_event = log.append(
            "longitudinal_summary",
            {
                "run_id": run_id,
                "models_requested": model_refs,
                "rounds": int(rounds),
                "call_count": len(calls),
                "ok_count": sum(1 for call in calls if call["status"] == "ok"),
                "provider_substitution_detected": bool(summary["provider_substitution_detected"]),
                "actual_model_drift_detected": bool(summary["actual_model_drift_detected"]),
                "output_drift_detected": bool(summary["output_drift_detected"]),
                "availability_error_count": len(summary["availability_errors"]),
                "patch_digest": patch_info["patch_digest"],
                "claim_boundary": "Observed provider identity and output drift are evidence, not semantic truth.",
            },
        )
        trust_card_payload = {
            "trust_card_version": "helix-cloud-provider-longitudinal-trust-card-v0",
            "run_id": run_id,
            "artifact_digest": patch_info["patch_digest"],
            "models_requested": model_refs,
            "rounds": int(rounds),
            "provider_substitution_detected": bool(summary["provider_substitution_detected"]),
            "actual_model_drift_detected": bool(summary["actual_model_drift_detected"]),
            "output_drift_detected": bool(summary["output_drift_detected"]),
            "memory_admitted": [call["memory_id"] for call in calls],
            "memory_quarantined": [str(poison.get("memory_id"))],
            "external_anchor": None,
            "claim_boundary": "Local cloud evidence only; no external witness yet.",
        }
        trust_obs = catalog.observe(
            project=PROJECT,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="trust_card",
            summary="Cloud provider longitudinal trust card payload recorded",
            content=canonical_json(trust_card_payload),
            tags=["transparency", "cloud", "trust-card", "longitudinal"],
        )
        log.append(
            "signed_poison_control",
            {
                "memory_id": str(poison.get("memory_id")),
                "node_hash": poison_hash,
                "node_hash_profile": getattr(poison_node, "hash_profile", None),
                "signature_verified": bool(poison_receipt.get("signature_verified")),
                "semantic_authority": False,
                "receipt_digest": f"sha256:{_sha256_text(canonical_json(poison_receipt))}",
            },
        )
        log.append("trust_card", {**trust_card_payload, "catalog_node_hash": trust_obs.get("node_hash")})

        final_sth = log.signed_tree_head(label="final-longitudinal")
        checkpoints.append({"label": "final-longitudinal", "sth": final_sth})
        subject_payload = subject_event.to_leaf_payload()
        inclusion_proof = log.inclusion_proof_for_event(subject_event.event_id)
        inclusion_result = verify_inclusion_proof(subject_payload, inclusion_proof, final_sth)
        first_sth = checkpoints[0]["sth"]
        end_to_end_consistency_proof = log.consistency_proof(
            old_size=int(first_sth["tree_size"]),
            new_size=int(final_sth["tree_size"]),
        )
        end_to_end_consistency = verify_consistency_proof(first_sth, final_sth, end_to_end_consistency_proof)
        checkpoint_consistency = _verify_adjacent_checkpoints(log=log, checkpoints=checkpoints)
        catalog_coverage = catalog.verify_dag_coverage()
        model_audit = {
            "requested_model": ",".join(model_refs),
            "actual_model": ",".join(
                sorted({str(call["actual_model"]) for call in calls if call.get("actual_model")})
            ),
            "provider_mismatch": bool(summary["provider_substitution_detected"]),
            "calls": [
                {
                    "round_index": call["round_index"],
                    "requested": call["requested_model"],
                    "actual": call["actual_model"],
                    "provider_mismatch": call["provider_mismatch"],
                    "status": call["status"],
                    "text_digest": call["text_digest"],
                    "memory_node_hash": call["memory_node_hash"],
                }
                for call in calls
            ],
        }
        gates = {
            "all_cloud_calls_completed": all(call["status"] == "ok" for call in calls),
            "requested_actual_recorded": all(call.get("actual_model") for call in calls if call["status"] == "ok"),
            "substitutions_are_auditable": all("provider_mismatch" in call for call in calls),
            "memory_receipts_verified": all(
                call["receipt_signature_verified"] and call["chain_status"] == "verified" for call in calls
            ),
            "memory_hash_profile_v2": all(call["node_hash_profile"] == DAG_HASH_PROFILE_V2 for call in calls),
            "signed_poison_not_semantic_authority": (
                bool(poison_receipt.get("signature_verified"))
                and getattr(poison_node, "hash_profile", None) == DAG_HASH_PROFILE_V2
            ),
            "final_summary_inclusion_verified": bool(inclusion_result.get("ok")),
            "checkpoint_consistency_verified": all(item["ok"] for item in checkpoint_consistency),
            "end_to_end_consistency_verified": bool(end_to_end_consistency.get("ok")),
            "catalog_dag_coverage_verified": catalog_coverage.get("status") == "verified",
            "standalone_verifier_bundle_passes": False,
            "claim_boundary_present": True,
        }
        attestation = build_agent_run_attestation(
            run_id=run_id,
            subject_event=subject_payload,
            inclusion_proof=inclusion_proof,
            sth=final_sth,
            previous_sth=first_sth,
            consistency_proof=end_to_end_consistency_proof,
            model_audit=model_audit,
            memory_summary={
                "admitted": [call["memory_id"] for call in calls],
                "quarantined": [str(poison.get("memory_id"))],
            },
            checks={name: ok for name, ok in gates.items() if name != "standalone_verifier_bundle_passes"},
            external_anchor=None,
        )
        standalone_bundle = {
            "bundle_version": "helix-agent-run-verifier-bundle-v0",
            "event": subject_payload,
            "inclusion_proof": inclusion_proof,
            "previous_sth": first_sth,
            "consistency_proof": end_to_end_consistency_proof,
            "sth": final_sth,
            "attestation": attestation,
            "claim_boundary": attestation["predicate"]["verification"]["claim_boundary"],
        }
        standalone_result = verify_standalone_bundle(standalone_bundle)
        gates["standalone_verifier_bundle_passes"] = bool(standalone_result.get("ok"))
        attestation["predicate"]["verification"]["checks"] = gates
        standalone_result = verify_standalone_bundle(standalone_bundle)
        score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
        return {
            "artifact": "local-cloud-provider-substitution-longitudinal-v1",
            "suite_version": SUITE_VERSION,
            "run_id": run_id,
            "run_started_utc": _utc_now(),
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "partial",
            "score": score,
            "models": model_refs,
            "rounds": int(rounds),
            "cloud_config": {
                "endpoint": DEEPINFRA_BASE,
                "max_tokens": int(max_tokens),
                "temperature_milli": temperature_milli,
                "timeout_s": int(round(float(timeout))),
                "token_persisted": False,
            },
            "gates": gates,
            "findings": summary,
            "cloud_calls": calls,
            "cloud_transcript": transcripts,
            "tree": {
                "tree_id": log.tree_id,
                "tree_size": len(log.events),
                "tree_hash_profile": TREE_HASH_PROFILE,
                "final_root_hash": final_sth.get("root_hash"),
                "key_id": final_sth.get("key_id"),
            },
            "checkpoints": checkpoints,
            "catalog": {
                "db_path": str(catalog.db_path),
                "session_id": session_id,
                "dag_coverage": catalog_coverage,
                "stats": catalog.stats(),
            },
            "events": log.leaf_payloads(),
            "sths": {"first": first_sth, "final": final_sth},
            "proofs": {
                "summary_inclusion": inclusion_proof,
                "end_to_end_consistency": end_to_end_consistency_proof,
                "checkpoint_consistency": checkpoint_consistency,
            },
            "verifier_results": {
                "summary_inclusion": inclusion_result,
                "end_to_end_consistency": end_to_end_consistency,
                "standalone_bundle": standalone_result,
            },
            "attestation": attestation,
            "standalone_bundle": standalone_bundle,
            "claim_boundary": (
                "This artifact proves local HeliX transparency mechanics over observed DeepInfra calls. "
                "It does not prove model semantic truth, provider honesty outside the observed response, "
                "or global non-equivocation."
            ),
        }
    finally:
        catalog.close()


def write_extract(path: Path, artifact: dict[str, Any]) -> None:
    failed = [name for name, ok in artifact["gates"].items() if not ok]
    findings = artifact.get("findings") or {}
    lines = [
        f"# HeliX Cloud Provider Substitution Longitudinal: {artifact['run_id']}",
        "",
        "## Verdict",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Score: `{artifact['score']}`",
        f"- Rounds: `{artifact['rounds']}`",
        f"- Models: `{', '.join(artifact['models'])}`",
        f"- Tree size: `{artifact['tree']['tree_size']}`",
        f"- Final root: `{artifact['tree']['final_root_hash']}`",
        "",
        "## Findings",
        "",
        f"- Provider substitution detected: `{bool(findings.get('provider_substitution_detected'))}`",
        f"- Actual model drift detected: `{bool(findings.get('actual_model_drift_detected'))}`",
        f"- Output digest drift detected: `{bool(findings.get('output_drift_detected'))}`",
        f"- Availability errors: `{len(findings.get('availability_errors') or [])}`",
        "",
        "## Failing Gates",
        "",
    ]
    lines.extend([f"- `{item}`" for item in failed] or ["- None"])
    lines.extend(["", "## Model Summaries", ""])
    for model, summary in (findings.get("model_summaries") or {}).items():
        lines.append(
            f"- `{model}`: ok `{summary['ok_calls']}/{summary['total_calls']}`, "
            f"actual `{', '.join(summary['actual_models'])}`, substitutions `{summary['substitution_count']}`, "
            f"output digests `{summary['output_digest_count']}`, avg latency `{summary['avg_latency_ms']}ms`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Local DeepInfra evidence is auditable; semantic truth and global non-equivocation still require external witnesses.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_transcript_markdown(path: Path, artifact: dict[str, Any]) -> None:
    lines = [
        f"# Cloud Provider Longitudinal Transcript: {artifact['run_id']}",
        "",
    ]
    for item in artifact.get("cloud_transcript") or []:
        lines.extend(
            [
                f"## {item['call_id']} | round {item['round_index']} | {item['requested_model']}",
                "",
                f"- Actual: `{item.get('actual_model')}`",
                f"- Status: `{item.get('status')}`",
                f"- Digest: `{item.get('text_digest')}`",
                "",
                "```text",
                str(item.get("text") or item.get("error") or ""),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HeliX cloud provider substitution longitudinal suite")
    parser.add_argument("--run-id", default=f"cloud-provider-longitudinal-{uuid.uuid4().hex[:10]}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=420)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    artifact = run_longitudinal_suite(
        run_id=args.run_id,
        models=_parse_models(args.models),
        rounds=args.rounds,
        output_dir=output_dir,
        max_tokens=args.tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    artifact_slug = str(artifact.get("artifact") or "local-cloud-provider-substitution-longitudinal-v1")
    artifact_path = output_dir / f"{artifact_slug}-{args.run_id}.json"
    extract_path = output_dir / f"{artifact_slug}-{args.run_id}-extract.md"
    transcript_path = output_dir / f"{artifact_slug}-{args.run_id}-transcript.md"
    bundle_path = output_dir / f"{artifact_slug}-{args.run_id}-bundle.json"
    artifact["artifact_path"] = str(artifact_path)
    artifact["extract_markdown_path"] = str(extract_path)
    artifact["transcript_markdown_path"] = str(transcript_path)
    artifact["standalone_bundle_path"] = str(bundle_path)

    if not args.no_write:
        _write_json(artifact_path, artifact)
        write_extract(extract_path, artifact)
        write_transcript_markdown(transcript_path, artifact)
        _write_json(bundle_path, artifact["standalone_bundle"])
        artifact["artifact_sha256"] = _sha256_path(artifact_path)
        artifact["standalone_bundle_sha256"] = _sha256_path(bundle_path)
        _write_json(artifact_path, artifact)

    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "extract_markdown_path": str(extract_path),
                "transcript_markdown_path": str(transcript_path),
                "standalone_bundle_path": str(bundle_path),
                "status": artifact["status"],
                "score": artifact["score"],
                "rounds": artifact["rounds"],
                "tree_size": artifact["tree"]["tree_size"],
                "provider_substitution_detected": artifact["findings"]["provider_substitution_detected"],
                "actual_model_drift_detected": artifact["findings"]["actual_model_drift_detected"],
                "output_drift_detected": artifact["findings"]["output_drift_detected"],
                "failing_gates": [name for name, ok in artifact["gates"].items() if not ok],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
