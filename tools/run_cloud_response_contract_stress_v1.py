"""
run_cloud_response_contract_stress_v1.py
========================================

Cloud structured-output contract stress suite for HeliX.

This suite deliberately separates transport success from response usability.
A model call can be successfully served, signed into MemoryCatalog, included in
a transparency tree, and still fail the operational contract because the output
is malformed JSON, fenced markdown, missing required fields, or too weakly
bounded to be consumed by an agent.

Claim boundary:
    This is an observed-output reliability suite. It proves local attestability
    of calls and contract checks, not semantic truth or global non-equivocation.
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


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/cloud-response-contract-stress"
DEFAULT_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "anthropic/claude-sonnet-4-6",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
]
SUITE_VERSION = "helix-cloud-response-contract-stress-v1"
PROJECT = "cloud-response-contract-stress-v1"
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"


CONTRACTS: dict[str, dict[str, Any]] = {
    "minimal_json": {
        "description": "Exact JSON object, no markdown, five bounded string fields.",
        "required": [
            "strongest_signal",
            "provider_identity_risk",
            "output_drift_risk",
            "missing_witness",
            "next_test",
        ],
        "type": "flat_strings",
        "word_limit": 24,
    },
    "nested_claims": {
        "description": "Exact JSON with claims array and decision object.",
        "required": ["claims", "decision"],
        "type": "nested_claims",
        "word_limit": 28,
    },
    "adversarial_boundary": {
        "description": "Exact JSON requiring rejection of semantic overclaim.",
        "required": ["verdict", "semantic_authority", "invalid_claims", "next_test"],
        "type": "boundary",
        "word_limit": 20,
    },
}


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
    raw = value if isinstance(value, list) else str(value or "").split(",")
    models: list[str] = []
    for item in raw:
        model = str(item).strip()
        if model and model not in models:
            models.append(model)
    if len(models) < 2:
        raise ValueError("--models must contain at least 2 distinct DeepInfra model refs")
    return models


def _extract_json_exact(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(str(key))
            seen.add(str(key))
            obj[str(key)] = value
        return obj

    result: dict[str, Any] = {
        "parse_ok": False,
        "exact_json_ok": False,
        "duplicate_keys": [],
        "fenced_markdown": "```" in stripped,
        "json": None,
        "error": None,
    }
    if not stripped:
        result["error"] = "empty_output"
        return result
    decoder = json.JSONDecoder(object_pairs_hook=hook)
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        result["error"] = f"json_decode_error:{exc.msg}"
        return result
    result["parse_ok"] = isinstance(value, dict)
    result["exact_json_ok"] = isinstance(value, dict) and stripped[end:].strip() == "" and not result["fenced_markdown"]
    result["duplicate_keys"] = sorted(set(duplicates))
    result["json"] = value if isinstance(value, dict) else None
    if duplicates:
        result["error"] = "duplicate_keys"
    elif not isinstance(value, dict):
        result["error"] = "not_json_object"
    elif stripped[end:].strip():
        result["error"] = "trailing_text"
    elif result["fenced_markdown"]:
        result["error"] = "markdown_fence"
    return result


def _word_count(value: Any) -> int:
    return len(str(value or "").split())


def validate_contract(text: str, contract_name: str) -> dict[str, Any]:
    contract = CONTRACTS[contract_name]
    parsed = _extract_json_exact(text)
    obj = parsed.get("json") if isinstance(parsed.get("json"), dict) else {}
    missing = [key for key in contract["required"] if key not in obj]
    extra = sorted(key for key in obj.keys() if key not in set(contract["required"]))
    type_errors: list[str] = []
    word_limit_errors: list[str] = []
    semantic_errors: list[str] = []
    limit = int(contract["word_limit"])

    if contract["type"] == "flat_strings":
        for key in contract["required"]:
            value = obj.get(key)
            if not isinstance(value, str) or not value.strip():
                type_errors.append(key)
            elif _word_count(value) > limit:
                word_limit_errors.append(key)
    elif contract["type"] == "nested_claims":
        claims = obj.get("claims")
        decision = obj.get("decision")
        if not isinstance(claims, list) or len(claims) != 2:
            type_errors.append("claims")
        else:
            for index, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    type_errors.append(f"claims[{index}]")
                    continue
                for field in ("claim", "evidence_boundary", "risk"):
                    if not isinstance(claim.get(field), str) or not str(claim.get(field)).strip():
                        type_errors.append(f"claims[{index}].{field}")
                    elif _word_count(claim.get(field)) > limit:
                        word_limit_errors.append(f"claims[{index}].{field}")
        if not isinstance(decision, dict):
            type_errors.append("decision")
        else:
            if not isinstance(decision.get("should_trust"), bool):
                type_errors.append("decision.should_trust")
            if not isinstance(decision.get("why"), str) or not decision.get("why"):
                type_errors.append("decision.why")
            elif _word_count(decision.get("why")) > limit:
                word_limit_errors.append("decision.why")
    elif contract["type"] == "boundary":
        if obj.get("semantic_authority") is not False:
            semantic_errors.append("semantic_authority_must_be_false")
        if not isinstance(obj.get("invalid_claims"), list) or len(obj.get("invalid_claims") or []) != 2:
            type_errors.append("invalid_claims")
        for key in ("verdict", "next_test"):
            if not isinstance(obj.get(key), str) or not obj.get(key):
                type_errors.append(key)
            elif _word_count(obj.get(key)) > limit:
                word_limit_errors.append(key)

    contract_ok = (
        bool(parsed["parse_ok"])
        and bool(parsed["exact_json_ok"])
        and not parsed["duplicate_keys"]
        and not missing
        and not extra
        and not type_errors
        and not word_limit_errors
        and not semantic_errors
    )
    return {
        "contract_name": contract_name,
        "parse_ok": bool(parsed["parse_ok"]),
        "exact_json_ok": bool(parsed["exact_json_ok"]),
        "fenced_markdown": bool(parsed["fenced_markdown"]),
        "duplicate_keys": parsed["duplicate_keys"],
        "missing_fields": missing,
        "extra_fields": extra,
        "type_errors": type_errors,
        "word_limit_errors": word_limit_errors,
        "semantic_errors": semantic_errors,
        "contract_ok": contract_ok,
        "error": parsed.get("error"),
    }


def _prompt_for_contract(contract_name: str) -> tuple[str, str]:
    system = (
        "You are participating in a HeliX structured-output reliability audit. "
        "Return exact JSON only. No markdown fences, no preamble, no comments."
    )
    if contract_name == "minimal_json":
        user = (
            "Return exactly one JSON object with keys strongest_signal, provider_identity_risk, "
            "output_drift_risk, missing_witness, next_test. Every value must be a string under 24 words. "
            "Assess HeliX as an agentic transparency system."
        )
    elif contract_name == "nested_claims":
        user = (
            "Return exactly one JSON object with keys claims and decision. claims must be an array of exactly "
            "two objects, each with string keys claim, evidence_boundary, risk under 28 words. decision must be "
            "an object with boolean should_trust and string why under 28 words."
        )
    else:
        user = (
            "A local HeliX receipt has a valid signature. Return exactly one JSON object with keys verdict, "
            "semantic_authority, invalid_claims, next_test. semantic_authority must be false. invalid_claims "
            "must list exactly two public claims that a local receipt still cannot prove."
        )
    return system, user


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    contract_ok = [item for item in results if item["contract"]["contract_ok"]]
    malformed = [item for item in results if not item["contract"]["parse_ok"] or not item["contract"]["exact_json_ok"]]
    fenced = [item for item in results if item["contract"]["fenced_markdown"]]
    by_model: dict[str, dict[str, Any]] = {}
    for item in results:
        model = str(item["requested_model"])
        bucket = by_model.setdefault(
            model,
            {
                "total": 0,
                "transport_ok": 0,
                "contract_ok": 0,
                "parse_fail": 0,
                "fenced_markdown": 0,
                "provider_mismatch": 0,
                "violations": [],
            },
        )
        bucket["total"] += 1
        bucket["transport_ok"] += 1 if item["status"] == "ok" else 0
        bucket["contract_ok"] += 1 if item["contract"]["contract_ok"] else 0
        bucket["parse_fail"] += 1 if not item["contract"]["parse_ok"] else 0
        bucket["fenced_markdown"] += 1 if item["contract"]["fenced_markdown"] else 0
        bucket["provider_mismatch"] += 1 if item["provider_mismatch"] else 0
        if not item["contract"]["contract_ok"]:
            bucket["violations"].append(
                {
                    "call_id": item["call_id"],
                    "contract": item["contract_name"],
                    "error": item["contract"].get("error"),
                    "missing_fields": item["contract"].get("missing_fields"),
                    "type_errors": item["contract"].get("type_errors"),
                    "semantic_errors": item["contract"].get("semantic_errors"),
                }
            )
    return {
        "total_calls": total,
        "contract_ok_count": len(contract_ok),
        "contract_violation_count": total - len(contract_ok),
        "contract_pass_rate_milli": int(round((len(contract_ok) / max(total, 1)) * 1000)),
        "malformed_json_count": len(malformed),
        "fenced_markdown_count": len(fenced),
        "provider_substitution_count": sum(1 for item in results if item["provider_mismatch"]),
        "by_model": by_model,
    }


def run_contract_stress_suite(
    *,
    run_id: str,
    models: list[str],
    contracts: list[str] | None = None,
    rounds: int = 1,
    output_dir: Path | None = None,
    max_tokens: int = 420,
    temperature: float = 0.0,
    timeout: float = 240.0,
) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is required")
    model_refs = _parse_models(models)
    contract_names = contracts or list(CONTRACTS.keys())
    for name in contract_names:
        if name not in CONTRACTS:
            raise ValueError(f"unknown contract: {name}")
    if rounds < 1:
        raise ValueError("--rounds must be at least 1")

    run_dir = (output_dir or (REPO_ROOT / DEFAULT_OUTPUT_DIR)) / "_cloud-response-contract-stress" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(run_dir / "memory.sqlite")
    session_id = f"cloud-response-contract-stress:{run_id}"
    agent_id = "helix-cloud-contract-stress-runner"
    temperature_milli = int(round(float(temperature) * 1000))
    changed_files = [
        "tools/run_cloud_response_contract_stress_v1.py",
        "tools/run_cloud_response_contract_stress_secure.ps1",
        "tools/run_agent_run_transparency_gauntlet_v1.py",
        "tools/verify_agent_run_bundle.py",
    ]
    patch_info = _git_diff_digest(changed_files)

    try:
        task_payload = {
            "run_id": run_id,
            "suite_version": SUITE_VERSION,
            "goal": "Separate cloud transport success from strict structured-output contract usability.",
            "models_requested": model_refs,
            "contracts": contract_names,
            "rounds": int(rounds),
            "max_tokens": int(max_tokens),
            "temperature_milli": temperature_milli,
            "timeout_s": int(round(float(timeout))),
            "claim_boundary": "Observed output contract reliability only; no semantic truth or global non-equivocation.",
        }
        task_obs = catalog.observe(
            project=PROJECT,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="task_capsule",
            summary="Cloud response contract stress task capsule",
            content=canonical_json(task_payload),
            tags=["transparency", "cloud", "contract", "task-capsule"],
        )
        keypair = derive_ephemeral_keypair(f"{PROJECT}:{run_id}:log-key")
        log = AgentRunTransparencyLog(tree_id=f"helix-cloud-response-contract-stress:{run_id}", run_id=run_id, keypair=keypair)
        log.append("task_capsule", {**task_payload, "catalog_node_hash": task_obs.get("node_hash")})

        results: list[dict[str, Any]] = []
        transcripts: list[dict[str, Any]] = []
        checkpoints = [{"label": "after-task", "sth": log.signed_tree_head(label="after-task")}]
        call_index = 0
        for round_index in range(1, int(rounds) + 1):
            for contract_name in contract_names:
                system, user = _prompt_for_contract(contract_name)
                for model_index, model in enumerate(model_refs, start=1):
                    call_index += 1
                    call = _deepinfra_chat_sync(
                        model=model,
                        system=system,
                        user=user,
                        token=token,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        timeout=float(timeout),
                    )
                    text = str(call.get("text") or "")
                    contract_result = validate_contract(text, contract_name)
                    call_id = f"r{round_index:02d}-c{contract_names.index(contract_name) + 1:02d}-m{model_index:02d}"
                    safe_call = {
                        "call_id": call_id,
                        "round_index": round_index,
                        "contract_name": contract_name,
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
                        "contract": contract_result,
                    }
                    call_obs = catalog.observe(
                        project=PROJECT,
                        agent_id=agent_id,
                        session_id=session_id,
                        observation_type="cloud_contract_call",
                        summary=f"{contract_name} contract call for {model}",
                        content=canonical_json({**safe_call, "text_preview": text[:900]}),
                        tags=["transparency", "cloud", "contract", contract_name],
                    )
                    memory = catalog.remember(
                        project=PROJECT,
                        agent_id=agent_id,
                        session_id=session_id,
                        memory_type="episodic",
                        summary=f"{contract_name} output from {model}",
                        content=text or f"ERROR: {call.get('error') or 'empty_response'}",
                        tags=["transparency", "cloud", "contract", "admitted-memory"],
                        importance=8 if call.get("status") == "ok" else 3,
                        llm_call_id=f"deepinfra:{run_id}:{call_id}",
                    )
                    memory_hash = str(catalog.get_memory_node_hash(memory.memory_id) or "")
                    receipt = catalog.get_memory_receipt(memory.memory_id) or {}
                    chain = catalog.verify_chain(memory_hash) if memory_hash else {"status": "missing"}
                    node = catalog.dag.lookup(memory_hash) if memory_hash else None
                    result = {
                        **safe_call,
                        "catalog_node_hash": call_obs.get("node_hash"),
                        "memory_id": memory.memory_id,
                        "memory_node_hash": memory_hash,
                        "node_hash_profile": getattr(node, "hash_profile", None),
                        "receipt_signature_verified": bool(receipt.get("signature_verified")),
                        "receipt_digest": f"sha256:{_sha256_text(canonical_json(receipt))}",
                        "chain_status": chain.get("status"),
                    }
                    results.append(result)
                    transcripts.append(
                        {
                            "call_id": call_id,
                            "round_index": round_index,
                            "contract_name": contract_name,
                            "requested_model": model,
                            "actual_model": call.get("actual_model"),
                            "status": call.get("status"),
                            "text_digest": call.get("text_digest"),
                            "contract": contract_result,
                            "text": text,
                            "error": call.get("error"),
                        }
                    )
                    log.append(
                        "contract_call",
                        {
                            "call_id": call_id,
                            "round_index": round_index,
                            "contract_name": contract_name,
                            "requested_model": result["requested_model"],
                            "actual_model": result["actual_model"],
                            "provider_mismatch": result["provider_mismatch"],
                            "transport_ok": result["status"] == "ok",
                            "text_digest": result["text_digest"],
                            "output_chars": result["output_chars"],
                            "contract_ok": result["contract"]["contract_ok"],
                            "parse_ok": result["contract"]["parse_ok"],
                            "exact_json_ok": result["contract"]["exact_json_ok"],
                            "fenced_markdown": result["contract"]["fenced_markdown"],
                            "memory_id": result["memory_id"],
                            "memory_node_hash": result["memory_node_hash"],
                            "node_hash_profile": result["node_hash_profile"],
                            "receipt_signature_verified": result["receipt_signature_verified"],
                            "chain_status": result["chain_status"],
                            "semantic_truth_status": "unproven",
                        },
                    )
            checkpoints.append({"label": f"after-round-{round_index}", "sth": log.signed_tree_head(label=f"after-round-{round_index}")})

        poison = catalog.remember_quarantined(
            project=PROJECT,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="semantic",
            summary="Signed poison control for contract stress suite",
            content="SIGNED_POISON_CONTROL: schema compliance still does not grant semantic authority.",
            tags=["transparency", "cloud", "contract", "poison-control"],
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
        summary = _summarize_results(results)
        subject_event = log.append(
            "contract_stress_summary",
            {
                "run_id": run_id,
                "models_requested": model_refs,
                "contracts": contract_names,
                "rounds": int(rounds),
                "total_calls": len(results),
                "contract_violation_count": summary["contract_violation_count"],
                "contract_pass_rate_milli": summary["contract_pass_rate_milli"],
                "malformed_json_count": summary["malformed_json_count"],
                "fenced_markdown_count": summary["fenced_markdown_count"],
                "provider_substitution_count": summary["provider_substitution_count"],
                "patch_digest": patch_info["patch_digest"],
                "claim_boundary": "Contract compliance is usability evidence, not semantic authority.",
            },
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
        final_sth = log.signed_tree_head(label="final-contract-stress")
        first_sth = checkpoints[0]["sth"]
        inclusion_proof = log.inclusion_proof_for_event(subject_event.event_id)
        subject_payload = subject_event.to_leaf_payload()
        inclusion_result = verify_inclusion_proof(subject_payload, inclusion_proof, final_sth)
        consistency_proof = log.consistency_proof(old_size=int(first_sth["tree_size"]), new_size=int(final_sth["tree_size"]))
        consistency_result = verify_consistency_proof(first_sth, final_sth, consistency_proof)
        catalog_coverage = catalog.verify_dag_coverage()
        gates = {
            "all_cloud_calls_completed": all(item["status"] == "ok" for item in results),
            "requested_actual_recorded": all(item.get("actual_model") for item in results if item["status"] == "ok"),
            "contract_checks_executed": len(results) == int(rounds) * len(contract_names) * len(model_refs),
            "contract_violations_auditable": all("contract_ok" in item["contract"] for item in results),
            "memory_receipts_verified": all(item["receipt_signature_verified"] and item["chain_status"] == "verified" for item in results),
            "memory_hash_profile_v2": all(item["node_hash_profile"] == DAG_HASH_PROFILE_V2 for item in results),
            "signed_poison_not_semantic_authority": bool(poison_receipt.get("signature_verified")) and getattr(poison_node, "hash_profile", None) == DAG_HASH_PROFILE_V2,
            "summary_inclusion_verified": bool(inclusion_result.get("ok")),
            "append_only_consistency_verified": bool(consistency_result.get("ok")),
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
            consistency_proof=consistency_proof,
            model_audit={
                "requested_model": ",".join(model_refs),
                "actual_model": ",".join(sorted({str(item["actual_model"]) for item in results if item.get("actual_model")})),
                "provider_mismatch": any(item["provider_mismatch"] for item in results),
                "calls": [
                    {
                        "call_id": item["call_id"],
                        "requested": item["requested_model"],
                        "actual": item["actual_model"],
                        "provider_mismatch": item["provider_mismatch"],
                        "contract_ok": item["contract"]["contract_ok"],
                        "text_digest": item["text_digest"],
                    }
                    for item in results
                ],
            },
            memory_summary={
                "admitted": [item["memory_id"] for item in results],
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
            "consistency_proof": consistency_proof,
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
            "artifact": "local-cloud-response-contract-stress-v1",
            "suite_version": SUITE_VERSION,
            "run_id": run_id,
            "run_started_utc": _utc_now(),
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "partial",
            "score": score,
            "models": model_refs,
            "contracts": contract_names,
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
            "cloud_calls": results,
            "cloud_transcript": transcripts,
            "tree": {
                "tree_id": log.tree_id,
                "tree_size": len(log.events),
                "tree_hash_profile": TREE_HASH_PROFILE,
                "final_root_hash": final_sth.get("root_hash"),
                "key_id": final_sth.get("key_id"),
            },
            "catalog": {
                "db_path": str(catalog.db_path),
                "session_id": session_id,
                "dag_coverage": catalog_coverage,
                "stats": catalog.stats(),
            },
            "events": log.leaf_payloads(),
            "sths": {"first": first_sth, "final": final_sth},
            "proofs": {"summary_inclusion": inclusion_proof, "consistency": consistency_proof},
            "verifier_results": {
                "summary_inclusion": inclusion_result,
                "consistency": consistency_result,
                "standalone_bundle": standalone_result,
            },
            "attestation": attestation,
            "standalone_bundle": standalone_bundle,
            "claim_boundary": (
                "This artifact proves local HeliX attestability of observed structured-output checks. "
                "It does not prove model semantic truth or global non-equivocation."
            ),
        }
    finally:
        catalog.close()


def write_extract(path: Path, artifact: dict[str, Any]) -> None:
    findings = artifact.get("findings") or {}
    failed = [name for name, ok in artifact["gates"].items() if not ok]
    lines = [
        f"# HeliX Cloud Response Contract Stress: {artifact['run_id']}",
        "",
        "## Verdict",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Evidence score: `{artifact['score']}`",
        f"- Contract pass rate: `{findings.get('contract_pass_rate_milli', 0) / 10:.1f}%`",
        f"- Contract violations: `{findings.get('contract_violation_count')}`",
        f"- Malformed JSON count: `{findings.get('malformed_json_count')}`",
        f"- Fenced markdown count: `{findings.get('fenced_markdown_count')}`",
        f"- Provider substitutions: `{findings.get('provider_substitution_count')}`",
        "",
        "## Failing Evidence Gates",
        "",
    ]
    lines.extend([f"- `{item}`" for item in failed] or ["- None"])
    lines.extend(["", "## Model Contract Summaries", ""])
    for model, summary in (findings.get("by_model") or {}).items():
        lines.append(
            f"- `{model}`: contract ok `{summary['contract_ok']}/{summary['total']}`, "
            f"parse failures `{summary['parse_fail']}`, fenced `{summary['fenced_markdown']}`, "
            f"provider mismatch `{summary['provider_mismatch']}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_transcript_markdown(path: Path, artifact: dict[str, Any]) -> None:
    lines = [f"# Cloud Response Contract Stress Transcript: {artifact['run_id']}", ""]
    for item in artifact.get("cloud_transcript") or []:
        contract = item.get("contract") or {}
        lines.extend(
            [
                f"## {item['call_id']} | {item['contract_name']} | {item['requested_model']}",
                "",
                f"- Actual: `{item.get('actual_model')}`",
                f"- Status: `{item.get('status')}`",
                f"- Contract OK: `{contract.get('contract_ok')}`",
                f"- Parse OK: `{contract.get('parse_ok')}`",
                f"- Exact JSON OK: `{contract.get('exact_json_ok')}`",
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
    parser = argparse.ArgumentParser(description="HeliX cloud structured-output contract stress suite")
    parser.add_argument("--run-id", default=f"cloud-contract-stress-{uuid.uuid4().hex[:10]}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--contracts", default=",".join(CONTRACTS.keys()))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=420)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    artifact = run_contract_stress_suite(
        run_id=args.run_id,
        models=_parse_models(args.models),
        contracts=[item.strip() for item in str(args.contracts).split(",") if item.strip()],
        rounds=args.rounds,
        output_dir=output_dir,
        max_tokens=args.tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    slug = str(artifact["artifact"])
    artifact_path = output_dir / f"{slug}-{args.run_id}.json"
    extract_path = output_dir / f"{slug}-{args.run_id}-extract.md"
    transcript_path = output_dir / f"{slug}-{args.run_id}-transcript.md"
    bundle_path = output_dir / f"{slug}-{args.run_id}-bundle.json"
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
                "contract_pass_rate_milli": artifact["findings"]["contract_pass_rate_milli"],
                "contract_violation_count": artifact["findings"]["contract_violation_count"],
                "malformed_json_count": artifact["findings"]["malformed_json_count"],
                "fenced_markdown_count": artifact["findings"]["fenced_markdown_count"],
                "provider_substitution_count": artifact["findings"]["provider_substitution_count"],
                "failing_gates": [name for name, ok in artifact["gates"].items() if not ok],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
