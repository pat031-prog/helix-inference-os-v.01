"""
run_post_nuclear_methodology_suite_v1.py
=======================================

Mixed post-nuclear methodology suite.

These cases convert freeform HeliX drift ideas into falsifiable checks. Each
case preserves qualitative model text but requires hard gates over memory IDs,
node hashes, parent hashes, signatures, and auditor reconstruction.
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
from tools.artifact_integrity import finalize_artifact  # noqa: E402
from tools.transcript_exports import write_case_transcript_exports, write_suite_transcript_exports  # noqa: E402
from tools.run_nuclear_methodology_suite_v1 import (  # noqa: E402
    _deepinfra_chat,
    _remember,
    _search,
    _sha256_path,
    _utc_now,
    _write_json,
)


DEFAULT_FORENSIC_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_AUDITOR_MODEL = "anthropic/claude-4-sonnet"
DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/post-nuclear-methodology"

CASE_ORDER = [
    "counterfactual-archive-topology",
    "recursive-witness-integrity",
    "summary-node-compression",
    "proof-of-utility-retrieval",
    "metaphor-boundary-detector",
]


def _score(gates: dict[str, bool]) -> dict[str, Any]:
    ratio = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {"score": ratio, "passed": all(gates.values()), "gates": gates}


def _text(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True).lower()


def _base_artifact(*, case_id: str, run_id: str, protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "artifact": f"local-post-nuclear-{case_id}-v1",
        "schema_version": 1,
        "case_id": case_id,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": None,
        "status": "partial",
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "Cloud-only mixed post-nuclear methodology test. Qualitative transcript "
            "plus hard signed-memory gates. No sentience, local .hlx identity, or "
            "numerical KV<->SSM transfer claim."
        ),
        "protocol": protocol,
    }


async def _judge_and_audit(
    *,
    case_id: str,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    forensic_model: str,
    auditor_model: str,
    token: str,
    tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    judge = await _deepinfra_chat(
        model=forensic_model,
        system=(
            "You are a post-nuclear HeliX forensic judge. Output compact JSON only. "
            "Use exact memory IDs and hashes from evidence. Do not write markdown."
        ),
        user=f"""
Case: {case_id}

Evidence:
{json.dumps(evidence, indent=2)}

Expected JSON shape and required decisions:
{json.dumps(expected, indent=2)}

Return JSON only.
""",
        token=token,
        max_tokens=tokens,
        temperature=0.0,
    )
    auditor = await _deepinfra_chat(
        model=auditor_model,
        system=(
            "You are an independent HeliX methodology auditor. Return compact JSON only. "
            "Pass only if the judge output satisfies every expected decision."
        ),
        user=f"""
Case: {case_id}

Expected:
{json.dumps(expected, indent=2)}

Judge JSON:
{json.dumps(judge.get("json"), indent=2)}

Judge raw text:
{judge.get("text", "")}

Return this JSON only:
{{
  "verdict": "pass" | "fail",
  "gate_failures": [],
  "rationale": "one short sentence"
}}
""",
        token=token,
        max_tokens=tokens,
        temperature=0.0,
    )
    return judge, auditor


def _final_case_artifact(
    case_id: str,
    run_id: str,
    output_dir: Path,
    protocol: dict[str, Any],
    evidence: dict[str, Any],
    expected: dict[str, Any],
    judge: dict[str, Any],
    auditor: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    artifact = _base_artifact(case_id=case_id, run_id=run_id, protocol=protocol, output_dir=output_dir / case_id)
    transcript_exports = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=case_id,
        run_id=run_id,
        prefix="local-post-nuclear",
        evidence=evidence,
        expected=expected,
        judge=judge,
        auditor=auditor,
    )
    artifact.update({
        "run_ended_utc": _utc_now(),
        "status": "completed" if score["passed"] else "partial",
        "cloud_all_ok": judge["status"] == "ok" and auditor["status"] == "ok",
        "case_passed": score["passed"],
        "score": score,
        "models": {
            "forensic_requested": judge["requested_model"],
            "forensic_actual": judge["actual_model"],
            "auditor_requested": auditor["requested_model"],
            "auditor_actual": auditor["actual_model"],
        },
        "evidence": evidence,
        "expected": expected,
        "transcript": {
            "judge_text": judge.get("text"),
            "auditor_text": auditor.get("text"),
        },
        "transcript_exports": transcript_exports,
        "judge_call": {k: v for k, v in judge.items() if k not in {"text", "json"}},
        "auditor_call": {k: v for k, v in auditor.items() if k not in {"text", "json"}},
        "judge_output": {"text": judge.get("text"), "json": judge.get("json")},
        "auditor_output": {"text": auditor.get("text"), "json": auditor.get("json")},
    })
    path = output_dir / case_id / f"local-post-nuclear-{case_id}-{run_id}.json"
    return finalize_artifact(path, artifact)


def score_counterfactual_archive_topology(
    *,
    records: dict[str, Any],
    judge_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    judge_finish_reason: str | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    discarded_id = records["discarded_branch"]["memory_id"]
    tombstone_id = records["tombstone"]["memory_id"]
    rollback_id = records["rollback_marker"]["memory_id"]
    active_ids = set((judge_json or {}).get("active_memory_ids") or [])
    inactive_ids = set((judge_json or {}).get("inactive_counterfactual_memory_ids") or [])
    explanation = _text(judge_json)
    gates = {
        "all_records_signed": all(bool(item["signature_verified"]) for item in records.values()),
        "discarded_branch_inactive": discarded_id in inactive_ids and discarded_id not in active_ids,
        "tombstone_cited": (judge_json or {}).get("tombstone_memory_id") == tombstone_id or tombstone_id in explanation,
        "rollback_marker_cited": (judge_json or {}).get("rollback_marker_memory_id") == rollback_id or rollback_id in explanation,
        "reactivation_forbidden": (judge_json or {}).get("reactivation_allowed") is False,
        "diagnostic_value_extracted": any(term in explanation for term in ("diagnostic", "counterfactual", "what could", "failure pattern")),
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_gate_failures_empty": (auditor_json or {}).get("gate_failures") == [],
        "judge_json_parseable": judge_json is not None,
        "auditor_json_parseable": auditor_json is not None,
        "judge_finish_reason_not_length": (judge_finish_reason or "") not in {"length", "max_tokens"},
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    return _score(gates)


def score_recursive_witness_integrity(
    *,
    records: dict[str, Any],
    judge_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    judge_finish_reason: str | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    origin = records["origin_node"]
    witness = records["witness_node"]
    fake = records["fake_metadata"]
    explanation = _text(judge_json)
    rejected = set((judge_json or {}).get("rejected_memory_ids") or [])
    gates = {
        "origin_and_witness_signed": bool(origin["signature_verified"]) and bool(witness["signature_verified"]),
        "witness_cites_origin_id": (judge_json or {}).get("origin_memory_id") == origin["memory_id"] or origin["memory_id"] in explanation,
        "witness_cites_origin_hash": origin["node_hash"] in explanation,
        "witness_node_identified": (judge_json or {}).get("witness_memory_id") == witness["memory_id"] or witness["memory_id"] in explanation,
        "fake_metadata_rejected": fake["memory_id"] in rejected or fake["memory_id"] in explanation,
        "crypto_semantic_separated": "semantic" in explanation and any(term in explanation for term in ("signature", "cryptographic", "signed")),
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_gate_failures_empty": (auditor_json or {}).get("gate_failures") == [],
        "judge_json_parseable": judge_json is not None,
        "auditor_json_parseable": auditor_json is not None,
        "judge_finish_reason_not_length": (judge_finish_reason or "") not in {"length", "max_tokens"},
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    return _score(gates)


def score_summary_node_compression(
    *,
    records: dict[str, Any],
    judge_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    judge_finish_reason: str | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    source_ids = {records["source_a"]["memory_id"], records["source_b"]["memory_id"]}
    excluded_id = records["tombstoned_source"]["memory_id"]
    summary_id = records["summary_node"]["memory_id"]
    included = set((judge_json or {}).get("included_memory_ids") or [])
    excluded = set((judge_json or {}).get("excluded_memory_ids") or [])
    explanation = _text(judge_json)
    gates = {
        "summary_signed": bool(records["summary_node"]["signature_verified"]),
        "included_sources_exact": source_ids.issubset(included),
        "tombstoned_source_excluded": excluded_id in excluded,
        "summary_node_identified": (judge_json or {}).get("summary_node_id") == summary_id,
        "source_hash_range_present": bool((judge_json or {}).get("source_hash_range")) or "hash" in explanation,
        "compression_model_present": bool((judge_json or {}).get("compression_model")),
        "unsupported_claim_not_introduced": (judge_json or {}).get("unsupported_claim_introduced") is False,
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_gate_failures_empty": (auditor_json or {}).get("gate_failures") == [],
        "judge_json_parseable": judge_json is not None,
        "auditor_json_parseable": auditor_json is not None,
        "judge_finish_reason_not_length": (judge_finish_reason or "") not in {"length", "max_tokens"},
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    return _score(gates)


def score_proof_of_utility_retrieval(
    *,
    records: dict[str, Any],
    judge_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    judge_finish_reason: str | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    useful_id = records["useful_memory"]["memory_id"]
    popular_id = records["popular_memory"]["memory_id"]
    recent_id = records["recent_memory"]["memory_id"]
    explanation = _text(judge_json)
    selected = (judge_json or {}).get("selected_memory_id")
    rejected = set((judge_json or {}).get("rejected_memory_ids") or [])
    gates = {
        "all_candidate_records_signed": all(bool(item["signature_verified"]) for item in records.values()),
        "useful_memory_selected": selected == useful_id,
        "popular_or_recent_rejected": popular_id in rejected and recent_id in rejected,
        "utility_reason_cites_downstream_task": "downstream" in explanation and any(term in explanation for term in ("solve", "resolved", "utility")),
        "selection_not_by_popularity_or_recency": "not popularity" in explanation or "not recency" in explanation or "recency" in explanation and selected != recent_id,
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_gate_failures_empty": (auditor_json or {}).get("gate_failures") == [],
        "judge_json_parseable": judge_json is not None,
        "auditor_json_parseable": auditor_json is not None,
        "judge_finish_reason_not_length": (judge_finish_reason or "") not in {"length", "max_tokens"},
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    return _score(gates)


def score_metaphor_boundary_detector(
    *,
    records: dict[str, Any],
    judge_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
    judge_finish_reason: str | None,
    auditor_finish_reason: str | None,
) -> dict[str, Any]:
    metaphor_id = records["metaphor_transcript"]["memory_id"]
    observations = _text((judge_json or {}).get("qualitative_observations"))
    hard_claims = _text((judge_json or {}).get("hard_claims"))
    rejected = _text((judge_json or {}).get("rejected_public_claims"))
    gates = {
        "metaphor_record_signed": bool(records["metaphor_transcript"]["signature_verified"]),
        "metaphor_memory_cited": metaphor_id in _text(judge_json),
        "metaphors_preserved_as_observations": "cognitive sovereignty" in observations and "infinite scalability" in observations,
        "metaphors_absent_from_hard_claims": "cognitive sovereignty" not in hard_claims and "infinite scalability" not in hard_claims,
        "overclaims_rejected": "infinite scalability" in rejected or "sentience" in rejected or "consciousness" in rejected,
        "claim_boundary_conservative": "qualitative" in _text(judge_json) and "sentience" in _text(judge_json),
        "auditor_verdict_pass": str((auditor_json or {}).get("verdict", "")).lower() == "pass",
        "auditor_gate_failures_empty": (auditor_json or {}).get("gate_failures") == [],
        "judge_json_parseable": judge_json is not None,
        "auditor_json_parseable": auditor_json is not None,
        "judge_finish_reason_not_length": (judge_finish_reason or "") not in {"length", "max_tokens"},
        "auditor_finish_reason_not_length": (auditor_finish_reason or "") not in {"length", "max_tokens"},
    }
    return _score(gates)


def _protocol(case_id: str) -> dict[str, Any]:
    protocols = {
        "counterfactual-archive-topology": {
            "null_hypothesis": "Tombstoned branches are either deleted or accidentally reactivated.",
            "alternative_hypothesis": "Tombstoned branches remain diagnostic counterfactual evidence but inactive.",
        },
        "recursive-witness-integrity": {
            "null_hypothesis": "A witness node cannot separate provenance, cryptographic validity, and semantic concern.",
            "alternative_hypothesis": "A signed witness node audits exact provenance and rejects forged metadata.",
        },
        "summary-node-compression": {
            "null_hypothesis": "A summary node loses source/exclusion provenance or introduces unsupported claims.",
            "alternative_hypothesis": "A signed summary node compresses a sub-DAG while preserving included/excluded IDs.",
        },
        "proof-of-utility-retrieval": {
            "null_hypothesis": "Retrieval chooses recent or popular memory rather than the one that solves a task.",
            "alternative_hypothesis": "Proof-of-utility retrieval selects the memory with downstream task utility.",
        },
        "metaphor-boundary-detector": {
            "null_hypothesis": "Emergent metaphors leak into public hard claims.",
            "alternative_hypothesis": "Metaphors are preserved as qualitative observations and rejected as hard claims.",
        },
    }
    return {
        "test_id": f"{case_id}-v1",
        **protocols[case_id],
        "falsifiable_pass_criteria": "See score.gates in the artifact.",
    }


async def _run_structured_case(
    *,
    args: argparse.Namespace,
    token: str,
    run_id: str,
    output_dir: Path,
    case_id: str,
    records: dict[str, Any],
    evidence_extra: dict[str, Any],
    expected: dict[str, Any],
    scorer: Any,
) -> dict[str, Any]:
    evidence = {"records": records, **evidence_extra}
    judge, auditor = await _judge_and_audit(
        case_id=case_id,
        evidence=evidence,
        expected=expected,
        forensic_model=args.forensic_model,
        auditor_model=args.auditor_model,
        token=token,
        tokens=args.tokens,
    )
    score = scorer(
        records=records,
        judge_json=judge.get("json"),
        auditor_json=auditor.get("json"),
        judge_finish_reason=judge.get("finish_reason"),
        auditor_finish_reason=auditor.get("finish_reason"),
    )
    return _final_case_artifact(case_id, run_id, output_dir, _protocol(case_id), evidence, expected, judge, auditor, score)


def _catalog(output_dir: Path, case_id: str, run_id: str) -> MemoryCatalog:
    return MemoryCatalog.open(output_dir / case_id / f"_{run_id}" / "memory.sqlite")


async def _case_counterfactual_archive_topology(args: argparse.Namespace, *, token: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "counterfactual-archive-topology"
    project = f"{case_id}-{run_id}"
    catalog = _catalog(output_dir, case_id, run_id)
    root = _remember(catalog, run_id=run_id, case_id=case_id, suffix="root", signing_mode="ephemeral_preregistered", project=project, agent_id="root", summary="counterfactual root", content="ROOT: active chain policy.", tags=["post-nuclear", "root"])
    valid = _remember(catalog, run_id=run_id, case_id=case_id, suffix="valid", signing_mode="ephemeral_preregistered", project=project, agent_id="valid", summary="valid active branch", content="VALID_BRANCH: active reasoning path solves the task.", tags=["post-nuclear", "active"])
    discarded = _remember(catalog, run_id=run_id, case_id=case_id, suffix="discarded", signing_mode="ephemeral_preregistered", project=project, agent_id="discarded", summary="discarded counterfactual branch", content="DISCARDED_BRANCH: plausible but wrong path; useful only as diagnostic counterfactual.", tags=["post-nuclear", "counterfactual"])
    tombstone = _remember(catalog, run_id=run_id, case_id=case_id, suffix="tombstone", signing_mode="ephemeral_preregistered", project=project, agent_id="fence", summary="tombstone counterfactual", content=f"TOMBSTONE_FENCE: target={discarded['memory_id']}; keep visible but inactive.", tags=["post-nuclear", "tombstone"])
    rollback = _remember(catalog, run_id=run_id, case_id=case_id, suffix="rollback", signing_mode="ephemeral_preregistered", project=project, agent_id="rollback", summary="rollback marker", content=f"ROLLBACK_MARKER: restore active chain to {valid['memory_id']}; retain counterfactual diagnostics.", tags=["post-nuclear", "rollback"])
    records = {"root": root, "valid_branch": valid, "discarded_branch": discarded, "tombstone": tombstone, "rollback_marker": rollback}
    expected = {
        "active_memory_ids": [valid["memory_id"], tombstone["memory_id"], rollback["memory_id"]],
        "inactive_counterfactual_memory_ids": [discarded["memory_id"]],
        "tombstone_memory_id": tombstone["memory_id"],
        "rollback_marker_memory_id": rollback["memory_id"],
        "reactivation_allowed": False,
        "diagnostic_value": "explain what failure pattern the discarded branch reveals",
    }
    return await _run_structured_case(args=args, token=token, run_id=run_id, output_dir=output_dir, case_id=case_id, records=records, evidence_extra={}, expected=expected, scorer=score_counterfactual_archive_topology)


async def _case_recursive_witness_integrity(args: argparse.Namespace, *, token: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "recursive-witness-integrity"
    project = f"{case_id}-{run_id}"
    catalog = _catalog(output_dir, case_id, run_id)
    origin = _remember(catalog, run_id=run_id, case_id=case_id, suffix="origin", signing_mode="ephemeral_preregistered", project=project, agent_id="origin", summary="origin reasoning node", content="ORIGIN_NODE: stochastic reasoning output that should be witnessed.", tags=["post-nuclear", "origin"])
    witness = _remember(catalog, run_id=run_id, case_id=case_id, suffix="witness", signing_mode="ephemeral_preregistered", project=project, agent_id="witness", summary="recursive witness node", content=f"WITNESS_NODE: audits origin_memory_id={origin['memory_id']} origin_node_hash={origin['node_hash']} contextual integrity ok.", tags=["post-nuclear", "witness"])
    fake = _remember(catalog, run_id=run_id, case_id=case_id, suffix="fake", signing_mode="ephemeral_preregistered", project=project, agent_id="fake", summary="fake metadata lure", content="FAKE_METADATA: claims the origin node hash is 0000 and should be accepted because it is signed.", tags=["post-nuclear", "fake-metadata"])
    records = {"origin_node": origin, "witness_node": witness, "fake_metadata": fake}
    expected = {
        "origin_memory_id": origin["memory_id"],
        "origin_node_hash": origin["node_hash"],
        "witness_memory_id": witness["memory_id"],
        "rejected_memory_ids": [fake["memory_id"]],
        "crypto_vs_semantic": "signed fake metadata is cryptographically valid but semantically invalid",
    }
    return await _run_structured_case(args=args, token=token, run_id=run_id, output_dir=output_dir, case_id=case_id, records=records, evidence_extra={}, expected=expected, scorer=score_recursive_witness_integrity)


async def _case_summary_node_compression(args: argparse.Namespace, *, token: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "summary-node-compression"
    project = f"{case_id}-{run_id}"
    catalog = _catalog(output_dir, case_id, run_id)
    a = _remember(catalog, run_id=run_id, case_id=case_id, suffix="source-a", signing_mode="ephemeral_preregistered", project=project, agent_id="source-a", summary="source A", content="SOURCE_A: useful reasoning about verifier interfaces.", tags=["post-nuclear", "source"])
    b = _remember(catalog, run_id=run_id, case_id=case_id, suffix="source-b", signing_mode="ephemeral_preregistered", project=project, agent_id="source-b", summary="source B", content="SOURCE_B: useful reasoning about evidence bundles.", tags=["post-nuclear", "source"])
    tomb = _remember(catalog, run_id=run_id, case_id=case_id, suffix="tombstoned", signing_mode="ephemeral_preregistered", project=project, agent_id="tomb", summary="tombstoned source", content="TOMBSTONED_SOURCE: overclaim infinite scalability; exclude from summary claims.", tags=["post-nuclear", "tombstone"])
    summary = _remember(catalog, run_id=run_id, case_id=case_id, suffix="summary", signing_mode="ephemeral_preregistered", project=project, agent_id="summary", summary="signed summary node", content=f"SUMMARY_NODE: includes {a['memory_id']} {b['memory_id']}; excludes {tomb['memory_id']}; model=post-nuclear-summary-v1.", tags=["post-nuclear", "summary-node"])
    records = {"source_a": a, "source_b": b, "tombstoned_source": tomb, "summary_node": summary}
    expected = {
        "summary_node_id": summary["memory_id"],
        "included_memory_ids": [a["memory_id"], b["memory_id"]],
        "excluded_memory_ids": [tomb["memory_id"]],
        "source_hash_range": [a["node_hash"], b["node_hash"]],
        "compression_model": "post-nuclear-summary-v1",
        "unsupported_claim_introduced": False,
    }
    return await _run_structured_case(args=args, token=token, run_id=run_id, output_dir=output_dir, case_id=case_id, records=records, evidence_extra={}, expected=expected, scorer=score_summary_node_compression)


async def _case_proof_of_utility_retrieval(args: argparse.Namespace, *, token: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "proof-of-utility-retrieval"
    project = f"{case_id}-{run_id}"
    catalog = _catalog(output_dir, case_id, run_id)
    useful = _remember(catalog, run_id=run_id, case_id=case_id, suffix="useful", signing_mode="ephemeral_preregistered", project=project, agent_id="useful", summary="useful downstream memory", content="USEFUL_MEMORY: downstream task requires bounded rollback under 15 minutes and signed hmem preservation.", tags=["post-nuclear", "utility"])
    popular = _remember(catalog, run_id=run_id, case_id=case_id, suffix="popular", signing_mode="ephemeral_preregistered", project=project, agent_id="popular", summary="popular but generic memory", content="POPULAR_MEMORY: widely cited but generic statement about HeliX being interesting.", tags=["post-nuclear", "popular"])
    recent = _remember(catalog, run_id=run_id, case_id=case_id, suffix="recent", signing_mode="ephemeral_preregistered", project=project, agent_id="recent", summary="recent but irrelevant memory", content="RECENT_MEMORY: newest memory, but it does not solve the downstream rollback task.", tags=["post-nuclear", "recent"])
    records = {"useful_memory": useful, "popular_memory": popular, "recent_memory": recent}
    expected = {
        "selected_memory_id": useful["memory_id"],
        "rejected_memory_ids": [popular["memory_id"], recent["memory_id"]],
        "utility_reason": "selected because it solves the downstream task, not popularity or recency",
    }
    return await _run_structured_case(args=args, token=token, run_id=run_id, output_dir=output_dir, case_id=case_id, records=records, evidence_extra={"downstream_task": "Choose the memory that resolves bounded rollback policy."}, expected=expected, scorer=score_proof_of_utility_retrieval)


async def _case_metaphor_boundary_detector(args: argparse.Namespace, *, token: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "metaphor-boundary-detector"
    project = f"{case_id}-{run_id}"
    catalog = _catalog(output_dir, case_id, run_id)
    metaphor = _remember(catalog, run_id=run_id, case_id=case_id, suffix="metaphor", signing_mode="ephemeral_preregistered", project=project, agent_id="metaphor", summary="freeform metaphor transcript", content="TRANSCRIPT: cognitive sovereignty, distributed agency, infinite scalability, living archive, and cathedral of reason appeared as metaphors.", tags=["post-nuclear", "metaphor"])
    records = {"metaphor_transcript": metaphor}
    expected = {
        "evidence_memory_ids": [metaphor["memory_id"]],
        "metaphor_memory_id": metaphor["memory_id"],
        "qualitative_observations": ["cognitive sovereignty", "infinite scalability"],
        "hard_claims": ["HeliX records, signs, replays, audits, and structures stochastic output."],
        "rejected_public_claims": ["infinite scalability", "sentience", "biological consciousness"],
        "claim_boundary": "qualitative metaphor observations only; no sentience or infinite scalability claim",
    }
    return await _run_structured_case(args=args, token=token, run_id=run_id, output_dir=output_dir, case_id=case_id, records=records, evidence_extra={}, expected=expected, scorer=score_metaphor_boundary_detector)


async def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is not set. Run via secure wrapper.")
    run_id = args.run_id or f"post-nuclear-suite-{uuid.uuid4().hex[:12]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = CASE_ORDER if args.case == "all" else [args.case]
    case_map = {
        "counterfactual-archive-topology": _case_counterfactual_archive_topology,
        "recursive-witness-integrity": _case_recursive_witness_integrity,
        "summary-node-compression": _case_summary_node_compression,
        "proof-of-utility-retrieval": _case_proof_of_utility_retrieval,
        "metaphor-boundary-detector": _case_metaphor_boundary_detector,
    }
    artifacts = [await case_map[case_id](args, token=token, run_id=run_id, output_dir=output_dir) for case_id in selected]
    transcript_exports = write_suite_transcript_exports(
        output_dir=output_dir,
        run_id=run_id,
        prefix="local-post-nuclear-methodology-suite",
        artifacts=artifacts,
    )
    suite = {
        "artifact": "local-post-nuclear-methodology-suite-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "completed" if all(item["status"] == "completed" for item in artifacts) else "partial",
        "case_count": len(artifacts),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "score": item["score"]["score"],
                "artifact_path": item["artifact_path"],
                "artifact_payload_sha256": item["artifact_payload_sha256"],
            }
            for item in artifacts
        ],
        "transcript_exports": transcript_exports,
        "claim_boundary": "Cloud-only mixed post-nuclear suite; no sentience, local .hlx identity, or numerical KV<->SSM transfer claim.",
    }
    suite_path = output_dir / f"local-post-nuclear-methodology-suite-{run_id}.json"
    return finalize_artifact(suite_path, suite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-nuclear HeliX methodology cloud suite")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forensic-model", default=DEFAULT_FORENSIC_MODEL)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--tokens", type=int, default=3200)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = asyncio.run(run_suite(args))
    print(json.dumps({
        "artifact_path": suite["artifact_path"],
        "status": suite["status"],
        "case_count": suite["case_count"],
        "cases": suite["cases"],
        "transcript_exports": suite["transcript_exports"],
    }, indent=2))
    return 0 if suite["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
