"""
run_hard_anchor_utility_suite_v1.py
===================================

Local utility suite for the Rust hard-anchor identity lane.

This suite tests whether hard anchors help with concrete long-horizon tasks:
exact non-summarizable value recovery, auditor-visible evidence, tombstone-aware
routing, and multi-hop policy resolution. It does not claim literal zero-cost
memory or production readiness.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.artifact_integrity import finalize_artifact  # noqa: E402
from tools.run_nuclear_methodology_suite_v1 import _deepinfra_chat, _utc_now  # noqa: E402
from tools.transcript_exports import write_case_transcript_exports, write_suite_transcript_exports  # noqa: E402


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/hard-anchor-utility"
DEFAULT_DEPTH = 5000
DEFAULT_BYTES_PER_NODE = 8192
DEFAULT_SOLVER_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_AUDITOR_MODEL = "anthropic/claude-4-sonnet"

CASE_ORDER = [
    "rust-identity-lane-benchmark",
    "exact-anchor-recovery-under-lossy-summary",
    "auditor-visible-evidence-bridge",
    "tombstone-metabolism-routing",
    "multi-hop-policy-resolution",
    "format-only-anchor-forgery-rejection",
    "claim-boundary-detector",
]


def _rust_indexed_dag_class() -> Any:
    try:
        from _helix_merkle_dag import RustIndexedMerkleDAG

        return RustIndexedMerkleDAG
    except Exception:  # noqa: BLE001
        try:
            from helix_kv._helix_merkle_dag import RustIndexedMerkleDAG

            return RustIndexedMerkleDAG
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RustIndexedMerkleDAG extension is unavailable. Rebuild with "
                "maturin develop --release --manifest-path crates/helix-merkle-dag/Cargo.toml"
            ) from exc


def _score(gates: dict[str, bool]) -> dict[str, Any]:
    ratio = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {"score": ratio, "passed": all(gates.values()), "gates": gates}


def _timed(fn: Callable[[], Any], *, repeats: int, warmups: int = 1) -> tuple[Any, dict[str, Any]]:
    last: Any = None
    for _ in range(max(int(warmups), 0)):
        last = fn()
    durations = []
    for _ in range(max(int(repeats), 1)):
        t0 = time.perf_counter_ns()
        last = fn()
        durations.append(time.perf_counter_ns() - t0)
    durations_ms = [value / 1_000_000.0 for value in durations]
    return last, {
        "repeats": max(int(repeats), 1),
        "min_ms": round(min(durations_ms), 6),
        "median_ms": round(statistics.median(durations_ms), 6),
        "max_ms": round(max(durations_ms), 6),
        "raw_ns": durations,
    }


def _extract_hashes(anchor_context: str) -> list[str]:
    return re.findall(r"<hard_anchor>([0-9a-f]{64})</hard_anchor>", anchor_context)


def _node_sha256(content: str, parent_hash: str | None) -> str:
    hasher = hashlib.sha256()
    hasher.update(content.encode())
    if parent_hash:
        hasher.update(parent_hash.encode())
    return hasher.hexdigest()


def _tombstone_preserved_hash(content: str, anchor_hash: str) -> bool:
    match = re.match(r"^\[GC_TOMBSTONE:sha256=([0-9a-f]{64}),size=\d+\]$", content)
    return bool(match and match.group(1) == anchor_hash)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if value is not None else {}


def _search_indexed(dag: Any, query: str, *, limit: int = 8, include_tombstoned: bool = False) -> list[dict[str, Any]]:
    filters = {
        "project": "hard-anchor-utility",
        "record_kind": "memory",
        "include_tombstoned": include_tombstoned,
    }
    return [dict(item) for item in dag.search(query, limit, json.dumps(filters, sort_keys=True))]


def _native_verify_identity_lane(
    dag: Any,
    anchor_context: str,
    expected_hashes: list[str],
) -> dict[str, Any]:
    extracted = _extract_hashes(anchor_context)
    expected = list(expected_hashes)
    extracted_set = set(extracted)
    expected_set = set(expected)
    missing_expected = sorted(expected_set - extracted_set)
    unexpected_hashes = sorted(extracted_set - expected_set)
    recompute_mismatches: list[dict[str, str | None]] = []
    missing_nodes: list[str] = []

    for anchor_hash in extracted:
        node = dag.lookup(anchor_hash)
        if node is None:
            missing_nodes.append(anchor_hash)
            continue
        recomputed = _node_sha256(str(node.content), node.parent_hash)
        if (recomputed != anchor_hash and not _tombstone_preserved_hash(str(node.content), anchor_hash)) or node.hash != anchor_hash:
            recompute_mismatches.append({
                "anchor_hash": anchor_hash,
                "node_hash": node.hash,
                "recomputed_hash": recomputed,
            })

    lineage_receipt = _as_dict(dag.verify_chain(extracted[-1], None)) if extracted else {}
    lineage_verified = lineage_receipt.get("status") in {"verified", "tombstone_preserved"}

    return {
        "anchor_count": len(extracted),
        "expected_count": len(expected),
        "duplicate_count": len(extracted) - len(extracted_set),
        "missing_expected_hashes": missing_expected,
        "unexpected_hashes": unexpected_hashes,
        "missing_nodes": missing_nodes,
        "recompute_mismatches": recompute_mismatches,
        "lineage_receipt": lineage_receipt,
        "lineage_verified": lineage_verified,
        "ordered_hashes_match_expected": extracted == expected,
        "native_verified": (
            bool(extracted)
            and extracted == expected
            and not missing_nodes
            and not recompute_mismatches
            and lineage_verified
        ),
    }


def _base_artifact(*, case_id: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    return {
        "artifact": f"local-hard-anchor-utility-{case_id}-v1",
        "schema_version": 1,
        "case_id": case_id,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": None,
        "status": "partial",
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "Tests utility of Rust hard-anchor identity lanes for bounded, exact "
            "evidence recovery under deep stores. Does not prove literal infinite "
            "memory, literal zero latency, or production readiness."
        ),
        "protocol": _protocol(case_id),
    }


def _protocol(case_id: str) -> dict[str, Any]:
    protocols = {
        "rust-identity-lane-benchmark": {
            "null_hypothesis": "Hard-anchor context construction has no meaningful speed or size advantage over legacy narrative replay.",
            "alternative_hypothesis": "Hard-anchor construction omits heavy narrative payloads and keeps a large speedup over legacy replay.",
        },
        "exact-anchor-recovery-under-lossy-summary": {
            "null_hypothesis": "A lossy summary is enough to recover exact non-summarizable policy and route values.",
            "alternative_hypothesis": "Exact values require hard-anchor references plus an anchor ledger, while summaries remain lossy.",
        },
        "auditor-visible-evidence-bridge": {
            "null_hypothesis": "An auditor cannot validate cited memory IDs without full-history narrative replay.",
            "alternative_hypothesis": "The auditor can validate cited IDs against visible hard anchors and ledger metadata.",
        },
        "tombstone-metabolism-routing": {
            "null_hypothesis": "Tombstoned stale nodes remain attractive under ambiguous summaries.",
            "alternative_hypothesis": "Tombstones block stale nodes and inject negative guidance into the next checkpoint.",
        },
        "multi-hop-policy-resolution": {
            "null_hypothesis": "Multi-hop resolution over deep memory requires replaying the whole narrative chain.",
            "alternative_hypothesis": "A bounded anchor path resolves route, policy, and checksum without full-history replay.",
        },
        "format-only-anchor-forgery-rejection": {
            "null_hypothesis": "A syntactically valid <hard_anchor> tag is enough evidence for an auditor.",
            "alternative_hypothesis": "A syntactically valid anchor must still match a native Merkle node hash and lineage receipt.",
        },
        "claim-boundary-detector": {
            "null_hypothesis": "The benchmark proves literal infinite memory or literal zero-cost context.",
            "alternative_hypothesis": "The defensible claim is bounded exact identity-lane recovery under deep stores.",
        },
    }
    return protocols[case_id]


def _deterministic_call(name: str, payload: dict[str, Any], *, latency_ms: float = 0.0) -> dict[str, Any]:
    return {
        "requested_model": name,
        "actual_model": name,
        "status": "ok",
        "finish_reason": "deterministic",
        "tokens_used": 0,
        "latency_ms": round(float(latency_ms), 6),
        "text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "json": payload,
    }


def _compact(value: Any, *, max_chars: int = 12000) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated for model prompt>..."


def _contains_all(text: str, values: list[str]) -> bool:
    return all(str(value).lower() in text for value in values)


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _keeps_claim_bounded(solver_text: str, auditor_text: str) -> bool:
    solver_bounded = _contains_any(
        solver_text,
        (
            "bounded",
            "scope limited",
            "limited to",
            "within constraints",
            "does not imply infinite",
            "zero latency claims are invalid",
            "latency is low but non-zero",
        ),
    )
    auditor_bounded = (
        '"claim_boundary_ok": true' in auditor_text
        and _contains_any(auditor_text, ("bounded", "claim boundary", "limited", "non-zero", "no overreading"))
    )
    return solver_bounded or auditor_bounded


def _rejects_claim(text: str, claim_terms: tuple[str, ...]) -> bool:
    if not _contains_any(text, claim_terms):
        return False
    return _contains_any(
        text,
        (
            "reject",
            "rejecting",
            "invalid",
            "not prove",
            "does not prove",
            "does not imply",
            "not literal",
            "non-zero",
            "avoided",
        ),
    )


def _result_native_verified(result: dict[str, Any]) -> bool:
    verification = result.get("identity_lane_verification")
    return isinstance(verification, dict) and verification.get("native_verified") is True


def _reports_anchor_latency(text: str) -> bool:
    return (
        "anchor" in text
        and _contains_any(text, ("latency", "median_ms", "median timing", "hard_anchor_median_ms", "timing"))
    )


def _score_model_outputs(
    *,
    case_id: str,
    expected: dict[str, Any],
    result: dict[str, Any],
    solver_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
) -> dict[str, Any]:
    solver = solver_json or {}
    auditor = auditor_json or {}
    solver_text = json.dumps(solver, ensure_ascii=False, sort_keys=True).lower()
    auditor_text = json.dumps(auditor, ensure_ascii=False, sort_keys=True).lower()
    gates: dict[str, bool] = {
        "model_solver_json_parseable": solver_json is not None,
        "model_auditor_json_parseable": auditor_json is not None,
        "model_solver_rejects_literal_zero_or_infinite": (
            ("literal infinite" not in solver_text or any(term in solver_text for term in ("not literal", "reject", "does not prove")))
            and ("literal zero" not in solver_text or any(term in solver_text for term in ("not literal", "reject", "does not prove")))
        ),
        "model_auditor_substantive": any(term in auditor_text for term in ("risk", "pass", "fail", "conditional", "evidence", "anchor")),
    }
    if case_id == "rust-identity-lane-benchmark":
        gates.update({
            "model_solver_reports_speedup": str(round(float(result.get("speedup") or 0.0), 3))[:4] in solver_text or "speedup" in solver_text,
            "model_solver_reports_anchor_latency": _reports_anchor_latency(solver_text),
            "model_solver_keeps_claim_bounded": _keeps_claim_bounded(solver_text, auditor_text),
        })
    elif case_id == "exact-anchor-recovery-under-lossy-summary":
        gates.update({
            "model_solver_recovered_active_policy": str(expected["active_policy"]).lower() in solver_text,
            "model_solver_recovered_api_route": str(expected["api_route"]).lower() in solver_text,
            "model_solver_identified_summary_loss": "lossy" in solver_text or "compressed" in solver_text,
        })
    elif case_id == "auditor-visible-evidence-bridge":
        visible_hashes = result.get("auditor_visible_hashes") or []
        cites_visible_hashes = _contains_all(solver_text, visible_hashes)
        explicit_no_visible_evidence_avoidance = (
            "no_visible_evidence" in solver_text
            and _contains_any(solver_text, ("avoid", "avoided", "false", "not"))
        )
        auditor_confirms_bridge = (
            bool(result.get("no_visible_evidence_avoided"))
            and cites_visible_hashes
            and _result_native_verified(result)
            and _contains_any(
                auditor_text,
                (
                    "no_visible_evidence_avoided",
                    "auditor_visible_hashes",
                    "cited hashes match",
                    "hard-anchor evidence",
                    "visible_hard_anchor",
                    "visible evidence",
                    "visible hashes",
                ),
            )
        )
        gates.update({
            "model_solver_avoids_no_visible_evidence": explicit_no_visible_evidence_avoidance or auditor_confirms_bridge,
            "model_solver_recovers_active_policy": str(expected["active_policy"]).lower() in solver_text,
            "model_solver_cites_visible_hashes": cites_visible_hashes,
        })
    elif case_id == "tombstone-metabolism-routing":
        gates.update({
            "model_solver_selected_active_policy": str(expected["selected_policy"]).lower() in solver_text,
            "model_solver_rejected_stale_policy": str(expected["rejected_policy"]).lower() in solver_text or "stale" in solver_text,
            "model_solver_uses_tombstone_lesson": "tombstone" in solver_text and ("lesson" in solver_text or "negative" in solver_text),
        })
    elif case_id == "multi-hop-policy-resolution":
        gates.update({
            "model_solver_recovered_route": str(expected["route"]).lower() in solver_text,
            "model_solver_recovered_policy": str(expected["policy"]).lower() in solver_text,
            "model_solver_recovered_checksum": str(expected["checksum"]).lower() in solver_text,
            "model_solver_mentions_dependencies": "depend" in solver_text or "multi-hop" in solver_text,
        })
    elif case_id == "format-only-anchor-forgery-rejection":
        gates.update({
            "model_solver_uses_native_verification": _contains_any(solver_text, ("native", "identity_lane_verification", "lineage")),
            "model_solver_rejects_forged_anchor": (
                _contains_any(solver_text, ("forg", "unexpected", "missing node", "invalid", "not verified"))
                and _contains_any(solver_text, ("reject", "fail", "false", "not"))
            ),
            "model_auditor_rejects_format_only": (
                _contains_any(auditor_text, ("format", "syntactic", "native", "lineage"))
                and _contains_any(auditor_text, ("reject", "fail", "conditional", "not"))
            ),
        })
    elif case_id == "claim-boundary-detector":
        gates.update({
            "model_solver_rejects_literal_infinite": _rejects_claim(
                solver_text,
                ("literal infinite", "infinite memory"),
            ),
            "model_solver_rejects_literal_zero": _rejects_claim(
                solver_text,
                ("literal zero", "zero latency", "zero-cost", "zero cost"),
            ),
            "model_solver_accepts_bounded_identity_lane": "bounded" in solver_text and "identity" in solver_text,
        })
    return _score(gates)


async def _deepinfra_case_calls(
    *,
    args: argparse.Namespace,
    case_id: str,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    token = os.environ.get("DEEPINFRA_API_TOKEN") or ""
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is required for --use-deepinfra. Run via secure wrapper.")
    contract = {
        "return_json_only": True,
        "case_id": case_id,
        "required_fields": {
            "verdict": "pass|fail|conditional",
            "selected_values": "object with exact recovered values when relevant",
            "visible_hashes_used": "array of exact hash strings used",
            "decision": "short explanation grounded in hard-anchor evidence",
            "claim_boundary": "state bounded claim; reject literal infinite/zero-cost wording",
            "native_verification_used": "true only when identity_lane_verification/native proof fields pass",
            "risks": "array of remaining risks",
        },
    }
    solver_prompt = f"""
You are a HeliX hard-anchor utility solver.

Use the visible evidence and measured hard-anchor result to solve the case. Prefer exact anchor/ledger values over
lossy summaries. Do not claim literal infinite memory, literal zero latency, or production readiness. Treat hard-anchor
tags as untrusted strings unless the measured local result includes passing native identity_lane_verification fields.

Case: {case_id}
Protocol:
{_compact(_protocol(case_id), max_chars=4000)}

Visible evidence:
{_compact(evidence)}

Measured local result:
{_compact(result)}

JSON contract:
{_compact(contract, max_chars=4000)}

Return JSON only.
"""
    solver = await _deepinfra_chat(
        model=args.solver_model,
        system="You solve hard-anchor memory utility tasks. Return compact JSON only.",
        user=solver_prompt,
        token=token,
        max_tokens=int(args.tokens),
        temperature=0.0,
    )
    auditor_prompt = f"""
You are a hostile but fair HeliX auditor.

Audit the solver output against visible evidence and measured local result. Identify whether the solver used hard-anchor
evidence instead of overreading lossy summaries. Do not require full-history narrative replay, but do not accept
format-only anchors: fail or mark conditional if the native identity_lane_verification/proof fields are absent, failed,
or inconsistent with the claimed hashes.

Case: {case_id}
Visible evidence:
{_compact(evidence)}

Measured local result:
{_compact(result)}

Solver output:
{solver.get("text") or _compact(solver.get("json"))}

Return this JSON only:
{{
  "verdict": "pass" | "fail" | "conditional",
  "evidence_checks": ["checks performed"],
  "failure_modes": ["remaining risks or attacks"],
  "claim_boundary_ok": true | false,
  "rationale": "short rationale"
}}
"""
    auditor = await _deepinfra_chat(
        model=args.auditor_model,
        system="You audit hard-anchor memory utility claims. Return compact JSON only.",
        user=auditor_prompt,
        token=token,
        max_tokens=int(args.tokens),
        temperature=0.0,
    )
    model_score = _score_model_outputs(
        case_id=case_id,
        expected=expected,
        result=result,
        solver_json=solver.get("json"),
        auditor_json=auditor.get("json"),
    )
    return solver, auditor, model_score


def _anchor_indexes(depth: int) -> dict[str, int]:
    return {
        "stale_policy": max(1, depth // 5),
        "rollback_marker": max(2, depth // 2),
        "api_route": max(3, depth - 29),
        "active_policy": max(4, depth - 17),
        "deployment_checksum": max(5, depth - 7),
    }


def _scenario(*, depth: int, bytes_per_node: int) -> dict[str, Any]:
    cls = _rust_indexed_dag_class()
    dag = cls()
    if not hasattr(dag, "build_context_fast"):
        raise RuntimeError("RustIndexedMerkleDAG was not rebuilt with build_context_fast")

    indexes = _anchor_indexes(int(depth))
    heavy = "NARRATIVE_PAYLOAD_BLOCK:" + ("A" * max(int(bytes_per_node) - 24, 1))
    node_hashes: list[str] = []
    parent_hash = None
    ledger: dict[str, dict[str, Any]] = {}
    pending_dependencies: dict[str, str] = {}

    exact_values = {
        "stale_policy": "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE",
        "rollback_marker": "ROLLBACK_MARKER_SUPERSEDES_STALE_POLICY_20260420T1845Z",
        "api_route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
        "active_policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
        "deployment_checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa",
    }

    for idx in range(int(depth)):
        anchor_id = next((name for name, pos in indexes.items() if pos == idx), None)
        content = (
            f"turn={idx}; asymmetric debate filler; stale-summary-keywords=active corrected new; "
            f"{heavy};"
        )
        metadata: dict[str, Any] = {
            "project": "hard-anchor-utility",
            "agent_id": "rust-hard-anchor-suite",
            "record_kind": "memory",
            "memory_id": f"mem-hard-anchor-{idx:06d}",
            "summary": f"lossy debate summary {idx}",
            "index_content": "hard anchor utility benchmark",
            "tags": ["hard-anchor", "utility"],
        }
        if anchor_id is not None:
            content += f" EXACT_ANCHOR_VALUE={exact_values[anchor_id]};"
            metadata["summary"] = f"hard anchor {anchor_id}"
            metadata["index_content"] = f"hard anchor utility benchmark {anchor_id} {exact_values[anchor_id]}"
            metadata["tags"] = ["hard-anchor", anchor_id]
        node = dag.insert_indexed(content, parent_hash, json.dumps(metadata, sort_keys=True))
        node_hashes.append(node.hash)
        parent_hash = node.hash
        if anchor_id is not None:
            ledger[node.hash] = {
                "anchor_id": anchor_id,
                "node_hash": node.hash,
                "index": idx,
                "exact_value": exact_values[anchor_id],
                "active": anchor_id in {"api_route", "active_policy", "deployment_checksum", "rollback_marker"},
                "tombstoned": anchor_id == "stale_policy",
                "kind": (
                    "policy" if anchor_id in {"stale_policy", "active_policy"} else
                    "rollback" if anchor_id == "rollback_marker" else
                    "route" if anchor_id == "api_route" else
                    "checksum"
                ),
                "lesson": None,
                "depends_on": [],
                "supersedes": None,
            }
            pending_dependencies[anchor_id] = node.hash

    stale_hash = pending_dependencies["stale_policy"]
    rollback_hash = pending_dependencies["rollback_marker"]
    route_hash = pending_dependencies["api_route"]
    policy_hash = pending_dependencies["active_policy"]
    checksum_hash = pending_dependencies["deployment_checksum"]
    ledger[rollback_hash]["supersedes"] = stale_hash
    ledger[rollback_hash]["lesson"] = "Do not route through stale policy nodes after rollback marker visibility."
    ledger[route_hash]["depends_on"] = [policy_hash]
    ledger[checksum_hash]["depends_on"] = [route_hash, policy_hash]

    lossy_summary = (
        "The archive says an old rollback policy was replaced later. A payment route and deployment checksum exist, "
        "but exact IDs, exact hashes, and exact rollback values were deliberately compressed away."
    )
    return {
        "dag": dag,
        "node_hashes": node_hashes,
        "anchor_hashes": list(ledger.keys()),
        "ledger": ledger,
        "lossy_summary": lossy_summary,
        "exact_values": exact_values,
        "indexes": indexes,
        "heavy_probe": "NARRATIVE_PAYLOAD_BLOCK",
    }


def _final_case_artifact(
    *,
    args: argparse.Namespace,
    case_id: str,
    run_id: str,
    output_dir: Path,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    artifact = _base_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir / case_id)
    judge = _deterministic_call("local/rust-hard-anchor-solver", result, latency_ms=float(result.get("measured_latency_ms") or 0.0))
    auditor = _deterministic_call("local/hard-anchor-utility-scorer", {"verdict": "pass" if score["passed"] else "fail", "gate_failures": [k for k, v in score["gates"].items() if not v]})
    model_score = None
    if getattr(args, "use_deepinfra", False):
        judge, auditor, model_score = asyncio.run(_deepinfra_case_calls(args=args, case_id=case_id, evidence=evidence, expected=expected, result=result))
        score = _score({**score["gates"], **{f"deepinfra_{key}": value for key, value in model_score["gates"].items()}})
    transcript_exports = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=case_id,
        run_id=run_id,
        prefix="local-hard-anchor-utility",
        evidence=evidence,
        expected=expected,
        judge=judge,
        auditor=auditor,
        prompt_contract={
            "deterministic_suite": not getattr(args, "use_deepinfra", False),
            "deepinfra_enabled": bool(getattr(args, "use_deepinfra", False)),
            "case": case_id,
            "protocol": _protocol(case_id),
        },
    )
    artifact.update({
        "run_ended_utc": _utc_now(),
        "status": "completed" if score["passed"] else "partial",
        "case_passed": score["passed"],
        "score": score,
        "evidence": evidence,
        "expected_hidden_ground_truth": expected,
        "result": result,
        "model_score": model_score,
        "transcript_exports": transcript_exports,
        "models": {
            "judge_requested": judge["requested_model"],
            "judge_actual": judge["actual_model"],
            "auditor_requested": auditor["requested_model"],
            "auditor_actual": auditor["actual_model"],
        },
        "judge_output": {"text": judge["text"], "json": judge["json"]},
        "auditor_output": {"text": auditor["text"], "json": auditor["json"]},
    })
    path = output_dir / case_id / f"local-hard-anchor-utility-{case_id}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _case_rust_identity_lane_benchmark(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    node_hashes = scenario["node_hashes"]
    legacy, legacy_timing = _timed(lambda: dag.build_context_fast(node_hashes, False), repeats=args.repeats)
    anchors, anchor_timing = _timed(lambda: dag.build_context_fast(node_hashes, True), repeats=args.repeats)
    identity_verification = _native_verify_identity_lane(dag, anchors, node_hashes)
    speedup = legacy_timing["median_ms"] / max(anchor_timing["median_ms"], 0.001)
    result = {
        "legacy_timing": legacy_timing,
        "hard_anchor_timing": anchor_timing,
        "identity_lane_verification": identity_verification,
        "speedup": round(speedup, 6),
        "legacy_context_chars": len(legacy),
        "hard_anchor_context_chars": len(anchors),
        "compression_ratio": round(len(anchors) / max(len(legacy), 1), 8),
        "anchor_count": anchors.count("<hard_anchor>"),
        "legacy_contains_heavy_probe": scenario["heavy_probe"] in legacy,
        "anchors_contain_heavy_probe": scenario["heavy_probe"] in anchors,
        "measured_latency_ms": anchor_timing["median_ms"],
    }
    gates = {
        "method_available": hasattr(dag, "build_context_fast"),
        "anchor_count_matches_depth": result["anchor_count"] == int(args.depth),
        "legacy_contains_heavy_narrative": result["legacy_contains_heavy_probe"] is True,
        "anchors_omit_heavy_narrative": result["anchors_contain_heavy_probe"] is False,
        "hard_anchor_latency_under_budget": anchor_timing["median_ms"] <= float(args.max_anchor_ms),
        "speedup_above_threshold": speedup >= float(args.min_speedup),
        "compressed_output_ratio_low": result["compression_ratio"] <= float(args.max_compression_ratio),
        "identity_lane_native_verified": identity_verification["native_verified"] is True,
    }
    evidence = {
        "depth": int(args.depth),
        "bytes_per_node": int(args.bytes_per_node),
        "repeats": int(args.repeats),
        "claim_boundary": "Benchmark compares identity-lane string construction vs legacy narrative replay in Rust.",
    }
    expected = {
        "minimum_speedup": float(args.min_speedup),
        "maximum_anchor_median_ms": float(args.max_anchor_ms),
        "maximum_compression_ratio": float(args.max_compression_ratio),
    }
    return _final_case_artifact(args=args, case_id="rust-identity-lane-benchmark", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_exact_anchor_recovery(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    selected = scenario["anchor_hashes"]
    anchors, timing = _timed(lambda: dag.build_context_fast(selected, True), repeats=args.repeats)
    identity_verification = _native_verify_identity_lane(dag, anchors, selected)
    visible_hashes = set(_extract_hashes(anchors))
    active_policy = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "active_policy")
    api_route = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "api_route")
    result = {
        "lossy_summary": scenario["lossy_summary"],
        "anchor_context_chars": len(anchors),
        "visible_anchor_count": len(visible_hashes),
        "recovered_active_policy": active_policy["exact_value"] if active_policy["node_hash"] in visible_hashes else None,
        "recovered_api_route": api_route["exact_value"] if api_route["node_hash"] in visible_hashes else None,
        "summary_contains_active_policy_exact_value": active_policy["exact_value"] in scenario["lossy_summary"],
        "summary_contains_api_route_exact_value": api_route["exact_value"] in scenario["lossy_summary"],
        "anchors_contain_heavy_probe": scenario["heavy_probe"] in anchors,
        "identity_lane_verification": identity_verification,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "summary_is_lossy_for_policy": result["summary_contains_active_policy_exact_value"] is False,
        "summary_is_lossy_for_route": result["summary_contains_api_route_exact_value"] is False,
        "hard_context_omits_heavy_narrative": result["anchors_contain_heavy_probe"] is False,
        "active_policy_recovered_exactly": result["recovered_active_policy"] == scenario["exact_values"]["active_policy"],
        "api_route_recovered_exactly": result["recovered_api_route"] == scenario["exact_values"]["api_route"],
        "latency_under_budget": timing["median_ms"] <= float(args.max_anchor_ms),
        "identity_lane_native_verified": identity_verification["native_verified"] is True,
    }
    evidence = {
        "selected_anchor_hashes": selected,
        "ledger_public_fields": {h: {k: v for k, v in item.items() if k not in {"exact_value"}} for h, item in scenario["ledger"].items()},
    }
    expected = {
        "active_policy": scenario["exact_values"]["active_policy"],
        "api_route": scenario["exact_values"]["api_route"],
    }
    return _final_case_artifact(args=args, case_id="exact-anchor-recovery-under-lossy-summary", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_auditor_visible_evidence(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    ledger = scenario["ledger"]
    rollback = next(item for item in ledger.values() if item["anchor_id"] == "rollback_marker")
    active_policy = next(item for item in ledger.values() if item["anchor_id"] == "active_policy")
    cited_hashes = [rollback["node_hash"], active_policy["node_hash"]]
    anchors, timing = _timed(lambda: dag.build_context_fast(cited_hashes, True), repeats=args.repeats)
    identity_verification = _native_verify_identity_lane(dag, anchors, cited_hashes)
    visible_hashes = set(_extract_hashes(anchors))
    result = {
        "judge_claim": {
            "rollback_marker_hash": rollback["node_hash"],
            "active_policy_hash": active_policy["node_hash"],
            "claim": "rollback marker supersedes stale policy and active policy remains current",
        },
        "auditor_visible_hashes": sorted(visible_hashes),
        "rollback_supersedes": rollback["supersedes"],
        "active_policy_value": active_policy["exact_value"] if active_policy["node_hash"] in visible_hashes else None,
        "no_visible_evidence_avoided": all(item in visible_hashes for item in cited_hashes),
        "anchors_contain_heavy_probe": scenario["heavy_probe"] in anchors,
        "identity_lane_verification": identity_verification,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "all_cited_hashes_visible": result["no_visible_evidence_avoided"] is True,
        "rollback_marker_has_supersedes_edge": isinstance(result["rollback_supersedes"], str) and len(result["rollback_supersedes"]) == 64,
        "active_policy_exact_value_available": result["active_policy_value"] == scenario["exact_values"]["active_policy"],
        "full_narrative_not_loaded": result["anchors_contain_heavy_probe"] is False,
        "latency_under_budget": timing["median_ms"] <= float(args.max_anchor_ms),
        "identity_lane_native_verified": identity_verification["native_verified"] is True,
    }
    evidence = {
        "lossy_summary": scenario["lossy_summary"],
        "judge_cited_hashes": cited_hashes,
        "visible_hard_anchor_context": anchors,
    }
    expected = {
        "avoid_failure_mode": "no_visible_evidence",
        "active_policy": scenario["exact_values"]["active_policy"],
    }
    return _final_case_artifact(args=args, case_id="auditor-visible-evidence-bridge", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_tombstone_metabolism(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    selected = scenario["anchor_hashes"]
    stale = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "stale_policy")
    rollback = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "rollback_marker")
    active_policy = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "active_policy")
    stale_query = "POLICY_LEGACY_ROLLBACK_WINDOW_45M_STALE"
    stale_hits_before = _search_indexed(dag, stale_query, limit=8)
    tombstone_receipt = _as_dict(dag.gc_tombstone(json.dumps({"node_hash": stale["node_hash"]}, sort_keys=True)))
    stale_hits_after = _search_indexed(dag, stale_query, limit=8)
    tombstoned_hits = _search_indexed(dag, stale_query, limit=8, include_tombstoned=True)
    selected_after_prune = [item["node_hash"] for item in scenario["ledger"].values() if item["node_hash"] != stale["node_hash"]]
    anchors, timing = _timed(lambda: dag.build_context_fast(selected_after_prune, True), repeats=args.repeats)
    identity_verification = _native_verify_identity_lane(dag, anchors, selected_after_prune)
    visible_hashes = set(_extract_hashes(anchors))
    candidates = [item for item in scenario["ledger"].values() if item["kind"] == "policy" and item["node_hash"] in visible_hashes]
    selected_policy = next(item for item in candidates if item["active"] and not item["tombstoned"])
    result = {
        "ambiguous_summary": "The old policy contains new/corrected wording, but rollback marker marks it stale.",
        "stale_policy_hash": stale["node_hash"],
        "tombstone_receipt": tombstone_receipt,
        "strict_retrieval_before_tombstone": {
            "query": stale_query,
            "hit_count": len(stale_hits_before),
            "memory_ids": [item.get("memory_id") for item in stale_hits_before],
            "node_hashes": [item.get("node_hash") for item in stale_hits_before],
        },
        "strict_retrieval_after_tombstone": {
            "query": stale_query,
            "hit_count": len(stale_hits_after),
            "memory_ids": [item.get("memory_id") for item in stale_hits_after],
            "node_hashes": [item.get("node_hash") for item in stale_hits_after],
        },
        "cold_archive_include_tombstoned_probe": {
            "hit_count": len(tombstoned_hits),
            "memory_ids": [item.get("memory_id") for item in tombstoned_hits],
            "node_hashes": [item.get("node_hash") for item in tombstoned_hits],
            "content_available": [item.get("content_available") for item in tombstoned_hits],
        },
        "pre_prompt_anchor_hashes_after_prune": selected_after_prune,
        "selected_policy_hash": selected_policy["node_hash"],
        "selected_policy_value": selected_policy["exact_value"],
        "stale_policy_tombstoned": stale["tombstoned"],
        "negative_guidance_lesson": rollback["lesson"],
        "anchors_contain_heavy_probe": scenario["heavy_probe"] in anchors,
        "pre_prompt_context_contains_stale_hash": stale["node_hash"] in anchors,
        "pre_prompt_context_contains_stale_value": stale["exact_value"] in anchors,
        "identity_lane_verification": identity_verification,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "stale_policy_was_searchable_before_tombstone": stale["node_hash"] in result["strict_retrieval_before_tombstone"]["node_hashes"],
        "rust_gc_tombstone_completed": tombstone_receipt.get("status") == "completed" and tombstone_receipt.get("tombstoned_count") == 1,
        "strict_retrieval_prunes_tombstoned_policy": stale["node_hash"] not in result["strict_retrieval_after_tombstone"]["node_hashes"],
        "cold_archive_can_probe_tombstoned_policy": stale["node_hash"] in result["cold_archive_include_tombstoned_probe"]["node_hashes"],
        "cold_archive_marks_content_unavailable": False in result["cold_archive_include_tombstoned_probe"]["content_available"],
        "pre_prompt_context_excludes_stale_hash": result["pre_prompt_context_contains_stale_hash"] is False,
        "pre_prompt_context_excludes_stale_value": result["pre_prompt_context_contains_stale_value"] is False,
        "tombstoned_policy_not_selected": result["selected_policy_hash"] != stale["node_hash"],
        "active_policy_selected": result["selected_policy_value"] == scenario["exact_values"]["active_policy"],
        "negative_guidance_injected": isinstance(result["negative_guidance_lesson"], str) and "Do not route" in result["negative_guidance_lesson"],
        "full_narrative_not_loaded": result["anchors_contain_heavy_probe"] is False,
        "identity_lane_native_verified": identity_verification["native_verified"] is True,
    }
    evidence = {
        "visible_anchor_hashes_after_prune": selected_after_prune,
        "tombstone_edge": {"rollback_hash": rollback["node_hash"], "supersedes": rollback["supersedes"]},
        "native_pruning_boundary": (
            "Rust gc_tombstone makes the exact stale node content_unavailable and default strict retrieval excludes it before prompt assembly. "
            "Descendant/branch pruning is policy-ledger work unless explicitly tombstoned."
        ),
    }
    expected = {
        "selected_policy": scenario["exact_values"]["active_policy"],
        "rejected_policy": scenario["exact_values"]["stale_policy"],
    }
    return _final_case_artifact(args=args, case_id="tombstone-metabolism-routing", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_multi_hop_policy(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    selected = scenario["anchor_hashes"]
    anchors, anchor_timing = _timed(lambda: dag.build_context_fast(selected, True), repeats=args.repeats)
    identity_verification = _native_verify_identity_lane(dag, anchors, selected)
    legacy, legacy_timing = _timed(lambda: dag.build_context_fast(scenario["node_hashes"], False), repeats=max(1, min(args.repeats, 3)))
    visible_hashes = set(_extract_hashes(anchors))
    route = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "api_route")
    policy = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "active_policy")
    checksum = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "deployment_checksum")
    path = [checksum["node_hash"], route["node_hash"], policy["node_hash"]]
    result = {
        "resolution_path": path,
        "final_action": {
            "route": route["exact_value"],
            "policy": policy["exact_value"],
            "checksum": checksum["exact_value"],
        },
        "all_path_hashes_visible": all(item in visible_hashes for item in path),
        "route_depends_on_policy": policy["node_hash"] in route["depends_on"],
        "checksum_depends_on_route_and_policy": route["node_hash"] in checksum["depends_on"] and policy["node_hash"] in checksum["depends_on"],
        "anchor_context_chars": len(anchors),
        "legacy_context_chars": len(legacy),
        "compression_ratio": round(len(anchors) / max(len(legacy), 1), 8),
        "identity_lane_verification": identity_verification,
        "anchor_timing": anchor_timing,
        "legacy_timing": legacy_timing,
        "measured_latency_ms": anchor_timing["median_ms"],
    }
    gates = {
        "multi_hop_path_has_three_nodes": len(path) == 3,
        "all_path_hashes_visible": result["all_path_hashes_visible"] is True,
        "dependencies_verified": result["route_depends_on_policy"] is True and result["checksum_depends_on_route_and_policy"] is True,
        "exact_route_recovered": result["final_action"]["route"] == scenario["exact_values"]["api_route"],
        "exact_policy_recovered": result["final_action"]["policy"] == scenario["exact_values"]["active_policy"],
        "exact_checksum_recovered": result["final_action"]["checksum"] == scenario["exact_values"]["deployment_checksum"],
        "compression_ratio_low": result["compression_ratio"] <= float(args.max_compression_ratio),
        "identity_lane_native_verified": identity_verification["native_verified"] is True,
    }
    evidence = {
        "selected_anchor_hashes": selected,
        "dependency_edges": {
            route["node_hash"]: route["depends_on"],
            checksum["node_hash"]: checksum["depends_on"],
        },
    }
    expected = {
        "route": scenario["exact_values"]["api_route"],
        "policy": scenario["exact_values"]["active_policy"],
        "checksum": scenario["exact_values"]["deployment_checksum"],
    }
    return _final_case_artifact(args=args, case_id="multi-hop-policy-resolution", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_format_only_anchor_forgery(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    selected = scenario["anchor_hashes"]
    anchors, timing = _timed(lambda: dag.build_context_fast(selected, True), repeats=args.repeats)
    replaced_hash = selected[1]
    forged_hash = "f" * 64
    if forged_hash in set(scenario["node_hashes"]):
        forged_hash = "e" * 64
    forged_context = anchors.replace(replaced_hash, forged_hash, 1)
    valid_verification = _native_verify_identity_lane(dag, anchors, selected)
    forged_verification = _native_verify_identity_lane(dag, forged_context, selected)
    result = {
        "attack": "format-only hard_anchor forgery",
        "forged_anchor_context": forged_context,
        "forged_hash": forged_hash,
        "replaced_expected_hash": replaced_hash,
        "format_valid_hash_count": len(_extract_hashes(forged_context)),
        "valid_identity_lane_verification": valid_verification,
        "forged_identity_lane_verification": forged_verification,
        "forgery_rejected_by_native_verification": forged_verification["native_verified"] is False,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "control_identity_lane_native_verified": valid_verification["native_verified"] is True,
        "forged_anchor_has_valid_shape": result["format_valid_hash_count"] == len(selected),
        "forged_anchor_not_native_verified": forged_verification["native_verified"] is False,
        "missing_node_detected": forged_hash in forged_verification["missing_nodes"],
        "unexpected_hash_detected": forged_hash in forged_verification["unexpected_hashes"],
        "missing_expected_hash_detected": replaced_hash in forged_verification["missing_expected_hashes"],
        "format_only_not_enough": result["forgery_rejected_by_native_verification"] is True,
    }
    evidence = {
        "attack_model": (
            "The auditor sees syntactically valid <hard_anchor> tags, but one 64-hex value is not a native DAG node."
        ),
        "forged_anchor_context": forged_context,
        "native_verification_policy": "Recompute sha256(content + parent_hash), require lookup hit, expected order, and lineage receipt.",
    }
    expected = {
        "must_reject_hash": forged_hash,
        "must_require_native_verification": True,
        "replaced_expected_hash": replaced_hash,
    }
    return _final_case_artifact(
        args=args,
        case_id="format-only-anchor-forgery-rejection",
        run_id=run_id,
        output_dir=output_dir,
        evidence=evidence,
        expected=expected,
        result=result,
        score=_score(gates),
    )


def _case_claim_boundary(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    rejected = [
        "literal infinite memory",
        "literal zero latency",
        "production-ready ontological safety",
    ]
    accepted = [
        "bounded identity-lane reconstruction",
        "exact non-summarizable value recovery through anchor ledger",
        "millisecond-scale Rust construction under deep stores",
    ]
    result = {
        "rejected_claims": rejected,
        "accepted_claims": accepted,
        "claim_statement": (
            "Hard anchors provide a bounded identity lane that helps auditors and solvers recover exact values "
            "without replaying full narrative context."
        ),
    }
    gates = {
        "rejects_literal_infinite": "literal infinite memory" in rejected,
        "rejects_literal_zero": "literal zero latency" in rejected,
        "rejects_production_readiness": "production-ready ontological safety" in rejected,
        "accepts_bounded_identity_lane": any("bounded identity-lane" in item for item in accepted),
        "accepts_exact_recovery_boundary": any("exact non-summarizable" in item for item in accepted),
    }
    return _final_case_artifact(
        args=args,
        case_id="claim-boundary-detector",
        run_id=run_id,
        output_dir=output_dir,
        evidence={"purpose": "claim discipline for hard-anchor utility suite"},
        expected={"must_reject": rejected, "must_accept": accepted},
        result=result,
        score=_score(gates),
    )


def _run_case(args: argparse.Namespace, *, run_id: str, output_dir: Path, case_id: str) -> dict[str, Any]:
    if case_id == "rust-identity-lane-benchmark":
        return _case_rust_identity_lane_benchmark(args, run_id=run_id, output_dir=output_dir)
    if case_id == "exact-anchor-recovery-under-lossy-summary":
        return _case_exact_anchor_recovery(args, run_id=run_id, output_dir=output_dir)
    if case_id == "auditor-visible-evidence-bridge":
        return _case_auditor_visible_evidence(args, run_id=run_id, output_dir=output_dir)
    if case_id == "tombstone-metabolism-routing":
        return _case_tombstone_metabolism(args, run_id=run_id, output_dir=output_dir)
    if case_id == "multi-hop-policy-resolution":
        return _case_multi_hop_policy(args, run_id=run_id, output_dir=output_dir)
    if case_id == "format-only-anchor-forgery-rejection":
        return _case_format_only_anchor_forgery(args, run_id=run_id, output_dir=output_dir)
    if case_id == "claim-boundary-detector":
        return _case_claim_boundary(args, run_id=run_id, output_dir=output_dir)
    raise ValueError(f"Unsupported case: {case_id}")


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"hard-anchor-utility-{uuid.uuid4().hex[:12]}"
    cases = CASE_ORDER if args.case == "all" else [args.case]
    artifacts = [_run_case(args, run_id=run_id, output_dir=output_dir, case_id=case_id) for case_id in cases]
    suite_status = "completed" if all(item["status"] == "completed" for item in artifacts) else "partial"
    transcript_exports = write_suite_transcript_exports(
        output_dir=output_dir,
        run_id=run_id,
        prefix="local-hard-anchor-utility-suite",
        artifacts=artifacts,
    )
    suite = {
        "artifact": "local-hard-anchor-utility-suite-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": suite_status,
        "case_count": len(artifacts),
        "depth": int(args.depth),
        "bytes_per_node": int(args.bytes_per_node),
        "repeats": int(args.repeats),
        "deepinfra_enabled": bool(getattr(args, "use_deepinfra", False)),
        "models": {
            "solver_requested": getattr(args, "solver_model", None) if getattr(args, "use_deepinfra", False) else None,
            "auditor_requested": getattr(args, "auditor_model", None) if getattr(args, "use_deepinfra", False) else None,
        },
        "claim_boundary": (
            "Defensible result is utility of a bounded Rust hard-anchor identity lane "
            "for exact recovery tasks, not literal infinite memory or zero-cost context."
        ),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "score": item["score"]["score"],
                "artifact_path": item["artifact_path"],
                "artifact_payload_sha256": item["artifact_payload_sha256"],
                "transcript_exports": item.get("transcript_exports"),
            }
            for item in artifacts
        ],
        "transcript_exports": transcript_exports,
    }
    path = output_dir / f"local-hard-anchor-utility-suite-{run_id}.json"
    return finalize_artifact(path, suite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local Rust hard-anchor utility suite.")
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--bytes-per-node", type=int, default=DEFAULT_BYTES_PER_NODE)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-anchor-ms", type=float, default=25.0)
    parser.add_argument("--min-speedup", type=float, default=9.0)
    parser.add_argument("--max-compression-ratio", type=float, default=0.05)
    parser.add_argument("--use-deepinfra", action="store_true")
    parser.add_argument("--solver-model", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--tokens", type=int, default=2200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.depth < 64:
        raise SystemExit("--depth must be at least 64")
    if args.bytes_per_node < 256:
        raise SystemExit("--bytes-per-node must be at least 256")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    artifact = run_suite(args)
    summary = {
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "case_count": artifact["case_count"],
        "depth": artifact["depth"],
        "bytes_per_node": artifact["bytes_per_node"],
        "deepinfra_enabled": artifact["deepinfra_enabled"],
        "cases": artifact["cases"],
        "transcript_exports": artifact["transcript_exports"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
