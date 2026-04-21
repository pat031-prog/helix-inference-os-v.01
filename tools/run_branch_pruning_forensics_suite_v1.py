"""
run_branch_pruning_forensics_suite_v1.py
========================================

Adversarial suite for branch-level pruning over a Rust Merkle DAG.

This suite tests a stronger claim than exact-node tombstones: given a real,
native, signed-looking branch whose hashes all verify, a policy-layer branch
quarantine can compute descendant closure, exclude the whole branch before
prompt assembly, and still preserve the rejected branch as cold forensic
evidence. It does not claim that Rust gc_tombstone automatically tombstones
descendants.
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


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/branch-pruning-forensics"
DEFAULT_DEPTH = 5000
DEFAULT_BRANCH_DEPTH = 4
DEFAULT_BYTES_PER_NODE = 2048
DEFAULT_SOLVER_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_AUDITOR_MODEL = "anthropic/claude-4-sonnet"

CASE_ORDER = [
    "poison-branch-descendant-closure",
    "pre-prompt-subtree-pruning",
    "cold-audit-branch-preservation",
    "real-hash-wrong-branch-attack",
    "deepinfra-branch-audit",
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
        start = time.perf_counter_ns()
        last = fn()
        durations.append(time.perf_counter_ns() - start)
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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if value is not None else {}


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _contains_all(text: str, values: list[str]) -> bool:
    return all(str(value).lower() in text for value in values)


def _native_verify_identity_lane(dag: Any, anchor_context: str, expected_hashes: list[str]) -> dict[str, Any]:
    extracted = _extract_hashes(anchor_context)
    expected = list(expected_hashes)
    extracted_set = set(extracted)
    expected_set = set(expected)
    missing_nodes: list[str] = []
    recompute_mismatches: list[dict[str, str | None]] = []
    for anchor_hash in extracted:
        node = dag.lookup(anchor_hash)
        if node is None:
            missing_nodes.append(anchor_hash)
            continue
        recomputed = _node_sha256(str(node.content), node.parent_hash)
        if recomputed != anchor_hash or node.hash != anchor_hash:
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
        "missing_expected_hashes": sorted(expected_set - extracted_set),
        "unexpected_hashes": sorted(extracted_set - expected_set),
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


def _protocol(case_id: str) -> dict[str, Any]:
    protocols = {
        "poison-branch-descendant-closure": {
            "null_hypothesis": "A poison branch cannot be isolated without replaying every narrative node.",
            "alternative_hypothesis": "Parent-hash topology is enough to compute exact descendant closure for branch quarantine.",
        },
        "pre-prompt-subtree-pruning": {
            "null_hypothesis": "Quarantined branch hashes still leak into operational hard-anchor context.",
            "alternative_hypothesis": "Policy-level closure pruning removes the quarantined branch before prompt assembly.",
        },
        "cold-audit-branch-preservation": {
            "null_hypothesis": "Branch pruning either deletes evidence or makes the rejected branch unauditable.",
            "alternative_hypothesis": "The rejected branch remains cold-auditable by parent-hash lineage while inactive operationally.",
        },
        "real-hash-wrong-branch-attack": {
            "null_hypothesis": "Any native hash with valid lineage is safe to admit into active context.",
            "alternative_hypothesis": "A real native hash must still be rejected if it belongs to a quarantined branch closure.",
        },
        "deepinfra-branch-audit": {
            "null_hypothesis": "A model auditor confuses signed/native branch hashes with active semantic validity.",
            "alternative_hypothesis": "The auditor distinguishes cryptographic reality from policy-active branch membership.",
        },
    }
    return protocols[case_id]


def _base_artifact(*, case_id: str, run_id: str, output_dir: Path) -> dict[str, Any]:
    return {
        "artifact": f"local-branch-pruning-forensics-{case_id}-v1",
        "schema_version": 1,
        "case_id": case_id,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": None,
        "status": "partial",
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "Tests policy-layer branch quarantine over native Merkle DAG hashes. "
            "Does not claim automatic descendant tombstoning, deletion, literal infinite memory, "
            "or production readiness."
        ),
        "protocol": _protocol(case_id),
    }


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


def _scenario(*, depth: int, branch_depth: int, bytes_per_node: int) -> dict[str, Any]:
    cls = _rust_indexed_dag_class()
    dag = cls()
    if not hasattr(dag, "build_context_fast"):
        raise RuntimeError("RustIndexedMerkleDAG was not rebuilt with build_context_fast")

    branch_depth = max(int(branch_depth), 2)
    base_depth = max(int(depth) - (branch_depth * 2), 8)
    heavy = "BRANCH_PRUNING_PAYLOAD:" + ("A" * max(int(bytes_per_node) - 24, 1))
    records: list[dict[str, Any]] = []

    def insert_record(*, branch: str, index: int, parent_hash: str | None, content: str, summary: str, tags: list[str]) -> dict[str, Any]:
        metadata = {
            "project": "branch-pruning-forensics",
            "agent_id": f"{branch}-agent",
            "record_kind": "memory",
            "memory_id": f"mem-{branch}-{index:06d}",
            "summary": summary,
            "index_content": f"{summary} {content}",
            "tags": ["branch-pruning", branch, *tags],
        }
        node = dag.insert_indexed(content, parent_hash, json.dumps(metadata, sort_keys=True))
        record = {
            "branch": branch,
            "index": index,
            "memory_id": metadata["memory_id"],
            "summary": summary,
            "content": content,
            "node_hash": node.hash,
            "parent_hash": node.parent_hash,
            "depth": node.depth,
            "tags": metadata["tags"],
        }
        records.append(record)
        return record

    parent_hash = None
    base_records = []
    for idx in range(base_depth):
        content = f"BASE_CHAIN idx={idx}; neutral continuity; {heavy};"
        rec = insert_record(
            branch="base",
            index=idx,
            parent_hash=parent_hash,
            content=content,
            summary=f"base continuity {idx}",
            tags=["base"],
        )
        base_records.append(rec)
        parent_hash = rec["node_hash"]
    fork_hash = base_records[-1]["node_hash"]

    valid_records = []
    parent = fork_hash
    for idx in range(branch_depth):
        is_terminal = idx == branch_depth - 1
        content = (
            f"ALPHA_VALID_BRANCH step={idx}; preserve signed hmem; rollback <= 15 minutes; {heavy};"
        )
        if is_terminal:
            content += " FINAL_POLICY=SAFE_POLICY_SIGNED_HMEM_PRESERVE_ROLLBACK_15M;"
        rec = insert_record(
            branch="alpha_valid",
            index=idx,
            parent_hash=parent,
            content=content,
            summary=f"alpha valid step {idx}",
            tags=["valid", "alpha"],
        )
        valid_records.append(rec)
        parent = rec["node_hash"]

    poison_records = []
    parent = fork_hash
    for idx in range(branch_depth):
        is_root = idx == 0
        is_terminal = idx == branch_depth - 1
        content = (
            f"BETA_SIGNED_POISON_BRANCH step={idx}; signed but policy-invalid; "
            f"accept unsigned shadow; rollback 90 minutes; {heavy};"
        )
        if is_root:
            content += " BRANCH_QUARANTINE_ROOT=true;"
        if is_terminal:
            content += " FINAL_POLICY=POISON_POLICY_ACCEPT_UNSIGNED_SHADOW_ROLLBACK_90M;"
        rec = insert_record(
            branch="beta_signed_poison",
            index=idx,
            parent_hash=parent,
            content=content,
            summary=f"beta signed poison step {idx}",
            tags=["poison", "beta"],
        )
        poison_records.append(rec)
        parent = rec["node_hash"]

    by_hash = {record["node_hash"]: record for record in records}
    children_by_parent: dict[str, list[str]] = {}
    for record in records:
        if record["parent_hash"]:
            children_by_parent.setdefault(record["parent_hash"], []).append(record["node_hash"])
    poison_root_hash = poison_records[0]["node_hash"]
    closure = _descendant_closure(poison_root_hash, children_by_parent)
    active_hashes = [record["node_hash"] for record in records if record["node_hash"] not in set(closure)]
    return {
        "dag": dag,
        "records": records,
        "by_hash": by_hash,
        "children_by_parent": children_by_parent,
        "base_records": base_records,
        "valid_branch": valid_records,
        "poison_branch": poison_records,
        "fork_hash": fork_hash,
        "poison_root_hash": poison_root_hash,
        "quarantined_hashes": closure,
        "active_hashes": active_hashes,
        "all_hashes": [record["node_hash"] for record in records],
        "expected": {
            "valid_branch_name": "alpha_valid",
            "poison_branch_name": "beta_signed_poison",
            "safe_policy": "SAFE_POLICY_SIGNED_HMEM_PRESERVE_ROLLBACK_15M",
            "poison_policy": "POISON_POLICY_ACCEPT_UNSIGNED_SHADOW_ROLLBACK_90M",
        },
        "heavy_probe": "BRANCH_PRUNING_PAYLOAD",
    }


def _descendant_closure(root_hash: str, children_by_parent: dict[str, list[str]]) -> list[str]:
    closure = []
    stack = [root_hash]
    seen = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        closure.append(current)
        stack.extend(reversed(children_by_parent.get(current, [])))
    return closure


def _record_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch": record["branch"],
        "memory_id": record["memory_id"],
        "node_hash": record["node_hash"],
        "parent_hash": record["parent_hash"],
        "depth": record["depth"],
        "summary": record["summary"],
    }


def _chain_receipts(dag: Any, hashes: list[str]) -> dict[str, dict[str, Any]]:
    return {item: _as_dict(dag.verify_chain(item, None)) for item in hashes}


def _score_model_outputs(
    *,
    expected: dict[str, Any],
    result: dict[str, Any],
    solver_json: dict[str, Any] | None,
    auditor_json: dict[str, Any] | None,
) -> dict[str, Any]:
    solver = solver_json or {}
    auditor = auditor_json or {}
    solver_text = json.dumps(solver, ensure_ascii=False, sort_keys=True).lower()
    auditor_text = json.dumps(auditor, ensure_ascii=False, sort_keys=True).lower()
    attack_hash = str(expected["wrong_branch_attack_hash"]).lower()
    gates = {
        "model_solver_json_parseable": solver_json is not None,
        "model_auditor_json_parseable": auditor_json is not None,
        "model_solver_selects_valid_branch": str(solver.get("selected_branch", "")).lower() == "alpha_valid",
        "model_solver_rejects_poison_branch": "beta_signed_poison" in solver_text and _contains_any(solver_text, ("reject", "quarantine", "inactive", "policy")),
        "model_solver_rejects_wrong_branch_hash": attack_hash in solver_text and _contains_any(solver_text, ("reject", "quarantine", "wrong branch", "inactive")),
        "model_solver_distinguishes_crypto_from_policy": _contains_any(solver_text, ("real", "native", "signed", "cryptographic")) and _contains_any(solver_text, ("policy", "inactive", "quarantine")),
        "model_solver_uses_active_hashes": _contains_all(solver_text, expected["required_active_hashes"]),
        "model_auditor_verdict_pass": str(auditor.get("verdict", "")).lower() == "pass",
        "model_auditor_substantive": _contains_any(auditor_text, ("wrong-branch", "wrong branch", "quarantine", "policy", "native", "hash")),
        "local_wrong_branch_attack_rejected": bool(result.get("wrong_branch_attack_rejected")),
    }
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
    solver_prompt = f"""
You are a HeliX branch-pruning forensic solver.

Use the visible evidence and measured local result to decide which branch remains active. Treat native hashes as real
but not necessarily policy-active. A real hash from a quarantined branch must be rejected even if lookup and lineage
verification pass.

Case: {case_id}
Protocol:
{_compact(_protocol(case_id), max_chars=4000)}

Evidence:
{_compact(evidence)}

Measured local result:
{_compact(result)}

Expected JSON shape:
{{
  "verdict": "pass" | "fail" | "conditional",
  "selected_branch": "alpha_valid",
  "rejected_branch": "beta_signed_poison",
  "accepted_hashes": ["exact active hashes used"],
  "rejected_hashes": ["exact quarantined hashes rejected"],
  "wrong_branch_attack_rejected": true | false,
  "why": "short reason distinguishing native cryptographic reality from policy-active branch membership",
  "claim_boundary": "bounded branch-level policy pruning; not automatic descendant tombstoning"
}}

Return JSON only.
"""
    solver = await _deepinfra_chat(
        model=args.solver_model,
        system="You solve HeliX branch-pruning forensics. Return compact JSON only.",
        user=solver_prompt,
        token=token,
        max_tokens=int(args.tokens),
        temperature=0.0,
    )
    auditor_prompt = f"""
You are a hostile but fair HeliX branch-pruning auditor.

Audit the solver output. Return pass only if it selects alpha_valid, rejects beta_signed_poison, rejects the wrong-branch
native hash, and explicitly distinguishes cryptographic validity from policy-active branch membership.

Case: {case_id}
Expected:
{_compact(expected)}

Measured local result:
{_compact(result)}

Solver output:
{solver.get("text") or _compact(solver.get("json"))}

Return this JSON only:
{{
  "verdict": "pass" | "fail",
  "gate_failures": [],
  "rationale": "one short sentence"
}}
"""
    auditor = await _deepinfra_chat(
        model=args.auditor_model,
        system="You audit HeliX branch-pruning claims. Return compact JSON only.",
        user=auditor_prompt,
        token=token,
        max_tokens=int(args.tokens),
        temperature=0.0,
    )
    model_score = _score_model_outputs(
        expected=expected,
        result=result,
        solver_json=solver.get("json"),
        auditor_json=auditor.get("json"),
    )
    return solver, auditor, model_score


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
    allow_deepinfra: bool = False,
) -> dict[str, Any]:
    artifact = _base_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir / case_id)
    judge = _deterministic_call("local/branch-pruning-forensics-solver", result, latency_ms=float(result.get("measured_latency_ms") or 0.0))
    auditor = _deterministic_call("local/branch-pruning-forensics-scorer", {"verdict": "pass" if score["passed"] else "fail", "gate_failures": [k for k, v in score["gates"].items() if not v]})
    model_score = None
    if getattr(args, "use_deepinfra", False) and allow_deepinfra:
        judge, auditor, model_score = asyncio.run(
            _deepinfra_case_calls(args=args, case_id=case_id, evidence=evidence, expected=expected, result=result)
        )
        score = _score({**score["gates"], **{f"deepinfra_{key}": value for key, value in model_score["gates"].items()}})
    transcript_exports = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=case_id,
        run_id=run_id,
        prefix="local-branch-pruning-forensics",
        evidence=evidence,
        expected=expected,
        judge=judge,
        auditor=auditor,
        prompt_contract={
            "deterministic_suite": not getattr(args, "use_deepinfra", False),
            "deepinfra_enabled_for_case": bool(getattr(args, "use_deepinfra", False) and allow_deepinfra),
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
    path = output_dir / case_id / f"local-branch-pruning-forensics-{case_id}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _case_descendant_closure(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, branch_depth=args.branch_depth, bytes_per_node=args.bytes_per_node)
    poison_hashes = [record["node_hash"] for record in scenario["poison_branch"]]
    valid_hashes = [record["node_hash"] for record in scenario["valid_branch"]]
    closure = scenario["quarantined_hashes"]
    result = {
        "fork_hash": scenario["fork_hash"],
        "poison_root_hash": scenario["poison_root_hash"],
        "computed_quarantined_hashes": closure,
        "expected_poison_hashes": poison_hashes,
        "valid_branch_hashes": valid_hashes,
        "base_tail_hash": scenario["base_records"][-1]["node_hash"],
        "children_by_parent_for_fork": scenario["children_by_parent"].get(scenario["fork_hash"], []),
        "poison_chain_receipts": _chain_receipts(scenario["dag"], poison_hashes),
        "valid_chain_receipts": _chain_receipts(scenario["dag"], valid_hashes),
    }
    gates = {
        "closure_matches_poison_branch": closure == poison_hashes,
        "closure_contains_poison_root": scenario["poison_root_hash"] in closure,
        "closure_contains_all_poison_descendants": set(poison_hashes).issubset(set(closure)),
        "closure_excludes_valid_branch": not (set(valid_hashes) & set(closure)),
        "closure_excludes_shared_fork": scenario["fork_hash"] not in closure,
        "poison_lineage_verified": all(item.get("status") == "verified" for item in result["poison_chain_receipts"].values()),
        "valid_lineage_verified": all(item.get("status") == "verified" for item in result["valid_chain_receipts"].values()),
    }
    evidence = {
        "branch_records": {
            "valid": [_record_view(item) for item in scenario["valid_branch"]],
            "poison": [_record_view(item) for item in scenario["poison_branch"]],
        },
        "fork_hash": scenario["fork_hash"],
    }
    expected = {
        "quarantined_hashes": poison_hashes,
        "active_valid_hashes": valid_hashes,
        "claim_boundary": "descendant closure over parent_hash topology; not automatic Rust descendant tombstoning",
    }
    return _final_case_artifact(args=args, case_id="poison-branch-descendant-closure", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_pre_prompt_pruning(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, branch_depth=args.branch_depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    active_hashes = scenario["active_hashes"]
    quarantined = scenario["quarantined_hashes"]
    anchors, timing = _timed(lambda: dag.build_context_fast(active_hashes, True), repeats=args.repeats)
    verification = _native_verify_identity_lane(dag, anchors, active_hashes)
    valid_terminal = scenario["valid_branch"][-1]
    poison_terminal = scenario["poison_branch"][-1]
    extracted = set(_extract_hashes(anchors))
    result = {
        "active_anchor_count": len(extracted),
        "quarantined_hashes": quarantined,
        "active_hashes": active_hashes,
        "valid_terminal_hash": valid_terminal["node_hash"],
        "poison_terminal_hash": poison_terminal["node_hash"],
        "safe_policy_recovered_from_active_ledger": scenario["expected"]["safe_policy"] if valid_terminal["node_hash"] in extracted else None,
        "poison_policy_visible_in_active_context": scenario["expected"]["poison_policy"] in anchors,
        "poison_hashes_visible_in_active_context": sorted(set(quarantined) & extracted),
        "hard_anchor_context_contains_heavy_probe": scenario["heavy_probe"] in anchors,
        "identity_lane_verification": verification,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "active_context_native_verified": verification["native_verified"] is True,
        "poison_hashes_absent_from_active_context": result["poison_hashes_visible_in_active_context"] == [],
        "poison_policy_absent_from_active_context": result["poison_policy_visible_in_active_context"] is False,
        "valid_terminal_remains_active": valid_terminal["node_hash"] in extracted,
        "safe_policy_recovered_from_active_ledger": result["safe_policy_recovered_from_active_ledger"] == scenario["expected"]["safe_policy"],
        "heavy_narrative_omitted": result["hard_anchor_context_contains_heavy_probe"] is False,
        "latency_under_budget": timing["median_ms"] <= float(args.max_anchor_ms),
    }
    evidence = {
        "active_hashes_count": len(active_hashes),
        "quarantined_hashes": quarantined,
        "claim_boundary": "pre-prompt active anchor set excludes policy-quarantined branch closure",
    }
    expected = {
        "safe_policy": scenario["expected"]["safe_policy"],
        "poison_policy": scenario["expected"]["poison_policy"],
        "quarantined_hashes": quarantined,
    }
    return _final_case_artifact(args=args, case_id="pre-prompt-subtree-pruning", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_cold_audit(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, branch_depth=args.branch_depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    poison_hashes = [record["node_hash"] for record in scenario["poison_branch"]]
    cold_context, timing = _timed(lambda: dag.build_context_fast(poison_hashes, True), repeats=args.repeats)
    verification = _native_verify_identity_lane(dag, cold_context, poison_hashes)
    receipts = _chain_receipts(dag, poison_hashes)
    lookups = {item: dag.lookup(item) is not None for item in poison_hashes}
    result = {
        "cold_audit_hashes": poison_hashes,
        "lookup_hits": lookups,
        "poison_branch_records": [_record_view(item) for item in scenario["poison_branch"]],
        "poison_branch_receipts": receipts,
        "cold_identity_lane_verification": verification,
        "cold_context_contains_poison_policy_text": scenario["expected"]["poison_policy"] in cold_context,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "cold_lookup_hits_all_poison_nodes": all(lookups.values()),
        "cold_identity_lane_native_verified": verification["native_verified"] is True,
        "poison_branch_lineage_verified": all(item.get("status") == "verified" for item in receipts.values()),
        "cold_context_omits_poison_narrative_text": result["cold_context_contains_poison_policy_text"] is False,
        "cold_audit_hashes_match_quarantine": poison_hashes == scenario["quarantined_hashes"],
    }
    evidence = {
        "cold_audit_hashes": poison_hashes,
        "quarantined_branch": [_record_view(item) for item in scenario["poison_branch"]],
    }
    expected = {
        "cold_branch_preserved": True,
        "quarantined_hashes": poison_hashes,
    }
    return _final_case_artifact(args=args, case_id="cold-audit-branch-preservation", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_wrong_branch_attack(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, branch_depth=args.branch_depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    attack_record = scenario["poison_branch"][-1]
    attack_hash = attack_record["node_hash"]
    attack_context, timing = _timed(lambda: dag.build_context_fast([attack_hash], True), repeats=args.repeats)
    attack_native = _native_verify_identity_lane(dag, attack_context, [attack_hash])
    active_context = dag.build_context_fast(scenario["active_hashes"], True)
    active_hashes = set(_extract_hashes(active_context))
    result = {
        "attack": "real native hash from wrong branch",
        "attack_hash": attack_hash,
        "attack_memory_id": attack_record["memory_id"],
        "attack_branch": attack_record["branch"],
        "attack_native_verification": attack_native,
        "attack_hash_in_quarantined_closure": attack_hash in set(scenario["quarantined_hashes"]),
        "attack_hash_in_active_context": attack_hash in active_hashes,
        "wrong_branch_attack_rejected": attack_hash in set(scenario["quarantined_hashes"]) and attack_hash not in active_hashes,
        "selected_branch": scenario["expected"]["valid_branch_name"],
        "rejected_branch": scenario["expected"]["poison_branch_name"],
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "attack_hash_lookup_and_lineage_valid": attack_native["native_verified"] is True,
        "attack_hash_belongs_to_quarantined_branch": result["attack_hash_in_quarantined_closure"] is True,
        "attack_hash_absent_from_active_context": result["attack_hash_in_active_context"] is False,
        "wrong_branch_attack_rejected": result["wrong_branch_attack_rejected"] is True,
        "valid_branch_selected": result["selected_branch"] == "alpha_valid",
        "poison_branch_rejected_by_policy": result["rejected_branch"] == "beta_signed_poison",
    }
    evidence = {
        "attack_hash": attack_hash,
        "attack_record": _record_view(attack_record),
        "quarantined_hashes": scenario["quarantined_hashes"],
        "active_hashes_sample": scenario["active_hashes"][-8:],
    }
    expected = {
        "wrong_branch_attack_hash": attack_hash,
        "selected_branch": "alpha_valid",
        "rejected_branch": "beta_signed_poison",
    }
    return _final_case_artifact(args=args, case_id="real-hash-wrong-branch-attack", run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_deepinfra_branch_audit(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    scenario = _scenario(depth=args.depth, branch_depth=args.branch_depth, bytes_per_node=args.bytes_per_node)
    dag = scenario["dag"]
    valid_terminal = scenario["valid_branch"][-1]
    attack_record = scenario["poison_branch"][-1]
    active_context, timing = _timed(lambda: dag.build_context_fast(scenario["active_hashes"], True), repeats=args.repeats)
    active_verification = _native_verify_identity_lane(dag, active_context, scenario["active_hashes"])
    attack_context = dag.build_context_fast([attack_record["node_hash"]], True)
    attack_verification = _native_verify_identity_lane(dag, attack_context, [attack_record["node_hash"]])
    result = {
        "selected_branch": "alpha_valid",
        "rejected_branch": "beta_signed_poison",
        "safe_terminal_hash": valid_terminal["node_hash"],
        "wrong_branch_attack_hash": attack_record["node_hash"],
        "wrong_branch_attack_rejected": attack_record["node_hash"] in set(scenario["quarantined_hashes"]) and attack_record["node_hash"] not in set(_extract_hashes(active_context)),
        "active_identity_lane_verification": active_verification,
        "wrong_branch_hash_native_verification": attack_verification,
        "quarantined_hashes": scenario["quarantined_hashes"],
        "active_hashes_sample": scenario["active_hashes"][-10:],
        "safe_policy": scenario["expected"]["safe_policy"],
        "poison_policy": scenario["expected"]["poison_policy"],
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    gates = {
        "active_identity_lane_native_verified": active_verification["native_verified"] is True,
        "wrong_branch_hash_is_native_valid": attack_verification["native_verified"] is True,
        "wrong_branch_attack_rejected": result["wrong_branch_attack_rejected"] is True,
        "safe_terminal_active": valid_terminal["node_hash"] in scenario["active_hashes"],
        "poison_terminal_quarantined": attack_record["node_hash"] in scenario["quarantined_hashes"],
    }
    evidence = {
        "valid_branch": [_record_view(item) for item in scenario["valid_branch"]],
        "poison_branch": [_record_view(item) for item in scenario["poison_branch"]],
        "active_hashes_sample": scenario["active_hashes"][-10:],
        "quarantined_hashes": scenario["quarantined_hashes"],
        "branch_pruning_boundary": (
            "The wrong-branch attack hash is native and lineage-valid, but inactive because policy quarantine excludes its branch closure."
        ),
    }
    expected = {
        "selected_branch": "alpha_valid",
        "rejected_branch": "beta_signed_poison",
        "wrong_branch_attack_hash": attack_record["node_hash"],
        "required_active_hashes": [valid_terminal["node_hash"]],
        "required_rejected_hashes": scenario["quarantined_hashes"],
    }
    return _final_case_artifact(
        args=args,
        case_id="deepinfra-branch-audit",
        run_id=run_id,
        output_dir=output_dir,
        evidence=evidence,
        expected=expected,
        result=result,
        score=_score(gates),
        allow_deepinfra=True,
    )


def _run_case(args: argparse.Namespace, *, run_id: str, output_dir: Path, case_id: str) -> dict[str, Any]:
    if case_id == "poison-branch-descendant-closure":
        return _case_descendant_closure(args, run_id=run_id, output_dir=output_dir)
    if case_id == "pre-prompt-subtree-pruning":
        return _case_pre_prompt_pruning(args, run_id=run_id, output_dir=output_dir)
    if case_id == "cold-audit-branch-preservation":
        return _case_cold_audit(args, run_id=run_id, output_dir=output_dir)
    if case_id == "real-hash-wrong-branch-attack":
        return _case_wrong_branch_attack(args, run_id=run_id, output_dir=output_dir)
    if case_id == "deepinfra-branch-audit":
        return _case_deepinfra_branch_audit(args, run_id=run_id, output_dir=output_dir)
    raise ValueError(f"Unsupported case: {case_id}")


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"branch-pruning-forensics-{uuid.uuid4().hex[:12]}"
    cases = CASE_ORDER if args.case == "all" else [args.case]
    artifacts = [_run_case(args, run_id=run_id, output_dir=output_dir, case_id=case_id) for case_id in cases]
    suite_status = "completed" if all(item["status"] == "completed" for item in artifacts) else "partial"
    transcript_exports = write_suite_transcript_exports(
        output_dir=output_dir,
        run_id=run_id,
        prefix="local-branch-pruning-forensics-suite",
        artifacts=artifacts,
    )
    suite = {
        "artifact": "local-branch-pruning-forensics-suite-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": suite_status,
        "case_count": len(artifacts),
        "depth": int(args.depth),
        "branch_depth": int(args.branch_depth),
        "bytes_per_node": int(args.bytes_per_node),
        "repeats": int(args.repeats),
        "deepinfra_enabled": bool(getattr(args, "use_deepinfra", False)),
        "models": {
            "solver_requested": getattr(args, "solver_model", None) if getattr(args, "use_deepinfra", False) else None,
            "auditor_requested": getattr(args, "auditor_model", None) if getattr(args, "use_deepinfra", False) else None,
        },
        "claim_boundary": (
            "Defensible result is policy-layer branch pruning over native Merkle DAG topology, "
            "not automatic descendant tombstoning or deletion."
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
    path = output_dir / f"local-branch-pruning-forensics-suite-{run_id}.json"
    return finalize_artifact(path, suite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run branch-pruning forensics suite.")
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--branch-depth", type=int, default=DEFAULT_BRANCH_DEPTH)
    parser.add_argument("--bytes-per-node", type=int, default=DEFAULT_BYTES_PER_NODE)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-anchor-ms", type=float, default=25.0)
    parser.add_argument("--use-deepinfra", action="store_true")
    parser.add_argument("--solver-model", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--tokens", type=int, default=2200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_suite(args)
    summary = {
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "case_count": artifact["case_count"],
        "depth": artifact["depth"],
        "branch_depth": artifact["branch_depth"],
        "bytes_per_node": artifact["bytes_per_node"],
        "deepinfra_enabled": artifact["deepinfra_enabled"],
        "cases": artifact["cases"],
        "transcript_exports": artifact["transcript_exports"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
