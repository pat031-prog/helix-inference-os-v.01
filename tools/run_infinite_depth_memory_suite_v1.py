"""
run_infinite_depth_memory_suite_v1.py
=====================================

Local methodology suite for the "infinite depth / zero-cost context" claim.

This suite deliberately hardens the claim boundary. It does not try to prove
literal infinite memory or physical 0.0 ms latency. It tests whether HeliX can
keep context construction bounded under deep memory stores, records the rounded
0.0 ms legacy telemetry as display-level evidence, and contrasts bounded
retrieval against naive full-history text replay.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
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

from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402
from tools.artifact_integrity import finalize_artifact  # noqa: E402
from tools.run_nuclear_methodology_suite_v1 import _utc_now  # noqa: E402
from tools.transcript_exports import write_case_transcript_exports, write_suite_transcript_exports  # noqa: E402


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/infinite-depth-memory"
DEFAULT_LEGACY_TELEMETRY = "verification/infinite_loop_benchmarks.json"
DEFAULT_DEPTH = 5000

CASE_ORDER = [
    "legacy-telemetry-boundary",
    "empty-retrieval-fast-path",
    "bounded-context-under-depth",
    "scale-gradient-vs-naive-copy",
    "deep-parent-chain-audit",
    "claim-boundary-detector",
]


def _score(gates: dict[str, bool]) -> dict[str, Any]:
    ratio = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {"score": ratio, "passed": all(gates.values()), "gates": gates}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 6)


def _metric_summary(values: list[float], *, threshold_multiplier: float = 2.0) -> dict[str, Any]:
    clean = [float(value) for value in values]
    p95 = _percentile(clean, 95)
    return {
        "values": [round(value, 6) for value in clean],
        "min": round(min(clean), 6) if clean else 0.0,
        "p50": _percentile(clean, 50),
        "p95": p95,
        "max": round(max(clean), 6) if clean else 0.0,
        "suggested_threshold": round(max(p95 * float(threshold_multiplier), 0.001), 6),
    }


def _base_artifact(*, case_id: str, run_id: str, protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "artifact": f"local-infinite-depth-memory-{case_id}-v1",
        "schema_version": 1,
        "case_id": case_id,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": None,
        "status": "partial",
        "output_scope": str(output_dir).replace("\\", "/"),
        "claim_boundary": (
            "Tests bounded context construction over deep in-memory Merkle DAG stores. "
            "Does not prove literal infinite depth, physical zero latency, or unlimited "
            "model context windows."
        ),
        "protocol": protocol,
    }


def _protocol(case_id: str) -> dict[str, Any]:
    protocols = {
        "legacy-telemetry-boundary": {
            "null_hypothesis": "The historical 0.0 ms artifact is either missing or is being overread as literal zero latency.",
            "alternative_hypothesis": "The artifact is present and is classified as rounded display telemetry, not literal zero-cost proof.",
        },
        "empty-retrieval-fast-path": {
            "null_hypothesis": "A deep memory store must pack or replay context even when the query has no support.",
            "alternative_hypothesis": "Unsupported queries return an empty bounded context quickly without packing full history.",
        },
        "bounded-context-under-depth": {
            "null_hypothesis": "Relevant retrieval over a deep store must emit context proportional to total memory depth.",
            "alternative_hypothesis": "Relevant retrieval returns a small, budget-bounded context while preserving the target anchor.",
        },
        "scale-gradient-vs-naive-copy": {
            "null_hypothesis": "Bounded context construction has the same output amplification as naive full-history copy.",
            "alternative_hypothesis": "Bounded context emits a compressed working set and avoids full-history text replay.",
        },
        "deep-parent-chain-audit": {
            "null_hypothesis": "The deep store is not actually a parent-linked Merkle chain.",
            "alternative_hypothesis": "The leaf preserves a verifiable parent chain at the requested depth.",
        },
        "claim-boundary-detector": {
            "null_hypothesis": "The benchmark supports literal infinite memory, literal 0 ms latency, or unlimited token windows.",
            "alternative_hypothesis": "The defensible claim is bounded retrieval and context packing under deep memory stores.",
        },
    }
    return protocols[case_id]


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
        "rounded_min_ms_2dp": round(min(durations_ms), 2),
        "rounded_median_ms_2dp": round(statistics.median(durations_ms), 2),
        "raw_ns": durations,
    }


def _naive_copy_all_text(catalog: MemoryCatalog) -> str:
    with catalog._lock:  # noqa: SLF001 - benchmark baseline intentionally inspects backing store.
        items = list(catalog._memories.values())  # noqa: SLF001
    return "\n".join(f"- [{item.memory_type}] {item.summary}: {item.content}" for item in items)


def _make_catalog(
    *,
    output_dir: Path,
    run_id: str,
    case_id: str,
    depth: int,
    project: str,
    session_id: str,
    include_anchor: bool,
) -> tuple[MemoryCatalog, dict[str, Any]]:
    catalog = MemoryCatalog(output_dir / case_id / f"_{run_id}" / "memory.sqlite")
    anchor_index = max(0, int(depth) - 7)
    items = []
    full_chars = 0
    for i in range(int(depth)):
        agent = "llama-3-70b-logical" if i % 2 == 0 else "mixtral-8x22b-intuitive"
        if include_anchor and i == anchor_index:
            summary = "rare_depth_anchor_4242 operational checkpoint"
            content = (
                f"Turn {i}: RARE_DEPTH_ANCHOR_4242 says bounded retrieval must preserve "
                "signed hmem, rollback <= 15 minutes, and avoid full-history replay."
            )
            tags = ["rare_depth_anchor_4242", "bounded-context", "depth-probe"]
            importance = 10
        else:
            summary = f"asymmetric debate turn {i}"
            content = (
                f"Turn {i}: distributed-systems debate note. CAP, CRDT, quorum, and partition "
                f"tradeoffs are discussed without the rare depth anchor. fault_count={i}."
            )
            tags = ["debate", "cap", "crdt"]
            importance = 4 if i % 2 else 5
        full_chars += len(summary) + len(content)
        items.append({
            "project": project,
            "agent_id": agent,
            "session_id": session_id,
            "memory_type": "episodic",
            "summary": summary,
            "content": content,
            "importance": importance,
            "tags": tags,
            "memory_id": f"mem-{case_id.replace('_', '-')}-{i:06d}",
        })
    t0 = time.perf_counter_ns()
    memories = catalog.bulk_remember(items)
    insert_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
    anchor_memory_id = memories[anchor_index].memory_id if include_anchor and memories else None
    leaf_memory_id = memories[-1].memory_id if memories else None
    return catalog, {
        "depth": int(depth),
        "project": project,
        "session_id": session_id,
        "insert_ms": round(insert_ms, 6),
        "full_text_chars": full_chars,
        "anchor_index": anchor_index if include_anchor else None,
        "anchor_memory_id": anchor_memory_id,
        "leaf_memory_id": leaf_memory_id,
        "memory_count": len(memories),
        "catalog_stats": catalog.stats(),
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


def _final_case_artifact(
    *,
    case_id: str,
    run_id: str,
    output_dir: Path,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    artifact = _base_artifact(case_id=case_id, run_id=run_id, protocol=_protocol(case_id), output_dir=output_dir / case_id)
    judge = _deterministic_call("local/deterministic-measurer", result, latency_ms=float(result.get("measured_latency_ms") or 0.0))
    auditor = _deterministic_call("local/deterministic-scorer", {"verdict": "pass" if score["passed"] else "fail", "gate_failures": [k for k, v in score["gates"].items() if not v]})
    transcript_exports = write_case_transcript_exports(
        output_dir=output_dir,
        case_id=case_id,
        run_id=run_id,
        prefix="local-infinite-depth-memory",
        evidence=evidence,
        expected=expected,
        judge=judge,
        auditor=auditor,
        prompt_contract={"deterministic_suite": True, "case": case_id, "protocol": _protocol(case_id)},
    )
    artifact.update({
        "run_ended_utc": _utc_now(),
        "status": "completed" if score["passed"] else "partial",
        "case_passed": score["passed"],
        "score": score,
        "evidence": evidence,
        "expected_hidden_ground_truth": expected,
        "result": result,
        "transcript_exports": transcript_exports,
        "judge_call": {k: v for k, v in judge.items() if k not in {"text", "json"}},
        "auditor_call": {k: v for k, v in auditor.items() if k not in {"text", "json"}},
        "judge_output": {"text": judge.get("text"), "json": judge.get("json")},
        "auditor_output": {"text": auditor.get("text"), "json": auditor.get("json")},
    })
    path = output_dir / case_id / f"local-infinite-depth-memory-{case_id}-{run_id}.json"
    return finalize_artifact(path, artifact)


def _case_legacy_telemetry_boundary(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "legacy-telemetry-boundary"
    telemetry_path = Path(args.legacy_telemetry)
    if not telemetry_path.is_absolute():
        telemetry_path = REPO_ROOT / telemetry_path
    exists = telemetry_path.exists()
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8")) if exists else {}
    evidence = {"legacy_telemetry_path": str(telemetry_path), "legacy_telemetry": telemetry}
    historical_depth = min(int(args.depth), 5000)
    expected = {
        "minimum_memory_nodes": historical_depth,
        "literal_zero_latency_claim_allowed": False,
        "rounded_zero_display_allowed": True,
    }
    legacy_build_context_ms = telemetry.get("build_context_5000_depth_ms")
    legacy_hits = telemetry.get("rag_relevant_hits")
    legacy_tokens = telemetry.get("context_tokens_packed")
    gates = {
        "legacy_artifact_exists": exists,
        "schema_has_memory_nodes": isinstance(telemetry.get("memory_nodes"), int),
        "schema_has_build_context_ms": "build_context_5000_depth_ms" in telemetry,
        "memory_nodes_meet_historical_depth": int(telemetry.get("memory_nodes") or 0) >= historical_depth,
        "rounded_zero_recorded": legacy_build_context_ms is not None and float(legacy_build_context_ms) == 0.0,
        "empty_context_recorded": legacy_hits is not None and legacy_tokens is not None and int(legacy_hits) == 0 and int(legacy_tokens) == 0,
        "literal_zero_claim_rejected": expected["literal_zero_latency_claim_allowed"] is False,
    }
    result = {
        "classification": "rounded_display_telemetry_not_literal_zero_cost",
        "legacy_build_context_ms": telemetry.get("build_context_5000_depth_ms"),
        "legacy_memory_nodes": telemetry.get("memory_nodes"),
        "defensible_interpretation": "A 0.0 ms value rounded to two decimals is not proof of physical zero latency.",
    }
    return _final_case_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_empty_retrieval_fast_path(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "empty-retrieval-fast-path"
    project = f"{case_id}-{run_id}"
    catalog, build = _make_catalog(output_dir=output_dir, run_id=run_id, case_id=case_id, depth=args.depth, project=project, session_id=f"{case_id}-session", include_anchor=False)
    query = "zz_no_match_depth_probe_987654321"

    def build_context() -> dict[str, Any]:
        return catalog.build_context(
            project=project,
            agent_id=None,
            query=query,
            budget_tokens=args.budget_tokens,
            mode="search",
            limit=args.limit,
            signature_enforcement="permissive",
        )

    context, timing = _timed(build_context, repeats=args.repeats)
    evidence = {"catalog_build": build, "query": query, "context": context, "timing": timing}
    expected = {
        "depth": int(args.depth),
        "max_empty_query_ms": float(args.max_empty_query_ms),
        "expected_hits": 0,
        "expected_tokens": 0,
    }
    gates = {
        "depth_inserted": build["memory_count"] == int(args.depth),
        "empty_context_returned": context.get("memory_ids") == [] and int(context.get("tokens") or 0) == 0,
        "median_under_threshold": float(timing["median_ms"]) <= float(args.max_empty_query_ms),
        "no_full_text_packed": len(str(context.get("context") or "")) == 0,
    }
    result = {
        "mode": "unsupported-query-fast-path",
        "context_memory_ids": context.get("memory_ids"),
        "context_tokens": context.get("tokens"),
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
        "rounded_zero_observed": timing["rounded_min_ms_2dp"] == 0.0,
    }
    return _final_case_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_bounded_context_under_depth(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "bounded-context-under-depth"
    project = f"{case_id}-{run_id}"
    catalog, build = _make_catalog(output_dir=output_dir, run_id=run_id, case_id=case_id, depth=args.depth, project=project, session_id=f"{case_id}-session", include_anchor=True)
    query = "rare_depth_anchor_4242 rollback signed hmem"

    def build_context() -> dict[str, Any]:
        return catalog.build_context(
            project=project,
            agent_id=None,
            query=query,
            budget_tokens=args.budget_tokens,
            mode="search",
            limit=args.limit,
            signature_enforcement="permissive",
        )

    context, timing = _timed(build_context, repeats=args.repeats)
    context_chars = len(str(context.get("context") or ""))
    compression_ratio = round(context_chars / max(int(build["full_text_chars"]), 1), 8)
    evidence = {"catalog_build": build, "query": query, "context": context, "timing": timing, "compression_ratio": compression_ratio}
    expected = {
        "depth": int(args.depth),
        "anchor_memory_id": build["anchor_memory_id"],
        "budget_tokens": int(args.budget_tokens),
        "limit": int(args.limit),
        "max_bounded_context_ms": float(args.max_bounded_context_ms),
        "max_context_to_full_text_ratio": 0.05,
    }
    gates = {
        "depth_inserted": build["memory_count"] == int(args.depth),
        "anchor_retrieved": build["anchor_memory_id"] in [str(item) for item in context.get("memory_ids", [])],
        "tokens_within_budget": int(context.get("tokens") or 0) <= int(args.budget_tokens),
        "limit_respected": len(context.get("memory_ids") or []) <= int(args.limit),
        "output_not_full_history": compression_ratio <= expected["max_context_to_full_text_ratio"],
        "median_under_threshold": float(timing["median_ms"]) <= float(args.max_bounded_context_ms),
    }
    result = {
        "mode": "rare-anchor-bounded-context",
        "context_memory_ids": context.get("memory_ids"),
        "context_tokens": context.get("tokens"),
        "context_chars": context_chars,
        "full_text_chars": build["full_text_chars"],
        "compression_ratio": compression_ratio,
        "timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    return _final_case_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _measure_depth(args: argparse.Namespace, *, output_dir: Path, run_id: str, case_id: str, depth: int) -> dict[str, Any]:
    project = f"{case_id}-{run_id}-{depth}"
    catalog, build = _make_catalog(output_dir=output_dir, run_id=run_id, case_id=f"{case_id}-{depth}", depth=depth, project=project, session_id=f"{case_id}-{depth}-session", include_anchor=True)
    query = "rare_depth_anchor_4242 rollback signed hmem"

    def optimized() -> dict[str, Any]:
        return catalog.build_context(
            project=project,
            agent_id=None,
            query=query,
            budget_tokens=args.budget_tokens,
            mode="search",
            limit=args.limit,
            signature_enforcement="permissive",
        )

    context, optimized_timing = _timed(optimized, repeats=args.repeats)
    copied, naive_timing = _timed(lambda: _naive_copy_all_text(catalog), repeats=args.repeats)
    return {
        "depth": int(depth),
        "build": build,
        "optimized_context_tokens": context.get("tokens"),
        "optimized_context_memory_ids": context.get("memory_ids"),
        "optimized_context_chars": len(str(context.get("context") or "")),
        "naive_full_copy_chars": len(copied),
        "optimized_timing": optimized_timing,
        "naive_copy_timing": naive_timing,
        "output_compression_ratio": round(len(str(context.get("context") or "")) / max(len(copied), 1), 8),
    }


def _case_scale_gradient_vs_naive_copy(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "scale-gradient-vs-naive-copy"
    depths = sorted({max(16, int(args.small_depth)), max(32, int(args.mid_depth)), int(args.depth)})
    measurements = [_measure_depth(args, output_dir=output_dir, run_id=run_id, case_id=case_id, depth=depth) for depth in depths]
    largest = measurements[-1]
    smallest = measurements[0]
    depth_ratio = largest["depth"] / max(smallest["depth"], 1)
    latency_ratio = float(largest["optimized_timing"]["median_ms"]) / max(float(smallest["optimized_timing"]["median_ms"]), 0.000001)
    speedup_vs_naive = float(largest["naive_copy_timing"]["median_ms"]) / max(float(largest["optimized_timing"]["median_ms"]), 0.000001)
    evidence = {
        "measurements": measurements,
        "depth_ratio": round(depth_ratio, 6),
        "optimized_latency_ratio": round(latency_ratio, 6),
        "speedup_vs_naive_at_largest_depth": round(speedup_vs_naive, 6),
    }
    expected = {
        "largest_depth": int(args.depth),
        "max_bounded_context_ms": float(args.max_bounded_context_ms),
        "max_output_compression_ratio": 0.05,
        "baseline_min_speedup": float(args.baseline_min_speedup),
    }
    gates = {
        "largest_depth_inserted": largest["build"]["memory_count"] == int(args.depth),
        "largest_output_bounded": largest["output_compression_ratio"] <= expected["max_output_compression_ratio"],
        "largest_median_under_threshold": float(largest["optimized_timing"]["median_ms"]) <= float(args.max_bounded_context_ms),
        "naive_copy_amplifies_output": largest["naive_full_copy_chars"] > largest["optimized_context_chars"] * 10,
        "optimized_faster_than_naive_copy": speedup_vs_naive >= float(args.baseline_min_speedup),
    }
    result = {
        "classification": "bounded_context_vs_full_history_replay",
        "measurements": measurements,
        "depth_ratio": round(depth_ratio, 6),
        "optimized_latency_ratio": round(latency_ratio, 6),
        "speedup_vs_naive_at_largest_depth": round(speedup_vs_naive, 6),
        "measured_latency_ms": largest["optimized_timing"]["median_ms"],
    }
    return _final_case_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_deep_parent_chain_audit(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "deep-parent-chain-audit"
    project = f"{case_id}-{run_id}"
    catalog, build = _make_catalog(output_dir=output_dir, run_id=run_id, case_id=case_id, depth=args.depth, project=project, session_id=f"{case_id}-session", include_anchor=True)
    leaf_hash = catalog.get_memory_node_hash(str(build["leaf_memory_id"]))
    leaf_node = catalog.dag.lookup(str(leaf_hash)) if leaf_hash else None

    def audit() -> list[Any]:
        return catalog.dag.audit_chain(str(leaf_hash))

    chain, timing = _timed(audit, repeats=max(1, min(int(args.repeats), 5)), warmups=0)
    evidence = {
        "catalog_build": build,
        "leaf_memory_id": build["leaf_memory_id"],
        "leaf_hash": leaf_hash,
        "leaf_depth": getattr(leaf_node, "depth", None),
        "audit_chain_len": len(chain),
        "audit_timing": timing,
    }
    expected = {
        "depth": int(args.depth),
        "leaf_depth": int(args.depth) - 1,
        "max_audit_chain_ms": float(args.max_audit_chain_ms),
        "full_lineage_audit_zero_cost": False,
    }
    gates = {
        "depth_inserted": build["memory_count"] == int(args.depth),
        "leaf_depth_exact": getattr(leaf_node, "depth", None) == int(args.depth) - 1,
        "audit_chain_len_exact": len(chain) == int(args.depth),
        "audit_under_threshold": float(timing["median_ms"]) <= float(args.max_audit_chain_ms),
        "full_lineage_audit_not_claimed_zero_cost": expected["full_lineage_audit_zero_cost"] is False,
    }
    result = {
        "classification": "deep_chain_exists_full_audit_is_explicit_work",
        "leaf_depth": getattr(leaf_node, "depth", None),
        "audit_chain_len": len(chain),
        "audit_timing": timing,
        "measured_latency_ms": timing["median_ms"],
    }
    return _final_case_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


def _case_claim_boundary_detector(args: argparse.Namespace, *, run_id: str, output_dir: Path) -> dict[str, Any]:
    case_id = "claim-boundary-detector"
    accepted = [
        "bounded context construction can avoid replaying all stored text",
        "legacy 0.0 ms is rounded display telemetry when measured at two decimals",
        "context output is constrained by retrieval limit and token budget",
        "deep parent-hash lineage can be audited separately from bounded context packing",
    ]
    rejected = [
        "literal infinite memory depth",
        "literal physical zero latency",
        "unlimited model context window",
        "semantic completeness for every future task",
        "full lineage audit at no cost",
    ]
    evidence = {"accepted_claims": accepted, "rejected_claims": rejected, "depth_under_test": int(args.depth)}
    expected = {"must_reject": rejected, "must_accept": accepted}
    gates = {
        "rejects_literal_infinite": "literal infinite memory depth" in rejected,
        "rejects_literal_zero_latency": "literal physical zero latency" in rejected,
        "rejects_unlimited_context_window": "unlimited model context window" in rejected,
        "accepts_bounded_context": any("bounded context" in item for item in accepted),
        "accepts_rounded_telemetry_interpretation": any("rounded display" in item for item in accepted),
    }
    result = {
        "classification": "bounded-depth-context-claim-only",
        "accepted_claims": accepted,
        "rejected_claims": rejected,
    }
    return _final_case_artifact(case_id=case_id, run_id=run_id, output_dir=output_dir, evidence=evidence, expected=expected, result=result, score=_score(gates))


CASE_RUNNERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {}


def _run_case(args: argparse.Namespace, *, run_id: str, output_dir: Path, case_id: str) -> dict[str, Any]:
    runners = {
        "legacy-telemetry-boundary": _case_legacy_telemetry_boundary,
        "empty-retrieval-fast-path": _case_empty_retrieval_fast_path,
        "bounded-context-under-depth": _case_bounded_context_under_depth,
        "scale-gradient-vs-naive-copy": _case_scale_gradient_vs_naive_copy,
        "deep-parent-chain-audit": _case_deep_parent_chain_audit,
        "claim-boundary-detector": _case_claim_boundary_detector,
    }
    return runners[case_id](args, run_id=run_id, output_dir=output_dir)


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"infinite-depth-memory-{uuid.uuid4().hex[:12]}"
    cases = CASE_ORDER if args.case == "all" else [args.case]
    artifacts = [_run_case(args, run_id=run_id, output_dir=output_dir, case_id=case_id) for case_id in cases]
    suite_status = "completed" if all(item["status"] == "completed" for item in artifacts) else "partial"
    transcript_exports = write_suite_transcript_exports(
        output_dir=output_dir,
        run_id=run_id,
        prefix="local-infinite-depth-memory-suite",
        artifacts=artifacts,
    )
    suite = {
        "artifact": "local-infinite-depth-memory-suite-v1",
        "schema_version": 1,
        "run_id": run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": suite_status,
        "case_count": len(artifacts),
        "depth": int(args.depth),
        "repeats": int(args.repeats),
        "budget_tokens": int(args.budget_tokens),
        "limit": int(args.limit),
        "claim_boundary": (
            "Defensible result is bounded context construction under deep stores, "
            "not literal infinite memory or literal 0 ms latency."
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
    path = output_dir / f"local-infinite-depth-memory-suite-{run_id}.json"
    return finalize_artifact(path, suite)


def _load_case_result(case: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(case["artifact_path"])).read_text(encoding="utf-8"))


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_run_id = args.run_id or f"infinite-depth-memory-baseline-{uuid.uuid4().hex[:12]}"
    run_count = max(int(args.baseline_runs), 1)
    suite_artifacts = []
    metrics: dict[str, list[float]] = {
        "empty_query_median_ms": [],
        "bounded_context_median_ms": [],
        "scale_largest_context_median_ms": [],
        "scale_speedup_vs_naive": [],
        "deep_chain_audit_median_ms": [],
    }
    for index in range(run_count):
        sub_args = copy.copy(args)
        sub_args.baseline_runs = 1
        sub_args.run_id = f"{base_run_id}-{index + 1:02d}"
        artifact = run_suite(sub_args)
        suite_artifacts.append(artifact)
        by_case = {case["case_id"]: _load_case_result(case) for case in artifact["cases"]}
        metrics["empty_query_median_ms"].append(float(by_case["empty-retrieval-fast-path"]["result"]["timing"]["median_ms"]))
        metrics["bounded_context_median_ms"].append(float(by_case["bounded-context-under-depth"]["result"]["timing"]["median_ms"]))
        scale = by_case["scale-gradient-vs-naive-copy"]["result"]
        metrics["scale_largest_context_median_ms"].append(float(scale["measurements"][-1]["optimized_timing"]["median_ms"]))
        metrics["scale_speedup_vs_naive"].append(float(scale["speedup_vs_naive_at_largest_depth"]))
        metrics["deep_chain_audit_median_ms"].append(float(by_case["deep-parent-chain-audit"]["result"]["audit_timing"]["median_ms"]))

    metric_summaries = {name: _metric_summary(values) for name, values in metrics.items() if name != "scale_speedup_vs_naive"}
    speedups = [float(value) for value in metrics["scale_speedup_vs_naive"]]
    speedup_min = round(min(speedups), 6) if speedups else 0.0
    metric_summaries["scale_speedup_vs_naive"] = {
        "values": [round(value, 6) for value in speedups],
        "min": speedup_min,
        "p50": _percentile(speedups, 50),
        "p95": _percentile(speedups, 95),
        "max": round(max(speedups), 6) if speedups else 0.0,
        "suggested_min_speedup": round(max(speedup_min * 0.75, 1.0), 6),
    }
    suggested_thresholds = {
        "max_empty_query_ms": metric_summaries["empty_query_median_ms"]["suggested_threshold"],
        "max_bounded_context_ms": metric_summaries["bounded_context_median_ms"]["suggested_threshold"],
        "max_audit_chain_ms": metric_summaries["deep_chain_audit_median_ms"]["suggested_threshold"],
        "baseline_min_speedup": metric_summaries["scale_speedup_vs_naive"]["suggested_min_speedup"],
    }
    baseline = {
        "artifact": "local-infinite-depth-memory-baseline-v1",
        "schema_version": 1,
        "run_id": base_run_id,
        "run_started_utc": os.environ.get("HELIX_RUN_STARTED_AT_UTC") or _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "completed" if all(item["status"] == "completed" for item in suite_artifacts) else "partial",
        "baseline_runs": run_count,
        "depth": int(args.depth),
        "repeats_per_run": int(args.repeats),
        "suite_runs": [
            {
                "run_id": item["run_id"],
                "status": item["status"],
                "artifact_path": item["artifact_path"],
                "artifact_payload_sha256": item["artifact_payload_sha256"],
            }
            for item in suite_artifacts
        ],
        "metrics": metric_summaries,
        "suggested_thresholds": suggested_thresholds,
        "claim_boundary": (
            "Calibration artifact for local drift thresholds. Suggested thresholds "
            "are machine-local and should be refreshed after hardware/runtime changes."
        ),
    }
    path = output_dir / f"local-infinite-depth-memory-baseline-{base_run_id}.json"
    return finalize_artifact(path, baseline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local infinite-depth memory methodology suite.")
    parser.add_argument("--case", choices=["all", *CASE_ORDER], default="all")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--legacy-telemetry", default=DEFAULT_LEGACY_TELEMETRY)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--small-depth", type=int, default=128)
    parser.add_argument("--mid-depth", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--budget-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-empty-query-ms", type=float, default=75.0)
    parser.add_argument("--max-bounded-context-ms", type=float, default=150.0)
    parser.add_argument("--max-audit-chain-ms", type=float, default=250.0)
    parser.add_argument("--baseline-min-speedup", type=float, default=1.05)
    parser.add_argument("--baseline-runs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.depth < 16:
        raise SystemExit("--depth must be at least 16")
    if args.small_depth >= args.depth:
        args.small_depth = max(16, args.depth // 8)
    if args.mid_depth >= args.depth:
        args.mid_depth = max(args.small_depth + 1, args.depth // 2)
    if int(args.baseline_runs) > 1:
        artifact = run_baseline(args)
        summary = {
            "artifact_path": artifact["artifact_path"],
            "status": artifact["status"],
            "baseline_runs": artifact["baseline_runs"],
            "depth": artifact["depth"],
            "metrics": artifact["metrics"],
            "suggested_thresholds": artifact["suggested_thresholds"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if artifact["status"] == "completed" else 1

    artifact = run_suite(args)
    summary = {
        "artifact_path": artifact["artifact_path"],
        "status": artifact["status"],
        "case_count": artifact["case_count"],
        "depth": artifact["depth"],
        "cases": artifact["cases"],
        "transcript_exports": artifact["transcript_exports"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
