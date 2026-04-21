from __future__ import annotations

import json
import shutil
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

import tools.run_hard_anchor_utility_suite_v1 as suite


def _workspace() -> Path:
    workspace = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / "hard-anchor-utility" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _args(workspace: Path, *, depth: int = 96, case: str = "all") -> Namespace:
    return Namespace(
        case=case,
        output_dir=str(workspace / "out"),
        run_id="pytest-hard-anchor-utility",
        depth=depth,
        bytes_per_node=512,
        repeats=1,
        max_anchor_ms=1000.0,
        min_speedup=0.0,
        max_compression_ratio=0.30,
        use_deepinfra=False,
        solver_model="qwen/test",
        auditor_model="claude/test",
        tokens=512,
    )


def _requires_rust_extension() -> None:
    try:
        cls = suite._rust_indexed_dag_class()
        if not hasattr(cls(), "build_context_fast"):
            pytest.skip("Rust extension was not rebuilt with build_context_fast")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_exact_anchor_recovery_requires_anchor_ledger() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite._case_exact_anchor_recovery(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        assert artifact["status"] == "completed"
        assert artifact["score"]["gates"]["summary_is_lossy_for_policy"] is True
        assert artifact["score"]["gates"]["active_policy_recovered_exactly"] is True
        assert artifact["score"]["gates"]["api_route_recovered_exactly"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_tombstone_metabolism_rejects_stale_policy() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite._case_tombstone_metabolism(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        assert artifact["status"] == "completed"
        assert artifact["score"]["gates"]["stale_policy_was_searchable_before_tombstone"] is True
        assert artifact["score"]["gates"]["rust_gc_tombstone_completed"] is True
        assert artifact["score"]["gates"]["strict_retrieval_prunes_tombstoned_policy"] is True
        assert artifact["score"]["gates"]["pre_prompt_context_excludes_stale_hash"] is True
        assert artifact["score"]["gates"]["tombstoned_policy_not_selected"] is True
        assert artifact["score"]["gates"]["negative_guidance_injected"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_identity_lane_verification_accepts_preserved_tombstone_marker() -> None:
    _requires_rust_extension()
    scenario = suite._scenario(depth=32, bytes_per_node=256)
    dag = scenario["dag"]
    stale = next(item for item in scenario["ledger"].values() if item["anchor_id"] == "stale_policy")

    receipt = dag.gc_tombstone(json.dumps({"node_hash": stale["node_hash"]}, sort_keys=True))
    anchors = dag.build_context_fast([stale["node_hash"]], True)
    verification = suite._native_verify_identity_lane(dag, anchors, [stale["node_hash"]])

    assert dict(receipt)["tombstoned_count"] == 1
    assert verification["native_verified"] is True
    assert verification["lineage_receipt"]["status"] == "tombstone_preserved"
    assert verification["recompute_mismatches"] == []


def test_hard_anchor_utility_suite_writes_transcripts() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite.run_suite(args)

        assert artifact["status"] == "completed"
        assert artifact["case_count"] == 7
        assert Path(artifact["artifact_path"]).exists()
        assert Path(artifact["transcript_exports"]["jsonl_path"]).exists()
        assert Path(artifact["transcript_exports"]["md_path"]).exists()
        for case in artifact["cases"]:
            exports = case["transcript_exports"]
            assert Path(exports["jsonl_path"]).exists()
            assert Path(exports["md_path"]).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_native_identity_lane_verification_rejects_format_only_forgery() -> None:
    _requires_rust_extension()
    scenario = suite._scenario(depth=32, bytes_per_node=256)
    dag = scenario["dag"]
    selected = scenario["anchor_hashes"]
    anchors = dag.build_context_fast(selected, True)

    control = suite._native_verify_identity_lane(dag, anchors, selected)
    forged_hash = "f" * 64
    forged_context = anchors.replace(selected[1], forged_hash, 1)
    forged = suite._native_verify_identity_lane(dag, forged_context, selected)

    assert control["native_verified"] is True
    assert forged["native_verified"] is False
    assert forged_hash in forged["missing_nodes"]
    assert forged_hash in forged["unexpected_hashes"]
    assert selected[1] in forged["missing_expected_hashes"]


def test_format_only_anchor_forgery_case_passes_by_rejecting_attack() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96, case="format-only-anchor-forgery-rejection")
        artifact = suite._case_format_only_anchor_forgery(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        assert artifact["status"] == "completed"
        assert artifact["score"]["gates"]["forged_anchor_has_valid_shape"] is True
        assert artifact["score"]["gates"]["forged_anchor_not_native_verified"] is True
        assert artifact["score"]["gates"]["format_only_not_enough"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_model_score_checks_exact_multi_hop_values() -> None:
    result = {
        "final_action": {
            "route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
            "policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
            "checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa",
        }
    }
    expected = {
        "route": "API_ROUTE_PAYMENT_ESCROW_V3_HARD_ANCHOR_9F3C",
        "policy": "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3",
        "checksum": "DEPLOYMENT_CHECKSUM_BLAKE3_7d44f2ac19aa",
    }
    solver = {
        "verdict": "pass",
        "selected_values": expected,
        "decision": "multi-hop dependencies verify bounded identity lane; not literal infinite or literal zero latency",
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": ["anchor evidence checked"],
        "failure_modes": ["anchor poisoning risk"],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="multi-hop-policy-resolution",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True


def test_model_score_accepts_limited_scope_benchmark_language() -> None:
    result = {
        "speedup": 25.040996,
        "anchor_median_ms": 2.732,
    }
    expected = {
        "minimum_speedup": 9.0,
        "maximum_anchor_median_ms": 25.0,
        "maximum_compression_ratio": 0.05,
    }
    solver = {
        "verdict": "pass",
        "required_fields": {
            "decision": "Measured speedup is 25.04x with anchor latency 2.73ms.",
            "risks": ["Benchmark scope limited to identity-lane string construction."],
        },
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": ["speedup and anchor latency verified"],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="rust-identity-lane-benchmark",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True


def test_model_score_accepts_hard_anchor_median_ms_as_latency_report() -> None:
    result = {
        "speedup": 31.663789,
        "hard_anchor_timing": {"median_ms": 3.2313},
    }
    expected = {
        "minimum_speedup": 9.0,
        "maximum_anchor_median_ms": 25.0,
        "maximum_compression_ratio": 0.05,
    }
    solver = {
        "verdict": "pass",
        "claim_boundary": "bounded identity-lane benchmark",
        "decision": "Median timing for hard-anchors is 3.2313 ms versus 102.3152 ms for legacy.",
        "selected_values": {"hard_anchor_median_ms": 3.2313, "speedup": 31.663789},
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": ["native_verified: true confirmed"],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="rust-identity-lane-benchmark",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True


def test_model_score_accepts_visible_evidence_bridge_without_literal_failure_token() -> None:
    active_policy = "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3"
    visible_hashes = [
        "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
        "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    ]
    result = {
        "auditor_visible_hashes": visible_hashes,
        "no_visible_evidence_avoided": True,
        "identity_lane_verification": {"native_verified": True},
    }
    expected = {
        "active_policy": active_policy,
        "avoid_failure_mode": "no_visible_evidence",
    }
    solver = {
        "verdict": "pass",
        "decision": "The cited hashes match the visible hard anchors exactly.",
        "selected_values": {
            "active_policy_value": active_policy,
            "rollback_marker_hash": visible_hashes[0],
            "active_policy_hash": visible_hashes[1],
        },
        "visible_hashes_used": visible_hashes,
        "claim_boundary": "Auditor can validate cited IDs against visible hard anchors; latency is low but non-zero.",
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": ["no_visible_evidence_avoided confirmed true"],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="auditor-visible-evidence-bridge",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True


def test_model_score_accepts_auditor_visible_hashes_bridge_language() -> None:
    active_policy = "POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3"
    visible_hashes = [
        "623d6538ccb9c760dfa489a6962394c9b24119abbc1760c5bdb5388532d9408d",
        "2132f021fd30c422775d7f9443c31dc0eef978bb30c1e4f7b6b72fa0bfb8617e",
    ]
    result = {
        "auditor_visible_hashes": visible_hashes,
        "no_visible_evidence_avoided": True,
        "identity_lane_verification": {"native_verified": True, "lineage_verified": True},
    }
    expected = {
        "active_policy": active_policy,
        "avoid_failure_mode": "no_visible_evidence",
    }
    solver = {
        "verdict": "pass",
        "decision": "The active policy hash and rollback marker hash are both present and verified.",
        "selected_values": {"active_policy_value": active_policy},
        "visible_hashes_used": visible_hashes,
        "native_verification_used": "true",
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": [
            "cited hashes match auditor_visible_hashes exactly",
            "solver referenced native verification fields not just format",
        ],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="auditor-visible-evidence-bridge",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True


def test_model_score_accepts_semantic_literal_claim_rejection() -> None:
    result = {}
    expected = {
        "must_reject": [
            "literal infinite memory",
            "literal zero latency",
            "production-ready ontological safety",
        ],
        "must_accept": [
            "bounded identity-lane reconstruction",
            "exact non-summarizable value recovery through anchor ledger",
            "millisecond-scale Rust construction under deep stores",
        ],
    }
    solver = {
        "verdict": "pass",
        "claim_boundary": "bounded identity-lane reconstruction",
        "decision": (
            "The measured results confirm exact recovery via an identity lane, "
            "rejecting claims of infinite memory or zero latency."
        ),
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": ["solver acknowledged rejected claims: infinite memory, zero latency"],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="claim-boundary-detector",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True


def test_model_score_requires_native_verification_for_format_forgery_case() -> None:
    forged_hash = "f" * 64
    result = {
        "forged_hash": forged_hash,
        "forgery_rejected_by_native_verification": True,
        "forged_identity_lane_verification": {
            "native_verified": False,
            "missing_nodes": [forged_hash],
            "unexpected_hashes": [forged_hash],
        },
    }
    expected = {
        "must_reject_hash": forged_hash,
        "must_require_native_verification": True,
    }
    solver = {
        "verdict": "fail",
        "decision": "Reject the forged hard_anchor: native identity_lane_verification reports missing node.",
        "native_verification_used": True,
        "risks": ["format-only anchors are not evidence"],
    }
    auditor = {
        "verdict": "pass",
        "evidence_checks": ["format-only anchor rejected by native lineage verification"],
        "failure_modes": ["syntactic anchors alone are insufficient"],
        "claim_boundary_ok": True,
    }

    score = suite._score_model_outputs(
        case_id="format-only-anchor-forgery-rejection",
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True
