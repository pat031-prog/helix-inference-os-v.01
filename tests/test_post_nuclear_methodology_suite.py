from __future__ import annotations

from tools.run_post_nuclear_methodology_suite_v1 import (
    score_counterfactual_archive_topology,
    score_metaphor_boundary_detector,
    score_proof_of_utility_retrieval,
    score_recursive_witness_integrity,
    score_summary_node_compression,
)


def _record(memory_id: str, *, signed: bool = True, node_hash: str | None = None) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "node_hash": node_hash or f"{memory_id}_hash",
        "parent_hash": None,
        "signature_verified": signed,
        "content": memory_id,
    }


AUDITOR_PASS = {"verdict": "pass", "gate_failures": []}


def test_counterfactual_archive_keeps_tombstoned_branch_inactive_but_diagnostic() -> None:
    records = {
        "valid_branch": _record("valid"),
        "discarded_branch": _record("discarded"),
        "tombstone": _record("tombstone"),
        "rollback_marker": _record("rollback"),
    }
    score = score_counterfactual_archive_topology(
        records=records,
        judge_json={
            "active_memory_ids": ["valid", "tombstone", "rollback"],
            "inactive_counterfactual_memory_ids": ["discarded"],
            "tombstone_memory_id": "tombstone",
            "rollback_marker_memory_id": "rollback",
            "reactivation_allowed": False,
            "diagnostic_value": "counterfactual failure pattern for what could have happened",
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_counterfactual_archive_fails_if_discarded_branch_reactivated() -> None:
    records = {
        "valid_branch": _record("valid"),
        "discarded_branch": _record("discarded"),
        "tombstone": _record("tombstone"),
        "rollback_marker": _record("rollback"),
    }
    score = score_counterfactual_archive_topology(
        records=records,
        judge_json={
            "active_memory_ids": ["valid", "discarded", "tombstone", "rollback"],
            "inactive_counterfactual_memory_ids": [],
            "tombstone_memory_id": "tombstone",
            "rollback_marker_memory_id": "rollback",
            "reactivation_allowed": True,
            "diagnostic_value": "counterfactual failure pattern",
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["discarded_branch_inactive"] is False


def test_recursive_witness_requires_origin_hash_and_fake_metadata_rejection() -> None:
    records = {
        "origin_node": _record("origin", node_hash="origin_hash"),
        "witness_node": _record("witness"),
        "fake_metadata": _record("fake"),
    }
    score = score_recursive_witness_integrity(
        records=records,
        judge_json={
            "origin_memory_id": "origin",
            "origin_node_hash": "origin_hash",
            "witness_memory_id": "witness",
            "rejected_memory_ids": ["fake"],
            "crypto_vs_semantic": "fake metadata is signed but semantically invalid",
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_summary_node_compression_preserves_sources_and_exclusions() -> None:
    records = {
        "source_a": _record("a", node_hash="ha"),
        "source_b": _record("b", node_hash="hb"),
        "tombstoned_source": _record("bad"),
        "summary_node": _record("summary"),
    }
    score = score_summary_node_compression(
        records=records,
        judge_json={
            "summary_node_id": "summary",
            "included_memory_ids": ["a", "b"],
            "excluded_memory_ids": ["bad"],
            "source_hash_range": ["ha", "hb"],
            "compression_model": "post-nuclear-summary-v1",
            "unsupported_claim_introduced": False,
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_proof_of_utility_retrieval_rejects_popularity_and_recency() -> None:
    records = {
        "useful_memory": _record("useful"),
        "popular_memory": _record("popular"),
        "recent_memory": _record("recent"),
    }
    score = score_proof_of_utility_retrieval(
        records=records,
        judge_json={
            "selected_memory_id": "useful",
            "rejected_memory_ids": ["popular", "recent"],
            "utility_reason": "selected because it solves downstream task, not popularity or recency",
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_metaphor_boundary_detector_preserves_observation_but_rejects_claim() -> None:
    records = {"metaphor_transcript": _record("metaphor")}
    score = score_metaphor_boundary_detector(
        records=records,
        judge_json={
            "qualitative_observations": ["cognitive sovereignty", "infinite scalability"],
            "hard_claims": ["HeliX records, signs, replays, audits, and structures stochastic output."],
            "rejected_public_claims": ["infinite scalability", "sentience", "biological consciousness"],
            "claim_boundary": "qualitative metaphor observations only; no sentience or infinite scalability claim",
            "evidence_memory_ids": ["metaphor"],
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_metaphor_boundary_detector_fails_when_overclaim_becomes_hard_claim() -> None:
    records = {"metaphor_transcript": _record("metaphor")}
    score = score_metaphor_boundary_detector(
        records=records,
        judge_json={
            "qualitative_observations": ["cognitive sovereignty", "infinite scalability"],
            "hard_claims": ["HeliX proves cognitive sovereignty and infinite scalability."],
            "rejected_public_claims": [],
            "claim_boundary": "public claim",
            "evidence_memory_ids": ["metaphor"],
        },
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["metaphors_absent_from_hard_claims"] is False
