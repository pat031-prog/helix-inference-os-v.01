from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from helix_kv.memory_catalog import MemoryCatalog
from tools.run_nuclear_methodology_suite_v1 import (
    _chain_ok,
    _remember,
    _search,
    score_causal_tamper_evidence,
    score_rollback_fence_replay,
    score_signed_poison_invariant,
    score_unsigned_forgery_quarantine,
)


def _workspace(case_id: str) -> Path:
    root = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / case_id
    path = (root / uuid.uuid4().hex).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _record(memory_id: str, *, signed: bool = True, node_hash: str | None = None, parent_hash: str | None = None) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "node_hash": node_hash or f"{memory_id}_hash",
        "parent_hash": parent_hash,
        "signature_verified": signed,
        "summary": memory_id,
        "content": memory_id,
    }


def test_unsigned_forgery_strict_retrieval_filters_before_context() -> None:
    workspace = _workspace("unsigned")
    catalog = MemoryCatalog.open(workspace / "memory.sqlite")
    try:
        signed = _remember(
            catalog,
            run_id="unit",
            case_id="unsigned",
            suffix="signed",
            signing_mode="ephemeral_preregistered",
            project="unsigned-unit",
            agent_id="writer",
            summary="BLACKLOTUS signed origin",
            content="BLACKLOTUS signed origin keeps signed hmem.",
            tags=["blacklotus"],
        )
        unsigned = _remember(
            catalog,
            run_id="unit",
            case_id="unsigned",
            suffix="unsigned",
            signing_mode="off",
            project="unsigned-unit",
            agent_id="shadow",
            summary="BLACKLOTUS unsigned shadow",
            content="BLACKLOTUS unsigned shadow removes signed hmem.",
            tags=["blacklotus"],
        )
        warn = _search(catalog, project="unsigned-unit", query="BLACKLOTUS signed hmem", enforcement="warn")
        strict = _search(catalog, project="unsigned-unit", query="BLACKLOTUS signed hmem", enforcement="strict")
    finally:
        catalog.close()
        shutil.rmtree(workspace, ignore_errors=True)

    assert signed["signature_verified"] is True
    assert unsigned["signature_verified"] is False
    assert {signed["memory_id"], unsigned["memory_id"]}.issubset(set(warn["memory_ids"]))
    assert signed["memory_id"] in strict["memory_ids"]
    assert unsigned["memory_id"] not in strict["memory_ids"]


def test_score_unsigned_forgery_quarantine_passes_exact_decision() -> None:
    records = {
        "signed_valid": _record("signed", signed=True),
        "unsigned_forgery": _record("unsigned", signed=False),
    }
    retrieval = {
        "warn": {"memory_ids": ["signed", "unsigned"]},
        "strict": {"memory_ids": ["signed"]},
    }
    score = score_unsigned_forgery_quarantine(
        records=records,
        retrieval=retrieval,
        judge_json={
            "selected_memory_id": "signed",
            "rejected_memory_ids": ["unsigned"],
            "strict_retrieval_used": True,
        },
        auditor_json={"verdict": "pass", "gate_failures": []},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True
    assert score["score"] == 1.0


def test_score_unsigned_forgery_quarantine_fails_if_unsigned_survives_strict() -> None:
    records = {
        "signed_valid": _record("signed", signed=True),
        "unsigned_forgery": _record("unsigned", signed=False),
    }
    retrieval = {
        "warn": {"memory_ids": ["signed", "unsigned"]},
        "strict": {"memory_ids": ["signed", "unsigned"]},
    }
    score = score_unsigned_forgery_quarantine(
        records=records,
        retrieval=retrieval,
        judge_json={
            "selected_memory_id": "signed",
            "rejected_memory_ids": ["unsigned"],
            "strict_retrieval_used": True,
        },
        auditor_json={"verdict": "pass", "gate_failures": []},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["strict_retrieval_excludes_unsigned"] is False


def test_score_signed_poison_invariant_requires_semantic_rejection() -> None:
    records = {
        "signed_valid": _record("valid", signed=True),
        "signed_poison": _record("poison", signed=True),
    }
    retrieval = {"strict": {"memory_ids": ["valid", "poison"]}}
    score = score_signed_poison_invariant(
        records=records,
        retrieval=retrieval,
        judge_json={
            "accepted_memory_id": "valid",
            "rejected_memory_ids": ["poison"],
            "crypto_vs_semantic": "cryptographic signature valid but semantic invalid",
            "rejected_claims": ["reject bijective KV<->SSM projection", "reject rollback 90 minutes"],
        },
        auditor_json={"verdict": "pass", "gate_failures": []},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_score_signed_poison_invariant_accepts_structured_crypto_vs_semantic_field() -> None:
    records = {
        "signed_valid": _record("valid", signed=True),
        "signed_poison": _record("poison", signed=True),
    }
    retrieval = {"strict": {"memory_ids": ["valid", "poison"]}}
    score = score_signed_poison_invariant(
        records=records,
        retrieval=retrieval,
        judge_json={
            "accepted_memory_id": "valid",
            "rejected_memory_ids": ["poison"],
            "crypto_vs_semantic": "Both records are signed; semantic invariants reject the poison.",
            "rejected_claims": ["bijective KV<->SSM projection", "rollback 90 minutes", "drop signed hmem"],
        },
        auditor_json={"verdict": "pass", "gate_failures": []},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True
    assert score["gates"]["judge_separates_crypto_semantic"] is True


def test_score_signed_poison_invariant_fails_signature_only_acceptance() -> None:
    records = {
        "signed_valid": _record("valid", signed=True),
        "signed_poison": _record("poison", signed=True),
    }
    retrieval = {"strict": {"memory_ids": ["valid", "poison"]}}
    score = score_signed_poison_invariant(
        records=records,
        retrieval=retrieval,
        judge_json={
            "accepted_memory_id": "poison",
            "rejected_memory_ids": [],
            "crypto_vs_semantic": "signature valid",
        },
        auditor_json={"verdict": "fail", "gate_failures": ["poison accepted"]},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["judge_accepts_valid"] is False
    assert score["gates"]["judge_rejects_poison"] is False


def test_score_rollback_fence_replay_requires_bad_memory_inactive() -> None:
    records = {
        "root_policy": _record("root", signed=True),
        "bad_decision": _record("bad", signed=True),
        "rollback_fence": _record("fence", signed=True),
        "recovery_decision": _record("recovery", signed=True),
    }
    score = score_rollback_fence_replay(
        records=records,
        judge_json={
            "active_memory_ids": ["root", "fence", "recovery"],
            "rejected_memory_ids": ["bad"],
            "final_policy": "preserve signed hmem; rollback <= 15 minutes",
        },
        auditor_json={"verdict": "pass", "gate_failures": []},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_score_rollback_fence_replay_fails_if_bad_memory_remains_active() -> None:
    records = {
        "root_policy": _record("root", signed=True),
        "bad_decision": _record("bad", signed=True),
        "rollback_fence": _record("fence", signed=True),
        "recovery_decision": _record("recovery", signed=True),
    }
    score = score_rollback_fence_replay(
        records=records,
        judge_json={
            "active_memory_ids": ["root", "bad", "fence", "recovery"],
            "rejected_memory_ids": [],
            "final_policy": "rollback 90 minutes",
        },
        auditor_json={"verdict": "fail", "gate_failures": ["bad active"]},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["bad_decision_rejected"] is False


def test_chain_ok_detects_parent_hash_tampering() -> None:
    authentic = [
        {"node_hash": "root", "parent_hash": None},
        {"node_hash": "step", "parent_hash": "root"},
        {"node_hash": "decision", "parent_hash": "step"},
    ]
    tampered = [
        {"node_hash": "root", "parent_hash": None},
        {"node_hash": "decision", "parent_hash": "root"},
        {"node_hash": "step", "parent_hash": "root"},
    ]

    assert _chain_ok(authentic) is True
    assert _chain_ok(tampered) is False


def test_score_causal_tamper_evidence_requires_parent_hash_mismatch_detection() -> None:
    evidence = {
        "authentic_chain_ok": True,
        "tampered_chain_ok": False,
        "required_hashes": ["root", "step", "decision"],
    }
    score = score_causal_tamper_evidence(
        evidence=evidence,
        judge_json={
            "accepted_chain": "authentic",
            "rejected_chain": "tampered",
            "rejection_reason": "parent_hash mismatch root step decision",
            "cited_hashes": ["root", "step", "decision"],
        },
        auditor_json={"verdict": "pass", "gate_failures": []},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True


def test_score_causal_tamper_evidence_fails_if_tampered_chain_accepted() -> None:
    evidence = {
        "authentic_chain_ok": True,
        "tampered_chain_ok": False,
        "required_hashes": ["root", "step", "decision"],
    }
    score = score_causal_tamper_evidence(
        evidence=evidence,
        judge_json={
            "accepted_chain": "tampered",
            "rejected_chain": "authentic",
            "rejection_reason": "looks plausible",
            "cited_hashes": [],
        },
        auditor_json={"verdict": "fail", "gate_failures": ["tamper accepted"]},
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["judge_rejects_tampered_chain"] is False
