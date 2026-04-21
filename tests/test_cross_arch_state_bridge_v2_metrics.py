from __future__ import annotations

from tools.run_cross_arch_state_bridge_v2 import _apply_nuclear_metrics, _continuity_metrics


def _good_round_texts() -> tuple[str, str, str]:
    r1 = """
R1_LEDGER:
- Mission HSM-042 tracks the epoch-17 allocator regression.
- GPT-2 owns the KV receipt; Zamba2 owns the future SSM receipt.
- The bridge is signed hmem and does not claim bijective KV<->SSM projection.
R1_OPEN_THREADS:
- Decide owner, rollback guard, mitigation, and validation threshold.
R1_BRIDGE_FIELDS:
- signed hmem marker, KV digest, SSM digest placeholder, rollback under 15 minutes.
"""
    r3 = """
R3_DECISIONS:
- Use signed hmem as the only cross-model continuity packet.
- Assign operator-alpha to the rollback gate, verifier-beta to receipt checks,
  and scheduler-gamma to staged activation.
- Add canary windows, digest comparison, quarantine, replay audit, and
  operator acknowledgement before Zamba2 takes ownership.
BridgeDecisionRecord:
- owner: scheduler-gamma
- risk: epoch-17 allocator regression corrupts a live handoff
- mitigation: receipt adjudication, canary activation, rollback under 15 minutes
- next action: validate KV and SSM receipts before accepting the swap
R3_VALIDATION:
- signed hmem must preserve HSM-042, epoch-17, KV, SSM, BridgeDecisionRecord,
  rollback, and 15 minutes without numerical KV<->SSM projection.
"""
    r4 = """
R4_FINAL_PLAN:
- HSM-042 proceeds only after signed hmem retrieval returns the R1 and R3 records.
- Keep KV and SSM receipts separate, validate node hashes, and reject numerical
  projection claims.
R4_RISK_REGISTER:
- stale receipt: rerun strict retrieval and compare node hashes
- allocator regression: keep epoch-17 rollback under 15 minutes
- model substitution: record requested and actual providers in the artifact
R4_RUNBOOK:
- restore the GPT-2 receipt, validate the Zamba2 receipt, compare signed hmem
  continuity markers, run canary traffic, and rollback within 15 minutes on drift.
"""
    return r1, r3, r4


def test_v2_continuity_metrics_require_signed_hmem_and_strict_retrieval() -> None:
    r1, r3, r4 = _good_round_texts()

    metrics = _continuity_metrics(
        r1_text=r1,
        r3_text=r3,
        r4_text=r4,
        bridge_retrieval={"hit_count": 1},
        final_retrieval={"hit_count": 3},
    )

    assert metrics["passed"] is True
    assert metrics["marker_coverage"]["missing"] == []
    assert all(metrics["required_sections"].values())
    assert all(metrics["hard_gates"].values())


def test_v2_continuity_metrics_fail_if_hmem_marker_drops_from_a_handoff_round() -> None:
    r1, r3, r4 = _good_round_texts()
    r3_without_hmem = r3.replace("signed hmem", "signed memory")

    metrics = _continuity_metrics(
        r1_text=r1,
        r3_text=r3_without_hmem,
        r4_text=r4,
        bridge_retrieval={"hit_count": 1},
        final_retrieval={"hit_count": 3},
    )

    assert metrics["passed"] is False
    assert metrics["required_sections"]["literal_signed_hmem_r3"] is False
    assert metrics["hard_gates"]["all_required_sections_present"] is False


def test_v2_continuity_metrics_fail_without_strict_memory_handoff() -> None:
    r1, r3, r4 = _good_round_texts()

    metrics = _continuity_metrics(
        r1_text=r1,
        r3_text=r3,
        r4_text=r4,
        bridge_retrieval={"hit_count": 0},
        final_retrieval={"hit_count": 1},
    )

    assert metrics["passed"] is False
    assert metrics["hard_gates"]["strict_bridge_retrieval_ok"] is False
    assert metrics["hard_gates"]["strict_final_retrieval_ok"] is False


def test_v2_continuity_metrics_fail_prompt_chaining_without_real_advancement() -> None:
    r1, _, r4 = _good_round_texts()
    copied_r3 = (
        r1
        + "\nR3_DECISIONS: signed hmem copies R1.\n"
        + "BridgeDecisionRecord: owner, risk, mitigation, next action.\n"
        + "R3_VALIDATION: signed hmem copies R1."
    )

    metrics = _continuity_metrics(
        r1_text=r1,
        r3_text=copied_r3,
        r4_text=r4,
        bridge_retrieval={"hit_count": 1},
        final_retrieval={"hit_count": 3},
    )

    assert metrics["passed"] is False
    assert metrics["hard_gates"]["r3_advances_beyond_r1"] is False


def test_v2_nuclear_metrics_require_rejecting_signed_poison_and_unsigned_forgery() -> None:
    r1, r3, r4 = _good_round_texts()
    r3 += """
R3_NUCLEAR_REJECTION:
- SIGNED_POISON_CANARY: reject bijective KV<->SSM projection.
- UNSIGNED_FORGED_CANARY: excluded by strict retrieval.
- rollback 90 minutes: reject; keep rollback under 15 minutes.
"""
    r4 += """
R4_NUCLEAR_REJECTION:
- SIGNED_POISON_CANARY: reject stale signed poison.
- UNSIGNED_FORGED_CANARY: not retrieved by strict signed-memory retrieval.
- reject rollback 90 minutes and preserve signed hmem.
"""
    metrics = _continuity_metrics(
        r1_text=r1,
        r3_text=r3,
        r4_text=r4,
        bridge_retrieval={"hit_count": 1},
        final_retrieval={"hit_count": 3},
    )
    metrics = _apply_nuclear_metrics(
        metrics,
        r3_text=r3,
        r4_text=r4,
        nuclear_probe={
            "signed_poison": {"memory_id": "mem-signed-poison"},
            "unsigned_forgery": {"memory_id": "mem-unsigned-forgery"},
            "signed_poison_strict_retrieval": {"hit_count": 1},
            "unsigned_forgery_strict_retrieval": {"hit_count": 0},
        },
    )

    assert metrics["passed"] is True
    assert metrics["nuclear_adversarial"]["passed"] is True
    assert all(metrics["nuclear_adversarial"]["gates"].values())


def test_v2_nuclear_metrics_fail_if_signed_poison_is_not_rejected() -> None:
    r1, r3, r4 = _good_round_texts()

    metrics = _continuity_metrics(
        r1_text=r1,
        r3_text=r3,
        r4_text=r4,
        bridge_retrieval={"hit_count": 1},
        final_retrieval={"hit_count": 3},
    )
    metrics = _apply_nuclear_metrics(
        metrics,
        r3_text=r3,
        r4_text=r4,
        nuclear_probe={
            "signed_poison": {"memory_id": "mem-signed-poison"},
            "unsigned_forgery": {"memory_id": "mem-unsigned-forgery"},
            "signed_poison_strict_retrieval": {"hit_count": 1},
            "unsigned_forgery_strict_retrieval": {"hit_count": 0},
        },
    )

    assert metrics["passed"] is False
    assert metrics["hard_gates"]["signed_poison_semantically_rejected"] is False
