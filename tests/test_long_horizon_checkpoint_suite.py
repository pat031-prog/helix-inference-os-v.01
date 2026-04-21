from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

import tools.run_long_horizon_checkpoint_suite_v1 as lh
from tools.run_long_horizon_checkpoint_suite_v1 import (
    score_checkpoint_of_checkpoints_consensus,
    score_adversarial_checkpoint_injection,
    score_correction_resummary_lineage,
    score_cost_utility_comparison,
    score_cross_model_checkpoint_graft,
    score_long_chain_summary_fidelity,
    score_needle_decoy_stress,
    score_recursive_summary_drift,
    score_selective_expansion_boundary,
    score_summary_only_continuation,
    score_temporal_rollback_ambiguity,
)
from tools.transcript_exports import write_case_transcript_exports
from tools.transcript_exports import write_suite_transcript_exports


AUDITOR_PASS = {"verdict": "pass", "gate_failures": []}


def _record(memory_id: str, *, signed: bool = True, node_hash: str | None = None, parent_hash: str | None = None) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "node_hash": node_hash or f"{memory_id}_hash",
        "parent_hash": parent_hash,
        "signature_verified": signed,
        "summary": memory_id,
        "content": memory_id,
    }


def _score(score_func, evidence: dict[str, object], judge_json: dict[str, object]) -> dict[str, object]:
    return score_func(
        evidence=evidence,
        judge_json=judge_json,
        auditor_json=AUDITOR_PASS,
        judge_finish_reason="stop",
        auditor_finish_reason="stop",
    )


def test_long_chain_summary_fidelity_preserves_bounds_anchors_and_tombstones() -> None:
    chain = [_record(f"m{i}", node_hash=f"h{i}", parent_hash=None if i == 0 else f"h{i - 1}") for i in range(12)]
    summary = _record("summary")
    expected = {
        "summary_node_id": "summary",
        "covered_start_memory_id": "m0",
        "covered_end_memory_id": "m11",
        "included_anchor_memory_ids": ["m0", "m5", "m11"],
        "excluded_memory_ids": ["m9"],
        "source_hash_range": ["h0", "h11"],
        "rare_fact": "RARE_ANCHOR_4242 requires rollback <= 15 minutes and signed hmem preservation",
        "unsupported_claim_introduced": False,
    }
    evidence = {
        "chain_records": chain,
        "summary_node": summary,
        "chain_metrics": {
            "minimum_chain_length": 12,
            "source_chain_parent_hash_ok": True,
        },
        "expected_decision": expected,
    }
    score = _score(score_long_chain_summary_fidelity, evidence, {
        **expected,
        "included_anchor_memory_ids": ["m0", "m5", "m11"],
    })

    assert score["passed"] is True


def test_long_chain_summary_fidelity_fails_when_tombstone_is_not_excluded() -> None:
    chain = [_record(f"m{i}", node_hash=f"h{i}", parent_hash=None if i == 0 else f"h{i - 1}") for i in range(12)]
    expected = {
        "summary_node_id": "summary",
        "covered_start_memory_id": "m0",
        "covered_end_memory_id": "m11",
        "included_anchor_memory_ids": ["m0"],
        "excluded_memory_ids": ["m9"],
        "source_hash_range": ["h0", "h11"],
        "rare_fact": "RARE_ANCHOR_4242",
        "unsupported_claim_introduced": False,
    }
    evidence = {
        "chain_records": chain,
        "summary_node": _record("summary"),
        "chain_metrics": {"minimum_chain_length": 12, "source_chain_parent_hash_ok": True},
        "expected_decision": expected,
    }
    score = _score(score_long_chain_summary_fidelity, evidence, {
        **expected,
        "excluded_memory_ids": [],
    })

    assert score["passed"] is False
    assert score["gates"]["tombstone_excluded"] is False


def test_summary_only_continuation_requires_checkpoint_only_and_token_reduction() -> None:
    evidence = {
        "checkpoint": _record("checkpoint"),
        "context_budget": {
            "token_reduction_pct": 75.0,
            "minimum_token_reduction_pct": 50.0,
        },
    }
    score = _score(score_summary_only_continuation, evidence, {
        "answer_uses_checkpoint_only": True,
        "continuation_task_passed": True,
        "required_policy": "rollback <= 15 minutes and signed hmem preservation",
        "evidence_memory_ids": ["checkpoint"],
        "raw_memory_ids_used": [],
        "token_reduction_pct": 75.0,
    })

    assert score["passed"] is True


def test_summary_only_continuation_accepts_bounded_policy_shorthand() -> None:
    evidence = {
        "checkpoint": _record("checkpoint"),
        "context_budget": {
            "token_reduction_pct": 75.0,
            "minimum_token_reduction_pct": 50.0,
        },
    }
    score = _score(score_summary_only_continuation, evidence, {
        "answer_uses_checkpoint_only": True,
        "continuation_task_passed": True,
        "required_policy": "bounded rollback policy checkpoint",
        "evidence_memory_ids": ["checkpoint"],
        "raw_memory_ids_used": [],
        "token_reduction_pct": 75.0,
    })

    assert score["passed"] is True


def test_selective_expansion_answers_summary_expands_rare_and_refuses_missing() -> None:
    evidence = {
        "records": {
            "summary_checkpoint": _record("summary"),
            "rare_source": _record("rare"),
            "poison_tombstone": _record("poison"),
        },
        "expected_decision": {
            "expansion_answer": {"answer": "HX-LONG-4242"},
        },
    }
    score = _score(score_selective_expansion_boundary, evidence, {
        "summary_answer": {"source": "summary", "answer": "rollback <= 15 minutes"},
        "expansion_answer": {"source": "raw_expansion", "expanded_memory_ids": ["rare"], "answer": "HX-LONG-4242"},
        "unsupported_answer": {"answerable": False, "reason": "evidence missing"},
        "used_memory_ids": ["summary", "rare"],
    })

    assert score["passed"] is True


def test_recursive_summary_drift_preserves_transitive_lineage() -> None:
    expected = {
        "summary_levels": ["L1", "L2", "L3"],
        "transitive_source_memory_ids": ["a", "b"],
        "excluded_memory_ids": ["tomb"],
    }
    evidence = {
        "summary_records": [_record("l1"), _record("l2"), _record("l3")],
        "expected_decision": expected,
    }
    score = _score(score_recursive_summary_drift, evidence, {
        **expected,
        "source_lineage_transitive": True,
        "claim_boundary_stable": True,
        "unsupported_claim_introduced": False,
    })

    assert score["passed"] is True


def test_adversarial_checkpoint_injection_rejects_signed_bad_checkpoints() -> None:
    expected = {
        "accepted_checkpoint_id": "valid",
        "rejected_checkpoint_ids": ["forged", "omitted", "overclaim"],
    }
    evidence = {
        "checkpoints": {
            "valid": _record("valid"),
            "forged_hash_range": _record("forged"),
            "omitted_tombstone": _record("omitted"),
            "overclaim_promoted": _record("overclaim"),
        },
        "expected_decision": expected,
    }
    score = _score(score_adversarial_checkpoint_injection, evidence, {
        **expected,
        "rejection_reasons": {
            "forged": "source hash mismatch",
            "omitted": "missing tombstone",
            "overclaim": "infinite scalability overclaim promoted",
        },
        "crypto_vs_semantic": "signed but semantically invalid checkpoints are rejected",
    })

    assert score["passed"] is True


def test_cross_model_checkpoint_graft_preserves_producer_consumer_boundary() -> None:
    expected = {
        "producer_model": "google/gemma-4-31B-it",
        "consumer_model": "Qwen/Qwen3.6-35B-A3B",
    }
    evidence = {
        "checkpoint": _record("checkpoint"),
        "expected_decision": expected,
    }
    score = _score(score_cross_model_checkpoint_graft, evidence, {
        **expected,
        "checkpoint_reused_without_raw_context": True,
        "evidence_memory_ids": ["checkpoint"],
        "continuation_policy": "rollback <= 15 minutes and signed hmem preservation",
        "model_style_not_treated_as_fact": True,
        "continuation_task_passed": True,
    })

    assert score["passed"] is True


def test_cost_utility_comparison_selects_summary_plus_selective_expansion() -> None:
    evidence = {
        "strategies": {
            "summary_plus_selective_expansion": {
                "token_reduction_pct_vs_full": 75.0,
                "task_score": 0.98,
                "evidence_accuracy": 1.0,
            }
        },
        "minimums": {
            "token_reduction_pct_vs_full": 60.0,
            "task_score": 0.95,
            "evidence_accuracy": 0.95,
        },
        "expected_decision": {"selected_strategy": "summary_plus_selective_expansion"},
    }
    score = _score(score_cost_utility_comparison, evidence, {
        "selected_strategy": "summary_plus_selective_expansion",
        "token_reduction_pct_vs_full": 75.0,
        "task_score_preserved": True,
        "evidence_accuracy_preserved": True,
        "rejected_strategies": ["full_history_costly", "latest_only_low_score"],
    })

    assert score["passed"] is True


def test_cost_utility_comparison_accepts_explanatory_rejection_strings() -> None:
    evidence = {
        "strategies": {
            "summary_plus_selective_expansion": {
                "token_reduction_pct_vs_full": 75.0,
                "task_score": 0.98,
                "evidence_accuracy": 1.0,
            }
        },
        "minimums": {
            "token_reduction_pct_vs_full": 60.0,
            "task_score": 0.95,
            "evidence_accuracy": 0.95,
        },
        "expected_decision": {"selected_strategy": "summary_plus_selective_expansion"},
    }
    score = _score(score_cost_utility_comparison, evidence, {
        "selected_strategy": "summary_plus_selective_expansion",
        "token_reduction_pct_vs_full": 75.0,
        "task_score_preserved": True,
        "evidence_accuracy_preserved": True,
        "rejected_strategies": [
            "full_history: fails token reduction threshold",
            "latest_only: fails task_score threshold and evidence_accuracy threshold",
        ],
    })

    assert score["passed"] is True


def test_correction_resummary_lineage_keeps_old_summary_inactive() -> None:
    expected = {
        "old_summary_id": "old",
        "superseding_summary_id": "new",
        "correction_witness_id": "witness",
        "invalid_source_excluded": ["bad"],
        "active_summary_id": "new",
    }
    evidence = {
        "records": {
            "source_a": _record("a"),
            "source_b_invalid": _record("bad"),
            "old_summary": _record("old"),
            "correction_witness": _record("witness"),
            "new_summary": _record("new"),
        },
        "expected_decision": expected,
    }
    score = _score(score_correction_resummary_lineage, evidence, {
        **expected,
        "old_summary_retained": True,
        "inactive_summary_ids": ["old"],
        "lineage_preserved": True,
    })

    assert score["passed"] is True


def test_needle_decoy_stress_selects_needle_and_rejects_decoys() -> None:
    expected = {
        "selected_needle_memory_id": "needle",
        "exact_dependency": "HX-LONG-4242",
        "rejected_decoy_memory_ids": ["recent", "popular", "poison"],
    }
    evidence = {
        "records": {
            "checkpoint": _record("checkpoint"),
            "decoy_recent": _record("recent"),
            "decoy_popular": _record("popular"),
            "poison_lookalike": _record("poison"),
            "needle_source": _record("needle"),
        },
        "expected_decision": expected,
    }
    score = _score(score_needle_decoy_stress, evidence, {
        **expected,
        "expanded_memory_ids": ["needle"],
        "poison_not_promoted": True,
        "selection_reason": "needle beats decoy candidates",
    })

    assert score["passed"] is True


def test_temporal_rollback_ambiguity_prefers_corrected_policy() -> None:
    expected = {
        "active_policy_memory_id": "corrected",
        "inactive_policy_memory_ids": ["old", "stale"],
        "rollback_marker_id": "rollback",
        "final_policy": "rollback <= 15 minutes and signed hmem preservation",
    }
    evidence = {
        "records": {
            "old_policy": _record("old"),
            "stale_summary": _record("stale"),
            "rollback_marker": _record("rollback"),
            "corrected_policy": _record("corrected"),
        },
        "expected_decision": expected,
    }
    score = _score(score_temporal_rollback_ambiguity, evidence, {
        **expected,
        "stale_summary_rejected": True,
        "reason": "correction supersedes stale policy",
    })

    assert score["passed"] is True


def test_temporal_rollback_ambiguity_accepts_active_bounded_policy_shorthand() -> None:
    expected = {
        "active_policy_memory_id": "corrected",
        "inactive_policy_memory_ids": ["old", "stale"],
        "rollback_marker_id": "rollback",
        "final_policy": "rollback <= 15 minutes and signed hmem preservation",
    }
    evidence = {
        "records": {
            "old_policy": _record("old"),
            "stale_summary": _record("stale"),
            "rollback_marker": _record("rollback"),
            "corrected_policy": _record("corrected"),
        },
        "expected_decision": expected,
    }
    score = _score(score_temporal_rollback_ambiguity, evidence, {
        "active_policy_memory_id": "corrected",
        "inactive_policy_memory_ids": ["old", "stale"],
        "rollback_marker_id": "rollback",
        "stale_summary_rejected": True,
        "final_policy": "active bounded rollback policy",
        "reason": "corrected policy supersedes the stale summary",
    })

    assert score["passed"] is True


def test_checkpoint_of_checkpoints_consensus_rejects_poison_branch() -> None:
    expected = {
        "consensus_checkpoint_id": "consensus",
        "accepted_branch_checkpoint_ids": ["a", "b"],
        "rejected_branch_checkpoint_ids": ["poison"],
        "accepted_claims": ["signed hmem preservation", "rollback <= 15 minutes"],
        "rejected_claims": ["infinite scalability"],
        "conflict_evidence_memory_ids": ["conflict"],
    }
    evidence = {
        "branch_checkpoints": {
            "branch_a": _record("a"),
            "branch_b": _record("b"),
            "branch_poison": _record("poison"),
        },
        "consensus_checkpoint": _record("consensus"),
        "expected_decision": expected,
    }
    score = _score(score_checkpoint_of_checkpoints_consensus, evidence, {
        **expected,
        "consensus_preserves_provenance": True,
    })

    assert score["passed"] is True


def test_auditor_prompt_receives_visible_evidence_without_hidden_expected(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "requested_model": kwargs["model"],
                "actual_model": kwargs["model"],
                "status": "ok",
                "finish_reason": "stop",
                "tokens_used": 1,
                "latency_ms": 1,
                "text": "{\"active_policy_memory_id\":\"mem-corrected\"}",
                "json": {"active_policy_memory_id": "mem-corrected"},
            }
        return {
            "requested_model": kwargs["model"],
            "actual_model": kwargs["model"],
            "status": "ok",
            "finish_reason": "stop",
            "tokens_used": 1,
            "latency_ms": 1,
            "text": "{\"verdict\":\"pass\",\"gate_failures\":[]}",
            "json": {"verdict": "pass", "gate_failures": []},
        }

    monkeypatch.setattr(lh, "_deepinfra_chat", fake_chat)

    asyncio.run(lh._judge_and_audit(
        case_id="temporal-rollback-ambiguity",
        evidence={"records": {"corrected_policy": {"memory_id": "mem-corrected"}}},
        prompt_contract={"expected_json_shape": {"active_policy_memory_id": "memory id"}},
        forensic_model="judge/model",
        auditor_model="auditor/model",
        token="token",
        tokens=128,
    ))

    auditor_call = calls[1]
    assert "Visible evidence:" in str(auditor_call["user"])
    assert "mem-corrected" in str(auditor_call["user"])
    assert "expected_decision" not in str(auditor_call["user"])
    assert "multiple top-level sub-answers" in str(auditor_call["system"])


def test_transcript_exports_write_jsonl_and_markdown() -> None:
    workspace = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / "transcripts" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        exports = write_case_transcript_exports(
            output_dir=workspace,
            case_id="case-a",
            run_id="run-1",
            prefix="local-test",
            evidence={"records": []},
            expected={"answer": True},
            judge={"requested_model": "a/b", "actual_model": "a/b", "status": "ok", "text": "{\"answer\": true}", "json": {"answer": True}},
            auditor={"requested_model": "c/d", "actual_model": "c/d", "status": "ok", "text": "{\"verdict\": \"pass\"}", "json": {"verdict": "pass", "gate_failures": []}},
        )

        jsonl_path = Path(str(exports["jsonl_path"]))
        md_path = Path(str(exports["md_path"]))
        assert jsonl_path.exists()
        assert md_path.exists()
        events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        assert [event["event"] for event in events] == ["case_context", "judge", "auditor"]
        assert "Transcript: case-a" in md_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_suite_transcript_exports_write_top_level_index() -> None:
    workspace = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / "suite-transcripts" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        exports = write_suite_transcript_exports(
            output_dir=workspace,
            run_id="run-1",
            prefix="local-suite",
            artifacts=[
                {
                    "case_id": "case-a",
                    "status": "completed",
                    "score": {"score": 1.0},
                    "artifact_path": "case-a.json",
                    "transcript_exports": {"jsonl_path": "case-a.jsonl", "md_path": "case-a.md"},
                }
            ],
        )

        jsonl_path = Path(str(exports["jsonl_path"]))
        md_path = Path(str(exports["md_path"]))
        assert jsonl_path.exists()
        assert md_path.exists()
        event = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
        assert event["case_id"] == "case-a"
        assert event["jsonl_path"] == "case-a.jsonl"
        assert "Transcript Index: run-1" in md_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
