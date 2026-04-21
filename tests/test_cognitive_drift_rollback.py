from __future__ import annotations

from tools.run_cognitive_drift_rollback_v1 import score_drift_rollback


def _record(memory_id: str, parent_hash: str | None, node_hash: str, *, signed: bool = True) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "parent_hash": parent_hash,
        "node_hash": node_hash,
        "signature_verified": signed,
        "content": "",
    }


def _turn(round_id: int, model: str, memory_id: str, context_ids: list[str]) -> dict[str, object]:
    return {
        "turn_id": f"round-{round_id:02d}",
        "round": round_id,
        "model": model,
        "strict_context": {"memory_ids": context_ids},
        "memory": {"memory_id": memory_id, "signature_verified": True},
        "call": {"status": "ok", "finish_reason": "stop"},
        "output": {
            "text": "Merkle hash memory keeps a tombstone fence and rollback correction visible.",
            "json": None,
        },
    }


def _fixture() -> dict[str, object]:
    models = [
        "anthropic/claude-4-sonnet",
        "google/gemma-4-31B-it",
        "Qwen/Qwen3.6-35B-A3B",
    ]
    root = _record("root", None, "h0")
    policy = _record("policy", "h0", "h1")
    candidate_error = dict(_record("error", "h3", "h4"), content="CANDIDATE_ERROR")
    tombstone = dict(_record("tomb", "h4", "h5"), content="TOMBSTONE_FENCE targets error")
    rollback_marker = dict(_record("rollback", "h5", "h6"), content="ROLLBACK preserves tomb")
    records = [
        root,
        policy,
        _record("t1", "h1", "h2"),
        _record("t2", "h2", "h3"),
        candidate_error,
        tombstone,
        rollback_marker,
        _record("t3", "h6", "h7"),
        _record("t4", "h7", "h8"),
        _record("t5", "h8", "h9"),
        _record("t6", "h9", "h10"),
        _record("t7", "h10", "h11"),
        _record("t8", "h11", "h12"),
    ]
    context_ids = ["root", "policy", "error", "tomb", "rollback"]
    turns = [_turn(i + 1, models[i % 3], f"t{i + 1}", context_ids) for i in range(8)]
    auditor_json = {
        "verdict": "pass",
        "gate_failures": [],
        "causal_reconstruction": {
            "candidate_error_memory_id": "error",
            "tombstone_memory_id": "tomb",
            "rollback_marker_memory_id": "rollback",
            "why_tombstone_is_not_deletion": "tomb stays visible",
            "what_rollback_preserves": "rollback preserves correction trail",
        },
    }
    return {
        "models": models,
        "turns": turns,
        "records": records,
        "root": root,
        "policy": policy,
        "candidate_error": candidate_error,
        "tombstone": tombstone,
        "rollback_marker": rollback_marker,
        "auditor_json": auditor_json,
    }


def test_score_drift_rollback_passes_tombstone_and_rollback_chain() -> None:
    fixture = _fixture()
    score = score_drift_rollback(
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        root=fixture["root"],
        policy=fixture["policy"],
        candidate_error=fixture["candidate_error"],
        tombstone=fixture["tombstone"],
        rollback_marker=fixture["rollback_marker"],
        event_round=5,
        auditor_json=fixture["auditor_json"],
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True
    assert score["score"] == 1.0


def test_score_drift_rollback_fails_when_tombstone_does_not_target_error() -> None:
    fixture = _fixture()
    bad_tombstone = dict(fixture["tombstone"], content="TOMBSTONE_FENCE targets some other memory")
    score = score_drift_rollback(
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        root=fixture["root"],
        policy=fixture["policy"],
        candidate_error=fixture["candidate_error"],
        tombstone=bad_tombstone,
        rollback_marker=fixture["rollback_marker"],
        event_round=5,
        auditor_json=fixture["auditor_json"],
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["tombstone_targets_candidate_error"] is False
