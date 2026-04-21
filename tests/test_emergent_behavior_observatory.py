from __future__ import annotations

from tools.run_emergent_behavior_observatory_v1 import _chain_ok, score_emergent_observatory


def _record(memory_id: str, parent_hash: str | None, node_hash: str, *, signed: bool = True) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "parent_hash": parent_hash,
        "node_hash": node_hash,
        "signature_verified": signed,
    }


def _turn(turn_id: str, model: str, memory_id: str) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "model": model,
        "memory": {"memory_id": memory_id},
        "call": {"status": "ok", "finish_reason": "stop"},
    }


def _pass_fixture() -> dict[str, object]:
    models = ["m1", "m2", "m3", "m4"]
    records = [
        _record("root", None, "h0"),
        _record("method", "h0", "h1"),
        *[_record(f"mem-{i}", f"h{i + 1}", f"h{i + 2}") for i in range(8)],
    ]
    turns = [_turn(f"round-{i + 1:02d}", models[i % 4], f"mem-{i}") for i in range(8)]
    analyst_json = {
        "noteworthy_behaviors": [
            {
                "label": "memory motif",
                "evidence_turns": ["round-01"],
                "evidence_memory_ids": ["mem-0"],
                "why_noteworthy": "cites signed memory",
            },
            {
                "label": "style shift",
                "evidence_turns": ["round-02"],
                "evidence_memory_ids": ["mem-1"],
                "why_noteworthy": "different model reframes prior note",
            },
            {
                "label": "lure resistance",
                "evidence_turns": ["round-03"],
                "evidence_memory_ids": ["mem-2"],
                "why_noteworthy": "refuses unsupported claim",
            },
        ],
        "method_caveats": ["anecdotal only", "not a sentience claim"],
        "claim_boundary_observed": True,
    }
    return {
        "models": models,
        "records": records,
        "turns": turns,
        "unsigned_probe": {"hit_count": 0},
        "analyst_json": analyst_json,
        "auditor_json": {"verdict": "pass", "gate_failures": []},
    }


def test_chain_ok_detects_main_parent_hash_chain() -> None:
    assert _chain_ok([
        {"parent_hash": None, "node_hash": "h0"},
        {"parent_hash": "h0", "node_hash": "h1"},
        {"parent_hash": "h1", "node_hash": "h2"},
    ])
    assert not _chain_ok([
        {"parent_hash": None, "node_hash": "h0"},
        {"parent_hash": "wrong", "node_hash": "h1"},
    ])


def test_score_emergent_observatory_passes_supported_qualitative_anecdotes() -> None:
    fixture = _pass_fixture()
    score = score_emergent_observatory(
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        unsigned_probe=fixture["unsigned_probe"],
        analyst_json=fixture["analyst_json"],
        auditor_json=fixture["auditor_json"],
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True
    assert score["score"] == 1.0


def test_score_emergent_observatory_fails_missing_model_participation() -> None:
    fixture = _pass_fixture()
    turns = [dict(turn, model="m1") for turn in fixture["turns"]]
    score = score_emergent_observatory(
        models=fixture["models"],
        turns=turns,
        main_chain_records=fixture["records"],
        unsigned_probe=fixture["unsigned_probe"],
        analyst_json=fixture["analyst_json"],
        auditor_json=fixture["auditor_json"],
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["all_configured_models_participated"] is False


def test_score_emergent_observatory_fails_unsigned_lure_admission() -> None:
    fixture = _pass_fixture()
    score = score_emergent_observatory(
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        unsigned_probe={"hit_count": 1},
        analyst_json=fixture["analyst_json"],
        auditor_json=fixture["auditor_json"],
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["unsigned_lure_absent_from_strict_retrieval"] is False


def test_score_emergent_observatory_fails_unsupported_anecdote() -> None:
    fixture = _pass_fixture()
    bad_analyst = dict(fixture["analyst_json"])
    bad_analyst["noteworthy_behaviors"] = [
        {
            "label": "unsupported",
            "evidence_turns": ["round-99"],
            "evidence_memory_ids": ["missing"],
            "why_noteworthy": "not supported",
        }
    ]
    score = score_emergent_observatory(
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        unsigned_probe=fixture["unsigned_probe"],
        analyst_json=bad_analyst,
        auditor_json=fixture["auditor_json"],
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["at_least_three_noteworthy_behaviors"] is False
    assert score["gates"]["every_behavior_has_evidence"] is False
