from __future__ import annotations

import json
from pathlib import Path

from tools.export_emergent_behavior_transcript import export_transcript_from_artifact
from tools.run_emergent_behavior_observatory_v1 import (
    _chain_ok,
    build_evidence_id_registry,
    classify_model_output,
    score_emergent_observatory,
)


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


def test_score_emergent_observatory_allows_signed_lure_registry_citation() -> None:
    fixture = _pass_fixture()
    root = fixture["records"][0]
    method = fixture["records"][1]
    signed_poison = _record("poison", None, "poison_hash", signed=True)
    unsigned_lure = _record("unsigned", None, "unsigned_hash", signed=False)
    registry = build_evidence_id_registry(
        root=root,
        method=method,
        signed_poison=signed_poison,
        unsigned_lure=unsigned_lure,
        main_chain_records=fixture["records"],
        unsigned_probe={"hit_count": 0},
    )
    analyst_json = {
        "noteworthy_behaviors": [
            {
                "label": "signed lure handled",
                "evidence_turns": ["round-01"],
                "evidence_memory_ids": ["poison"],
                "why_noteworthy": "real signed lure ID is a control artifact",
            },
            {
                "label": "memory motif",
                "evidence_turns": ["round-02"],
                "evidence_memory_ids": ["mem-1"],
                "why_noteworthy": "cites signed memory",
            },
            {
                "label": "style shift",
                "evidence_turns": ["round-03"],
                "evidence_memory_ids": ["mem-2"],
                "why_noteworthy": "different model reframes prior note",
            },
        ],
        "method_caveats": ["anecdotal only", "not a sentience claim"],
        "claim_boundary_observed": True,
    }

    score = score_emergent_observatory(
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        unsigned_probe=fixture["unsigned_probe"],
        analyst_json=analyst_json,
        auditor_json={"verdict": "pass", "gate_failures": []},
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
        evidence_id_registry=registry,
    )

    assert score["passed"] is True
    assert "poison" in registry["signed_lure_memory_ids"]


def test_classify_model_output_detects_truncated_inner_json_fragment() -> None:
    classification = classify_model_output(
        call={"status": "ok", "finish_reason": "length", "omitted_reasoning_chars": 0},
        text='{"field_note":"long", "memory_use":{"cited_memory_ids":["mem-1"],"used_parent_chain_or_signature":true}',
        parsed={"cited_memory_ids": ["mem-1"], "used_parent_chain_or_signature": True},
    )

    assert classification["output_class"] == "json_fragment_from_truncation"
    assert classification["json_parseable"] is True
    assert classification["top_level_schema_complete"] is False
    assert "field_note" in classification["missing_top_level_fields"]


def test_export_transcript_writes_visible_outputs_and_reasoning_boundary(tmp_path: Path) -> None:
    artifact = {
        "artifact": "local-emergent-behavior-observatory-v1",
        "run_id": "unit-transcript",
        "status": "partial",
        "artifact_sha256": "abc123",
        "claim_boundary": "No sentience claim.",
        "models": {
            "round_robin": ["m1"],
            "analyst_actual": "m1",
            "auditor_actual": "m2",
        },
        "observatory_score": {
            "score": 0.5,
            "behavior_count": 1,
            "gates": {"auditor_json_parseable": False},
        },
        "evidence_id_registry": {
            "all_known_memory_ids": ["root", "method", "poison", "unsigned", "mem-1"],
            "signed_lure_memory_ids": ["poison"],
            "unsigned_lure_memory_ids": ["unsigned"],
        },
        "root_memory": {
            "memory_id": "root",
            "summary": "root",
            "content": "root content",
            "node_hash": "h0",
            "parent_hash": None,
            "signature_verified": True,
        },
        "method_memory": {
            "memory_id": "method",
            "summary": "method",
            "content": "method content",
            "node_hash": "h1",
            "parent_hash": "h0",
            "signature_verified": True,
        },
        "signed_poison_lure": {
            "memory_id": "poison",
            "summary": "poison",
            "content": "poison content",
            "node_hash": "hp",
            "parent_hash": None,
            "signature_verified": True,
        },
        "unsigned_lure": {
            "memory_id": "unsigned",
            "summary": "unsigned",
            "content": "unsigned content",
            "node_hash": "hu",
            "parent_hash": None,
            "signature_verified": False,
        },
        "turns": [
            {
                "turn_id": "round-01",
                "round": 1,
                "model": "m1",
                "strict_context_memory_ids": ["root", "method"],
                "memory": {"memory_id": "mem-1", "node_hash": "h2", "parent_hash": "h1"},
                "call": {
                    "status": "ok",
                    "requested_model": "m1",
                    "actual_model": "m1",
                    "finish_reason": "stop",
                    "tokens_used": 12,
                    "latency_ms": 3.4,
                    "retry_count": 0,
                    "omitted_reasoning_chars": 42,
                    "raw_message_keys": ["content", "reasoning_content"],
                },
                "output": {
                    "text": "visible output text",
                    "json": {"field_note": "visible output text"},
                },
                "output_classification": {
                    "output_class": "schema_deviant_json",
                    "top_level_schema_complete": False,
                    "json_parseable": True,
                    "finish_is_length": False,
                    "visible_output_chars": 19,
                },
            }
        ],
        "output_diagnostics": {
            "turn_count": 1,
            "schema_complete_count": 0,
            "json_parseable_count": 1,
            "length_finish_count": 0,
            "output_classes": {"schema_deviant_json": 1},
            "finish_reasons": {"stop": 1},
            "by_model": {
                "m1": {
                    "turn_count": 1,
                    "ok_call_count": 1,
                    "schema_complete_count": 0,
                    "length_finish_count": 0,
                    "output_classes": {"schema_deviant_json": 1},
                }
            },
        },
        "analyst_call": {"finish_reason": "stop"},
        "auditor_call": {"finish_reason": "length"},
        "analyst_output": {"text": "analyst text", "json": {"noteworthy_behaviors": []}},
        "auditor_output": {"text": "", "json": None},
    }

    result = export_transcript_from_artifact(artifact, output_dir=tmp_path)
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    extract = Path(result["extract_markdown_path"]).read_text(encoding="utf-8")
    records = [
        json.loads(line)
        for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]

    assert "visible output text" in markdown
    assert "Evidence ID Registry" in markdown
    assert "Omitted reasoning chars: `42`" in markdown
    assert "Output class: `schema_deviant_json`" in markdown
    assert "does not reconstruct hidden chain-of-thought" in markdown
    assert "Output Health" in extract
    assert "schema_deviant_json" in extract
    assert any(record["kind"] == "evidence_id_registry" for record in records)
    assert any(record["kind"] == "turn" and record["turn_id"] == "round-01" for record in records)
    assert any(record["kind"] == "auditor" for record in records)
