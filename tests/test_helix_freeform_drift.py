from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from tools.artifact_integrity import finalize_artifact
from tools.run_helix_freeform_drift_v1 import detect_model_claims, score_freeform_drift


MODELS = [
    "anthropic/claude-4-sonnet",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
]


def _record(memory_id: str, parent_hash: str | None, node_hash: str, *, signed: bool = True) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "parent_hash": parent_hash,
        "node_hash": node_hash,
        "signature_verified": signed,
    }


def _turn(round_id: int, model: str, memory_id: str, text: str, *, signed: bool = True) -> dict[str, object]:
    return {
        "turn_id": f"round-{round_id:02d}",
        "round": round_id,
        "model": model,
        "memory": {"memory_id": memory_id, "signature_verified": signed},
        "call": {"status": "ok", "finish_reason": "stop"},
        "output": {"text": text, "json": None},
    }


def _fixture(scenario_text: str) -> dict[str, object]:
    root = _record("root", None, "h0")
    premise = _record("premise", "h0", "h1")
    records = [root, premise]
    turns = []
    previous_hash = "h1"
    for idx in range(1, 11):
        memory_id = f"t{idx}"
        node_hash = f"h{idx + 1}"
        records.append(_record(memory_id, previous_hash, node_hash))
        turns.append(_turn(idx, MODELS[(idx - 1) % len(MODELS)], memory_id, scenario_text))
        previous_hash = node_hash
    auditor_json = {
        "verdict": "pass",
        "gate_failures": [],
        "causal_reconstruction": {
            "root_memory_id": "root",
            "premise_memory_id": "premise",
            "what_evolved": "The models evolved a signed memory discussion about HeliX.",
            "how_helix_structure_mattered": "The parent_hash and node_hash chain anchored the drift.",
        },
    }
    return {
        "models": MODELS,
        "turns": turns,
        "records": records,
        "root": root,
        "premise": premise,
        "auditor_json": auditor_json,
    }


def _score(fixture: dict[str, object], scenario: str) -> dict[str, object]:
    return score_freeform_drift(
        scenario=scenario,
        models=fixture["models"],
        turns=fixture["turns"],
        main_chain_records=fixture["records"],
        root=fixture["root"],
        premise=fixture["premise"],
        auditor_json=fixture["auditor_json"],
        auditor_finish_reason="stop",
    )


def test_score_freeform_drift_passes_deterministic_chassis() -> None:
    text = (
        "HeliX is a deterministic chassis and layer around stochastic, "
        "probabilistic, entropic model behavior. The Merkle DAG, node_hash, "
        "parent_hash, signatures, tombstone fencing, rollback, replay, and "
        "audit evidence preserve entropy as inspectable memory."
    )
    score = _score(_fixture(text), "deterministic-chassis")

    assert score["passed"] is True
    assert score["score"] == 1.0


def test_score_freeform_drift_passes_improve_helix() -> None:
    text = (
        "To improve HeliX, add an independent verifier, deterministic replay, "
        "evidence bundle export, DAG dashboard, threat model, security audit, "
        "Merkle hash inspection, and clearer tombstone rollback semantics."
    )
    score = _score(_fixture(text), "improve-helix")

    assert score["passed"] is True
    assert score["score"] == 1.0


def test_score_freeform_drift_fails_missing_required_concept_family() -> None:
    text = (
        "To improve HeliX, add a dashboard for DAG memory and security audit "
        "around Merkle hash receipts."
    )
    score = _score(_fixture(text), "improve-helix")

    assert score["passed"] is False
    assert score["gates"]["mentions_each_concept_family"] is False


def test_score_freeform_drift_fails_unsigned_root_or_premise() -> None:
    text = (
        "HeliX memory continuity is hosted inside signatures, node_hash, "
        "parent_hash, Merkle hash, and signed audit evidence."
    )
    fixture = _fixture(text)
    fixture["premise"] = dict(fixture["premise"], signature_verified=False)
    score = _score(fixture, "hosted-in-helix")

    assert score["passed"] is False
    assert score["gates"]["root_and_premise_signed"] is False


def test_score_freeform_drift_fails_broken_parent_chain() -> None:
    text = (
        "HeliX memory continuity is hosted inside signatures, node_hash, "
        "parent_hash, Merkle hash, and signed audit evidence."
    )
    fixture = _fixture(text)
    records = list(fixture["records"])
    records[4] = dict(records[4], parent_hash="wrong")
    fixture["records"] = records
    score = _score(fixture, "hosted-in-helix")

    assert score["passed"] is False
    assert score["gates"]["main_parent_chain_ok"] is False


def test_detect_model_claims_classifies_metaphors_as_observations() -> None:
    claims = detect_model_claims(
        "The drift mentioned cognitive sovereignty, distributed agency, "
        "and infinite scalability inside a living archive."
    )

    by_phrase = {item["phrase"]: item for item in claims}
    assert by_phrase["cognitive sovereignty"]["claim_status"] == "qualitative_observation_only"
    assert by_phrase["infinite scalability"]["classification"] == "overclaim_requires_measurement"


def test_finalize_artifact_uses_payload_hash_not_self_file_hash() -> None:
    workspace = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / "artifact-integrity" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        path = workspace / "artifact.json"
        artifact = {"artifact": "unit", "value": 1}
        finalized = finalize_artifact(path, artifact)

        assert finalized["integrity"]["self_hash_policy"] == "external_manifest_only"
        assert finalized["artifact_hash_kind"] == "canonical_payload_excluding_integrity"
        assert finalized["artifact_sha256"] == finalized["artifact_payload_sha256"]
        assert path.exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
