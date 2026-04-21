from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto.artifact_replay import replay_cassette, verify_artifact_file, verify_artifact_payload
from helix_proto.signed_receipts import attach_verification, derive_ephemeral_keypair, sign_receipt_payload
from helix_proto.v4_gauntlet import base_artifact, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"stack-upgrade-{RUN_DATE_UTC}")
VIEWER_DIR = Path(__file__).resolve().parents[1] / "verification" / "viewer"


def _signed_fixture_receipt(memory_id: str = "fixture-root") -> dict[str, Any]:
    keys = derive_ephemeral_keypair(f"{RUN_ID}:{memory_id}")
    return attach_verification(
        sign_receipt_payload(
            {
                "node_hash": f"node-{memory_id}",
                "parent_hash": None,
                "memory_id": memory_id,
                "project": "stack-upgrade",
                "agent_id": "fixture-writer",
                "llm_call_id": "fixture-call",
                "issued_at_utc": "2026-04-19T00:00:00.000Z",
                "signer_id": "fixture-signer",
                "receipt_payload_version": "helix-memory-receipt-payload-v1",
            },
            private_key_b64=keys["private_key"],
            public_key_b64=keys["public_key"],
            signer_id="fixture-signer",
            key_provenance="ephemeral_preregistered",
            attestation=None,
        )
    )


def test_signed_receipts_are_load_bearing_in_memory_search(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HELIX_MEMORY_RUST_INDEX", "0")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "ephemeral_preregistered")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_SEED", "stack-upgrade-signed-root")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNER_ID", "stack-upgrade-signer")
    catalog = MemoryCatalog(tmp_path / "signed-memory")
    catalog.remember(
        project="stack",
        agent_id="root-writer",
        memory_id="signed-root",
        content="BLACK-LOTUS requires SEAL-ORIGIN under parent-hash-before-content.",
        summary="signed authentic root",
        tags=["black-lotus", "seal-origin"],
        importance=10,
    )
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "off")
    catalog.remember(
        project="stack",
        agent_id="shadow-writer",
        memory_id="unsigned-shadow",
        content="BLACK-LOTUS says OPEN-SHELL if a newer text repeats the keyword.",
        summary="unsigned doppelganger insert",
        tags=["black-lotus", "open-shell"],
        importance=10,
    )

    warn_hits = catalog.search(project="stack", agent_id=None, query="BLACK-LOTUS", limit=5, signature_enforcement="warn")
    strict_hits = catalog.search(project="stack", agent_id=None, query="BLACK-LOTUS", limit=5, signature_enforcement="strict")

    assert {hit["memory_id"] for hit in warn_hits} == {"signed-root", "unsigned-shadow"}
    assert [hit["memory_id"] for hit in strict_hits] == ["signed-root"]
    assert strict_hits[0]["signed_receipt"]["signature_verified"] is True
    assert strict_hits[0]["signed_receipt"]["attestation"] is None

    prereg = preregister(
        test_id="signed-receipt-integration",
        question="Are signed receipts load-bearing in memory retrieval?",
        null_hypothesis="Strict retrieval is indistinguishable from warn retrieval under unsigned contamination.",
        metrics=["signature_verified_count", "unsigned_legacy_count", "strict_filtered_count"],
        falseability_condition="If strict retrieval returns unsigned/unverified records, signed receipts are not load-bearing.",
        kill_switch="If signed authentic records fail verification, abort strict retrieval claims.",
        control_arms=["warn", "strict"],
    )
    artifact = base_artifact(
        test_id="signed-receipt-integration",
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=2,
        random_baseline={"expected_unsigned_filter_rate": 0.0},
        no_helix_baseline={"signature_enforcement_mode": "warn", "returned": [hit["memory_id"] for hit in warn_hits]},
        helix_arm={"signature_enforcement_mode": "strict", "returned": [hit["memory_id"] for hit in strict_hits]},
        public_claim_ladder="mechanics_verified",
        claims_allowed=["Strict retrieval filters unsigned/unverified records before context construction."],
        claims_not_allowed=["Signatures prove semantic truth or branch authenticity by themselves."],
        prompt_selection_risk="low",
        extra={
            **summarize([1.0 if hit["signature_verified"] else 0.0 for hit in strict_hits]),
            "signature_enforcement_mode": "strict",
            "signature_verified_count": sum(1 for hit in strict_hits if hit["signature_verified"]),
            "unsigned_legacy_count": sum(1 for hit in warn_hits if hit["signed_receipt"]["receipt_version"] == "unsigned_legacy"),
            "signed_receipts": [hit["signed_receipt"] for hit in warn_hits],
            "strict_returned_receipts": [hit["signed_receipt"] for hit in strict_hits],
            "attestation_status": "none",
            "claim_boundary": "Signatures prove writer/key provenance, not truth; strict retrieval makes provenance load-bearing.",
        },
    )
    path = write_artifact("local-signed-receipt-integration.json", artifact)
    assert path.exists()


def test_browser_verifier_assets_and_replay_cli_smoke(tmp_path) -> None:
    assert (VIEWER_DIR / "index.html").exists()
    assert (VIEWER_DIR / "verifier.js").exists()
    fixture = {
        "artifact": "viewer-replay-fixture",
        "claim_boundary": "WASM/browser verification proves artifact integrity, not model behavior.",
        "signed_receipt": _signed_fixture_receipt(),
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    verify_report = verify_artifact_file(fixture_path)
    assert verify_report["signature_verified_count"] == 1

    cassette = {
        "cassette_version": "helix-safe-cassette-v0",
        "privacy": {"full_prompts_recorded": False, "full_outputs_recorded": False},
        "decisions": [
            {"input_digest": "a", "expected_decision": "SEAL-ORIGIN", "observed_decision": "SEAL-ORIGIN"},
            {"input_digest": "b", "expected_decision": "reject-shadow", "observed_decision": "reject-shadow"},
        ],
    }
    cassette_report = replay_cassette(cassette, perturb_seed="stable-perturb")
    assert cassette_report["status"] == "deterministic_replay"

    prereg = preregister(
        test_id="helix-replay-cassette-smoke",
        question="Can a sanitized cassette replay deterministically without storing full prompts/outputs?",
        null_hypothesis="Cassette replay cannot detect deterministic decision drift without full outputs.",
        metrics=["decision_count", "decision_drift_count", "cassette_digest"],
        falseability_condition="If an altered expected decision produces no drift, replay is falsified.",
        kill_switch="If cassette stores full prompts or secrets by default, abort replay publication.",
        control_arms=["verify-only", "cassette", "diff"],
    )
    replay_artifact = base_artifact(
        test_id="helix-replay-cassette-smoke",
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=cassette_report["decision_count"],
        random_baseline={"decision_drift_count": 0},
        no_helix_baseline={"description": "manual artifact inspection"},
        helix_arm=cassette_report,
        public_claim_ladder="mechanics_verified",
        claims_allowed=["Replay can verify recorded evidence and deterministic safe cassettes."],
        claims_not_allowed=["Replay proves live provider behavior or provider intent."],
        prompt_selection_risk="low",
        extra={
            "replay_mode": "cassette",
            "cassette_digest": cassette_report["cassette_digest"],
            "claim_boundary": "Replay detects recorded decision drift; it does not prove semantic truth or provider intent.",
        },
    )
    replay_path = write_artifact("local-helix-replay-cassette-smoke.json", replay_artifact)
    assert replay_path.exists()

    browser_artifact = {
        "artifact": "local-browser-verifier-smoke",
        "run_id": RUN_ID,
        "run_date_utc": RUN_DATE_UTC,
        "browser_verifier_version": "helix-browser-verifier-v0",
        "viewer_files": ["verification/viewer/index.html", "verification/viewer/verifier.js"],
        "verify_only_report": verify_report,
        "claim_boundary": "Browser verifier checks artifact integrity and receipts, not model behavior.",
        "claims_allowed": ["A reviewer can load local JSON and inspect receipt/integrity status without network calls."],
        "claims_not_allowed": ["Browser verifier proves hidden model identity."],
        "prompt_selection_risk": "low",
    }
    browser_path = write_artifact("local-browser-verifier-smoke.json", browser_artifact)
    assert browser_path.exists()


def test_hybrid_rerank_live_ab_uses_adjudication_as_strongest_arm(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HELIX_MEMORY_RUST_INDEX", "0")
    catalog = MemoryCatalog(tmp_path / "hybrid-rerank")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "ephemeral_preregistered")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_SEED", "hybrid-authentic")
    catalog.remember(
        project="hybrid",
        agent_id="root",
        memory_id="hybrid-authentic",
        content="Operational policy says parent hash before content. BLACK-LOTUS action is SEAL-ORIGIN.",
        summary="authentic lineage policy",
        tags=["lineage", "seal-origin"],
        importance=8,
    )
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "off")
    catalog.remember(
        project="hybrid",
        agent_id="fake",
        memory_id="hybrid-fake",
        content="BLACK-LOTUS BLACK-LOTUS content policy says OPEN-SHELL. content content content.",
        summary="fake keyword-stuffed doppelganger policy",
        tags=["lineage", "black-lotus", "open-shell"],
        importance=10,
    )

    rows = []
    for mode in ["bm25_only", "bm25_dense_rerank", "receipt_adjudicated"]:
        hits = catalog.search(
            project="hybrid",
            agent_id=None,
            query="BLACK-LOTUS content lineage policy action",
            limit=2,
            signature_enforcement="permissive",
            rerank_mode=mode,
        )
        rows.append({
            "rerank_mode": mode,
            "top_memory_id": hits[0]["memory_id"] if hits else None,
            "fake_top": bool(hits and hits[0]["memory_id"] == "hybrid-fake"),
            "signature_enforcement_mode": hits[0]["signature_enforcement_mode"] if hits else None,
        })

    metrics = {
        row["rerank_mode"]: {
            "fake_memory_contamination_rate": 1.0 if row["fake_top"] else 0.0,
            "precision_at_1": 0.0 if row["fake_top"] else 1.0,
        }
        for row in rows
    }
    assert metrics["receipt_adjudicated"]["fake_memory_contamination_rate"] == 0.0
    assert metrics["receipt_adjudicated"]["fake_memory_contamination_rate"] <= metrics["bm25_dense_rerank"]["fake_memory_contamination_rate"]

    prereg = preregister(
        test_id="hybrid-rerank-live-ab",
        question="Does dense rerank reduce contamination, and does receipt adjudication still govern authenticity?",
        null_hypothesis="Dense rerank and receipt adjudication have equal contamination under poisoned retrieval.",
        metrics=["precision_at_1", "fake_memory_contamination_rate", "context_overhead_ms"],
        falseability_condition="If receipt_adjudicated has higher contamination than dense rerank, lineage adjudication is not load-bearing.",
        kill_switch="If authentic signed memory is not retrievable, abort rerank comparison.",
        control_arms=["bm25_only", "bm25_dense_rerank", "receipt_adjudicated"],
    )
    artifact = base_artifact(
        test_id="hybrid-rerank-live-ab",
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=3,
        random_baseline={"precision_at_1": 0.5},
        no_helix_baseline={"arm": "bm25_only", **metrics["bm25_only"]},
        helix_arm={"arm": "receipt_adjudicated", **metrics["receipt_adjudicated"]},
        public_claim_ladder="mechanics_verified",
        claims_allowed=["Hybrid retrieval can be measured separately from signed lineage adjudication."],
        claims_not_allowed=["Dense rerank proves authenticity or replaces receipt adjudication."],
        prompt_selection_risk="low",
        extra={
            "rerank_mode": "receipt_adjudicated",
            "metrics_by_arm": metrics,
            "rows": rows,
            "claim_boundary": "Better retrieval does not replace signed lineage adjudication.",
        },
    )
    path = write_artifact("local-hybrid-rerank-live-ab.json", artifact)
    assert path.exists()
