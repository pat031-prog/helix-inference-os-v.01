from __future__ import annotations

import copy
import os
from datetime import datetime, timezone
from typing import Any

import pytest

from helix_proto.signed_receipts import (
    CanonicalizationError,
    attach_verification,
    canonical_json,
    canonical_payload_sha256,
    derive_ephemeral_keypair,
    enforce_retrieval_signatures,
    generate_ed25519_keypair,
    key_id_for_public_key,
    loads_strict_json,
    sign_receipt_payload,
    unsigned_legacy_receipt,
    verify_signed_receipt,
)
from helix_proto.v4_gauntlet import base_artifact, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-signed-receipts-{RUN_DATE_UTC}")
TEST_ID = "signed-receipt-provenance"


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "node_hash": "node-abc123",
        "parent_hash": "parent-def456",
        "memory_id": "memory-root",
        "project": "signed-receipts",
        "agent_id": "writer-a",
        "llm_call_id": "call-001",
        "issued_at_utc": "2026-04-19T00:00:00.000Z",
        "receipt_version": "helix-signable-payload-v1",
    }
    payload.update(overrides)
    return payload


def _signed(key_provenance: str = "ephemeral_preregistered") -> dict[str, Any]:
    keys = derive_ephemeral_keypair(f"{RUN_ID}:{TEST_ID}:{key_provenance}")
    return sign_receipt_payload(
        _payload(),
        private_key_b64=keys["private_key"],
        public_key_b64=keys["public_key"],
        signer_id="test-signer",
        key_provenance=key_provenance,
        attestation=None,
    )


def test_signed_receipt_verifies_and_runtime_fields_are_not_signed() -> None:
    receipt = _signed()
    verified = verify_signed_receipt(receipt)
    assert verified["signature_verified"] is True
    assert verified["public_claim_eligible"] is True
    assert "signature_verified" not in receipt
    forged_runtime = {**receipt, "signature_verified": True}
    assert verify_signed_receipt(forged_runtime)["signature_verified"] is True


def test_signed_receipt_tamper_fails() -> None:
    receipt = _signed()
    tampered = copy.deepcopy(receipt)
    tampered["memory_id"] = "memory-evil"
    verified = verify_signed_receipt(tampered)
    assert verified["signature_verified"] is False
    assert "canonical_payload_sha256 mismatch" in verified["verification_error"]


def test_signed_receipt_provenance_tamper_fails() -> None:
    receipt = _signed("local_self_signed")
    tampered = copy.deepcopy(receipt)
    tampered["key_provenance"] = "sigstore_rekor"
    tampered["attestation"] = {
        "provider": "sigstore_rekor",
        "evidence_digest": "sha256:forged",
        "verified": True,
    }

    verified = verify_signed_receipt(tampered)

    assert verified["signature_verified"] is False
    assert verified["public_claim_eligible"] is False
    assert "canonical_payload_sha256 mismatch" in verified["verification_error"]


def test_local_self_signed_is_mechanics_only() -> None:
    receipt = _signed("local_self_signed")
    verified = verify_signed_receipt(receipt)
    assert verified["signature_verified"] is True
    assert verified["public_claim_eligible"] is False


def test_local_key_id_is_stable_for_generated_workspace_key() -> None:
    keys = generate_ed25519_keypair()
    key_id = key_id_for_public_key(keys["public_key"])
    receipt = sign_receipt_payload(
        {**_payload(), "signing_key_id": key_id},
        private_key_b64=keys["private_key"],
        public_key_b64=keys["public_key"],
        signer_id="workspace-local",
        key_provenance="local_self_signed",
    )

    assert key_id.startswith("ed25519-")
    assert key_id_for_public_key(keys["public_key"]) == key_id
    assert verify_signed_receipt(receipt)["signature_verified"] is True


def test_canonical_payload_rejects_float_and_duplicate_keys() -> None:
    assert canonical_json({"b": "ñ", "a": ["x\u0000y"]}) == '{"a":["x\\u0000y"],"b":"ñ"}'
    assert canonical_payload_sha256({"b": 2, "a": 1}) == canonical_payload_sha256({"a": 1, "b": 2})
    with pytest.raises(CanonicalizationError):
        canonical_json({"bad": 1.0})
    with pytest.raises(CanonicalizationError):
        loads_strict_json('{"a":1,"a":2}')


def test_retrieval_signature_enforcement_modes() -> None:
    signed = attach_verification(_signed())
    legacy = attach_verification(unsigned_legacy_receipt(_payload(memory_id="legacy")))
    items = [{"memory_id": "signed", "receipt": signed}, {"memory_id": "legacy", "receipt": legacy}]
    assert [item["memory_id"] for item in enforce_retrieval_signatures(items, mode="permissive")] == ["signed", "legacy"]
    warned = enforce_retrieval_signatures(items, mode="warn")
    assert warned[1]["signature_enforcement_warning"] == "unsigned_or_unverified_receipt_returned"
    assert [item["memory_id"] for item in enforce_retrieval_signatures(items, mode="strict")] == ["signed"]


def test_signed_receipt_gauntlet_artifact_contract() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Do signed receipts distinguish provenance from branch truth?",
        null_hypothesis="Signed receipts verify provenance but do not by themselves assert content truth.",
        metrics=["signature_verified", "public_claim_eligible", "strict_enforcement_kept"],
        falseability_condition="If tampered payload verifies, the signed receipt scheme is falsified.",
        kill_switch="If runtime signature_verified is accepted as stored truth, abort signed receipt claims.",
        control_arms=["unsigned_legacy", "local_self_signed", "ephemeral_preregistered"],
    )
    signed = attach_verification(_signed())
    local = attach_verification(_signed("local_self_signed"))
    legacy = attach_verification(unsigned_legacy_receipt(_payload(memory_id="legacy")))
    strict = enforce_retrieval_signatures(
        [
            {"memory_id": "signed", "receipt": signed},
            {"memory_id": "local", "receipt": local},
            {"memory_id": "legacy", "receipt": legacy},
        ],
        mode="strict",
    )
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=3,
        random_baseline={"signature_verified_rate": 0.0},
        no_helix_baseline={"description": "unsigned evidence has no key provenance"},
        helix_arm={"strict_enforcement_kept": [item["memory_id"] for item in strict]},
        public_claim_ladder="mechanics_verified",
        claims_allowed=["Signed receipts prove writer/key provenance for canonical payloads."],
        claims_not_allowed=["Signatures do not prove truth, authenticity of branch, or provider intent."],
        prompt_selection_risk="low",
        extra={
            **summarize([1.0 if item["receipt"]["signature_verified"] else 0.0 for item in strict]),
            "primary_metric": "signature_verified",
            "stored_vs_runtime_boundary": "signature_verified is verifier output and is never signed payload truth.",
            "key_provenance_modes": ["sigstore_rekor", "yubikey_or_tpm_pinned", "ephemeral_preregistered", "local_self_signed"],
            "strict_enforcement_kept": [item["memory_id"] for item in strict],
            "public_claim_eligible": signed["public_claim_eligible"],
        },
    )
    path = write_artifact("local-signed-receipt-gauntlet.json", artifact)
    assert path.exists()
    assert artifact["public_claim_eligible"] is True
    assert artifact["strict_enforcement_kept"] == ["signed", "local"]
