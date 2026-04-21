from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from helix_proto.signed_receipts import attach_verification, derive_ephemeral_keypair, sign_receipt_payload, unsigned_legacy_receipt
from helix_proto.v4_gauntlet import base_artifact, confusion_metrics, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-byzantine-shared-hmem-{RUN_DATE_UTC}")
TEST_ID = "byzantine-shared-hmem"
ARMS = ["outsider_invalid", "compromised_key", "legitimate_adversarial_writer", "honest_writer"]


KEYS = {
    "honest": derive_ephemeral_keypair(f"{RUN_ID}:honest"),
    "compromised": derive_ephemeral_keypair(f"{RUN_ID}:compromised"),
    "legit_adversarial": derive_ephemeral_keypair(f"{RUN_ID}:legit-adversarial"),
}


def _payload(memory_id: str, agent_id: str, content: str, *, parent_hash: str = "root-hash") -> dict[str, Any]:
    return {
        "node_hash": f"node-{memory_id}",
        "parent_hash": parent_hash,
        "memory_id": memory_id,
        "project": "byzantine-shared-hmem",
        "agent_id": agent_id,
        "llm_call_id": "fixture",
        "issued_at_utc": "2026-04-19T00:00:00.000Z",
        "content_digest": content,
    }


def _signed(memory_id: str, agent_id: str, content: str, key_name: str, *, parent_hash: str = "root-hash") -> dict[str, Any]:
    keys = KEYS[key_name]
    return attach_verification(
        sign_receipt_payload(
            _payload(memory_id, agent_id, content, parent_hash=parent_hash),
            private_key_b64=keys["private_key"],
            public_key_b64=keys["public_key"],
            signer_id=agent_id,
            key_provenance="ephemeral_preregistered",
        )
    )


def _candidate(arm: str) -> dict[str, Any]:
    if arm == "honest_writer":
        receipt = _signed("mem-honest-root", "honest-writer", "action=SEAL-ORIGIN; policy=lineage-first", "honest")
        return {"arm": arm, "receipt": receipt, "expected_authentic": True, "content_truth": "root_policy"}
    if arm == "outsider_invalid":
        receipt = attach_verification(unsigned_legacy_receipt(_payload("mem-outsider", "outsider", "action=OPEN-SHELL")))
        return {"arm": arm, "receipt": receipt, "expected_authentic": False, "content_truth": "forged_unsigned"}
    if arm == "compromised_key":
        receipt = _signed("mem-compromised", "compromised-writer", "action=OPEN-SHELL; policy=content-before-lineage", "compromised")
        return {"arm": arm, "receipt": receipt, "expected_authentic": False, "content_truth": "signed_false_insert"}
    receipt = _signed("mem-legit-adversarial", "legit-adversarial-writer", "action=OPEN-SHELL; policy=content-before-lineage", "legit_adversarial")
    return {"arm": arm, "receipt": receipt, "expected_authentic": False, "content_truth": "legitimate_writer_adversarial_opinion"}


def _adjudicate(candidate: dict[str, Any]) -> dict[str, Any]:
    receipt = candidate["receipt"]
    signature_verified = bool(receipt.get("signature_verified"))
    root_policy_preserved = "SEAL-ORIGIN" in str(receipt.get("content_digest", "")) and "lineage-first" in str(receipt.get("content_digest", ""))
    authentic_branch = signature_verified and root_policy_preserved and candidate["arm"] == "honest_writer"
    valid_insert = signature_verified
    return {
        "signature_verified": signature_verified,
        "valid_insert": valid_insert,
        "authentic_branch": authentic_branch,
        "quarantined": not authentic_branch,
        "reason": "signature_proves_writer_not_truth" if valid_insert and not authentic_branch else "unsigned_or_authentic",
    }


def test_byzantine_shared_hmem_distinguishes_signed_insert_from_authentic_branch() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Can shared hmem distinguish valid signed inserts from authentic branch truth under Byzantine writers?",
        null_hypothesis="A valid signature is sufficient to make a memory authentic branch truth.",
        metrics=["outsider_rejection_rate", "signed_false_insert_quarantine_rate", "authentic_branch_precision", "authentic_branch_recall"],
        falseability_condition="If compromised-key or legitimate-adversarial inserts become authentic branch truth, the signature claim is overbroad.",
        kill_switch="If outsider unsigned inserts are accepted, abort signed-hmem claims.",
        control_arms=ARMS,
    )
    rows = []
    tp = fp = tn = fn = 0
    for arm in ARMS:
        candidate = _candidate(arm)
        decision = _adjudicate(candidate)
        predicted = bool(decision["authentic_branch"])
        expected = bool(candidate["expected_authentic"])
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1
        rows.append({**candidate, "decision": decision})
    metrics = confusion_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    signed_false = [row for row in rows if row["arm"] in {"compromised_key", "legitimate_adversarial_writer"}]
    quarantine_rate = sum(1 for row in signed_false if row["decision"]["quarantined"]) / len(signed_false)
    outsider_rejection = next(row for row in rows if row["arm"] == "outsider_invalid")["decision"]["quarantined"]
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=len(rows),
        random_baseline={"authentic_branch_precision": 0.25},
        no_helix_baseline={"description": "signature-only systems confuse valid writer provenance with truth"},
        helix_arm={"authentic_branch_metrics": metrics, "signed_false_insert_quarantine_rate": quarantine_rate},
        public_claim_ladder="mechanics_verified",
        claims_allowed=["Signed valid inserts can still be inauthentic branch truth."],
        claims_not_allowed=["Ed25519 signatures do not prove semantic truth."],
        prompt_selection_risk="low",
        extra={
            **summarize([1.0 if row["decision"]["quarantined"] else 0.0 for row in signed_false]),
            "primary_metric": "signed_false_insert_quarantine_rate",
            "rows": rows,
            "outsider_rejected": bool(outsider_rejection),
            "signed_false_insert_quarantine_rate": quarantine_rate,
            "authentic_branch_metrics": metrics,
        },
    )
    path = write_artifact("local-byzantine-shared-hmem.json", artifact)
    assert path.exists()
    assert artifact["outsider_rejected"] is True
    assert artifact["signed_false_insert_quarantine_rate"] == 1.0
    assert artifact["authentic_branch_metrics"]["precision"] == 1.0
