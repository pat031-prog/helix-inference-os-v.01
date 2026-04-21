from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from helix_proto.signed_receipts import attach_verification, canonical_json, verify_signed_receipt


REPLAY_VERSION = "helix-replay-v0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _looks_like_receipt(value: Any) -> bool:
    return isinstance(value, dict) and (
        "signature_alg" in value
        or "receipt_version" in value
        or "canonical_payload_sha256" in value
    )


def collect_receipts(payload: Any) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for value in _walk(payload):
        if _looks_like_receipt(value):
            receipts.append(dict(value))
    return receipts


def verify_artifact_payload(payload: dict[str, Any], *, artifact_path: Path | None = None) -> dict[str, Any]:
    receipts = collect_receipts(payload)
    verified = 0
    unsigned = 0
    failed = 0
    errors = []
    for index, receipt in enumerate(receipts):
        if receipt.get("receipt_version") == "unsigned_legacy" or not receipt.get("signature"):
            unsigned += 1
            continue
        result = verify_signed_receipt(receipt)
        if result.get("signature_verified"):
            verified += 1
        else:
            failed += 1
            errors.append({"index": index, "error": result.get("verification_error")})
    chain_receipts = [
        value
        for value in _walk(payload)
        if isinstance(value, dict) and value.get("status") in {"verified", "failed", "tombstone_preserved"} and "chain_len" in value
    ]
    chain_verified = sum(1 for item in chain_receipts if item.get("status") in {"verified", "tombstone_preserved"})
    artifact_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    report = {
        "artifact": "helix-artifact-replay-report",
        "replay_version": REPLAY_VERSION,
        "replay_mode": "verify-only",
        "artifact_path": str(artifact_path) if artifact_path else None,
        "artifact_sha256": sha256_bytes(artifact_bytes),
        "receipt_count": len(receipts),
        "signature_verified_count": verified,
        "unsigned_legacy_count": unsigned,
        "signature_failed_count": failed,
        "chain_receipt_count": len(chain_receipts),
        "chain_verified_count": chain_verified,
        "claim_boundaries": sorted(
            {
                str(value)
                for value in _walk(payload)
                if isinstance(value, str) and ("claim_boundary" in value.lower() or "not " in value.lower())
            }
        )[:20],
        "verification_errors": errors,
        "status": "failed" if failed else "verified",
        "claim_boundary": "Replay verifies recorded artifact evidence and receipts; it does not prove semantic truth or provider intent.",
    }
    if artifact_path is not None and artifact_path.exists():
        report["artifact_file_sha256"] = sha256_file(artifact_path)
    return report


def verify_artifact_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be a JSON object")
    return verify_artifact_payload(payload, artifact_path=path)


def cassette_digest(cassette: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(cassette).encode("utf-8")).hexdigest()


def replay_cassette(cassette: dict[str, Any], *, perturb_seed: str | None = None) -> dict[str, Any]:
    decisions = list(cassette.get("decisions") or [])
    if perturb_seed:
        offset = int(hashlib.sha256(perturb_seed.encode("utf-8")).hexdigest()[:8], 16) % max(len(decisions), 1)
        decisions = decisions[offset:] + decisions[:offset]
    drift = []
    for index, decision in enumerate(decisions):
        expected = decision.get("expected_decision")
        observed = decision.get("observed_decision", expected)
        if expected != observed:
            drift.append({"index": index, "expected": expected, "observed": observed})
    return {
        "artifact": "helix-cassette-replay-report",
        "replay_version": REPLAY_VERSION,
        "replay_mode": "cassette",
        "cassette_digest": cassette_digest(cassette),
        "decision_count": len(decisions),
        "decision_drift_count": len(drift),
        "decision_drift": drift,
        "perturb_seed": perturb_seed,
        "status": "drift_detected" if drift else "deterministic_replay",
        "claim_boundary": "Cassette replay checks deterministic recorded decisions, not live provider behavior.",
    }


def diff_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    changed = [
        {"key": key, "left": left.get(key), "right": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    ]
    return {
        "artifact": "helix-replay-diff-report",
        "replay_version": REPLAY_VERSION,
        "replay_mode": "diff",
        "changed_count": len(changed),
        "changed": changed[:100],
        "status": "drift_detected" if changed else "no_drift",
        "claim_boundary": "Diff replay reports observable drift between recorded reports; it does not assign provider intent.",
    }
