"""Signed receipt helpers for HeliX verification artifacts.

The signature proves provenance for a canonical receipt payload. It does not
prove that the signed content is true or that a branch is authentic.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


SIGNATURE_ALG = "ed25519"
RECEIPT_VERSION = "helix-signed-receipt-v1"
CANONICALIZATION = "helix-jcs-v0-rfc8785-compatible-no-floats"
ACCEPTED_PUBLIC_PROVENANCE = {"sigstore_rekor", "yubikey_or_tpm_pinned", "ephemeral_preregistered"}
KEY_PROVENANCE_MODES = (*sorted(ACCEPTED_PUBLIC_PROVENANCE), "local_self_signed")
STORED_SIGNATURE_FIELDS = {
    "signature_alg",
    "signature",
    "public_key",
    "canonical_payload_sha256",
    "receipt_version",
    "key_provenance",
    "attestation",
    "canonicalization",
}
RUNTIME_VERIFICATION_FIELDS = {
    "signature_verified",
    "verified_at_utc",
    "verifier_version",
    "verification_error",
    "public_claim_eligible",
}
SignatureEnforcementMode = Literal["permissive", "warn", "strict"]


class CanonicalizationError(ValueError):
    """Raised when a payload cannot be encoded under the receipt profile."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64decode(value: str) -> bytes:
    return base64.b64decode(str(value).encode("ascii"), validate=True)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalizationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def loads_strict_json(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)


def _normalize_for_canonical(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite float at {path}")
        raise CanonicalizationError(f"floats are not allowed in signed receipt payloads: {path}")
    if isinstance(value, list):
        return [_normalize_for_canonical(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [_normalize_for_canonical(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string object key at {path}: {key!r}")
            normalized[key] = _normalize_for_canonical(item, f"{path}.{key}")
        return normalized
    raise CanonicalizationError(f"unsupported canonical JSON type at {path}: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return canonical JSON for the HeliX signed-receipt profile.

    This intentionally rejects floats instead of pretending Python's JSON
    encoder is a full RFC 8785 implementation for all numeric edge cases.
    """

    normalized = _normalize_for_canonical(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_ephemeral_keypair(seed: str) -> dict[str, str]:
    private_bytes = hashlib.sha256(str(seed).encode("utf-8")).digest()
    private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    public_key = private_key.public_key()
    return {
        "private_key": b64encode(
            private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        ),
        "public_key": b64encode(
            public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ),
        "key_provenance": "ephemeral_preregistered",
    }


def generate_ed25519_keypair() -> dict[str, str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return {
        "private_key": b64encode(
            private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        ),
        "public_key": b64encode(
            public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ),
        "key_provenance": "local_self_signed",
    }


def key_id_for_public_key(public_key_b64: str) -> str:
    digest = hashlib.sha256(b64decode(public_key_b64)).hexdigest()
    return f"ed25519-{digest[:16]}"


def signable_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in STORED_SIGNATURE_FIELDS and key not in RUNTIME_VERIFICATION_FIELDS
    }


def sign_receipt_payload(
    payload: dict[str, Any],
    *,
    private_key_b64: str,
    signer_id: str,
    key_provenance: str,
    public_key_b64: str | None = None,
    attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if key_provenance not in KEY_PROVENANCE_MODES:
        raise ValueError(f"unsupported key_provenance={key_provenance!r}")
    signable = signable_payload(dict(payload))
    signable["signer_id"] = str(signer_id)
    digest = canonical_payload_sha256(signable)
    private_key = Ed25519PrivateKey.from_private_bytes(b64decode(private_key_b64))
    signature = private_key.sign(canonical_json(signable).encode("utf-8"))
    if public_key_b64 is None:
        public_key_b64 = b64encode(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
    return {
        **signable,
        "receipt_version": RECEIPT_VERSION,
        "canonicalization": CANONICALIZATION,
        "signature_alg": SIGNATURE_ALG,
        "signature": b64encode(signature),
        "public_key": public_key_b64,
        "signer_id": str(signer_id),
        "key_provenance": key_provenance,
        "canonical_payload_sha256": digest,
        "attestation": attestation,
    }


def verify_signed_receipt(receipt: dict[str, Any], *, verifier_version: str = "helix-receipt-verifier-v1") -> dict[str, Any]:
    try:
        if receipt.get("signature_alg") != SIGNATURE_ALG:
            raise ValueError("unsupported signature_alg")
        payload = signable_payload(receipt)
        digest = canonical_payload_sha256(payload)
        if receipt.get("canonical_payload_sha256") != digest:
            raise ValueError("canonical_payload_sha256 mismatch")
        public_key = Ed25519PublicKey.from_public_bytes(b64decode(str(receipt.get("public_key") or "")))
        public_key.verify(b64decode(str(receipt.get("signature") or "")), canonical_json(payload).encode("utf-8"))
        return {
            "signature_verified": True,
            "verified_at_utc": utc_now(),
            "verifier_version": verifier_version,
            "canonical_payload_sha256": digest,
            "key_provenance": receipt.get("key_provenance"),
            "public_claim_eligible": receipt.get("key_provenance") in ACCEPTED_PUBLIC_PROVENANCE,
        }
    except (InvalidSignature, ValueError, CanonicalizationError) as exc:
        return {
            "signature_verified": False,
            "verified_at_utc": utc_now(),
            "verifier_version": verifier_version,
            "verification_error": str(exc),
            "public_claim_eligible": False,
        }


def unsigned_legacy_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "receipt_version": "unsigned_legacy",
        "signature_alg": None,
        "signature": None,
        "public_key": None,
        "signer_id": None,
        "key_provenance": "unsigned_legacy",
        "canonical_payload_sha256": canonical_payload_sha256(payload),
        "attestation": None,
    }


def attach_verification(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("receipt_version") == "unsigned_legacy" or not receipt.get("signature"):
        return {
            **receipt,
            "signature_verified": False,
            "verified_at_utc": utc_now(),
            "verifier_version": "helix-receipt-verifier-v1",
            "verification_error": "unsigned_legacy",
        }
    return {**receipt, **verify_signed_receipt(receipt)}


def enforce_retrieval_signatures(
    items: Iterable[dict[str, Any]],
    *,
    mode: SignatureEnforcementMode = "strict",
) -> list[dict[str, Any]]:
    if mode not in {"permissive", "warn", "strict"}:
        raise ValueError(f"unsupported signature enforcement mode: {mode}")
    results: list[dict[str, Any]] = []
    for item in items:
        receipt = item.get("receipt") if isinstance(item.get("receipt"), dict) else item
        verified = bool(receipt.get("signature_verified"))
        annotated = dict(item)
        annotated["signature_enforcement_mode"] = mode
        annotated["signature_verified"] = verified
        if mode == "strict" and not verified:
            continue
        if mode == "warn" and not verified:
            annotated["signature_enforcement_warning"] = "unsigned_or_unverified_receipt_returned"
        results.append(annotated)
    return results
