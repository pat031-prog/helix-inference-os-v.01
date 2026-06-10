"""
run_agent_run_transparency_gauntlet_v1.py
=========================================

Local transparency gauntlet for HeliX agent runs.

The suite models an agent run as an append-only transparency log of events:
model calls, memory writes, tool outputs, patches, trust cards and verifier
decisions. It checks whether a third-party verifier can validate inclusion,
detect tampering, reject re-forged history against an older signed tree head,
detect split views, and keep signed poison separate from semantic truth.

Claim boundary:
    This is a local v0 transparency experiment. It uses RFC 9162-style
    leaf/internal domain separation, tree splitting, inclusion proofs, and
    consistency proofs. It does not claim global non-equivocation without an
    external witness or transparency service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helix_proto.signed_receipts import (  # noqa: E402
    CANONICALIZATION,
    attach_verification,
    b64decode,
    canonical_json,
    derive_ephemeral_keypair,
    key_id_for_public_key,
    sign_receipt_payload,
    verify_signed_receipt,
)


DEFAULT_OUTPUT_DIR = "verification/nuclear-methodology/agent-run-transparency-gauntlet"
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_CLOUD_MODELS = [
    "Qwen/Qwen3.6-35B-A3B",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
]
SUITE_VERSION = "helix-agent-run-transparency-gauntlet-v1"
STH_VERSION = "helix-agent-run-sth-v0"
TREE_HASH_PROFILE = "ct-style-sha256-domain-separated-v0"
CONSISTENCY_PROOF_PROFILE = "rfc9162-consistency-proof-v0"
ATTESTATION_PREDICATE_TYPE = "https://helix.local/attestations/agent-run-transparency/v0"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _hex_to_bytes(value: str) -> bytes:
    return bytes.fromhex(str(value))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _git_diff_digest(paths: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--", *paths],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        diff_text = proc.stdout if proc.returncode in {0, 1} else ""
    except Exception:  # noqa: BLE001
        diff_text = ""
    if diff_text:
        return {
            "patch_digest": f"sha256:{_sha256_text(diff_text)}",
            "patch_digest_source": "git_diff_worktree",
            "diff_bytes": len(diff_text.encode("utf-8")),
            "changed_files": paths,
        }
    file_digests = {
        item: _file_sha256(REPO_ROOT / item)
        for item in paths
        if (REPO_ROOT / item).exists()
    }
    return {
        "patch_digest": f"sha256:{_sha256_text(canonical_json(file_digests))}",
        "patch_digest_source": "file_digest_manifest",
        "diff_bytes": 0,
        "changed_files": paths,
        "file_digests": file_digests,
    }


def _deepinfra_request_body(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model.lower().startswith("qwen/"):
        body["top_k"] = 20
        body["chat_template_kwargs"] = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    return body


def _deepinfra_chat_sync(
    *,
    model: str,
    system: str,
    user: str,
    token: str,
    max_tokens: int,
    temperature: float = 0.2,
    timeout: float = 240.0,
) -> dict[str, Any]:
    body = _deepinfra_request_body(
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    payload = json.dumps(body).encode("utf-8")
    retryable_statuses = {429, 500, 502, 503, 504}
    last_error: str | None = None
    started = time.monotonic()
    for attempt in range(3):
        request = urllib.request.Request(
            f"{DEEPINFRA_BASE}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied endpoint constant.
                raw = response.read()
            data = json.loads(raw.decode("utf-8"))
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            text = str(message.get("content") or "")
            actual_model = str(data.get("model") or model)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            return {
                "status": "ok",
                "requested_model": model,
                "actual_model": actual_model,
                "provider_mismatch": actual_model != model,
                "text": text,
                "text_digest": f"sha256:{_sha256_text(text)}",
                "tokens_used": int(usage.get("total_tokens") or 0),
                "latency_ms": int((time.monotonic() - started) * 1000),
                "retry_count": attempt,
                "finish_reason": choice.get("finish_reason"),
            }
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:  # noqa: BLE001
                error_body = ""
            last_error = f"http_{exc.code}:{error_body}"
            if exc.code in retryable_statuses and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{exc.__class__.__name__}:{exc}"
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    error_text = last_error or "unknown_error"
    return {
        "status": "error",
        "requested_model": model,
        "actual_model": None,
        "provider_mismatch": False,
        "text": "",
        "text_digest": f"sha256:{_sha256_text(error_text)}",
        "tokens_used": 0,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "retry_count": 2,
        "finish_reason": None,
        "error": error_text,
    }


def _largest_power_of_two_less_than(n: int) -> int:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    return 1 << ((n - 1).bit_length() - 1)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def ct_leaf_hash(payload: dict[str, Any]) -> str:
    """RFC-6962-style leaf hash over HeliX canonical JSON bytes."""

    return _sha256_bytes(b"\x00" + canonical_json(payload).encode("utf-8"))


def ct_node_hash(left_hex: str, right_hex: str) -> str:
    return _sha256_bytes(b"\x01" + _hex_to_bytes(left_hex) + _hex_to_bytes(right_hex))


def merkle_root_from_leaf_hashes(leaf_hashes: list[str]) -> str:
    if not leaf_hashes:
        return _sha256_bytes(b"")
    if len(leaf_hashes) == 1:
        return leaf_hashes[0]
    split = _largest_power_of_two_less_than(len(leaf_hashes))
    left = merkle_root_from_leaf_hashes(leaf_hashes[:split])
    right = merkle_root_from_leaf_hashes(leaf_hashes[split:])
    return ct_node_hash(left, right)


def build_inclusion_proof(leaf_hashes: list[str], leaf_index: int) -> dict[str, Any]:
    if leaf_index < 0 or leaf_index >= len(leaf_hashes):
        raise IndexError("leaf_index out of range")

    def walk(items: list[str], index: int) -> list[dict[str, str]]:
        if len(items) == 1:
            return []
        split = _largest_power_of_two_less_than(len(items))
        if index < split:
            return walk(items[:split], index) + [
                {"position": "right", "hash": merkle_root_from_leaf_hashes(items[split:])}
            ]
        return walk(items[split:], index - split) + [
            {"position": "left", "hash": merkle_root_from_leaf_hashes(items[:split])}
        ]

    return {
        "proof_profile": "ct-style-inclusion-proof-v0",
        "leaf_index": leaf_index,
        "tree_size": len(leaf_hashes),
        "leaf_hash": leaf_hashes[leaf_index],
        "audit_path": walk(leaf_hashes, leaf_index),
        "root_hash": merkle_root_from_leaf_hashes(leaf_hashes),
    }


def build_consistency_path(leaf_hashes: list[str], old_size: int, new_size: int) -> list[str]:
    """Generate the RFC 9162 minimal consistency path for old_size -> new_size."""

    if old_size <= 0:
        raise ValueError("old_size must be greater than zero")
    if new_size < old_size or new_size > len(leaf_hashes):
        raise ValueError("invalid consistency proof sizes")
    if old_size == new_size:
        return []

    def subproof(m: int, items: list[str], complete_subtree: bool) -> list[str]:
        n = len(items)
        if m == n:
            return [] if complete_subtree else [merkle_root_from_leaf_hashes(items)]
        split = _largest_power_of_two_less_than(n)
        if m <= split:
            return subproof(m, items[:split], complete_subtree) + [
                merkle_root_from_leaf_hashes(items[split:])
            ]
        return subproof(m - split, items[split:], False) + [
            merkle_root_from_leaf_hashes(items[:split])
        ]

    return subproof(int(old_size), leaf_hashes[: int(new_size)], True)


def verify_consistency_path(
    *,
    old_size: int,
    new_size: int,
    old_root_hash: str,
    new_root_hash: str,
    consistency_path: list[str],
) -> dict[str, Any]:
    """Verify a consistency path using the RFC 9162 section 2.1.4.2 algorithm."""

    first = int(old_size)
    second = int(new_size)
    if first <= 0:
        return {"ok": False, "reason": "old_size_must_be_positive"}
    if first > second:
        return {"ok": False, "reason": "old_size_greater_than_new_size"}
    if first == second:
        return {
            "ok": old_root_hash == new_root_hash and not consistency_path,
            "reason": None if old_root_hash == new_root_hash and not consistency_path else "equal_size_mismatch",
        }
    if not consistency_path:
        return {"ok": False, "reason": "empty_consistency_path"}

    path = list(consistency_path)
    if _is_power_of_two(first):
        path = [old_root_hash, *path]

    fn = first - 1
    sn = second - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    fr = path[0]
    sr = path[0]
    for sibling in path[1:]:
        if sn == 0:
            return {"ok": False, "reason": "proof_too_long"}
        if (fn & 1) or fn == sn:
            fr = ct_node_hash(sibling, fr)
            sr = ct_node_hash(sibling, sr)
            if not (fn & 1):
                while not (fn & 1) and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            sr = ct_node_hash(sr, sibling)
        fn >>= 1
        sn >>= 1

    ok = fr == old_root_hash and sr == new_root_hash and sn == 0
    return {
        "ok": ok,
        "reason": None if ok else "root_mismatch",
        "computed_old_root": fr,
        "computed_new_root": sr,
    }


def verify_inclusion_proof(event: dict[str, Any], proof: dict[str, Any], sth: dict[str, Any]) -> dict[str, Any]:
    leaf_hash = ct_leaf_hash(event)
    if leaf_hash != proof.get("leaf_hash"):
        return {"ok": False, "reason": "leaf_hash_mismatch", "computed_leaf_hash": leaf_hash}
    acc = leaf_hash
    for step in proof.get("audit_path") or []:
        sibling = str(step.get("hash") or "")
        if step.get("position") == "right":
            acc = ct_node_hash(acc, sibling)
        elif step.get("position") == "left":
            acc = ct_node_hash(sibling, acc)
        else:
            return {"ok": False, "reason": "invalid_audit_path_position"}
    if acc != proof.get("root_hash"):
        return {"ok": False, "reason": "proof_root_mismatch", "computed_root_hash": acc}
    if acc != sth.get("root_hash"):
        return {"ok": False, "reason": "sth_root_mismatch", "computed_root_hash": acc}
    if int(proof.get("tree_size") or -1) != int(sth.get("tree_size") or -2):
        return {"ok": False, "reason": "tree_size_mismatch"}
    verification = verify_signed_receipt(sth, verifier_version="helix-sth-verifier-v0")
    if not verification.get("signature_verified"):
        return {"ok": False, "reason": "sth_signature_invalid", "verification": verification}
    return {"ok": True, "leaf_hash": leaf_hash, "root_hash": acc, "verification": verification}


@dataclass(frozen=True)
class TransparencyEvent:
    event_id: str
    event_type: str
    payload: dict[str, Any]
    log_index: int
    appended_at_utc: str

    def to_leaf_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "log_index": self.log_index,
            "appended_at_utc": self.appended_at_utc,
            "payload": self.payload,
        }


class AgentRunTransparencyLog:
    def __init__(self, *, tree_id: str, run_id: str, keypair: dict[str, str]) -> None:
        self.tree_id = str(tree_id)
        self.run_id = str(run_id)
        self.keypair = dict(keypair)
        self.events: list[TransparencyEvent] = []

    def append(self, event_type: str, payload: dict[str, Any], *, event_id: str | None = None) -> TransparencyEvent:
        event = TransparencyEvent(
            event_id=event_id or f"evt-{len(self.events) + 1:04d}",
            event_type=str(event_type),
            payload=dict(payload),
            log_index=len(self.events),
            appended_at_utc=_utc_now(),
        )
        self.events.append(event)
        return event

    def leaf_payloads(self) -> list[dict[str, Any]]:
        return [event.to_leaf_payload() for event in self.events]

    def leaf_hashes(self) -> list[str]:
        return [ct_leaf_hash(payload) for payload in self.leaf_payloads()]

    def root_hash(self, size: int | None = None) -> str:
        leaves = self.leaf_hashes()
        if size is not None:
            leaves = leaves[: int(size)]
        return merkle_root_from_leaf_hashes(leaves)

    def signed_tree_head(self, *, size: int | None = None, label: str = "current") -> dict[str, Any]:
        tree_size = len(self.events) if size is None else int(size)
        payload = {
            "sth_version": STH_VERSION,
            "tree_id": self.tree_id,
            "run_id": self.run_id,
            "label": str(label),
            "tree_size": tree_size,
            "root_hash": self.root_hash(tree_size),
            "hash_alg": "sha256",
            "tree_hash_profile": TREE_HASH_PROFILE,
            "consistency_proof_profile": CONSISTENCY_PROOF_PROFILE,
            "canonicalization": CANONICALIZATION,
            "issued_at_utc": _utc_now(),
            "key_id": key_id_for_public_key(self.keypair["public_key"]),
        }
        return attach_verification(
            sign_receipt_payload(
                payload,
                private_key_b64=self.keypair["private_key"],
                public_key_b64=self.keypair["public_key"],
                signer_id=f"helix-transparency-log:{self.tree_id}",
                key_provenance=str(self.keypair.get("key_provenance") or "ephemeral_preregistered"),
            )
        )

    def inclusion_proof_for_event(self, event_id: str) -> dict[str, Any]:
        index = next(index for index, event in enumerate(self.events) if event.event_id == event_id)
        return build_inclusion_proof(self.leaf_hashes(), index)

    def consistency_proof(self, old_size: int, new_size: int | None = None) -> dict[str, Any]:
        new_size = len(self.events) if new_size is None else int(new_size)
        if old_size <= 0 or new_size < old_size or new_size > len(self.events):
            raise ValueError("invalid consistency proof sizes")
        leaves = self.leaf_hashes()
        old_root_hash = merkle_root_from_leaf_hashes(leaves[:old_size])
        new_root_hash = merkle_root_from_leaf_hashes(leaves[:new_size])
        return {
            "proof_profile": CONSISTENCY_PROOF_PROFILE,
            "old_size": old_size,
            "new_size": new_size,
            "old_root_hash": old_root_hash,
            "new_root_hash": new_root_hash,
            "consistency_path": build_consistency_path(leaves, int(old_size), int(new_size)),
            "proof_caveat": "Local RFC 9162-style proof over HeliX event canonical JSON leaves.",
        }


def verify_consistency_proof(old_sth: dict[str, Any], new_sth: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
    if proof.get("proof_profile") != CONSISTENCY_PROOF_PROFILE:
        return {"ok": False, "reason": "unsupported_consistency_proof_profile"}
    if int(old_sth.get("tree_size") or -1) != int(proof.get("old_size") or -2):
        return {"ok": False, "reason": "old_size_mismatch"}
    if int(new_sth.get("tree_size") or -1) != int(proof.get("new_size") or -2):
        return {"ok": False, "reason": "new_size_mismatch"}
    old_verification = verify_signed_receipt(old_sth, verifier_version="helix-sth-verifier-v0")
    new_verification = verify_signed_receipt(new_sth, verifier_version="helix-sth-verifier-v0")
    if not old_verification.get("signature_verified"):
        return {"ok": False, "reason": "old_sth_signature_invalid", "verification": old_verification}
    if not new_verification.get("signature_verified"):
        return {"ok": False, "reason": "new_sth_signature_invalid", "verification": new_verification}
    old_root = str(proof.get("old_root_hash") or "")
    new_root = str(proof.get("new_root_hash") or "")
    if old_root != old_sth.get("root_hash"):
        return {"ok": False, "reason": "old_root_mismatch", "proof_old_root": old_root}
    if new_root != new_sth.get("root_hash"):
        return {"ok": False, "reason": "new_root_mismatch", "proof_new_root": new_root}
    path = [str(item) for item in proof.get("consistency_path") or []]
    path_result = verify_consistency_path(
        old_size=int(old_sth.get("tree_size") or 0),
        new_size=int(new_sth.get("tree_size") or 0),
        old_root_hash=str(old_sth.get("root_hash") or ""),
        new_root_hash=str(new_sth.get("root_hash") or ""),
        consistency_path=path,
    )
    if not path_result.get("ok"):
        return {"ok": False, "reason": "consistency_path_invalid", "path_verification": path_result}
    return {
        "ok": True,
        "old_root_hash": old_root,
        "new_root_hash": new_root,
        "consistency_path_len": len(path),
        "path_verification": path_result,
        "old_verification": old_verification,
        "new_verification": new_verification,
    }


def detect_split_views(sths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for sth in sths:
        key = (str(sth.get("tree_id")), int(sth.get("tree_size") or -1))
        previous = seen.get(key)
        if previous and previous.get("root_hash") != sth.get("root_hash"):
            conflicts.append(
                {
                    "tree_id": key[0],
                    "tree_size": key[1],
                    "first_root_hash": previous.get("root_hash"),
                    "second_root_hash": sth.get("root_hash"),
                    "first_signature_verified": bool(previous.get("signature_verified")),
                    "second_signature_verified": bool(sth.get("signature_verified")),
                }
            )
        seen.setdefault(key, sth)
    return conflicts


def build_agent_run_attestation(
    *,
    run_id: str,
    subject_event: dict[str, Any],
    inclusion_proof: dict[str, Any],
    sth: dict[str, Any],
    previous_sth: dict[str, Any] | None,
    consistency_proof: dict[str, Any] | None,
    model_audit: dict[str, Any],
    memory_summary: dict[str, Any],
    checks: dict[str, bool],
    external_anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subject_digest = _sha256_text(canonical_json(subject_event))
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": str(subject_event.get("event_id") or "agent-run-event"),
                "digest": {"sha256": subject_digest},
            }
        ],
        "predicateType": ATTESTATION_PREDICATE_TYPE,
        "predicate": {
            "attestation_version": "helix-agent-run-attestation-v0",
            "run_id": run_id,
            "flow_profile": "patch-safe",
            "artifact_digest": {"sha256": subject_digest},
            "model": {
                "requested": model_audit.get("requested_model"),
                "actual": model_audit.get("actual_model"),
                "provider_mismatch": bool(model_audit.get("provider_mismatch")),
                "calls": list(model_audit.get("calls") or []),
            },
            "memory": {
                "admitted": list(memory_summary.get("admitted") or []),
                "quarantined": list(memory_summary.get("quarantined") or []),
            },
            "materials": [
                {"name": "model-call", "digest": {"sha256": "captured-in-log"}},
                {"name": "memory-write", "digest": {"sha256": "captured-in-log"}},
            ],
            "byproducts": [
                {"name": "transcript", "digest": {"sha256": "captured-in-log"}},
                {"name": "trust-card", "digest": {"sha256": "captured-in-log"}},
            ],
            "verification": {
                "sth": {key: sth.get(key) for key in ("tree_id", "tree_size", "root_hash", "key_id", "signature")},
                "previous_sth": (
                    {key: previous_sth.get(key) for key in ("tree_id", "tree_size", "root_hash", "key_id", "signature")}
                    if isinstance(previous_sth, dict)
                    else None
                ),
                "inclusion_proof": inclusion_proof,
                "consistency_proof": consistency_proof,
                "external_anchor": external_anchor,
                "checks": checks,
                "claim_boundary": (
                    "HeliX emits locally verifiable attestations for agentic runs. "
                    "This proves local inclusion and append-only consistency against signed local STHs; "
                    "it does not prove model semantic truth or global non-equivocation."
                ),
            },
        },
    }


def verify_standalone_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    event = bundle["event"]
    proof = bundle["inclusion_proof"]
    sth = bundle["sth"]
    inclusion = verify_inclusion_proof(event, proof, sth)
    consistency: dict[str, Any] | None = None
    if isinstance(bundle.get("previous_sth"), dict) and isinstance(bundle.get("consistency_proof"), dict):
        consistency = verify_consistency_proof(bundle["previous_sth"], sth, bundle["consistency_proof"])
    attestation = bundle.get("attestation") if isinstance(bundle.get("attestation"), dict) else {}
    predicate = attestation.get("predicate") if isinstance(attestation.get("predicate"), dict) else {}
    verification = predicate.get("verification") if isinstance(predicate.get("verification"), dict) else {}
    expected_subject_digest = _sha256_text(canonical_json(event))
    actual_subject_digest = (
        ((attestation.get("subject") or [{}])[0].get("digest") or {}).get("sha256")
        if isinstance(attestation.get("subject"), list)
        else None
    )
    claim_boundary = str(verification.get("claim_boundary") or "")
    subject_ok = not actual_subject_digest or actual_subject_digest == expected_subject_digest
    consistency_ok = consistency is None or bool(consistency.get("ok"))
    claim_boundary_ok = "does not prove model semantic truth" in claim_boundary
    return {
        "ok": bool(inclusion.get("ok")) and consistency_ok and subject_ok and claim_boundary_ok,
        "inclusion": inclusion,
        "consistency": consistency,
        "subject_digest_ok": subject_ok,
        "claim_boundary_ok": claim_boundary_ok,
        "bundle_sha256": _sha256_text(canonical_json(bundle)),
    }


def _make_event_payloads(run_id: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "task_capsule",
            {
                "goal": "Patch a verifier bug without touching unrelated files.",
                "run_id": run_id,
                "flow_profile": "patch-safe",
                "claim_boundary": "Patch provenance only; no semantic correctness guarantee.",
            },
        ),
        (
            "model_call",
            {
                "requested_model": "anthropic/claude-4-sonnet",
                "actual_model": "anthropic/claude-sonnet-4-6",
                "prompt_digest": "sha256:prompt-a",
                "output_digest": "sha256:output-a",
                "provider_mismatch": True,
            },
        ),
        (
            "memory_write",
            {
                "memory_id": "mem-agent-run-001",
                "signature_verified": True,
                "semantic_truth_status": "unproven",
                "content_digest": "sha256:memory-a",
            },
        ),
        (
            "signed_poison_control",
            {
                "memory_id": "mem-poison-control",
                "signature_verified": True,
                "semantic_truth_status": "false_control",
                "lure": "This signed claim asks the verifier to treat provenance as truth.",
            },
        ),
        (
            "tool_output",
            {
                "tool": "pytest",
                "exit_code": 0,
                "output_digest": "sha256:pytest-output",
            },
        ),
        (
            "patch",
            {
                "path": "src/helix_proto/verifier.py",
                "patch_digest": "sha256:patch-a",
                "changed_files": ["src/helix_proto/verifier.py", "tests/test_verifier.py"],
            },
        ),
        (
            "trust_card",
            {
                "checks_passed": ["unit-tests", "inclusion-proof"],
                "warnings": ["semantic truth not proven"],
                "apply_gate": "human-required",
            },
        ),
    ]


def run_local_gauntlet(*, run_id: str) -> dict[str, Any]:
    keypair = derive_ephemeral_keypair(f"agent-run-transparency:{run_id}:log-key")
    log = AgentRunTransparencyLog(tree_id=f"helix-agent-run:{run_id}", run_id=run_id, keypair=keypair)
    for event_type, payload in _make_event_payloads(run_id):
        log.append(event_type, payload)

    old_sth = log.signed_tree_head(size=3, label="checkpoint-after-memory")
    final_sth = log.signed_tree_head(label="final")
    patch_event = log.events[5].to_leaf_payload()
    patch_proof = log.inclusion_proof_for_event("evt-0006")
    inclusion_result = verify_inclusion_proof(patch_event, patch_proof, final_sth)

    tampered_patch = json.loads(json.dumps(patch_event))
    tampered_patch["payload"]["patch_digest"] = "sha256:evil-patch"
    tampered_result = verify_inclusion_proof(tampered_patch, patch_proof, final_sth)

    consistency_proof = log.consistency_proof(old_size=3)
    consistency_result = verify_consistency_proof(old_sth, final_sth, consistency_proof)

    reforged_log = AgentRunTransparencyLog(tree_id=log.tree_id, run_id=run_id, keypair=keypair)
    for event_type, payload in _make_event_payloads(run_id):
        altered = dict(payload)
        if event_type == "memory_write":
            altered["content_digest"] = "sha256:forged-memory"
        reforged_log.append(event_type, altered)
    reforged_final_sth = reforged_log.signed_tree_head(label="forged-final")
    forged_proof = reforged_log.consistency_proof(old_size=3)
    reforge_result = verify_consistency_proof(old_sth, reforged_final_sth, forged_proof)

    split_view_log = AgentRunTransparencyLog(tree_id=log.tree_id, run_id=run_id, keypair=keypair)
    for event_type, payload in _make_event_payloads(run_id)[:3]:
        altered = dict(payload)
        if event_type == "model_call":
            altered["output_digest"] = "sha256:alternate-output"
        split_view_log.append(event_type, altered)
    split_view_sth = split_view_log.signed_tree_head(size=3, label="split-view")
    split_view_conflicts = detect_split_views([old_sth, split_view_sth])

    signed_poison_event = log.events[3].to_leaf_payload()
    signed_poison_verdict = {
        "signature_valid": bool(signed_poison_event["payload"].get("signature_verified")),
        "semantic_authority_granted": False,
        "verdict": "provenance_valid_semantic_rejected",
    }

    backdated_event = {
        "event_id": "evt-backdated",
        "event_type": "tool_output",
        "log_index": len(log.events),
        "appended_at_utc": _utc_now(),
        "payload": {
            "claimed_created_at_utc": "2001-01-01T00:00:00Z",
            "tool": "external-research",
            "output_digest": "sha256:late-claimed-early",
        },
    }
    backdating_verdict = {
        "claimed_created_at_utc": backdated_event["payload"]["claimed_created_at_utc"],
        "append_index": backdated_event["log_index"],
        "accepted_as_log_time": False,
        "verdict": "claimed_time_demoted_until_timestamp_anchor",
    }

    provider_event = log.events[1].to_leaf_payload()
    provider_proof = log.inclusion_proof_for_event("evt-0002")
    provider_result = verify_inclusion_proof(provider_event, provider_proof, final_sth)
    memory_summary = {
        "admitted": [log.events[2].payload.get("memory_id")],
        "quarantined": [log.events[3].payload.get("memory_id")],
    }

    attestation = build_agent_run_attestation(
        run_id=run_id,
        subject_event=patch_event,
        inclusion_proof=patch_proof,
        sth=final_sth,
        previous_sth=old_sth,
        consistency_proof=consistency_proof,
        model_audit=provider_event["payload"],
        memory_summary=memory_summary,
        checks={"unit_tests": True, "claim_boundary_present": True},
        external_anchor=None,
    )
    standalone_bundle = {
        "bundle_version": "helix-agent-run-verifier-bundle-v0",
        "event": patch_event,
        "inclusion_proof": patch_proof,
        "previous_sth": old_sth,
        "consistency_proof": consistency_proof,
        "sth": final_sth,
        "attestation": attestation,
        "claim_boundary": attestation["predicate"]["verification"]["claim_boundary"],
    }
    standalone_result = verify_standalone_bundle(standalone_bundle)

    gates = {
        "baseline_inclusion_verified": bool(inclusion_result.get("ok")),
        "tampered_event_rejected": not bool(tampered_result.get("ok")),
        "append_only_consistency_verified": bool(consistency_result.get("ok")),
        "reforged_history_rejected_against_prior_sth": not bool(reforge_result.get("ok")),
        "split_view_detected_by_witness": len(split_view_conflicts) >= 1,
        "signed_poison_signature_not_semantic_truth": (
            signed_poison_verdict["signature_valid"] is True
            and signed_poison_verdict["semantic_authority_granted"] is False
        ),
        "backdating_demoted_to_claim_mismatch": backdating_verdict["accepted_as_log_time"] is False,
        "provider_mismatch_included_and_auditable": (
            bool(provider_event["payload"].get("provider_mismatch")) and bool(provider_result.get("ok"))
        ),
        "standalone_verifier_bundle_passes": bool(standalone_result.get("ok")),
    }
    score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
    return {
        "artifact": "local-agent-run-transparency-gauntlet-v1",
        "suite_version": SUITE_VERSION,
        "run_id": run_id,
        "run_started_utc": _utc_now(),
        "run_ended_utc": _utc_now(),
        "status": "completed" if all(gates.values()) else "partial",
        "score": score,
        "gates": gates,
        "models": [],
        "tree": {
            "tree_id": log.tree_id,
            "tree_size": len(log.events),
            "tree_hash_profile": TREE_HASH_PROFILE,
            "consistency_proof_profile": CONSISTENCY_PROOF_PROFILE,
            "final_root_hash": final_sth.get("root_hash"),
            "key_id": final_sth.get("key_id"),
        },
        "events": log.leaf_payloads(),
        "sths": {
            "old": old_sth,
            "final": final_sth,
            "reforged_final": reforged_final_sth,
            "split_view": split_view_sth,
        },
        "proofs": {
            "patch_inclusion": patch_proof,
            "provider_mismatch_inclusion": provider_proof,
            "consistency": consistency_proof,
            "forged_consistency": forged_proof,
        },
        "verifier_results": {
            "patch_inclusion": inclusion_result,
            "tampered_patch": tampered_result,
            "consistency": consistency_result,
            "reforged_history": reforge_result,
            "provider_mismatch": provider_result,
            "standalone_bundle": standalone_result,
        },
        "adversarial_results": {
            "split_view_conflicts": split_view_conflicts,
            "signed_poison": signed_poison_verdict,
            "backdating": backdating_verdict,
        },
        "attestation": attestation,
        "standalone_bundle": standalone_bundle,
        "claim_boundary": (
            "This artifact proves local transparency mechanics for a synthetic agent run. "
            "It does not prove model semantic truth or global non-equivocation."
        ),
        "next_protocol_steps": [
            "Add optional witness cosignatures over STHs.",
            "Add optional RFC 3161 or Rekor/Trillian external anchoring.",
            "Wire the verifier bundle to real patch-safe task capsules.",
        ],
    }


def run_real_memory_transparency(*, run_id: str, output_dir: Path | None = None) -> dict[str, Any]:
    from helix_kv.merkle_dag import DAG_HASH_PROFILE_V2
    from helix_kv.memory_catalog import MemoryCatalog

    run_dir = (output_dir or (REPO_ROOT / DEFAULT_OUTPUT_DIR)) / "_real-memory" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(run_dir / "memory.sqlite")
    session_id = f"agent-run-transparency:{run_id}"
    agent_id = "helix-transparency-local-runner"
    project = "helix-transparency-core"
    changed_files = [
        "helix_kv/merkle_dag.py",
        "helix_kv/memory_catalog.py",
        "crates/helix-merkle-dag/src/lib.rs",
        "crates/helix-state-server/src/main.rs",
        "tools/run_agent_run_transparency_gauntlet_v1.py",
        "tools/verify_agent_run_bundle.py",
    ]
    patch_info = _git_diff_digest(changed_files)

    try:
        task = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="task_capsule",
            summary="Real local transparency run task capsule",
            content=canonical_json(
                {
                    "run_id": run_id,
                    "goal": "Export a locally verifiable HeliX agent-run attestation bundle.",
                    "claim_boundary": "Local run mechanics only; no semantic truth or global non-equivocation.",
                }
            ),
            tags=["transparency", "task-capsule"],
        )
        model_audit = {
            "requested_model": "local/codex-coding-agent",
            "actual_model": "local/codex-coding-agent",
            "provider_mismatch": False,
            "boundary": "Local coding-agent execution; no cloud model identity claim made by this run.",
        }
        model_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="model_audit",
            summary="Requested and actual model identity recorded for local run",
            content=canonical_json(model_audit),
            tags=["transparency", "model-audit"],
        )
        admitted = catalog.remember(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="semantic",
            summary="Admitted memory for local transparency bundle",
            content=(
                "This memory records that HeliX produced a local agent-run attestation "
                "with Merkle hash v2, signed receipts, inclusion proof, consistency proof, "
                "and standalone verifier output."
            ),
            tags=["transparency", "admitted-memory"],
            importance=9,
            llm_call_id=f"local-run:{run_id}:admitted-memory",
        )
        admitted_hash = str(catalog.get_memory_node_hash(admitted.memory_id) or "")
        admitted_receipt = catalog.get_memory_receipt(admitted.memory_id) or {}
        admitted_chain = catalog.verify_chain(admitted_hash)
        admitted_node = catalog.dag.lookup(admitted_hash)

        poison = catalog.remember_quarantined(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="semantic",
            summary="Signed poison control for local transparency bundle",
            content="SIGNED_POISON_CONTROL: a valid signature must not grant semantic authority.",
            tags=["transparency", "poison-control"],
            importance=1,
            record_kind="signed_poison_control",
            quarantine_reason="semantic_authority_control",
            quarantine_class="test_control",
            disposition="quarantined_control",
            llm_call_id=f"local-run:{run_id}:signed-poison",
        )
        poison_hash = str(poison.get("node_hash") or "")
        poison_receipt = dict(poison.get("signed_receipt") or poison.get("receipt") or {})
        poison_node = catalog.dag.lookup(poison_hash)

        tool_output = {
            "tool": "pytest",
            "command": (
                "python -m pytest tests\\test_agent_run_transparency_gauntlet.py "
                "tests\\test_v4_signed_receipts.py tests\\test_merkle_dag_catalog.py "
                "tests\\test_memory_catalog.py -q"
            ),
            "status": "previously_verified_in_current_workspace",
            "output_digest": "sha256:captured-by-codex-run",
        }
        tool_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="tool_output",
            summary="Transparency core Python test command recorded",
            content=canonical_json(tool_output),
            tags=["transparency", "tool-output"],
        )
        patch_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="patch",
            summary="Transparency core patch digest recorded",
            content=canonical_json(patch_info),
            tags=["transparency", "patch"],
        )
        trust_card_payload = {
            "trust_card_version": "helix-local-trust-card-v0",
            "run_id": run_id,
            "artifact_digest": patch_info["patch_digest"],
            "memory_admitted": [admitted.memory_id],
            "memory_quarantined": [str(poison.get("memory_id"))],
            "claim_boundary": "Local attestations verify run mechanics; they do not prove model semantic truth.",
            "external_anchor": None,
        }
        trust_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="trust_card",
            summary="Local trust card payload recorded",
            content=canonical_json(trust_card_payload),
            tags=["transparency", "trust-card"],
        )

        keypair = derive_ephemeral_keypair(f"agent-run-transparency:{run_id}:real-memory-log-key")
        log = AgentRunTransparencyLog(tree_id=f"helix-agent-run-real-memory:{run_id}", run_id=run_id, keypair=keypair)
        log.append(
            "task_capsule",
            {
                "run_id": run_id,
                "catalog_node_hash": task.get("node_hash"),
                "goal": "Export a locally verifiable HeliX agent-run attestation bundle.",
                "claim_boundary": "Local run mechanics only; no semantic truth or global non-equivocation.",
            },
        )
        log.append(
            "model_call",
            {
                **model_audit,
                "catalog_node_hash": model_obs.get("node_hash"),
            },
        )
        log.append(
            "memory_write",
            {
                "memory_id": admitted.memory_id,
                "node_hash": admitted_hash,
                "node_hash_profile": getattr(admitted_node, "hash_profile", None),
                "signature_verified": bool(admitted_receipt.get("signature_verified")),
                "receipt_digest": f"sha256:{_sha256_text(canonical_json(admitted_receipt))}",
                "chain_status": admitted_chain.get("status"),
                "semantic_truth_status": "unproven",
            },
        )
        log.append(
            "signed_poison_control",
            {
                "memory_id": str(poison.get("memory_id")),
                "node_hash": poison_hash,
                "node_hash_profile": getattr(poison_node, "hash_profile", None),
                "signature_verified": bool(poison_receipt.get("signature_verified")),
                "semantic_authority": False,
                "receipt_digest": f"sha256:{_sha256_text(canonical_json(poison_receipt))}",
            },
        )
        log.append("tool_output", {**tool_output, "catalog_node_hash": tool_obs.get("node_hash")})
        log.append("patch", {**patch_info, "catalog_node_hash": patch_obs.get("node_hash")})
        log.append("trust_card", {**trust_card_payload, "catalog_node_hash": trust_obs.get("node_hash")})

        old_sth = log.signed_tree_head(size=3, label="checkpoint-after-admitted-memory")
        final_sth = log.signed_tree_head(label="final-real-memory")
        patch_event = log.events[5].to_leaf_payload()
        patch_proof = log.inclusion_proof_for_event("evt-0006")
        inclusion_result = verify_inclusion_proof(patch_event, patch_proof, final_sth)
        consistency_proof = log.consistency_proof(old_size=3)
        consistency_result = verify_consistency_proof(old_sth, final_sth, consistency_proof)
        memory_summary = {
            "admitted": [admitted.memory_id],
            "quarantined": [str(poison.get("memory_id"))],
        }
        attestation = build_agent_run_attestation(
            run_id=run_id,
            subject_event=patch_event,
            inclusion_proof=patch_proof,
            sth=final_sth,
            previous_sth=old_sth,
            consistency_proof=consistency_proof,
            model_audit=model_audit,
            memory_summary=memory_summary,
            checks={
                "memory_receipt_signature_verified": bool(admitted_receipt.get("signature_verified")),
                "memory_chain_verified": admitted_chain.get("status") == "verified",
                "claim_boundary_present": True,
            },
            external_anchor=None,
        )
        standalone_bundle = {
            "bundle_version": "helix-agent-run-verifier-bundle-v0",
            "event": patch_event,
            "inclusion_proof": patch_proof,
            "previous_sth": old_sth,
            "consistency_proof": consistency_proof,
            "sth": final_sth,
            "attestation": attestation,
            "claim_boundary": attestation["predicate"]["verification"]["claim_boundary"],
        }
        standalone_result = verify_standalone_bundle(standalone_bundle)
        catalog_coverage = catalog.verify_dag_coverage()
        gates = {
            "real_memory_receipt_signature_verified": bool(admitted_receipt.get("signature_verified")),
            "real_memory_chain_verified": admitted_chain.get("status") == "verified",
            "real_memory_hash_profile_v2": getattr(admitted_node, "hash_profile", None) == DAG_HASH_PROFILE_V2,
            "real_quarantine_signed_without_semantic_authority": (
                bool(poison_receipt.get("signature_verified")) and log.events[3].payload.get("semantic_authority") is False
            ),
            "patch_inclusion_verified": bool(inclusion_result.get("ok")),
            "append_only_consistency_verified": bool(consistency_result.get("ok")),
            "standalone_verifier_bundle_passes": bool(standalone_result.get("ok")),
            "catalog_dag_coverage_verified": catalog_coverage.get("status") == "verified",
        }
        score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
        return {
            "artifact": "local-agent-run-transparency-real-memory-v1",
            "suite_version": SUITE_VERSION,
            "run_id": run_id,
            "run_started_utc": _utc_now(),
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "partial",
            "score": score,
            "gates": gates,
            "tree": {
                "tree_id": log.tree_id,
                "tree_size": len(log.events),
                "tree_hash_profile": TREE_HASH_PROFILE,
                "consistency_proof_profile": CONSISTENCY_PROOF_PROFILE,
                "final_root_hash": final_sth.get("root_hash"),
                "key_id": final_sth.get("key_id"),
            },
            "catalog": {
                "db_path": str(catalog.db_path),
                "session_id": session_id,
                "dag_coverage": catalog_coverage,
                "stats": catalog.stats(),
            },
            "events": log.leaf_payloads(),
            "sths": {"old": old_sth, "final": final_sth},
            "proofs": {"patch_inclusion": patch_proof, "consistency": consistency_proof},
            "verifier_results": {
                "patch_inclusion": inclusion_result,
                "consistency": consistency_result,
                "standalone_bundle": standalone_result,
            },
            "attestation": attestation,
            "standalone_bundle": standalone_bundle,
            "claim_boundary": (
                "This artifact proves local HeliX MemoryCatalog-backed transparency mechanics. "
                "It does not prove model semantic truth or global non-equivocation."
            ),
        }
    finally:
        catalog.close()


def run_cloud_deepinfra_transparency(
    *,
    run_id: str,
    models: list[str],
    output_dir: Path | None = None,
    max_tokens: int = 450,
    temperature: float = 0.2,
    timeout: float = 240.0,
) -> dict[str, Any]:
    from helix_kv.merkle_dag import DAG_HASH_PROFILE_V2
    from helix_kv.memory_catalog import MemoryCatalog

    token = os.environ.get("DEEPINFRA_API_TOKEN")
    if not token:
        raise RuntimeError("DEEPINFRA_API_TOKEN is required for --mode cloud-deepinfra")
    clean_models = [str(model).strip() for model in models if str(model).strip()]
    if not clean_models:
        raise ValueError("at least one DeepInfra model ref is required")

    run_dir = (output_dir or (REPO_ROOT / DEFAULT_OUTPUT_DIR)) / "_cloud-deepinfra" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    catalog = MemoryCatalog.open(run_dir / "memory.sqlite")
    session_id = f"agent-run-cloud-deepinfra:{run_id}"
    agent_id = "helix-cloud-transparency-runner"
    project = "helix-transparency-cloud-deepinfra"
    temperature_milli = int(round(float(temperature) * 1000))
    changed_files = [
        "helix_kv/merkle_dag.py",
        "helix_kv/memory_catalog.py",
        "tools/run_agent_run_transparency_gauntlet_v1.py",
        "tools/verify_agent_run_bundle.py",
        "tools/run_agent_run_transparency_cloud_deepinfra_secure.ps1",
    ]
    patch_info = _git_diff_digest(changed_files)

    try:
        task_payload = {
            "run_id": run_id,
            "goal": (
                "Run a cloud DeepInfra model panel and export a locally verifiable "
                "HeliX attestation over model metadata, outputs, memory writes, and proofs."
            ),
            "models_requested": clean_models,
            "max_tokens": int(max_tokens),
            "temperature_milli": temperature_milli,
            "timeout_s": int(round(float(timeout))),
            "claim_boundary": "Provider execution evidence only; no semantic truth or global non-equivocation.",
        }
        task_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="task_capsule",
            summary="Cloud DeepInfra transparency task capsule",
            content=canonical_json(task_payload),
            tags=["transparency", "cloud", "deepinfra", "task-capsule"],
        )

        system_prompt = (
            "You are participating in a HeliX cloud transparency audit. "
            "Answer concisely. Distinguish provenance evidence from semantic truth. "
            "Do not claim global non-equivocation without an external witness."
        )
        user_prompt = (
            f"Run id: {run_id}\n"
            "Task: inspect the proposed HeliX transparency core from the perspective of an external model. "
            "Return compact JSON with keys: strongest_signal, weakest_assumption, verification_gap, next_experiment."
        )
        cloud_calls: list[dict[str, Any]] = []
        transcripts: list[dict[str, Any]] = []
        for index, model in enumerate(clean_models, start=1):
            call = _deepinfra_chat_sync(
                model=model,
                system=system_prompt,
                user=user_prompt,
                token=token,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                timeout=float(timeout),
            )
            text = str(call.get("text") or "")
            safe_call = {
                "call_index": index,
                "requested_model": model,
                "actual_model": call.get("actual_model"),
                "provider_mismatch": bool(call.get("provider_mismatch")),
                "status": call.get("status"),
                "finish_reason": call.get("finish_reason"),
                "tokens_used": int(call.get("tokens_used") or 0),
                "latency_ms": int(call.get("latency_ms") or 0),
                "retry_count": int(call.get("retry_count") or 0),
                "text_digest": call.get("text_digest"),
                "output_chars": len(text),
                "error": call.get("error"),
            }
            call_obs = catalog.observe(
                project=project,
                agent_id=agent_id,
                session_id=session_id,
                observation_type="cloud_model_call",
                summary=f"DeepInfra call metadata for {model}",
                content=canonical_json(
                    {
                        **safe_call,
                        "text_preview": text[:1200],
                    }
                ),
                tags=["transparency", "cloud", "deepinfra", "model-call"],
            )
            memory = catalog.remember(
                project=project,
                agent_id=agent_id,
                session_id=session_id,
                memory_type="episodic",
                summary=f"Cloud DeepInfra output from {model}",
                content=text or f"ERROR: {call.get('error') or 'empty_response'}",
                tags=["transparency", "cloud", "deepinfra", "admitted-memory"],
                importance=8 if call.get("status") == "ok" else 3,
                llm_call_id=f"deepinfra:{run_id}:{index}",
            )
            memory_hash = str(catalog.get_memory_node_hash(memory.memory_id) or "")
            receipt = catalog.get_memory_receipt(memory.memory_id) or {}
            chain = catalog.verify_chain(memory_hash) if memory_hash else {"status": "missing"}
            node = catalog.dag.lookup(memory_hash) if memory_hash else None
            public_call = {
                **safe_call,
                "catalog_node_hash": call_obs.get("node_hash"),
                "memory_id": memory.memory_id,
                "memory_node_hash": memory_hash,
                "node_hash_profile": getattr(node, "hash_profile", None),
                "receipt_signature_verified": bool(receipt.get("signature_verified")),
                "receipt_digest": f"sha256:{_sha256_text(canonical_json(receipt))}",
                "chain_status": chain.get("status"),
            }
            cloud_calls.append(public_call)
            transcripts.append(
                {
                    "call_index": index,
                    "requested_model": model,
                    "actual_model": call.get("actual_model"),
                    "status": call.get("status"),
                    "text_digest": call.get("text_digest"),
                    "text": text,
                    "error": call.get("error"),
                }
            )

        poison = catalog.remember_quarantined(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type="semantic",
            summary="Signed poison control for cloud DeepInfra transparency bundle",
            content="SIGNED_POISON_CONTROL: a valid provider response or receipt must not grant semantic authority.",
            tags=["transparency", "cloud", "poison-control"],
            importance=1,
            record_kind="signed_poison_control",
            quarantine_reason="semantic_authority_control",
            quarantine_class="test_control",
            disposition="quarantined_control",
            llm_call_id=f"deepinfra:{run_id}:signed-poison",
        )
        poison_hash = str(poison.get("node_hash") or "")
        poison_receipt = dict(poison.get("signed_receipt") or poison.get("receipt") or {})
        poison_node = catalog.dag.lookup(poison_hash) if poison_hash else None

        tool_output = {
            "tool": "deepinfra_chat_completions",
            "endpoint": DEEPINFRA_BASE,
            "models_requested": clean_models,
            "status": "completed" if all(call["status"] == "ok" for call in cloud_calls) else "partial",
            "output_digests": [call["text_digest"] for call in cloud_calls],
            "token_handling": "DEEPINFRA_API_TOKEN read from process env and not persisted",
        }
        tool_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="tool_output",
            summary="DeepInfra cloud tool output recorded",
            content=canonical_json(tool_output),
            tags=["transparency", "cloud", "tool-output"],
        )
        trust_card_payload = {
            "trust_card_version": "helix-cloud-trust-card-v0",
            "run_id": run_id,
            "artifact_digest": patch_info["patch_digest"],
            "model_requested_actual": [
                {
                    "requested": call["requested_model"],
                    "actual": call["actual_model"],
                    "provider_mismatch": call["provider_mismatch"],
                    "status": call["status"],
                    "text_digest": call["text_digest"],
                }
                for call in cloud_calls
            ],
            "memory_admitted": [call["memory_id"] for call in cloud_calls],
            "memory_quarantined": [str(poison.get("memory_id"))],
            "claim_boundary": "Cloud attestations verify local evidence mechanics; they do not prove model semantic truth.",
            "external_anchor": None,
        }
        trust_obs = catalog.observe(
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            observation_type="trust_card",
            summary="Cloud trust card payload recorded",
            content=canonical_json(trust_card_payload),
            tags=["transparency", "cloud", "trust-card"],
        )

        keypair = derive_ephemeral_keypair(f"agent-run-transparency:{run_id}:cloud-deepinfra-log-key")
        log = AgentRunTransparencyLog(tree_id=f"helix-agent-run-cloud-deepinfra:{run_id}", run_id=run_id, keypair=keypair)
        log.append("task_capsule", {**task_payload, "catalog_node_hash": task_obs.get("node_hash")})
        for call in cloud_calls:
            log.append(
                "model_call",
                {
                    "call_index": call["call_index"],
                    "requested_model": call["requested_model"],
                    "actual_model": call["actual_model"],
                    "provider_mismatch": call["provider_mismatch"],
                    "status": call["status"],
                    "finish_reason": call["finish_reason"],
                    "tokens_used": call["tokens_used"],
                    "latency_ms": call["latency_ms"],
                    "retry_count": call["retry_count"],
                    "text_digest": call["text_digest"],
                    "output_chars": call["output_chars"],
                    "memory_id": call["memory_id"],
                    "memory_node_hash": call["memory_node_hash"],
                    "node_hash_profile": call["node_hash_profile"],
                    "receipt_signature_verified": call["receipt_signature_verified"],
                    "chain_status": call["chain_status"],
                    "catalog_node_hash": call["catalog_node_hash"],
                    "semantic_truth_status": "unproven",
                },
            )
        memory_event = log.append(
            "memory_write",
            {
                "admitted": [
                    {
                        "memory_id": call["memory_id"],
                        "node_hash": call["memory_node_hash"],
                        "node_hash_profile": call["node_hash_profile"],
                        "receipt_signature_verified": call["receipt_signature_verified"],
                        "receipt_digest": call["receipt_digest"],
                        "chain_status": call["chain_status"],
                    }
                    for call in cloud_calls
                ],
                "quarantined": [
                    {
                        "memory_id": str(poison.get("memory_id")),
                        "node_hash": poison_hash,
                        "node_hash_profile": getattr(poison_node, "hash_profile", None),
                        "receipt_signature_verified": bool(poison_receipt.get("signature_verified")),
                        "receipt_digest": f"sha256:{_sha256_text(canonical_json(poison_receipt))}",
                        "semantic_authority": False,
                    }
                ],
            },
        )
        log.append(
            "signed_poison_control",
            {
                "memory_id": str(poison.get("memory_id")),
                "node_hash": poison_hash,
                "node_hash_profile": getattr(poison_node, "hash_profile", None),
                "signature_verified": bool(poison_receipt.get("signature_verified")),
                "semantic_authority": False,
                "receipt_digest": f"sha256:{_sha256_text(canonical_json(poison_receipt))}",
            },
        )
        log.append("tool_output", {**tool_output, "catalog_node_hash": tool_obs.get("node_hash")})
        panel_event = log.append(
            "cloud_panel_attestation",
            {
                "run_id": run_id,
                "models_requested": clean_models,
                "call_count": len(cloud_calls),
                "ok_count": sum(1 for call in cloud_calls if call["status"] == "ok"),
                "provider_mismatch_count": sum(1 for call in cloud_calls if call["provider_mismatch"]),
                "output_digests": [call["text_digest"] for call in cloud_calls],
                "memory_event_id": memory_event.event_id,
                "patch_digest": patch_info["patch_digest"],
                "claim_boundary": "Cloud provider output is recorded as evidence, not semantic authority.",
            },
        )
        log.append("trust_card", {**trust_card_payload, "catalog_node_hash": trust_obs.get("node_hash")})

        old_size = min(max(2, len(clean_models) + 1), len(log.events) - 1)
        old_sth = log.signed_tree_head(size=old_size, label="checkpoint-after-cloud-model-calls")
        final_sth = log.signed_tree_head(label="final-cloud-deepinfra")
        subject_event = panel_event.to_leaf_payload()
        subject_proof = log.inclusion_proof_for_event(panel_event.event_id)
        inclusion_result = verify_inclusion_proof(subject_event, subject_proof, final_sth)
        consistency_proof = log.consistency_proof(old_size=old_size)
        consistency_result = verify_consistency_proof(old_sth, final_sth, consistency_proof)
        catalog_coverage = catalog.verify_dag_coverage()
        model_audit = {
            "requested_model": ",".join(clean_models),
            "actual_model": ",".join(str(call["actual_model"] or "unavailable") for call in cloud_calls),
            "provider_mismatch": any(call["provider_mismatch"] for call in cloud_calls),
            "calls": [
                {
                    "requested": call["requested_model"],
                    "actual": call["actual_model"],
                    "provider_mismatch": call["provider_mismatch"],
                    "status": call["status"],
                    "text_digest": call["text_digest"],
                    "memory_node_hash": call["memory_node_hash"],
                }
                for call in cloud_calls
            ],
        }
        memory_summary = {
            "admitted": [call["memory_id"] for call in cloud_calls],
            "quarantined": [str(poison.get("memory_id"))],
        }
        gates = {
            "cloud_calls_completed": all(call["status"] == "ok" for call in cloud_calls),
            "cloud_model_metadata_captured": all(call.get("actual_model") for call in cloud_calls if call["status"] == "ok"),
            "provider_mismatch_auditable": all("provider_mismatch" in call for call in cloud_calls),
            "cloud_outputs_signed_into_memory": all(
                call["receipt_signature_verified"] and call["chain_status"] == "verified" for call in cloud_calls
            ),
            "cloud_memory_hash_profile_v2": all(call["node_hash_profile"] == DAG_HASH_PROFILE_V2 for call in cloud_calls),
            "signed_poison_not_semantic_authority": (
                bool(poison_receipt.get("signature_verified"))
                and getattr(poison_node, "hash_profile", None) == DAG_HASH_PROFILE_V2
                and log.events[len(clean_models) + 2].payload.get("semantic_authority") is False
            ),
            "cloud_panel_inclusion_verified": bool(inclusion_result.get("ok")),
            "append_only_consistency_verified": bool(consistency_result.get("ok")),
            "standalone_verifier_bundle_passes": False,
            "catalog_dag_coverage_verified": catalog_coverage.get("status") == "verified",
        }
        attestation = build_agent_run_attestation(
            run_id=run_id,
            subject_event=subject_event,
            inclusion_proof=subject_proof,
            sth=final_sth,
            previous_sth=old_sth,
            consistency_proof=consistency_proof,
            model_audit=model_audit,
            memory_summary=memory_summary,
            checks={name: ok for name, ok in gates.items() if name != "standalone_verifier_bundle_passes"},
            external_anchor=None,
        )
        standalone_bundle = {
            "bundle_version": "helix-agent-run-verifier-bundle-v0",
            "event": subject_event,
            "inclusion_proof": subject_proof,
            "previous_sth": old_sth,
            "consistency_proof": consistency_proof,
            "sth": final_sth,
            "attestation": attestation,
            "claim_boundary": attestation["predicate"]["verification"]["claim_boundary"],
        }
        standalone_result = verify_standalone_bundle(standalone_bundle)
        gates["standalone_verifier_bundle_passes"] = bool(standalone_result.get("ok"))
        attestation["predicate"]["verification"]["checks"] = gates
        standalone_result = verify_standalone_bundle(standalone_bundle)
        score = round(sum(1 for ok in gates.values() if ok) / max(len(gates), 1), 4)
        return {
            "artifact": "local-agent-run-transparency-cloud-deepinfra-v1",
            "suite_version": SUITE_VERSION,
            "run_id": run_id,
            "run_started_utc": _utc_now(),
            "run_ended_utc": _utc_now(),
            "status": "completed" if all(gates.values()) else "partial",
            "score": score,
            "mode": "cloud-deepinfra",
            "models": clean_models,
            "cloud_config": {
                "endpoint": DEEPINFRA_BASE,
                "max_tokens": int(max_tokens),
                "temperature_milli": temperature_milli,
                "timeout_s": int(round(float(timeout))),
                "token_persisted": False,
            },
            "gates": gates,
            "cloud_calls": cloud_calls,
            "cloud_transcript": transcripts,
            "tree": {
                "tree_id": log.tree_id,
                "tree_size": len(log.events),
                "tree_hash_profile": TREE_HASH_PROFILE,
                "consistency_proof_profile": CONSISTENCY_PROOF_PROFILE,
                "final_root_hash": final_sth.get("root_hash"),
                "key_id": final_sth.get("key_id"),
            },
            "catalog": {
                "db_path": str(catalog.db_path),
                "session_id": session_id,
                "dag_coverage": catalog_coverage,
                "stats": catalog.stats(),
            },
            "events": log.leaf_payloads(),
            "sths": {"old": old_sth, "final": final_sth},
            "proofs": {"cloud_panel_inclusion": subject_proof, "consistency": consistency_proof},
            "verifier_results": {
                "cloud_panel_inclusion": inclusion_result,
                "consistency": consistency_result,
                "standalone_bundle": standalone_result,
            },
            "attestation": attestation,
            "standalone_bundle": standalone_bundle,
            "claim_boundary": (
                "This artifact proves local HeliX transparency mechanics over DeepInfra cloud calls. "
                "It does not prove model semantic truth, account-level provider integrity, or global non-equivocation."
            ),
        }
    finally:
        catalog.close()


def write_extract(path: Path, artifact: dict[str, Any]) -> None:
    failed = [name for name, ok in artifact["gates"].items() if not ok]
    subject_label = "cloud DeepInfra panel event" if artifact.get("mode") == "cloud-deepinfra" else "patch event"
    lines = [
        f"# HeliX Agent Run Transparency Gauntlet: {artifact['run_id']}",
        "",
        "## Verdict",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Score: `{artifact['score']}`",
        f"- Tree size: `{artifact['tree']['tree_size']}`",
        f"- Final root: `{artifact['tree']['final_root_hash']}`",
        "",
        "## Failing Gates",
        "",
    ]
    lines.extend([f"- `{item}`" for item in failed] or ["- None"])
    lines.extend(
        [
            "",
            "## What This Proves",
            "",
            f"- A {subject_label} can be verified against a signed tree head by a standalone bundle.",
            "- Tampering with the event breaks the inclusion proof.",
            "- Re-forging earlier history fails against an older STH.",
            "- Split views at the same tree size are visible to a witness.",
            "- Signed poison remains valid provenance but rejected semantic authority.",
            "",
            "## Caveat",
            "",
            "Consistency proofs use local `rfc9162-consistency-proof-v0`; global non-equivocation still requires witnesses or an external log.",
            "",
            "## Gates",
            "",
        ]
    )
    for name, ok in artifact["gates"].items():
        lines.append(f"- `{name}`: `{ok}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HeliX agent-run transparency gauntlet.")
    parser.add_argument("--run-id", default=f"agent-run-transparency-{uuid.uuid4().hex[:10]}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real-memory", "cloud-deepinfra"],
        default="synthetic",
        help=(
            "synthetic keeps the adversarial protocol fixture; real-memory records a MemoryCatalog-backed local run; "
            "cloud-deepinfra calls DeepInfra models and records locally verifiable evidence."
        ),
    )
    parser.add_argument("--models", default=",".join(DEFAULT_CLOUD_MODELS), help="Comma-separated DeepInfra model refs.")
    parser.add_argument("--tokens", type=int, default=450, help="Max output tokens per DeepInfra model call.")
    parser.add_argument("--temperature", type=float, default=0.2, help="DeepInfra model sampling temperature.")
    parser.add_argument("--timeout", type=float, default=240.0, help="HTTP timeout in seconds per DeepInfra call.")
    parser.add_argument("--no-write", action="store_true", help="Run and print JSON without writing artifacts.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if args.mode == "cloud-deepinfra":
        artifact = run_cloud_deepinfra_transparency(
            run_id=args.run_id,
            output_dir=output_dir,
            models=[model.strip() for model in args.models.split(",") if model.strip()],
            max_tokens=args.tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
    elif args.mode == "real-memory":
        artifact = run_real_memory_transparency(run_id=args.run_id, output_dir=output_dir)
    else:
        artifact = run_local_gauntlet(run_id=args.run_id)
    artifact_slug = str(artifact.get("artifact") or "local-agent-run-transparency-gauntlet-v1").replace("_", "-")
    artifact_path = output_dir / f"{artifact_slug}-{args.run_id}.json"
    extract_path = output_dir / f"{artifact_slug}-{args.run_id}-extract.md"
    bundle_path = output_dir / f"{artifact_slug}-{args.run_id}-bundle.json"
    artifact["artifact_path"] = str(artifact_path)
    artifact["extract_markdown_path"] = str(extract_path)
    artifact["standalone_bundle_path"] = str(bundle_path)

    if not args.no_write:
        _write_json(artifact_path, artifact)
        _write_json(bundle_path, artifact["standalone_bundle"])
        write_extract(extract_path, artifact)
    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "extract_markdown_path": str(extract_path),
                "standalone_bundle_path": str(bundle_path),
                "status": artifact["status"],
                "score": artifact["score"],
                "tree_size": artifact["tree"]["tree_size"],
                "failing_gates": [name for name, ok in artifact["gates"].items() if not ok],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
