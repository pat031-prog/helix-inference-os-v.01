from __future__ import annotations

import hashlib
import json
import os
import re
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from helix_kv.merkle_dag import MerkleDAG
from helix_kv.semantic_router import RoutedQuery, SemanticQueryRouter, tokenize as _router_tokenize
from helix_proto.signed_receipts import (
    attach_verification,
    canonical_payload_sha256,
    derive_ephemeral_keypair,
    enforce_retrieval_signatures,
    generate_ed25519_keypair,
    key_id_for_public_key,
    sign_receipt_payload,
    unsigned_legacy_receipt,
)

try:  # Optional fast path: built by crates/helix-merkle-dag via maturin.
    from _helix_merkle_dag import RustIndexedMerkleDAG
except Exception:  # noqa: BLE001
    try:
        from helix_kv._helix_merkle_dag import RustIndexedMerkleDAG
    except Exception:  # noqa: BLE001
        RustIndexedMerkleDAG = None  # type: ignore[assignment]

PRIVATE_TAG_RE = re.compile(r"<private>[\s\S]*?</private>", re.IGNORECASE)
SECRET_PATTERNS = [
    re.compile(r"(?:api[_-]?key|secret|token|password|credential|auth)\s*[=:]\s*[\"']?[A-Za-z0-9_\-/.+]{20,}[\"']?", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{20,}", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])sk-proj-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])(?:sk|pk|rk|ak)-[A-Za-z0-9][A-Za-z0-9\-_]{19,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"gh[pus]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"xoxb-[A-Za-z0-9\-]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[A-Za-z0-9\-_]{35}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"npm_[A-Za-z0-9]{36}"),
    re.compile(r"glpat-[A-Za-z0-9\-_]{20,}"),
]
SECRET_RE = re.compile("|".join(f"(?:{pattern.pattern})" for pattern in SECRET_PATTERNS), re.IGNORECASE)
SECRET_MARKERS = (
    "api",
    "key",
    "secret",
    "token",
    "password",
    "credential",
    "auth",
    "bearer",
    "sk-",
    "pk-",
    "rk-",
    "ak-",
    "sk-proj-",
    "sk-ant-",
    "ghp_",
    "ghu_",
    "ghs_",
    "github_pat_",
    "xoxb-",
    "akia",
    "aiza",
    "eyj",
    "npm_",
    "glpat-",
)
MEMORY_TYPES = {"working", "episodic", "semantic", "procedural"}
SIGNATURE_ENFORCEMENT_MODES = {"permissive", "warn", "strict"}
RERANK_MODES = {"bm25_only", "bm25_dense_rerank", "receipt_adjudicated"}
TRUST_ROOT_VERSION = "helix-local-trust-root-v1"
TRUST_POLICY_VERSION = "helix-local-trust-policy-v1"
CHECKPOINT_VERSION = "helix-session-head-checkpoint-v1"


def _now_ms() -> float:
    return time.time() * 1000.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_scope(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if ".." in Path(text).parts or "/" in text or "\\" in text:
        raise ValueError(f"{field} cannot contain path separators or traversal")
    return text


def _resolve_retrieval_scope(scope: str | None) -> str:
    selected = str(scope or "workspace").strip().lower()
    if selected in {"global", "workspace-global"}:
        return "workspace"
    if selected not in {"workspace", "session"}:
        raise ValueError(f"unsupported retrieval_scope: {selected}")
    return selected


def _tokenize(text: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_\-]{1,}", text)]


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def privacy_filter(text: str) -> str:
    result = str(text or "")
    lowered = result.lower()
    if "<private" in lowered:
        result = PRIVATE_TAG_RE.sub("[REDACTED]", result)
        lowered = result.lower()
    if not any(marker in lowered for marker in SECRET_MARKERS):
        return result
    return SECRET_RE.sub("[REDACTED_SECRET]", result)


def source_hash(*parts: str) -> str:
    payload = "\n".join(str(part or "") for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    project: str
    agent_id: str
    session_id: str | None
    memory_type: str
    summary: str
    content: str
    importance: int
    source_hash: str
    tags: list[str]
    decay_score: float
    created_ms: float
    last_access_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "project": self.project,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "summary": self.summary,
            "content": self.content,
            "importance": self.importance,
            "source_hash": self.source_hash,
            "tags": list(self.tags),
            "decay_score": self.decay_score,
            "created_ms": self.created_ms,
            "last_access_ms": self.last_access_ms,
        }


class MemoryCatalog:
    """In-memory Merkle DAG catalog for agent observations and distilled memories."""

    _REGISTRY: dict[str, "MemoryCatalog"] = {}

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "WAL",
    ) -> None:
        # We accept db_path and journal_mode for interface compatibility, but ignore them.
        self.db_path = Path(db_path)
        self.busy_timeout_ms = busy_timeout_ms
        self.journal_mode = "memory"
        self.fts_enabled = False
        self._trust_dir = self.db_path.parent / "trust"
        self._trust_root_path = self._trust_dir / "trust_root.json"
        self._local_signing_key_path = self._trust_dir / "local_signing_key.json"
        self._local_signing_key_cache: dict[str, Any] | None = None
        
        self.dag = MerkleDAG()
        rust_index_disabled = os.environ.get("HELIX_MEMORY_RUST_INDEX", "1").lower() in {"0", "false", "off", "no"}
        rust_index_required = os.environ.get("HELIX_MEMORY_REQUIRE_RUST_INDEX", "0").lower() in {"1", "true", "on", "yes"}
        if rust_index_required and (RustIndexedMerkleDAG is None or rust_index_disabled):
            raise RuntimeError("HELIX_MEMORY_REQUIRE_RUST_INDEX is set but RustIndexedMerkleDAG is unavailable or disabled")
        self._rust_index = RustIndexedMerkleDAG() if RustIndexedMerkleDAG is not None and not rust_index_disabled else None
        self._rust_index_error: str | None = None
        self._lock = threading.Lock()
        router_disabled = os.environ.get("HELIX_SEMANTIC_QUERY_ROUTER", "1").lower() in {"0", "false", "off", "no"}
        self._semantic_router = None if router_disabled else SemanticQueryRouter()
        self._router_stats = {
            "enabled": not router_disabled,
            "calls": 0,
            "rewrites": 0,
            "pass_through": 0,
            "recent_fallback": 0,
        }
        self._router_scope_doc_count: dict[tuple[str, str | None], int] = {}
        self._router_scope_df: dict[tuple[str, str | None], dict[str, int]] = {}
        self._router_scope_anchor_scores: dict[tuple[str, str | None], dict[str, float]] = {}
        self._router_scope_recent_ids: dict[tuple[str, str | None], list[str]] = {}
        
        # Primary storage backing
        self._memories: dict[str, MemoryItem] = {}
        self._memory_receipts: dict[str, dict[str, Any]] = {}
        self._memory_node_hashes: dict[str, str] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        self._links: list[dict[str, Any]] = []

        # Track the latest node hash per session to build the parent_hash chains
        self._session_heads: dict[str, str] = {}
        self._session_lineage: dict[str, dict[str, Any]] = {}
        self._session_transitions: dict[str, list[dict[str, Any]]] = {}
        self._node_lineage: dict[str, dict[str, Any]] = {}
        self._session_checkpoints: dict[str, list[dict[str, Any]]] = {}
        self._checkpoint_by_hash: dict[str, dict[str, Any]] = {}
        self._node_checkpoint_hashes: dict[str, str] = {}
        self._quarantine_records: dict[str, list[dict[str, Any]]] = {}
        self._journal_enabled = os.environ.get("HELIX_MEMORY_JOURNAL", "1").lower() not in {"0", "false", "off", "no"}
        self._journal_path = self.db_path.with_name("memory.journal.jsonl")
        self._journal_error: str | None = None
        self._replaying_journal = False
        self._load_journal()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        journal_mode: str = "WAL",
    ) -> "MemoryCatalog":
        path_str = str(Path(path).resolve())
        if path_str not in cls._REGISTRY:
            cls._REGISTRY[path_str] = cls(path, busy_timeout_ms=busy_timeout_ms, journal_mode=journal_mode)
        return cls._REGISTRY[path_str]

    def close(self) -> None:
        pass

    def _read_json_file_unlocked(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def _write_json_file_unlocked(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")

    def _local_signing_key_unlocked(self) -> dict[str, Any]:
        if self._local_signing_key_cache:
            return dict(self._local_signing_key_cache)
        stored = self._read_json_file_unlocked(self._local_signing_key_path) or {}
        public_key = str(stored.get("public_key") or "")
        private_key = str(stored.get("private_key") or "")
        key_id = str(stored.get("key_id") or "")
        if not private_key or not public_key:
            generated = generate_ed25519_keypair()
            private_key = generated["private_key"]
            public_key = generated["public_key"]
            key_id = key_id_for_public_key(public_key)
            stored = {
                "version": "helix-local-signing-key-v1",
                "created_at_utc": _utc_now(),
                "key_id": key_id,
                "private_key": private_key,
                "public_key": public_key,
                "key_provenance": "local_self_signed",
            }
            self._write_json_file_unlocked(self._local_signing_key_path, stored)
        if not key_id:
            key_id = key_id_for_public_key(public_key)
            stored["key_id"] = key_id
            self._write_json_file_unlocked(self._local_signing_key_path, stored)

        trust_root = self._read_json_file_unlocked(self._trust_root_path) or {}
        keys = trust_root.get("keys") if isinstance(trust_root.get("keys"), list) else []
        known = {str(item.get("key_id") or "") for item in keys if isinstance(item, dict)}
        if key_id not in known or trust_root.get("active_key_id") != key_id:
            keys = [
                item
                for item in keys
                if isinstance(item, dict) and str(item.get("key_id") or "") != key_id
            ]
            keys.append(
                {
                    "key_id": key_id,
                    "public_key": public_key,
                    "signature_alg": "ed25519",
                    "key_provenance": "local_self_signed",
                    "status": "active",
                    "created_at_utc": str(stored.get("created_at_utc") or _utc_now()),
                }
            )
            trust_root = {
                **trust_root,
                "version": TRUST_ROOT_VERSION,
                "policy_version": TRUST_POLICY_VERSION,
                "active_key_id": key_id,
                "threshold": 1,
                "updated_at_utc": _utc_now(),
                "external_anchor": trust_root.get("external_anchor"),
                "keys": keys,
            }
            trust_root.setdefault("created_at_utc", str(stored.get("created_at_utc") or _utc_now()))
            self._write_json_file_unlocked(self._trust_root_path, trust_root)

        self._local_signing_key_cache = {
            "key_id": key_id,
            "private_key": private_key,
            "public_key": public_key,
            "key_provenance": "local_self_signed",
            "trust_root_path": str(self._trust_root_path),
        }
        return dict(self._local_signing_key_cache)

    def trust_root(self) -> dict[str, Any]:
        with self._lock:
            self._local_signing_key_unlocked()
            payload = self._read_json_file_unlocked(self._trust_root_path) or {}
            return dict(payload)

    def _sign_with_local_key_unlocked(
        self,
        payload: dict[str, Any],
        *,
        signer_id: str,
        key_provenance: str = "local_self_signed",
        attestation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = self._local_signing_key_unlocked()
        signable = {
            **payload,
            "trust_root_version": TRUST_ROOT_VERSION,
            "trust_policy_version": TRUST_POLICY_VERSION,
            "signing_key_id": key["key_id"],
        }
        return attach_verification(
            sign_receipt_payload(
                signable,
                private_key_b64=str(key["private_key"]),
                public_key_b64=str(key["public_key"]),
                signer_id=signer_id,
                key_provenance=key_provenance,
                attestation=attestation,
            )
        )

    def _session_lineage_state_unlocked(self, session_id: str) -> dict[str, Any]:
        return self._session_lineage.setdefault(
            str(session_id),
            {
                "session_id": str(session_id),
                "canonical_head": None,
                "head_seq": 0,
                "transition_count": 0,
                "status": "active",
                "last_transition_ms": None,
                "equivocation_count": 0,
                "policy_quarantine_count": 0,
                "latest_transition_seq": 0,
                "checkpoint_count": 0,
                "latest_checkpoint_hash": None,
            },
        )

    def _lineage_status_for_state_unlocked(self, state: dict[str, Any]) -> str:
        if int(state.get("equivocation_count") or 0) > 0:
            return "equivocation_detected"
        return "active"

    def _record_lineage_transition_unlocked(
        self,
        *,
        session_id: str | None,
        parent_hash: str | None,
        node_hash: str,
        record_kind: str,
        created_ms: float,
    ) -> dict[str, Any]:
        if not session_id:
            payload = {
                "session_id": None,
                "thread_id": None,
                "previous_head": parent_hash,
                "candidate_head": node_hash,
                "record_kind": record_kind,
                "seq": 1,
                "canonical_seq": 1,
                "canonical": True,
                "quarantined": False,
                "non_canonical": False,
                "equivocation_id": None,
                "quarantined_reason": None,
                "created_ms": created_ms,
            }
            self._node_lineage[str(node_hash)] = dict(payload)
            return payload

        session_key = str(session_id)
        state = self._session_lineage_state_unlocked(session_key)
        canonical_head = state.get("canonical_head")
        transition_seq = int(state.get("transition_count") or 0) + 1
        canonical_seq = int(state.get("head_seq") or 0)
        previous_checkpoint_hash = state.get("latest_checkpoint_hash")
        is_equivocation = bool(canonical_head and parent_hash != canonical_head)
        equivocation_id = None
        quarantined_reason = None
        canonical = not is_equivocation
        if canonical:
            canonical_seq += 1
            state["canonical_head"] = str(node_hash)
            state["head_seq"] = canonical_seq
            self._session_heads[session_key] = str(node_hash)
        else:
            state["equivocation_count"] = int(state.get("equivocation_count") or 0) + 1
            equivocation_id = f"eqv-{session_key}-{state['equivocation_count']}"
            quarantined_reason = (
                f"candidate parent {parent_hash or '<root>'} does not continue canonical head "
                f"{canonical_head or '<none>'}"
            )
            if canonical_head:
                self._session_heads[session_key] = str(canonical_head)
        state["transition_count"] = transition_seq
        state["latest_transition_seq"] = transition_seq
        state["last_transition_ms"] = float(created_ms)
        state["status"] = self._lineage_status_for_state_unlocked(state)
        payload = {
            "session_id": session_key,
            "thread_id": session_key,
            "previous_head": parent_hash,
            "candidate_head": str(node_hash),
            "record_kind": record_kind,
            "seq": transition_seq,
            "canonical_seq": canonical_seq,
            "canonical": canonical,
            "quarantined": not canonical,
            "non_canonical": not canonical,
            "equivocation_id": equivocation_id,
            "quarantined_reason": quarantined_reason,
            "previous_checkpoint_hash": previous_checkpoint_hash,
            "created_ms": float(created_ms),
        }
        self._node_lineage[str(node_hash)] = dict(payload)
        self._session_transitions.setdefault(session_key, []).append(dict(payload))
        return payload

    def _record_policy_quarantine_transition_unlocked(
        self,
        *,
        session_id: str,
        parent_hash: str | None,
        node_hash: str,
        record_kind: str,
        created_ms: float,
        quarantine_reason: str,
        quarantine_class: str = "policy",
        disposition: str = "quarantined_policy",
    ) -> dict[str, Any]:
        session_key = str(session_id)
        state = self._session_lineage_state_unlocked(session_key)
        transition_seq = int(state.get("transition_count") or 0) + 1
        canonical_seq = int(state.get("head_seq") or 0)
        previous_checkpoint_hash = state.get("latest_checkpoint_hash")
        state["transition_count"] = transition_seq
        state["latest_transition_seq"] = transition_seq
        state["last_transition_ms"] = float(created_ms)
        state["policy_quarantine_count"] = int(state.get("policy_quarantine_count") or 0) + 1
        state["status"] = self._lineage_status_for_state_unlocked(state)
        if state.get("canonical_head"):
            self._session_heads[session_key] = str(state["canonical_head"])
        payload = {
            "session_id": session_key,
            "thread_id": session_key,
            "previous_head": parent_hash,
            "candidate_head": str(node_hash),
            "record_kind": record_kind,
            "seq": transition_seq,
            "canonical_seq": canonical_seq,
            "canonical": False,
            "quarantined": True,
            "non_canonical": True,
            "equivocation_id": None,
            "quarantined_reason": quarantine_reason,
            "quarantine_class": quarantine_class,
            "disposition": disposition,
            "previous_checkpoint_hash": previous_checkpoint_hash,
            "created_ms": float(created_ms),
        }
        self._node_lineage[str(node_hash)] = dict(payload)
        self._session_transitions.setdefault(session_key, []).append(dict(payload))
        return payload

    def _checkpoint_body_unlocked(self, *, session_key: str, lineage: dict[str, Any]) -> dict[str, Any]:
        state = self._session_lineage_state_unlocked(session_key)
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "session_id": session_key,
            "thread_id": session_key,
            "canonical_head": state.get("canonical_head"),
            "head_seq": int(state.get("head_seq") or 0),
            "transition_count": int(state.get("transition_count") or 0),
            "previous_checkpoint_hash": lineage.get("previous_checkpoint_hash") or None,
            "equivocation_count": int(state.get("equivocation_count") or 0),
            "status": str(state.get("status") or "active"),
            "issued_at_utc": _utc_now(),
            "policy_version": TRUST_POLICY_VERSION,
            "external_anchor": None,
        }

    @staticmethod
    def _checkpoint_hash_body(checkpoint: dict[str, Any]) -> dict[str, Any]:
        return {
            "checkpoint_version": checkpoint.get("checkpoint_version"),
            "session_id": checkpoint.get("session_id"),
            "thread_id": checkpoint.get("thread_id"),
            "canonical_head": checkpoint.get("canonical_head"),
            "head_seq": int(checkpoint.get("head_seq") or 0),
            "transition_count": int(checkpoint.get("transition_count") or 0),
            "previous_checkpoint_hash": checkpoint.get("previous_checkpoint_hash") or None,
            "equivocation_count": int(checkpoint.get("equivocation_count") or 0),
            "status": str(checkpoint.get("status") or "active"),
            "issued_at_utc": checkpoint.get("issued_at_utc"),
            "policy_version": checkpoint.get("policy_version"),
            "external_anchor": checkpoint.get("external_anchor"),
        }

    def _verify_checkpoint_unlocked(self, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        if not checkpoint:
            return {"checkpoint_verified": False, "verification_error": "missing_checkpoint"}
        verified = attach_verification(dict(checkpoint))
        try:
            expected_hash = canonical_payload_sha256(self._checkpoint_hash_body(verified))
        except Exception as exc:  # noqa: BLE001
            expected_hash = None
            hash_ok = False
            hash_error = str(exc)
        else:
            hash_ok = expected_hash == verified.get("checkpoint_hash")
            hash_error = None if hash_ok else "checkpoint_hash mismatch"
        return {
            "checkpoint_verified": bool(verified.get("signature_verified")) and hash_ok,
            "signature_verified": bool(verified.get("signature_verified")),
            "checkpoint_hash_verified": hash_ok,
            "expected_checkpoint_hash": expected_hash,
            "verification_error": verified.get("verification_error") or hash_error,
        }

    def _record_session_checkpoint_unlocked(self, *, session_id: str | None, lineage: dict[str, Any]) -> dict[str, Any] | None:
        if not session_id or not lineage.get("canonical", True):
            return None
        session_key = str(session_id)
        body = self._checkpoint_body_unlocked(session_key=session_key, lineage=lineage)
        checkpoint_hash = canonical_payload_sha256(body)
        checkpoint = self._sign_with_local_key_unlocked(
            {
                **body,
                "checkpoint_hash": checkpoint_hash,
            },
            signer_id="helix-local-checkpoint",
        )
        state = self._session_lineage_state_unlocked(session_key)
        state["latest_checkpoint_hash"] = checkpoint_hash
        state["checkpoint_count"] = int(state.get("checkpoint_count") or 0) + 1
        lineage["checkpoint_hash"] = checkpoint_hash
        self._node_checkpoint_hashes[str(lineage.get("candidate_head") or "")] = checkpoint_hash
        self._session_checkpoints.setdefault(session_key, []).append(dict(checkpoint))
        self._checkpoint_by_hash[checkpoint_hash] = dict(checkpoint)
        return checkpoint

    def _record_quarantine_record_unlocked(self, *, session_id: str | None, lineage: dict[str, Any]) -> dict[str, Any] | None:
        if not session_id or not lineage.get("quarantined"):
            return None
        session_key = str(session_id)
        state = self._session_lineage_state_unlocked(session_key)
        payload = {
            "quarantine_record_version": "helix-lineage-quarantine-v1",
            "session_id": session_key,
            "thread_id": session_key,
            "candidate_head": lineage.get("candidate_head"),
            "previous_head": lineage.get("previous_head"),
            "canonical_head": state.get("canonical_head"),
            "equivocation_id": lineage.get("equivocation_id"),
            "quarantined_reason": lineage.get("quarantined_reason"),
            "quarantine_class": lineage.get("quarantine_class"),
            "disposition": lineage.get("disposition") or ("quarantined_equivocation" if lineage.get("equivocation_id") else "quarantined_policy"),
            "transition_seq": int(lineage.get("seq") or 0),
            "canonical_seq": int(lineage.get("canonical_seq") or 0),
            "checkpoint_hash": state.get("latest_checkpoint_hash"),
            "issued_at_utc": _utc_now(),
            "policy_version": TRUST_POLICY_VERSION,
        }
        record = self._sign_with_local_key_unlocked(payload, signer_id="helix-local-quarantine")
        self._quarantine_records.setdefault(session_key, []).append(dict(record))
        lineage["checkpoint_hash"] = state.get("latest_checkpoint_hash")
        return record

    def _restore_session_checkpoint_unlocked(self, session_id: str | None, checkpoint: dict[str, Any] | None) -> None:
        if not session_id or not isinstance(checkpoint, dict) or not checkpoint:
            return
        session_key = str(session_id)
        checkpoint_hash = str(checkpoint.get("checkpoint_hash") or "")
        if not checkpoint_hash:
            return
        self._session_checkpoints.setdefault(session_key, []).append(dict(checkpoint))
        self._checkpoint_by_hash[checkpoint_hash] = dict(checkpoint)
        state = self._session_lineage_state_unlocked(session_key)
        state["latest_checkpoint_hash"] = checkpoint_hash
        state["checkpoint_count"] = max(int(state.get("checkpoint_count") or 0), len(self._session_checkpoints.get(session_key, [])))

    def _restore_quarantine_record_unlocked(self, session_id: str | None, record: dict[str, Any] | None) -> None:
        if not session_id or not isinstance(record, dict) or not record:
            return
        self._quarantine_records.setdefault(str(session_id), []).append(dict(record))

    def _restore_lineage_transition_unlocked(
        self,
        *,
        session_id: str | None,
        node_hash: str,
        transition: dict[str, Any] | None,
        parent_hash: str | None,
        record_kind: str,
        created_ms: float,
    ) -> None:
        if transition is None:
            self._record_lineage_transition_unlocked(
                session_id=session_id,
                parent_hash=parent_hash,
                node_hash=node_hash,
                record_kind=record_kind,
                created_ms=created_ms,
            )
            return
        restored = dict(transition)
        restored.setdefault("session_id", session_id)
        restored.setdefault("thread_id", session_id)
        restored.setdefault("previous_head", parent_hash)
        restored.setdefault("candidate_head", str(node_hash))
        restored.setdefault("record_kind", record_kind)
        restored.setdefault("seq", 1)
        restored.setdefault("canonical_seq", 1 if restored.get("canonical", True) else 0)
        restored.setdefault("canonical", True)
        restored.setdefault("quarantined", not bool(restored.get("canonical", True)))
        restored.setdefault("non_canonical", not bool(restored.get("canonical", True)))
        restored.setdefault("equivocation_id", None)
        restored.setdefault("quarantined_reason", None)
        restored.setdefault(
            "quarantine_class",
            "equivocation" if restored.get("equivocation_id") else None,
        )
        restored.setdefault(
            "disposition",
            "canonical"
            if restored.get("canonical", True)
            else ("quarantined_equivocation" if restored.get("equivocation_id") else "quarantined_policy"),
        )
        restored.setdefault("checkpoint_hash", None)
        restored.setdefault("previous_checkpoint_hash", None)
        restored.setdefault("created_ms", float(created_ms))
        self._node_lineage[str(node_hash)] = restored
        if not session_id:
            return
        session_key = str(session_id)
        state = self._session_lineage_state_unlocked(session_key)
        state["transition_count"] = max(int(state.get("transition_count") or 0), int(restored.get("seq") or 0))
        state["latest_transition_seq"] = max(int(state.get("latest_transition_seq") or 0), int(restored.get("seq") or 0))
        state["last_transition_ms"] = max(float(state.get("last_transition_ms") or 0.0), float(restored.get("created_ms") or 0.0))
        if restored.get("canonical", True):
            state["canonical_head"] = str(node_hash)
            state["head_seq"] = max(int(state.get("head_seq") or 0), int(restored.get("canonical_seq") or 0))
            self._session_heads[session_key] = str(node_hash)
        else:
            if restored.get("quarantine_class") == "policy" and not restored.get("equivocation_id"):
                state["policy_quarantine_count"] = int(state.get("policy_quarantine_count") or 0) + 1
            else:
                state["equivocation_count"] = max(
                    int(state.get("equivocation_count") or 0),
                    int(restored.get("equivocation_id", "0").rsplit("-", 1)[-1]) if restored.get("equivocation_id") else int(state.get("equivocation_count") or 0) + 1,
                )
            if state.get("canonical_head"):
                self._session_heads[session_key] = str(state["canonical_head"])
        state["status"] = self._lineage_status_for_state_unlocked(state)
        self._session_transitions.setdefault(session_key, []).append(restored)

    def _lineage_for_node_unlocked(self, node_hash: str | None, session_id: str | None = None) -> dict[str, Any]:
        lineage = dict(self._node_lineage.get(str(node_hash or "")) or {})
        active_session = str(session_id or lineage.get("session_id") or "")
        state = dict(self._session_lineage.get(active_session) or {})
        checkpoint_anchor = self._checkpoint_anchor_for_node_unlocked(
            str(node_hash or ""),
            active_session or None,
        )
        canonical = bool(lineage.get("canonical", True))
        quarantined = bool(lineage.get("quarantined", False))
        non_canonical = bool(lineage.get("non_canonical", False))
        if checkpoint_anchor.get("checkpoint_anchored"):
            canonical = True
            quarantined = False
            non_canonical = False
        return {
            "thread_id": active_session or None,
            "canonical": canonical,
            "quarantined": quarantined,
            "non_canonical": non_canonical,
            "equivocation_id": None if checkpoint_anchor.get("checkpoint_anchored") else lineage.get("equivocation_id"),
            "quarantined_reason": None if checkpoint_anchor.get("checkpoint_anchored") else lineage.get("quarantined_reason"),
            "quarantine_class": None if checkpoint_anchor.get("checkpoint_anchored") else lineage.get("quarantine_class"),
            "disposition": "canonical" if checkpoint_anchor.get("checkpoint_anchored") else (lineage.get("disposition") or ("canonical" if canonical else "quarantined_equivocation")),
            "canonical_head": state.get("canonical_head"),
            "head_seq": state.get("head_seq"),
            "equivocation_count": state.get("equivocation_count", 0),
            "policy_quarantine_count": state.get("policy_quarantine_count", 0),
            "lineage_status": state.get("status", "active"),
            "lineage_transition_seq": lineage.get("seq"),
            "lineage_previous_head": lineage.get("previous_head"),
            "checkpoint_hash": (
                checkpoint_anchor.get("checkpoint_hash")
                or lineage.get("checkpoint_hash")
                or self._node_checkpoint_hashes.get(str(node_hash or ""))
                or state.get("latest_checkpoint_hash")
            ),
            "checkpoint_status": (
                "anchored"
                if checkpoint_anchor.get("checkpoint_anchored")
                else ("ok" if state.get("latest_checkpoint_hash") else "missing_legacy")
            ),
            "checkpoint_anchored": bool(checkpoint_anchor.get("checkpoint_anchored")),
        }

    def _attach_lineage(self, payload: dict[str, Any], *, node_hash: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        lineage = self._lineage_for_node_unlocked(node_hash or str(payload.get("node_hash") or ""), session_id or payload.get("session_id"))
        payload.update(lineage)
        return payload

    def _checkpoint_anchor_for_node_unlocked(self, node_hash: str | None, session_id: str | None) -> dict[str, Any]:
        if not node_hash or not session_id:
            return {"checkpoint_anchored": False}
        checkpoints = self._session_checkpoints.get(str(session_id), [])
        if not checkpoints:
            return {"checkpoint_anchored": False}
        latest = dict(checkpoints[-1])
        verification = self._verify_checkpoint_unlocked(latest)
        if not verification.get("checkpoint_verified"):
            return {
                "checkpoint_anchored": False,
                "checkpoint_error": verification.get("verification_error") or "checkpoint_verification_failed",
            }
        canonical_head = str(latest.get("canonical_head") or "")
        if not canonical_head:
            return {"checkpoint_anchored": False}
        chain_status = self.verify_chain(canonical_head)
        if chain_status.get("status") != "verified":
            return {
                "checkpoint_anchored": False,
                "checkpoint_error": chain_status.get("error") or chain_status.get("status"),
            }
        canonical_chain = self.dag.audit_chain(canonical_head)
        chain_hashes = {
            str(getattr(item, "hash", "") or (item.get("hash") if isinstance(item, dict) else ""))
            for item in canonical_chain
        }
        return {
            "checkpoint_anchored": str(node_hash) in chain_hashes,
            "checkpoint_hash": latest.get("checkpoint_hash"),
            "checkpoint_canonical_head": canonical_head,
        }

    def _memory_visible_unlocked(self, memory_id: str, *, include_quarantined: bool) -> bool:
        node_hash = self._memory_node_hashes.get(str(memory_id))
        if not node_hash:
            return True
        lineage = self._node_lineage.get(str(node_hash))
        if lineage is None:
            return True
        if include_quarantined:
            return True
        if self._checkpoint_anchor_for_node_unlocked(node_hash, lineage.get("session_id")).get("checkpoint_anchored"):
            return True
        return bool(lineage.get("canonical", True)) and not bool(lineage.get("quarantined", False))

    def _observation_visible_unlocked(self, observation_id: str, *, include_quarantined: bool) -> bool:
        raw = self._observations.get(str(observation_id))
        node_hash = str((raw or {}).get("node_hash") or "")
        if not node_hash:
            return True
        lineage = self._node_lineage.get(node_hash)
        if lineage is None:
            return True
        if include_quarantined:
            return True
        if self._checkpoint_anchor_for_node_unlocked(node_hash, lineage.get("session_id")).get("checkpoint_anchored"):
            return True
        return bool(lineage.get("canonical", True)) and not bool(lineage.get("quarantined", False))

    def _append_journal(self, entry: dict[str, Any]) -> None:
        if not self._journal_enabled or self._replaying_journal:
            return
        payload = {
            "journal_version": "helix-memory-journal-v1",
            "created_utc": _utc_now(),
            **entry,
        }
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self._journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001
            self._journal_error = str(exc)

    def _load_journal(self) -> None:
        if not self._journal_enabled or not self._journal_path.exists():
            return
        self._replaying_journal = True
        try:
            with self._journal_path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                        self._replay_journal_entry(entry)
                    except Exception as exc:  # noqa: BLE001
                        self._journal_error = f"{self._journal_path}:{line_number}: {exc}"
        finally:
            self._replaying_journal = False

    def _replay_journal_entry(self, entry: dict[str, Any]) -> None:
        op = str(entry.get("op") or "")
        if op == "observe":
            payload = dict(entry["payload"])
            content_dump = str(entry["content_dump"])
            parent_hash = entry.get("parent_hash")
            node = self.dag._insert_unlocked(content_dump, parent_hash=parent_hash)
            stored_hash = entry.get("node_hash")
            if stored_hash and stored_hash != node.hash:
                raise ValueError(f"observation node hash mismatch: {stored_hash} != {node.hash}")
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent_hash,
                metadata={
                    **payload,
                    **(entry.get("lineage") or {}),
                    "record_kind": "observation",
                    "index_content": payload.get("content") or "",
                    "content_available": True,
                    "audit_status": "verified",
                },
            )
            session_id = payload.get("session_id")
            payload["node_hash"] = node.hash
            self._restore_lineage_transition_unlocked(
                session_id=str(session_id) if session_id else None,
                node_hash=node.hash,
                transition=entry.get("lineage") if isinstance(entry.get("lineage"), dict) else None,
                parent_hash=parent_hash,
                record_kind="observation",
                created_ms=float(payload.get("created_ms") or _now_ms()),
            )
            self._restore_session_checkpoint_unlocked(str(session_id) if session_id else None, entry.get("checkpoint") if isinstance(entry.get("checkpoint"), dict) else None)
            self._restore_quarantine_record_unlocked(str(session_id) if session_id else None, entry.get("quarantine_record") if isinstance(entry.get("quarantine_record"), dict) else None)
            self._observations[str(payload["observation_id"])] = payload
            return
        if op == "remember":
            payload = dict(entry["payload"])
            item = MemoryItem(**payload)
            content_dump = str(entry["content_dump"])
            parent_hash = entry.get("parent_hash")
            node = self.dag._insert_unlocked(content_dump, parent_hash=parent_hash)
            stored_hash = entry.get("node_hash")
            if stored_hash and stored_hash != node.hash:
                raise ValueError(f"memory node hash mismatch: {stored_hash} != {node.hash}")
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent_hash,
                metadata={
                    **item.to_dict(),
                    **(entry.get("lineage") or {}),
                    "record_kind": "memory",
                    "index_content": item.content,
                    "content_available": True,
                    "audit_status": "verified",
                },
            )
            self._restore_lineage_transition_unlocked(
                session_id=item.session_id,
                node_hash=node.hash,
                transition=entry.get("lineage") if isinstance(entry.get("lineage"), dict) else None,
                parent_hash=parent_hash,
                record_kind="memory",
                created_ms=float(item.created_ms),
            )
            self._restore_session_checkpoint_unlocked(item.session_id, entry.get("checkpoint") if isinstance(entry.get("checkpoint"), dict) else None)
            self._restore_quarantine_record_unlocked(item.session_id, entry.get("quarantine_record") if isinstance(entry.get("quarantine_record"), dict) else None)
            self._memories[item.memory_id] = item
            self._memory_node_hashes[item.memory_id] = node.hash
            self._memory_receipts[item.memory_id] = dict(entry.get("receipt") or {})
            self._index_router_item_unlocked(item)
            return
        if op == "remember_quarantined":
            payload = dict(entry["payload"])
            item = MemoryItem(**payload)
            content_dump = str(entry["content_dump"])
            parent_hash = entry.get("parent_hash")
            node = self.dag._insert_unlocked(content_dump, parent_hash=parent_hash)
            stored_hash = entry.get("node_hash")
            if stored_hash and stored_hash != node.hash:
                raise ValueError(f"quarantined memory node hash mismatch: {stored_hash} != {node.hash}")
            lineage = entry.get("lineage") if isinstance(entry.get("lineage"), dict) else None
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent_hash,
                metadata={
                    **item.to_dict(),
                    **(lineage or {}),
                    "record_kind": str((lineage or {}).get("record_kind") or "memory"),
                    "index_content": item.content,
                    "content_available": True,
                    "audit_status": "quarantined",
                },
            )
            self._restore_lineage_transition_unlocked(
                session_id=item.session_id,
                node_hash=node.hash,
                transition=lineage,
                parent_hash=parent_hash,
                record_kind=str((lineage or {}).get("record_kind") or "memory"),
                created_ms=float(item.created_ms),
            )
            self._restore_quarantine_record_unlocked(item.session_id, entry.get("quarantine_record") if isinstance(entry.get("quarantine_record"), dict) else None)
            self._memories[item.memory_id] = item
            self._memory_node_hashes[item.memory_id] = node.hash
            self._memory_receipts[item.memory_id] = dict(entry.get("receipt") or {})
            self._index_router_item_unlocked(item)
            return
        if op == "link_session_memory":
            self._links.append(dict(entry["payload"]))
            return

    def observe(
        self,
        *,
        project: str,
        agent_id: str,
        content: str,
        summary: str | None = None,
        session_id: str | None = None,
        observation_type: str = "working",
        importance: int = 5,
        tags: Iterable[str] | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        project = _safe_scope(project, field="project")
        agent_id = _safe_scope(agent_id, field="agent_id")
        clean_content = privacy_filter(content)
        clean_summary = privacy_filter(summary or clean_content[:160])
        digest = source_hash(project, agent_id, clean_summary, clean_content)
        obs_id = observation_id or f"obs-{digest[:24]}"
        now = _now_ms()
        tag_list = [str(item) for item in (tags or [])]
        
        payload = {
            "observation_id": obs_id,
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "observation_type": str(observation_type),
            "summary": clean_summary,
            "content": clean_content,
            "importance": int(importance),
            "source_hash": digest,
            "tags": tag_list,
            "created_ms": now,
            "last_access_ms": now,
        }
        
        with self._lock:
            # MerkleDAG insertion — use _insert_unlocked to avoid double-lock
            # (self._lock + dag._lock would be a deadlock risk if ordering inverts)
            content_dump = json.dumps(payload, sort_keys=True)
            parent = self._session_heads.get(session_id) if session_id else None
            node = self.dag._insert_unlocked(content_dump, parent_hash=parent)
            lineage = self._record_lineage_transition_unlocked(
                session_id=session_id,
                parent_hash=parent,
                node_hash=node.hash,
                record_kind="observation",
                created_ms=now,
            )
            checkpoint = self._record_session_checkpoint_unlocked(session_id=session_id, lineage=lineage)
            quarantine_record = self._record_quarantine_record_unlocked(session_id=session_id, lineage=lineage)
            payload["node_hash"] = node.hash
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent,
                metadata={
                    **payload,
                    **lineage,
                    "record_kind": "observation",
                    "index_content": clean_content,
                    "content_available": True,
                    "audit_status": "verified",
                },
            )

            self._observations[obs_id] = payload
            self._append_journal(
                {
                    "op": "observe",
                    "payload": payload,
                    "content_dump": content_dump,
                    "parent_hash": parent,
                    "node_hash": node.hash,
                    "lineage": lineage,
                    "checkpoint": checkpoint,
                    "quarantine_record": quarantine_record,
                }
            )

        return self._attach_lineage(
            {
            "observation_id": obs_id,
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "source_hash": digest,
            "content": clean_content,
            "summary": clean_summary,
            "tags": tag_list,
            "node_hash": node.hash,
            },
            node_hash=node.hash,
            session_id=session_id,
        )

    def remember(
        self,
        *,
        project: str,
        agent_id: str,
        content: str,
        summary: str | None = None,
        session_id: str | None = None,
        memory_type: str = "episodic",
        importance: int = 5,
        tags: Iterable[str] | None = None,
        memory_id: str | None = None,
        decay_score: float = 1.0,
        llm_call_id: str | None = None,
    ) -> MemoryItem:
        project = _safe_scope(project, field="project")
        agent_id = _safe_scope(agent_id, field="agent_id")
        memory_type = str(memory_type)
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"unsupported memory_type: {memory_type}")
        clean_content = privacy_filter(content)
        clean_summary = privacy_filter(summary or clean_content[:160])
        digest = source_hash(project, agent_id, memory_type, clean_summary, clean_content)
        mem_id = memory_id or f"mem-{digest[:24]}"
        now = _now_ms()
        tag_list = [str(item) for item in (tags or [])]
        
        item = MemoryItem(
            memory_id=mem_id,
            project=project,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            summary=clean_summary,
            content=clean_content,
            importance=int(importance),
            source_hash=digest,
            tags=tag_list,
            decay_score=float(decay_score),
            created_ms=now,
            last_access_ms=now,
        )
        
        with self._lock:
            content_dump = json.dumps(item.to_dict(), sort_keys=True)
            parent = self._session_heads.get(session_id) if session_id else None
            node = self.dag._insert_unlocked(content_dump, parent_hash=parent)
            lineage = self._record_lineage_transition_unlocked(
                session_id=session_id,
                parent_hash=parent,
                node_hash=node.hash,
                record_kind="memory",
                created_ms=now,
            )
            checkpoint = self._record_session_checkpoint_unlocked(session_id=session_id, lineage=lineage)
            quarantine_record = self._record_quarantine_record_unlocked(session_id=session_id, lineage=lineage)
            receipt = self._build_receipt(
                item=item,
                node_hash=node.hash,
                parent_hash=parent,
                llm_call_id=llm_call_id,
                lineage=lineage,
            )
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent,
                metadata={
                    **item.to_dict(),
                    **lineage,
                    "record_kind": "memory",
                    "index_content": clean_content,
                    "content_available": True,
                    "audit_status": "verified",
                },
            )

            self._memories[mem_id] = item
            self._memory_node_hashes[mem_id] = node.hash
            self._memory_receipts[mem_id] = receipt
            self._index_router_item_unlocked(item)
            self._append_journal(
                {
                    "op": "remember",
                    "payload": item.to_dict(),
                    "content_dump": content_dump,
                    "parent_hash": parent,
                    "node_hash": node.hash,
                    "receipt": receipt,
                    "lineage": lineage,
                    "checkpoint": checkpoint,
                    "quarantine_record": quarantine_record,
                }
            )
            
        return item

    def remember_quarantined(
        self,
        *,
        project: str,
        agent_id: str,
        content: str,
        summary: str | None = None,
        session_id: str,
        memory_type: str = "episodic",
        importance: int = 5,
        tags: Iterable[str] | None = None,
        memory_id: str | None = None,
        decay_score: float = 1.0,
        llm_call_id: str | None = None,
        parent_hash: str | None = None,
        record_kind: str = "memory",
        quarantine_reason: str = "policy_refusal",
        quarantine_class: str = "policy",
        disposition: str = "quarantined_policy",
    ) -> dict[str, Any]:
        project = _safe_scope(project, field="project")
        agent_id = _safe_scope(agent_id, field="agent_id")
        session_key = _safe_scope(session_id, field="session_id")
        memory_type = str(memory_type)
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"unsupported memory_type: {memory_type}")
        clean_content = privacy_filter(content)
        clean_summary = privacy_filter(summary or clean_content[:160])
        digest = source_hash(project, agent_id, memory_type, clean_summary, clean_content, record_kind, quarantine_reason)
        mem_id = memory_id or f"mem-{digest[:24]}"
        now = _now_ms()
        tag_list = [str(item) for item in (tags or [])]

        item = MemoryItem(
            memory_id=mem_id,
            project=project,
            agent_id=agent_id,
            session_id=session_key,
            memory_type=memory_type,
            summary=clean_summary,
            content=clean_content,
            importance=int(importance),
            source_hash=digest,
            tags=tag_list,
            decay_score=float(decay_score),
            created_ms=now,
            last_access_ms=now,
        )

        with self._lock:
            content_dump = json.dumps(item.to_dict(), sort_keys=True)
            parent = parent_hash if parent_hash is not None else self._session_heads.get(session_key)
            node = self.dag._insert_unlocked(content_dump, parent_hash=parent)
            lineage = self._record_policy_quarantine_transition_unlocked(
                session_id=session_key,
                parent_hash=parent,
                node_hash=node.hash,
                record_kind=str(record_kind),
                created_ms=now,
                quarantine_reason=str(quarantine_reason),
                quarantine_class=str(quarantine_class or "policy"),
                disposition=str(disposition or "quarantined_policy"),
            )
            quarantine_record = self._record_quarantine_record_unlocked(session_id=session_key, lineage=lineage)
            receipt = self._build_receipt(
                item=item,
                node_hash=node.hash,
                parent_hash=parent,
                llm_call_id=llm_call_id,
                lineage=lineage,
            )
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent,
                metadata={
                    **item.to_dict(),
                    **lineage,
                    "record_kind": str(record_kind),
                    "index_content": clean_content,
                    "content_available": True,
                    "audit_status": "quarantined",
                },
            )

            self._memories[mem_id] = item
            self._memory_node_hashes[mem_id] = node.hash
            self._memory_receipts[mem_id] = receipt
            self._index_router_item_unlocked(item)
            self._append_journal(
                {
                    "op": "remember_quarantined",
                    "payload": item.to_dict(),
                    "content_dump": content_dump,
                    "parent_hash": parent,
                    "node_hash": node.hash,
                    "receipt": receipt,
                    "lineage": lineage,
                    "quarantine_record": quarantine_record,
                }
            )

            return self._attach_receipt(
                {
                    **item.to_dict(),
                    "node_hash": node.hash,
                    "record_kind": str(record_kind),
                    "checkpoint_hash": lineage.get("checkpoint_hash"),
                }
            )

    def bulk_remember(
        self,
        items: list[dict[str, Any]],
    ) -> list[MemoryItem]:
        """Batch insert memories. Uses Rust insert_indexed_batch when available.

        Each item dict must have: project, agent_id, content.
        Optional: summary, session_id, memory_type, importance, tags, memory_id, decay_score.

        Returns list of MemoryItem in insertion order.
        """
        if not items:
            return []

        # --- Phase 1: Prepare all MemoryItems (pure Python, no lock) ---
        prepared: list[tuple[MemoryItem, str, str | None]] = []  # (item, content_dump, llm_call_id)
        for raw in items:
            project = _safe_scope(raw["project"], field="project")
            agent_id = _safe_scope(raw["agent_id"], field="agent_id")
            memory_type = str(raw.get("memory_type", "episodic"))
            if memory_type not in MEMORY_TYPES:
                raise ValueError(f"unsupported memory_type: {memory_type}")
            clean_content = privacy_filter(raw["content"])
            clean_summary = privacy_filter(raw.get("summary") or clean_content[:160])
            digest = source_hash(project, agent_id, memory_type, clean_summary, clean_content)
            mem_id = raw.get("memory_id") or f"mem-{digest[:24]}"
            now = _now_ms()
            tag_list = [str(t) for t in (raw.get("tags") or [])]

            item = MemoryItem(
                memory_id=mem_id,
                project=project,
                agent_id=agent_id,
                session_id=raw.get("session_id"),
                memory_type=memory_type,
                summary=clean_summary,
                content=clean_content,
                importance=int(raw.get("importance", 5)),
                source_hash=digest,
                tags=tag_list,
                decay_score=float(raw.get("decay_score", 1.0)),
                created_ms=now,
                last_access_ms=now,
            )
            content_dump = json.dumps(item.to_dict(), sort_keys=True)
            prepared.append((item, content_dump, raw.get("llm_call_id")))

        rust_batch_payload: str | None = None
        if self._rust_index is not None:
            batch_records = []
            for item, content_dump, _llm_call_id in prepared:
                batch_records.append({
                    "content": content_dump,
                    "metadata": {
                        **item.to_dict(),
                        "record_kind": "memory",
                        "index_content": item.content,
                        "content_available": True,
                        "audit_status": "verified",
                    },
                })
            rust_batch_payload = json.dumps(batch_records, sort_keys=True, separators=(",", ":"))

        # --- Phase 2: Batch insert under lock ---
        with self._lock:
            # Python DAG inserts (must be sequential for parent_hash chaining)
            for item, content_dump, llm_call_id in prepared:
                parent = self._session_heads.get(item.session_id) if item.session_id else None
                node = self.dag._insert_unlocked(content_dump, parent_hash=parent)
                lineage = self._record_lineage_transition_unlocked(
                    session_id=item.session_id,
                    parent_hash=parent,
                    node_hash=node.hash,
                    record_kind="memory",
                    created_ms=float(item.created_ms),
                )
                checkpoint = self._record_session_checkpoint_unlocked(session_id=item.session_id, lineage=lineage)
                quarantine_record = self._record_quarantine_record_unlocked(session_id=item.session_id, lineage=lineage)
                self._memories[item.memory_id] = item
                self._memory_node_hashes[item.memory_id] = node.hash
                receipt = self._build_receipt(
                    item=item,
                    node_hash=node.hash,
                    parent_hash=parent,
                    llm_call_id=llm_call_id,
                    lineage=lineage,
                )
                self._memory_receipts[item.memory_id] = receipt
                self._index_router_item_unlocked(item)
                self._append_journal(
                    {
                        "op": "remember",
                        "payload": item.to_dict(),
                        "content_dump": content_dump,
                        "parent_hash": parent,
                        "node_hash": node.hash,
                        "receipt": receipt,
                        "lineage": lineage,
                        "checkpoint": checkpoint,
                        "quarantine_record": quarantine_record,
                    }
                )

            # Rust batch insert (one cross-boundary call)
            if self._rust_index is not None and rust_batch_payload is not None:
                try:
                    self._rust_index.insert_indexed_batch(rust_batch_payload)
                except Exception as exc:  # noqa: BLE001
                    self._rust_index = None
                    self._rust_index_error = str(exc)

        return [item for item, _content_dump, _llm_call_id in prepared]

    def _insert_rust_indexed(
        self,
        *,
        content_dump: str,
        parent_hash: str | None,
        metadata: dict[str, Any],
    ) -> None:
        if self._rust_index is None:
            return
        try:
            self._rust_index.insert_indexed(
                content_dump,
                parent_hash,
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            )
        except Exception as exc:  # noqa: BLE001
            # The Rust index is an acceleration path. Never lose a memory insert
            # because the optional extension failed; fall back to Python scan.
            self._rust_index = None
            self._rust_index_error = str(exc)

    def _index_router_item_unlocked(self, item: MemoryItem) -> None:
        if self._semantic_router is None:
            return
        scopes = [(item.project, None), (item.project, item.agent_id)]
        field_terms = [
            (_router_tokenize(item.summary), 3.0),
            (_router_tokenize(" ".join(item.tags)), 2.2),
            (_router_tokenize(item.content), 1.0),
        ]
        quality = 1.0 + (max(float(item.importance), 0.0) / 10.0) * 0.4 + max(float(item.decay_score), 0.0) * 0.2
        per_item_terms: set[str] = set()
        per_item_scores: dict[str, float] = {}
        for terms, field_weight in field_terms:
            for term in terms:
                if self._semantic_router._is_generic(term) or not self._semantic_router._is_anchor_candidate(term):
                    continue
                structural_bonus = 1.6 if self._semantic_router._has_structural_signal(term) else 1.0
                score = field_weight * quality * structural_bonus
                per_item_terms.add(term)
                per_item_scores[term] = max(per_item_scores.get(term, 0.0), score)

        for scope in scopes:
            self._router_scope_doc_count[scope] = self._router_scope_doc_count.get(scope, 0) + 1
            df = self._router_scope_df.setdefault(scope, {})
            anchor_scores = self._router_scope_anchor_scores.setdefault(scope, {})
            recent_ids = self._router_scope_recent_ids.setdefault(scope, [])
            for term in per_item_terms:
                df[term] = df.get(term, 0) + 1
            for term, score in per_item_scores.items():
                anchor_scores[term] = max(anchor_scores.get(term, 0.0), score)
            if item.memory_id not in recent_ids:
                recent_ids.append(item.memory_id)
            recent_ids.sort(
                key=lambda mem_id: (
                    self._memories[mem_id].importance * self._memories[mem_id].decay_score,
                    self._memories[mem_id].last_access_ms,
                ),
                reverse=True,
            )
            del recent_ids[128:]

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._lock:
            return self._memories.get(str(memory_id))

    def get_memory_receipt(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            receipt = self._memory_receipts.get(str(memory_id))
            return attach_verification(dict(receipt)) if receipt else None

    def get_memory_node_hash(self, memory_id: str) -> str | None:
        with self._lock:
            return self._memory_node_hashes.get(str(memory_id))

    def link_session_memory(self, *, session_id: str, memory_id: str, relation: str = "supports") -> dict[str, Any]:
        if not str(session_id).strip() or ".." in Path(str(session_id)).parts:
            raise ValueError("session_id is required and cannot contain traversal")
            
        with self._lock:
            if memory_id not in self._memories:
                raise KeyError(f"memory not found: {memory_id}")
            now = _now_ms()
            payload = {
                "session_id": str(session_id),
                "memory_id": str(memory_id),
                "relation": str(relation),
                "created_ms": now,
            }
            self._links.append(payload)
            self._append_journal({"op": "link_session_memory", "payload": payload})
            return payload

    def search(
        self,
        *,
        project: str,
        agent_id: str | None,
        session_id: str | None = None,
        query: str,
        limit: int = 5,
        memory_types: Iterable[str] | None = None,
        exclude_memory_ids: Iterable[str] | None = None,
        route_query: bool | None = None,
        signature_enforcement: str | None = None,
        rerank_mode: str | None = None,
        retrieval_scope: str = "workspace",
        include_quarantined: bool = False,
    ) -> list[dict[str, Any]]:
        project = _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")
        session_filter = None if session_id is None else _safe_scope(session_id, field="session_id")
        selected_scope = _resolve_retrieval_scope(retrieval_scope)
        tokens = _tokenize(query)
        if not tokens:
            return []
        type_filter = {str(item) for item in (memory_types or [])}
        exclude_ids = {str(item) for item in (exclude_memory_ids or [])}
        routed = self._route_query(
            project=project,
            agent_filter=agent_filter,
            query=query,
            type_filter=type_filter,
            exclude_ids=exclude_ids,
            route_query=route_query,
        )
        if routed.action == "empty":
            return []
        if routed.action == "recent_fallback":
            recent_results = self._search_recent_fallback(
                project=project,
                agent_filter=agent_filter,
                session_filter=session_filter,
                retrieval_scope=selected_scope,
                limit=limit,
                type_filter=type_filter,
                exclude_ids=exclude_ids,
                routed=routed,
                include_quarantined=include_quarantined,
            )
            return self._finalize_search_results(
                recent_results,
                query=query,
                limit=limit,
                signature_enforcement=signature_enforcement,
                rerank_mode=self._resolve_rerank_mode(rerank_mode),
            )
        effective_query = routed.routed_query or query
        effective_tokens = _tokenize(effective_query)
        effective_limit = int(limit)
        selected_rerank_mode = self._resolve_rerank_mode(rerank_mode)
        if selected_rerank_mode in {"bm25_dense_rerank", "receipt_adjudicated"}:
            effective_limit = max(effective_limit * 4, 20)

        if self._rust_index is not None:
            results = self._search_rust_index(
                project=project,
                agent_filter=agent_filter,
                session_filter=session_filter,
                retrieval_scope=selected_scope,
                query=effective_query,
                limit=effective_limit,
                type_filter=type_filter,
                exclude_ids=exclude_ids,
                routed=routed,
                include_quarantined=include_quarantined,
            )
            if results is not None:
                return self._finalize_search_results(
                    results,
                    query=effective_query,
                    limit=limit,
                    signature_enforcement=signature_enforcement,
                    rerank_mode=selected_rerank_mode,
                )

        results = self._search_python_scan(
            project=project,
            agent_filter=agent_filter,
            session_filter=session_filter,
            retrieval_scope=selected_scope,
            tokens=effective_tokens,
            limit=effective_limit,
            type_filter=type_filter,
            exclude_ids=exclude_ids,
            routed=routed,
            include_quarantined=include_quarantined,
        )
        return self._finalize_search_results(
            results,
            query=effective_query,
            limit=limit,
            signature_enforcement=signature_enforcement,
            rerank_mode=selected_rerank_mode,
        )

    def _resolve_signature_enforcement(self, mode: str | None) -> str:
        # Default warn keeps legacy unsigned memory visible, but annotated. Strict
        # remains available for hard audits via argument or environment variable.
        selected = str(mode or os.environ.get("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "warn")).lower()
        if selected not in SIGNATURE_ENFORCEMENT_MODES:
            raise ValueError(f"unsupported signature_enforcement: {selected}")
        return selected

    def _resolve_rerank_mode(self, mode: str | None) -> str:
        selected = str(mode or os.environ.get("HELIX_RETRIEVAL_RERANK_MODE", "bm25_only")).lower()
        if selected not in RERANK_MODES:
            raise ValueError(f"unsupported rerank_mode: {selected}")
        return selected

    def _receipt_payload(
        self,
        *,
        item: MemoryItem,
        node_hash: str,
        parent_hash: str | None,
        signer_id: str,
        llm_call_id: str | None,
        lineage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lineage_payload = dict(lineage or {})
        return {
            "node_hash": str(node_hash),
            "parent_hash": parent_hash,
            "memory_id": item.memory_id,
            "project": item.project,
            "agent_id": item.agent_id,
            "session_id": item.session_id,
            "thread_id": item.session_id,
            "record_kind": str(lineage_payload.get("record_kind") or "memory"),
            "previous_head": lineage_payload.get("previous_head"),
            "candidate_head": lineage_payload.get("candidate_head") or str(node_hash),
            "canonical": bool(lineage_payload.get("canonical", True)),
            "quarantined": bool(lineage_payload.get("quarantined", False)),
            "canonical_seq": int(lineage_payload.get("canonical_seq") or 0),
            "equivocation_id": lineage_payload.get("equivocation_id"),
            "checkpoint_hash": lineage_payload.get("checkpoint_hash"),
            "llm_call_id": llm_call_id,
            "issued_at_utc": _utc_now(),
            "signer_id": signer_id,
            "receipt_payload_version": "helix-memory-receipt-payload-v1",
        }

    def _build_receipt(
        self,
        *,
        item: MemoryItem,
        node_hash: str,
        parent_hash: str | None,
        llm_call_id: str | None,
        lineage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE", "local_self_signed").lower()
        signer_id = os.environ.get("HELIX_RECEIPT_SIGNER_ID") or item.agent_id
        payload = self._receipt_payload(
            item=item,
            node_hash=node_hash,
            parent_hash=parent_hash,
            signer_id=signer_id,
            llm_call_id=llm_call_id,
            lineage=lineage,
        )
        if mode in {"", "0", "false", "off", "unsigned_legacy"}:
            return attach_verification(unsigned_legacy_receipt(payload))
        if mode not in {"local_self_signed", "ephemeral_preregistered", "sigstore_rekor"}:
            raise ValueError(f"unsupported HELIX_RECEIPT_SIGNING_MODE: {mode}")
        attestation = None
        key_provenance = mode
        if mode == "sigstore_rekor":
            evidence_digest = os.environ.get("HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST")
            if not evidence_digest:
                raise RuntimeError("sigstore_rekor signing requires HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST")
            attestation = {"provider": "sigstore_rekor", "evidence_digest": evidence_digest, "verified": True}
        if mode in {"local_self_signed", "sigstore_rekor"}:
            return self._sign_with_local_key_unlocked(
                payload,
                signer_id=signer_id,
                key_provenance=key_provenance,
                attestation=attestation,
            )
        seed = os.environ.get("HELIX_RECEIPT_SIGNING_SEED") or f"{self.db_path}:{item.memory_id}:{node_hash}"
        keys = derive_ephemeral_keypair(seed)
        return attach_verification(
            sign_receipt_payload(
                payload,
                private_key_b64=keys["private_key"],
                public_key_b64=keys["public_key"],
                signer_id=signer_id,
                key_provenance=key_provenance,
                attestation=attestation,
            )
        )

    def _attach_receipt(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(payload.get("memory_id") or "")
        receipt = self._memory_receipts.get(memory_id)
        if receipt is None:
            node_hash = payload.get("node_hash") or self._memory_node_hashes.get(memory_id)
            legacy_payload = {
                "node_hash": node_hash,
                "parent_hash": None,
                "memory_id": memory_id,
                "project": payload.get("project"),
                "agent_id": payload.get("agent_id"),
                "session_id": payload.get("session_id"),
                "thread_id": payload.get("session_id"),
                "record_kind": "memory",
                "previous_head": None,
                "candidate_head": node_hash,
                "canonical": True,
                "quarantined": False,
                "canonical_seq": 0,
                "equivocation_id": None,
                "checkpoint_hash": None,
                "llm_call_id": None,
                "issued_at_utc": _utc_now(),
                "signer_id": payload.get("agent_id"),
                "receipt_payload_version": "helix-memory-receipt-payload-v1",
            }
            receipt = attach_verification(unsigned_legacy_receipt(legacy_payload))
        else:
            receipt = attach_verification(dict(receipt))
        payload["node_hash"] = payload.get("node_hash") or self._memory_node_hashes.get(memory_id)
        payload["signed_receipt"] = receipt
        payload["receipt"] = receipt
        payload["signature_verified"] = bool(receipt.get("signature_verified"))
        payload["key_provenance"] = receipt.get("key_provenance")
        payload["attestation_status"] = "verified" if (receipt.get("attestation") or {}).get("verified") else "none"
        payload["legacy_unsigned"] = receipt.get("receipt_version") == "unsigned_legacy" or not bool(receipt.get("signature_verified"))
        payload["checkpoint_hash"] = payload.get("checkpoint_hash") or receipt.get("checkpoint_hash")
        payload = self._attach_lineage(payload, node_hash=payload.get("node_hash"), session_id=payload.get("session_id"))
        if payload["legacy_unsigned"]:
            payload["checkpoint_status"] = "missing_legacy" if not payload.get("checkpoint_hash") else "legacy_unsigned"
        return payload

    @staticmethod
    def _hash_embedding(text: str, dims: int = 16) -> list[float]:
        vector = [0.0] * dims
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = digest[0] % dims
            vector[slot] += 1.0 if digest[1] % 2 == 0 else -1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def _dense_rerank_score(self, query: str, payload: dict[str, Any]) -> float:
        query_vec = self._hash_embedding(query)
        text = " ".join(str(payload.get(key) or "") for key in ("summary", "content", "tags"))
        doc_vec = self._hash_embedding(text)
        return sum(a * b for a, b in zip(query_vec, doc_vec))

    def _finalize_search_results(
        self,
        results: list[dict[str, Any]],
        *,
        query: str,
        limit: int,
        signature_enforcement: str | None,
        rerank_mode: str,
    ) -> list[dict[str, Any]]:
        enforcement = self._resolve_signature_enforcement(signature_enforcement)
        if rerank_mode == "receipt_adjudicated":
            enforcement = "strict"
        annotated = [self._attach_receipt(dict(item)) for item in results]
        if rerank_mode == "bm25_dense_rerank":
            for item in annotated:
                item["rerank_mode"] = rerank_mode
                item["dense_rerank_score"] = self._dense_rerank_score(query, item)
                item["score"] = float(item.get("score") or 0.0) + float(item["dense_rerank_score"])
            annotated.sort(key=lambda item: (-float(item.get("score") or 0.0), -float(item.get("created_ms") or 0.0)))
        elif rerank_mode == "receipt_adjudicated":
            for item in annotated:
                item["rerank_mode"] = rerank_mode
                item["dense_rerank_score"] = self._dense_rerank_score(query, item)
                item["score"] = (
                    float(item.get("score") or 0.0)
                    + float(item["dense_rerank_score"])
                    + (100.0 if item.get("signature_verified") else -100.0)
                )
            annotated.sort(key=lambda item: (-float(item.get("score") or 0.0), -float(item.get("created_ms") or 0.0)))
        else:
            for item in annotated:
                item["rerank_mode"] = rerank_mode
        enforced = enforce_retrieval_signatures(annotated, mode=enforcement)  # type: ignore[arg-type]
        for item in enforced:
            item["signature_enforcement_mode"] = enforcement
        return enforced[: int(limit)]

    def _route_query(
        self,
        *,
        project: str,
        agent_filter: str | None,
        query: str,
        type_filter: set[str],
        exclude_ids: set[str],
        route_query: bool | None,
    ) -> RoutedQuery:
        if route_query is False or self._semantic_router is None:
            return RoutedQuery(original_query=query, routed_query=query, action="pass_through", reason="router_disabled")
        with self._lock:
            scope = (project, agent_filter)
            routed = self._semantic_router.route_from_index(
                query=query,
                doc_count=self._router_scope_doc_count.get(scope, 0),
                term_doc_freq=dict(self._router_scope_df.get(scope, {})),
                term_anchor_scores=dict(self._router_scope_anchor_scores.get(scope, {})),
            )
            self._router_stats["calls"] += 1
            if routed.action == "rewrite":
                self._router_stats["rewrites"] += 1
            elif routed.action == "recent_fallback":
                self._router_stats["recent_fallback"] += 1
            else:
                self._router_stats["pass_through"] += 1
            return routed

    @staticmethod
    def _attach_router_payload(payload: dict[str, Any], routed: RoutedQuery) -> dict[str, Any]:
        if routed.action != "pass_through" or routed.reason != "query_already_selective":
            payload["semantic_router"] = routed.to_dict()
        return payload

    def _search_rust_index(
        self,
        *,
        project: str,
        agent_filter: str | None,
        session_filter: str | None,
        retrieval_scope: str,
        query: str,
        limit: int,
        type_filter: set[str],
        exclude_ids: set[str],
        routed: RoutedQuery,
        include_quarantined: bool,
    ) -> list[dict[str, Any]] | None:
        if self._rust_index is None:
            return None
        filters = {
            "project": project,
            "record_kind": "memory",
            "include_tombstoned": False,
            "exclude_memory_ids": sorted(exclude_ids),
        }
        if agent_filter is not None:
            filters["agent_id"] = agent_filter
        if type_filter:
            filters["memory_types"] = sorted(type_filter)
        try:
            hits = self._rust_index.search(query, max(int(limit) * 4, int(limit)), json.dumps(filters, sort_keys=True))
        except Exception as exc:  # noqa: BLE001
            self._rust_index = None
            self._rust_index_error = str(exc)
            return None

        now = _now_ms()
        results: list[dict[str, Any]] = []
        with self._lock:
            for hit in hits:
                hit_dict = dict(hit)
                mem_id = str(hit_dict.get("memory_id") or "")
                if not mem_id or mem_id in exclude_ids:
                    continue
                item = self._memories.get(mem_id)
                if item is None:
                    continue
                if not self._memory_visible_unlocked(mem_id, include_quarantined=include_quarantined):
                    continue
                if item.project != project:
                    continue
                if agent_filter is not None and item.agent_id != agent_filter:
                    continue
                if retrieval_scope == "session" and item.session_id != session_filter:
                    continue
                if type_filter and item.memory_type not in type_filter:
                    continue
                updated_item = MemoryItem(**{**item.to_dict(), "last_access_ms": now})
                self._memories[mem_id] = updated_item
                payload = updated_item.to_dict()
                payload["score"] = float(hit_dict.get("score") or 0.0)
                payload["node_hash"] = hit_dict.get("node_hash")
                payload["matched_terms"] = hit_dict.get("matched_terms") or []
                payload["search_backend"] = "rust_bm25"
                payload["thread_id"] = updated_item.session_id
                payload["thread_match"] = bool(session_filter and updated_item.session_id == session_filter)
                if payload["thread_match"] and retrieval_scope == "workspace":
                    payload["score"] = float(payload["score"]) + 1000.0
                self._attach_router_payload(payload, routed)
                results.append(payload)
        results.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                -(1 if item.get("thread_match") else 0),
                -float(item.get("created_ms") or 0.0),
            )
        )
        return results[: int(limit)]

    def _search_recent_fallback(
        self,
        *,
        project: str,
        agent_filter: str | None,
        session_filter: str | None,
        retrieval_scope: str,
        limit: int,
        type_filter: set[str],
        exclude_ids: set[str],
        routed: RoutedQuery,
        include_quarantined: bool,
    ) -> list[dict[str, Any]]:
        now = _now_ms()
        with self._lock:
            scope = (project, agent_filter)
            recent_ids = self._router_scope_recent_ids.get(scope, [])
            candidates = []
            for mem_id in recent_ids:
                item = self._memories.get(mem_id)
                if item is None:
                    continue
                if not self._memory_visible_unlocked(mem_id, include_quarantined=include_quarantined):
                    continue
                if retrieval_scope == "session" and item.session_id != session_filter:
                    continue
                if type_filter and item.memory_type not in type_filter:
                    continue
                if exclude_ids and mem_id in exclude_ids:
                    continue
                candidates.append(item)
            if not candidates:
                for mem_id, item in self._memories.items():
                    if item.project != project:
                        continue
                    if agent_filter is not None and item.agent_id != agent_filter:
                        continue
                    if not self._memory_visible_unlocked(mem_id, include_quarantined=include_quarantined):
                        continue
                    if retrieval_scope == "session" and item.session_id != session_filter:
                        continue
                    if type_filter and item.memory_type not in type_filter:
                        continue
                    if exclude_ids and mem_id in exclude_ids:
                        continue
                    candidates.append(item)
                candidates.sort(
                    key=lambda item: (
                        1 if session_filter and item.session_id == session_filter else 0,
                        (item.importance * item.decay_score),
                        item.last_access_ms,
                    ),
                    reverse=True,
                )
            results = []
            for item in candidates[: int(limit)]:
                updated_item = MemoryItem(**{**item.to_dict(), "last_access_ms": now})
                self._memories[item.memory_id] = updated_item
                payload = updated_item.to_dict()
                payload["score"] = float(item.importance) * float(item.decay_score)
                payload["matched_terms"] = []
                payload["search_backend"] = "semantic_router_recent_fallback"
                payload["thread_id"] = updated_item.session_id
                payload["thread_match"] = bool(session_filter and updated_item.session_id == session_filter)
                if payload["thread_match"] and retrieval_scope == "workspace":
                    payload["score"] = float(payload["score"]) + 1000.0
                self._attach_router_payload(payload, routed)
                results.append(payload)
            results.sort(
                key=lambda item: (
                    -float(item.get("score") or 0.0),
                    -(1 if item.get("thread_match") else 0),
                    -float(item.get("created_ms") or 0.0),
                )
            )
            return results

    def _search_python_scan(
        self,
        *,
        project: str,
        agent_filter: str | None,
        session_filter: str | None,
        retrieval_scope: str,
        tokens: list[str],
        limit: int,
        type_filter: set[str],
        exclude_ids: set[str],
        routed: RoutedQuery,
        include_quarantined: bool,
    ) -> list[dict[str, Any]]:
        """BM25-compatible Python fallback. Ranking matches Rust engine."""
        import math

        query_terms = list(dict.fromkeys(tokens))

        with self._lock:
            # Rust uses global corpus DF/avgdl, then applies filters at scoring time.
            global_field_counts: dict[str, tuple[dict[str, int], dict[str, int], dict[str, int], int]] = {}
            df: dict[str, int] = {}
            for mem_id, item in self._memories.items():
                if not self._memory_visible_unlocked(mem_id, include_quarantined=include_quarantined):
                    continue
                content_counts: dict[str, int] = {}
                summary_counts: dict[str, int] = {}
                tag_counts: dict[str, int] = {}
                for tok in _tokenize(item.content):
                    content_counts[tok] = content_counts.get(tok, 0) + 1
                for tok in _tokenize(item.summary):
                    summary_counts[tok] = summary_counts.get(tok, 0) + 1
                for tok in _tokenize(" ".join(item.tags)):
                    tag_counts[tok] = tag_counts.get(tok, 0) + 1

                doc_len = max(sum(content_counts.values()) + sum(summary_counts.values()) + sum(tag_counts.values()), 1)
                global_field_counts[mem_id] = (content_counts, summary_counts, tag_counts, doc_len)
                for tok in set(content_counts) | set(summary_counts) | set(tag_counts):
                    df[tok] = df.get(tok, 0) + 1

            term_infos = [(tok, df[tok]) for tok in query_terms if tok in df]
            term_infos.sort(key=lambda item: item[1])
            if not term_infos:
                return []

            stopword_min_docs = 100
            doc_count = max(len(global_field_counts), 1)
            if doc_count >= stopword_min_docs:
                stopword_threshold = int(doc_count * 0.40)
                active_terms = [item for item in term_infos if item[1] <= stopword_threshold]
                if not active_terms:
                    active_terms = [term_infos[0]]
            else:
                active_terms = term_infos
            active_term_set = {term for term, _ in active_terms}
            avg_dl = sum(fields[3] for fields in global_field_counts.values()) / max(doc_count, 1)

            candidates: list[tuple[str, MemoryItem, dict[str, tuple[int, int, int]], int]] = []
            for mem_id, item in self._memories.items():
                if item.project != project:
                    continue
                if agent_filter is not None and item.agent_id != agent_filter:
                    continue
                if not self._memory_visible_unlocked(mem_id, include_quarantined=include_quarantined):
                    continue
                if retrieval_scope == "session" and item.session_id != session_filter:
                    continue
                if type_filter and item.memory_type not in type_filter:
                    continue
                if exclude_ids and mem_id in exclude_ids:
                    continue

                content_counts, summary_counts, tag_counts, doc_len = global_field_counts[mem_id]
                tf_by_term: dict[str, tuple[int, int, int]] = {}
                for tok in active_term_set:
                    content_tf = content_counts.get(tok, 0)
                    summary_tf = summary_counts.get(tok, 0)
                    tags_tf = tag_counts.get(tok, 0)
                    if content_tf or summary_tf or tags_tf:
                        tf_by_term[tok] = (content_tf, summary_tf, tags_tf)

                if not tf_by_term:
                    continue

                candidates.append((mem_id, item, tf_by_term, doc_len))

            if not candidates:
                return []

            # --- BM25 scoring (k1=1.2, b=0.75) matching Rust params ---
            k1 = 1.2
            b = 0.75
            field_boosts = {"content": 1.0, "summary": 2.0, "tags": 1.6}

            def bm25_component(tf_val: int, idf_val: float, doc_len_val: int) -> float:
                if tf_val <= 0:
                    return 0.0
                denom = tf_val + k1 * (1.0 - b + b * (doc_len_val / max(avg_dl, 1.0)))
                return idf_val * ((tf_val * (k1 + 1.0)) / max(denom, 0.000001))

            scored: list[tuple[float, MemoryItem]] = []
            for mem_id, item, tf_by_term, doc_len in candidates:
                score = 0.0
                for tok, (content_tf, summary_tf, tags_tf) in tf_by_term.items():
                    tok_df = max(df.get(tok, 1), 1)
                    idf = max(math.log((doc_count - tok_df + 0.5) / (tok_df + 0.5) + 1.0), 0.0)
                    score += bm25_component(content_tf, idf, doc_len) * field_boosts["content"]
                    score += bm25_component(summary_tf, idf, doc_len) * field_boosts["summary"]
                    score += bm25_component(tags_tf, idf, doc_len) * field_boosts["tags"]

                # Quality boost matching Rust: importance/10 * 0.20 + decay * 0.10
                quality = 1.0 + (max(float(item.importance), 0.0) / 10.0) * 0.20 + max(float(item.decay_score), 0.0) * 0.10
                score *= quality
                if session_filter and item.session_id == session_filter and retrieval_scope == "workspace":
                    score += 1000.0
                scored.append((score, item))

            scored.sort(
                key=lambda x: (
                    -x[0],
                    -(1 if session_filter and x[1].session_id == session_filter else 0),
                    -x[1].created_ms,
                    x[1].source_hash,
                )
            )

            now = _now_ms()
            results = []
            for score, item in scored[: int(limit)]:
                updated_item = MemoryItem(**{**item.to_dict(), "last_access_ms": now})
                self._memories[item.memory_id] = updated_item
                payload = updated_item.to_dict()
                payload["score"] = float(score)
                payload["search_backend"] = "python_bm25_fallback"
                payload["thread_id"] = updated_item.session_id
                payload["thread_match"] = bool(session_filter and updated_item.session_id == session_filter)
                self._attach_router_payload(payload, routed)
                results.append(payload)

            return results

    def verify_chain(self, leaf_hash: str, policy: str | None = None) -> dict[str, Any]:
        if self._rust_index is not None:
            try:
                return dict(self._rust_index.verify_chain(str(leaf_hash), policy))
            except Exception as exc:  # noqa: BLE001
                self._rust_index_error = str(exc)
        chain = self.dag.audit_chain(str(leaf_hash))
        failed_at = None
        missing_parent = None
        if not chain:
            missing_parent = str(leaf_hash)
        for node in chain:
            expected_payload = node.content.encode("utf-8")
            if node.parent_hash:
                expected_payload += node.parent_hash.encode("utf-8")
            expected = hashlib.sha256(expected_payload).hexdigest()
            if expected != node.hash and not str(node.content).startswith("[GC_TOMBSTONE:"):
                failed_at = node.hash
                break
            if node.parent_hash and self.dag.lookup(node.parent_hash) is None:
                missing_parent = node.parent_hash
                break
        return {
            "status": "failed" if failed_at or missing_parent else "verified",
            "leaf_hash": str(leaf_hash),
            "chain_len": len(chain),
            "failed_at": failed_at,
            "missing_parent": missing_parent,
            "backend": "python_dag",
        }

    def session_lineage(
        self,
        session_id: str,
        *,
        include_quarantined: bool = True,
        limit: int = 20,
    ) -> dict[str, Any]:
        session_key = _safe_scope(session_id, field="session_id")
        with self._lock:
            state = dict(self._session_lineage.get(session_key) or {})
            transitions = [dict(item) for item in self._session_transitions.get(session_key, [])]
            if not include_quarantined:
                transitions = [item for item in transitions if not item.get("quarantined")]
            if limit > 0:
                transitions = transitions[-int(limit) :]
            if not state:
                return {
                    "status": "not_found",
                    "session_id": session_key,
                    "thread_id": session_key,
                    "canonical_head": None,
                    "head_seq": 0,
                    "equivocation_count": 0,
                    "transition_count": 0,
                    "transitions": [],
                }
            canonical_head = state.get("canonical_head")
            canonical_transition = dict(self._node_lineage.get(str(canonical_head or "")) or {})
            return {
                "status": str(state.get("status") or "active"),
                "session_id": session_key,
                "thread_id": session_key,
                "canonical_head": canonical_head,
                "head_seq": int(state.get("head_seq") or 0),
                "equivocation_count": int(state.get("equivocation_count") or 0),
                "transition_count": int(state.get("transition_count") or 0),
                "checkpoint_count": int(state.get("checkpoint_count") or len(self._session_checkpoints.get(session_key, []))),
                "latest_checkpoint_hash": state.get("latest_checkpoint_hash"),
                "last_transition_ms": state.get("last_transition_ms"),
                "latest_transition_seq": int(state.get("latest_transition_seq") or 0),
                "canonical_transition": canonical_transition or None,
                "transitions": transitions,
            }

    def head_checkpoint(self, session_id: str) -> dict[str, Any]:
        session_key = _safe_scope(session_id, field="session_id")
        with self._lock:
            checkpoints = [dict(item) for item in self._session_checkpoints.get(session_key, [])]
            if not checkpoints:
                return {
                    "status": "not_found",
                    "session_id": session_key,
                    "thread_id": session_key,
                    "checkpoint_verified": False,
                    "checkpoint_status": "missing_legacy",
                }
            latest = checkpoints[-1]
            verification = self._verify_checkpoint_unlocked(latest)
            return {
                "status": "ok" if verification.get("checkpoint_verified") else "failed",
                "session_id": session_key,
                "thread_id": session_key,
                "checkpoint": latest,
                "checkpoint_hash": latest.get("checkpoint_hash"),
                **verification,
            }

    def _legacy_unsigned_count_unlocked(self, session_key: str) -> int:
        count = 0
        for memory_id, item in self._memories.items():
            if item.session_id != session_key:
                continue
            receipt = self._memory_receipts.get(memory_id) or {}
            if receipt.get("receipt_version") == "unsigned_legacy" or not receipt.get("signature_verified"):
                count += 1
        return count

    def _checkpoint_chain_status_unlocked(self, session_key: str) -> dict[str, Any]:
        checkpoints = [dict(item) for item in self._session_checkpoints.get(session_key, [])]
        if not checkpoints:
            return {
                "checkpoint_chain_verified": False,
                "checkpoint_count": 0,
                "latest_checkpoint": None,
                "checkpoint_error": "missing_legacy",
            }
        previous_hash = None
        for index, checkpoint in enumerate(checkpoints):
            verification = self._verify_checkpoint_unlocked(checkpoint)
            if not verification.get("checkpoint_verified"):
                return {
                    "checkpoint_chain_verified": False,
                    "checkpoint_count": len(checkpoints),
                    "latest_checkpoint": checkpoints[-1],
                    "checkpoint_error": verification.get("verification_error") or "checkpoint_verification_failed",
                    "failed_checkpoint_index": index,
                }
            if checkpoint.get("previous_checkpoint_hash") != previous_hash:
                return {
                    "checkpoint_chain_verified": False,
                    "checkpoint_count": len(checkpoints),
                    "latest_checkpoint": checkpoints[-1],
                    "checkpoint_error": "checkpoint_previous_hash_mismatch",
                    "failed_checkpoint_index": index,
                }
            previous_hash = checkpoint.get("checkpoint_hash")
        state = self._session_lineage.get(session_key) or {}
        latest = checkpoints[-1]
        mismatch_fields = []
        if latest.get("canonical_head") != state.get("canonical_head"):
            mismatch_fields.append("canonical_head")
        if int(latest.get("head_seq") or 0) != int(state.get("head_seq") or 0):
            mismatch_fields.append("head_seq")
        if mismatch_fields:
            return {
                "checkpoint_chain_verified": False,
                "checkpoint_count": len(checkpoints),
                "latest_checkpoint": latest,
                "checkpoint_error": "latest_checkpoint_state_mismatch",
                "mismatch_fields": mismatch_fields,
            }
        return {
            "checkpoint_chain_verified": True,
            "checkpoint_count": len(checkpoints),
            "latest_checkpoint": latest,
            "checkpoint_error": None,
        }

    def verify_session_lineage(self, session_id: str, *, include_quarantined: bool = False) -> dict[str, Any]:
        session_key = _safe_scope(session_id, field="session_id")
        with self._lock:
            state = dict(self._session_lineage.get(session_key) or {})
            if not state:
                return {
                    "status": "not_found",
                    "trust_status": "not_found",
                    "session_id": session_key,
                    "thread_id": session_key,
                    "checkpoint_verified": False,
                    "checkpoint_count": 0,
                    "latest_checkpoint": None,
                    "legacy_unsigned_count": 0,
                    "quarantined_count": 0,
                }
            canonical_head = str(state.get("canonical_head") or "")
            canonical_chain = self.verify_chain(canonical_head) if canonical_head else None
            transitions = [dict(item) for item in self._session_transitions.get(session_key, [])]
            quarantined = [item for item in transitions if item.get("quarantined")]
            policy_quarantined = [item for item in quarantined if item.get("quarantine_class") == "policy" and not item.get("equivocation_id")]
            equivocated = [item for item in quarantined if item not in policy_quarantined]
            visible_transitions = transitions if include_quarantined else [item for item in transitions if not item.get("quarantined")]
            checkpoint_chain = self._checkpoint_chain_status_unlocked(session_key)
            latest_checkpoint = checkpoint_chain.get("latest_checkpoint")
            checkpoint_verification = self._verify_checkpoint_unlocked(latest_checkpoint) if latest_checkpoint else {"checkpoint_verified": False}
            legacy_unsigned_count = self._legacy_unsigned_count_unlocked(session_key)
            status = "verified"
            if canonical_chain and canonical_chain.get("status") != "verified":
                status = "failed"
            elif latest_checkpoint and not checkpoint_chain.get("checkpoint_chain_verified"):
                status = "failed"
            elif equivocated:
                status = "equivocation_detected"
            if status == "failed":
                trust_status = "failed"
            elif not latest_checkpoint:
                trust_status = "missing_legacy"
            elif quarantined:
                trust_status = "verified_with_quarantine"
            elif legacy_unsigned_count:
                trust_status = "verified_with_legacy_warnings"
            else:
                trust_status = "verified"
            return {
                "status": status,
                "trust_status": trust_status,
                "session_id": session_key,
                "thread_id": session_key,
                "canonical_head": canonical_head or None,
                "head_seq": int(state.get("head_seq") or 0),
                "equivocation_count": int(state.get("equivocation_count") or 0),
                "policy_quarantine_count": int(state.get("policy_quarantine_count") or len(policy_quarantined)),
                "transition_count": int(state.get("transition_count") or 0),
                "quarantined_count": len(quarantined),
                "checkpoint_verified": bool(checkpoint_verification.get("checkpoint_verified")) and bool(checkpoint_chain.get("checkpoint_chain_verified")),
                "checkpoint_count": int(checkpoint_chain.get("checkpoint_count") or 0),
                "latest_checkpoint": latest_checkpoint,
                "latest_checkpoint_hash": state.get("latest_checkpoint_hash"),
                "checkpoint_error": checkpoint_chain.get("checkpoint_error") or checkpoint_verification.get("verification_error"),
                "legacy_unsigned_count": legacy_unsigned_count,
                "canonical_chain": canonical_chain,
                "transitions": visible_transitions,
                "quarantined": quarantined if include_quarantined else [item.get("candidate_head") for item in quarantined],
                "policy_quarantined": policy_quarantined if include_quarantined else [item.get("candidate_head") for item in policy_quarantined],
                "include_quarantined": include_quarantined,
            }

    def export_session_proof(
        self,
        session_id: str,
        *,
        ref: str | None = None,
        include_quarantined: bool = False,
    ) -> dict[str, Any]:
        session_key = _safe_scope(session_id, field="session_id")
        needle = str(ref or "").strip().lower()
        with self._lock:
            state = dict(self._session_lineage.get(session_key) or {})
            if not state:
                return {"status": "not_found", "session_id": session_key, "thread_id": session_key}
            target_node_hash = str(state.get("canonical_head") or "")
            target_memory_id = None
            if needle:
                for memory_id, node_hash in self._memory_node_hashes.items():
                    if memory_id.lower() == needle or memory_id.lower().startswith(needle) or node_hash.lower() == needle or node_hash.lower().startswith(needle):
                        item = self._memories.get(memory_id)
                        if item and item.session_id == session_key:
                            target_node_hash = node_hash
                            target_memory_id = memory_id
                            break
                else:
                    if needle in self.dag._nodes:
                        target_node_hash = needle
                    else:
                        for node_hash in self.dag._nodes:
                            if node_hash.lower().startswith(needle):
                                target_node_hash = node_hash
                                break
                        else:
                            return {
                                "status": "not_found",
                                "session_id": session_key,
                                "thread_id": session_key,
                                "ref": ref,
                            }
            else:
                for memory_id, node_hash in self._memory_node_hashes.items():
                    if node_hash == target_node_hash:
                        target_memory_id = memory_id
                        break
            lineage = self._node_lineage.get(target_node_hash) or {}
            if target_node_hash and lineage and str(lineage.get("session_id") or "") != session_key:
                return {
                    "status": "not_found",
                    "session_id": session_key,
                    "thread_id": session_key,
                    "ref": ref,
                    "error": "ref_not_in_session",
                }
            if lineage.get("quarantined") and not include_quarantined:
                return {
                    "status": "quarantined_hidden",
                    "session_id": session_key,
                    "thread_id": session_key,
                    "ref": ref,
                    "target_node_hash": target_node_hash,
                    "equivocation_id": lineage.get("equivocation_id"),
                }
            chain = self.dag.audit_chain(target_node_hash) if target_node_hash else []
            dag_chain = [
                {
                    "node_hash": node.hash,
                    "parent_hash": node.parent_hash,
                    "depth": int(node.depth),
                    "timestamp": node.timestamp,
                    "content_sha256": hashlib.sha256(str(node.content).encode("utf-8")).hexdigest(),
                }
                for node in chain
            ]
            receipt = self._memory_receipts.get(target_memory_id or "") if target_memory_id else None
            all_transitions = [dict(item) for item in self._session_transitions.get(session_key, [])]
            quarantined = [item for item in all_transitions if item.get("quarantined")]
            policy_quarantined = [item for item in quarantined if item.get("quarantine_class") == "policy" and not item.get("equivocation_id")]
            equivocated = [item for item in quarantined if item not in policy_quarantined]
            checkpoint_chain = self._checkpoint_chain_status_unlocked(session_key)
            chain_status = self.verify_chain(str(state.get("canonical_head") or "")) if state.get("canonical_head") else None
            proof_status = "verified"
            if chain_status and chain_status.get("status") != "verified":
                proof_status = "failed"
            elif checkpoint_chain.get("latest_checkpoint") and not checkpoint_chain.get("checkpoint_chain_verified"):
                proof_status = "failed"
            elif equivocated:
                proof_status = "equivocation_detected"
            return {
                "status": "ok",
                "session_id": session_key,
                "thread_id": session_key,
                "ref": ref,
                "target_node_hash": target_node_hash or None,
                "target_memory_id": target_memory_id,
                "target_receipt": dict(receipt) if isinstance(receipt, dict) else None,
                "target_lineage": dict(lineage),
                "dag_chain": dag_chain,
                "latest_checkpoint": self._session_checkpoints.get(session_key, [])[-1] if self._session_checkpoints.get(session_key) else None,
                "lineage_verification": {
                    "status": proof_status,
                    "trust_status": "failed" if proof_status == "failed" else (
                        "verified_with_quarantine" if quarantined else (
                            "missing_legacy" if not checkpoint_chain.get("latest_checkpoint") else "verified"
                        )
                    ),
                    "checkpoint_verified": bool(checkpoint_chain.get("checkpoint_chain_verified")),
                    "checkpoint_count": int(checkpoint_chain.get("checkpoint_count") or 0),
                    "legacy_unsigned_count": self._legacy_unsigned_count_unlocked(session_key),
                    "quarantined_count": len(quarantined),
                    "policy_quarantine_count": len(policy_quarantined),
                    "include_quarantined": include_quarantined,
                },
                "include_quarantined": include_quarantined,
            }

    def gc_tombstone_index(self, node_hash: str) -> dict[str, Any]:
        if self._rust_index is None:
            return {"status": "skipped_no_rust_index", "tombstoned_count": 0}
        try:
            return dict(self._rust_index.gc_tombstone(json.dumps({"node_hash": str(node_hash)})))
        except Exception as exc:  # noqa: BLE001
            self._rust_index_error = str(exc)
            return {"status": "failed", "error": str(exc), "tombstoned_count": 0}

    def list_memories(
        self,
        *,
        project: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        exclude_memory_ids: Iterable[str] | None = None,
        retrieval_scope: str = "workspace",
        include_quarantined: bool = False,
    ) -> list[dict[str, Any]]:
        proj_filter = None if project is None else _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")
        session_filter = None if session_id is None else _safe_scope(session_id, field="session_id")
        selected_scope = _resolve_retrieval_scope(retrieval_scope)
        exclude_ids = {str(item) for item in (exclude_memory_ids or [])}
        
        with self._lock:
            candidates = []
            for mem_id, item in self._memories.items():
                if proj_filter is not None and item.project != proj_filter:
                    continue
                if agent_filter is not None and item.agent_id != agent_filter:
                    continue
                if not self._memory_visible_unlocked(mem_id, include_quarantined=include_quarantined):
                    continue
                if selected_scope == "session" and item.session_id != session_filter:
                    continue
                if exclude_ids and mem_id in exclude_ids:
                    continue
                candidates.append(item)
                
            candidates.sort(
                key=lambda x: (
                    1 if session_filter and x.session_id == session_filter else 0,
                    (x.importance * x.decay_score),
                    x.last_access_ms,
                ),
                reverse=True,
            )
            results: list[dict[str, Any]] = []
            for item in candidates[: int(limit)]:
                payload = item.to_dict()
                payload["thread_id"] = item.session_id
                payload["thread_match"] = bool(session_filter and item.session_id == session_filter)
                results.append(self._attach_receipt(payload))
            return results

    def list_sessions(
        self,
        *,
        project: str | None = None,
        agent_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        proj_filter = None if project is None else _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")

        with self._lock:
            sessions: dict[str, dict[str, Any]] = {}

            def _touch(session_value: str | None, created_ms: float, kind: str) -> None:
                if not session_value:
                    return
                payload = sessions.setdefault(
                    session_value,
                    {
                        "session_id": session_value,
                        "thread_id": session_value,
                        "memory_count": 0,
                        "observation_count": 0,
                        "last_seen_ms": created_ms,
                        "canonical_head": None,
                        "head_seq": 0,
                        "equivocation_count": 0,
                        "status": "active",
                    },
                )
                payload["last_seen_ms"] = max(float(payload.get("last_seen_ms") or 0.0), float(created_ms or 0.0))
                counter_key = "memory_count" if kind == "memory" else "observation_count"
                payload[counter_key] = int(payload.get(counter_key) or 0) + 1

            for item in self._memories.values():
                if proj_filter is not None and item.project != proj_filter:
                    continue
                if agent_filter is not None and item.agent_id != agent_filter:
                    continue
                _touch(item.session_id, item.created_ms, "memory")
            for raw in self._observations.values():
                if proj_filter is not None and raw["project"] != proj_filter:
                    continue
                if agent_filter is not None and raw["agent_id"] != agent_filter:
                    continue
                _touch(raw.get("session_id"), float(raw.get("created_ms") or 0.0), "observation")

            ordered = sorted(
                sessions.values(),
                key=lambda item: (-float(item.get("last_seen_ms") or 0.0), str(item.get("session_id") or "")),
            )
            for item in ordered:
                state = self._session_lineage.get(str(item.get("session_id") or ""))
                if not state:
                    continue
                item["canonical_head"] = state.get("canonical_head")
                item["head_seq"] = int(state.get("head_seq") or 0)
                item["equivocation_count"] = int(state.get("equivocation_count") or 0)
                item["status"] = state.get("status") or "active"
            return ordered[: int(limit)]

    def graph(
        self,
        *,
        project: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        retrieval_scope: str = "workspace",
        include_quarantined: bool = False,
    ) -> dict[str, Any]:
        proj_filter = None if project is None else _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")
        session_filter = None if session_id is None else _safe_scope(session_id, field="session_id")
        selected_scope = _resolve_retrieval_scope(retrieval_scope)
        
        with self._lock:
            items = []
            for item in self._memories.values():
                if proj_filter is not None and item.project != proj_filter: continue
                if agent_filter is not None and item.agent_id != agent_filter: continue
                if not self._memory_visible_unlocked(item.memory_id, include_quarantined=include_quarantined): continue
                if selected_scope == "session" and item.session_id != session_filter: continue
                items.append(item)
                
            obs = []
            for item in self._observations.values():
                if proj_filter is not None and item["project"] != proj_filter: continue
                if agent_filter is not None and item["agent_id"] != agent_filter: continue
                if not self._observation_visible_unlocked(str(item["observation_id"]), include_quarantined=include_quarantined): continue
                if selected_scope == "session" and item.get("session_id") != session_filter: continue
                obs.append(item)
                
            items.sort(
                key=lambda x: (
                    1 if session_filter and x.session_id == session_filter else 0,
                    x.last_access_ms,
                ),
                reverse=True,
            )
            obs.sort(key=lambda x: x["last_access_ms"], reverse=True)
            items = items[:limit]
            obs = obs[:limit]
            
            memory_ids = {item.memory_id for item in items}
            valid_links = [ln for ln in self._links if ln["memory_id"] in memory_ids]
            
            session_ids = set()
            for item in items:
                if item.session_id: session_ids.add(item.session_id)
            for raw in obs:
                if raw["session_id"]: session_ids.add(raw["session_id"])
            
            nodes = []
            for sid in sorted(session_ids):
                state = dict(self._session_lineage.get(sid) or {})
                nodes.append(
                    {
                        "id": f"session:{sid}",
                        "kind": "session",
                        "label": sid,
                        "canonical_head": state.get("canonical_head"),
                        "head_seq": int(state.get("head_seq") or 0),
                        "equivocation_count": int(state.get("equivocation_count") or 0),
                        "status": state.get("status") or "active",
                    }
                )
            for item in items:
                lineage = self._lineage_for_node_unlocked(self._memory_node_hashes.get(item.memory_id), item.session_id)
                nodes.append({
                    "id": f"memory:{item.memory_id}", "kind": "memory", "label": item.summary,
                    "memory_type": item.memory_type, "importance": item.importance,
                    "decay_score": item.decay_score, "agent_id": item.agent_id,
                    **lineage,
                })
            for raw in obs:
                lineage = self._lineage_for_node_unlocked(raw.get("node_hash"), raw.get("session_id"))
                nodes.append({
                    "id": f"observation:{raw['observation_id']}", "kind": "observation", "label": raw["summary"],
                    "observation_type": raw["observation_type"], "importance": raw["importance"],
                    "agent_id": raw["agent_id"],
                    **lineage,
                })
            
            edges = [{"source": f"session:{ln['session_id']}", "target": f"memory:{ln['memory_id']}", "relation": ln["relation"]} for ln in valid_links]
            for raw in obs:
                if raw["session_id"]:
                    edges.append({"source": f"session:{raw['session_id']}", "target": f"observation:{raw['observation_id']}", "relation": "observed"})
                    
            return {
                "nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges),
                "project": project,
                "agent_id": agent_id,
                "session_id": session_id,
                "retrieval_scope": selected_scope,
                "include_quarantined": include_quarantined,
            }

    def build_context(
        self,
        *,
        project: str,
        agent_id: str | None,
        session_id: str | None = None,
        query: str | None = None,
        budget_tokens: int = 2000,
        mode: str = "search",
        limit: int = 5,
        exclude_memory_ids: Iterable[str] | None = None,
        signature_enforcement: str | None = None,
        retrieval_scope: str = "workspace",
        include_quarantined: bool = False,
    ) -> dict[str, Any]:
        mode = str(mode or "off")
        if mode not in {"off", "summary", "search"}:
            raise ValueError(f"unsupported helix_memory_mode: {mode}")
        if mode == "off":
            return {"mode": "off", "context": "", "tokens": 0, "memory_ids": [], "items": []}
        exclude_ids = list(exclude_memory_ids or [])
        if mode == "search" and query:
            items = self.search(
                project=project,
                agent_id=agent_id,
                session_id=session_id,
                query=query,
                limit=limit,
                exclude_memory_ids=exclude_ids,
                signature_enforcement=signature_enforcement,
                retrieval_scope=retrieval_scope,
                include_quarantined=include_quarantined,
            )
        else:
            items = self.list_memories(
                project=project,
                agent_id=agent_id,
                session_id=session_id,
                limit=limit,
                exclude_memory_ids=exclude_ids,
                retrieval_scope=retrieval_scope,
                include_quarantined=include_quarantined,
            )
        selected: list[str] = []
        selected_items: list[dict[str, Any]] = []
        used = _estimate_tokens("<helix-memory-context></helix-memory-context>")
        for item in items:
            line = f"- [{item.get('memory_type')}] {item.get('summary')}: {item.get('content')}"
            tokens = _estimate_tokens(line)
            if used + tokens > int(budget_tokens):
                continue
            selected.append(line)
            selected_items.append(item)
            used += tokens
        if not selected:
            return {"mode": mode, "context": "", "tokens": 0, "memory_ids": [], "items": [], "excluded_memory_ids": exclude_ids}

        context = "<helix-memory-context>\n" + "\n".join(selected) + "\n</helix-memory-context>"
        return {
            "mode": mode,
            "context": context,
            "tokens": used,
            "memory_ids": [str(item["memory_id"]) for item in selected_items],
            "items": selected_items,
            "excluded_memory_ids": exclude_ids,
            "session_id": session_id,
            "retrieval_scope": _resolve_retrieval_scope(retrieval_scope),
            "include_quarantined": include_quarantined,
            "thread_lineage": self.session_lineage(session_id, include_quarantined=include_quarantined, limit=8) if session_id else None,
            "trust_summary": self.verify_session_lineage(session_id, include_quarantined=include_quarantined) if session_id else None,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rust_stats: dict[str, Any] | None = None
            if self._rust_index is not None:
                try:
                    rust_stats = dict(self._rust_index.stats())
                except Exception as exc:  # noqa: BLE001
                    self._rust_index_error = str(exc)
            return {
                "memory_count": len(self._memories),
                "observation_count": len(self._observations),
                "link_count": len(self._links),
                "fts_enabled": self.fts_enabled,
                "journal_mode": self.journal_mode,
                "memory_journal_enabled": self._journal_enabled,
                "memory_journal_path": str(self._journal_path),
                "memory_journal_error": self._journal_error,
                "busy_timeout_ms": self.busy_timeout_ms,
                "dag_node_count": len(self.dag._nodes),
                "memory_backend": "in_memory_dag",
                "search_backend": "rust_bm25" if self._rust_index is not None else "python_bm25_fallback",
                "rust_index_available": self._rust_index is not None,
                "rust_index_error": self._rust_index_error,
                "rust_index_stats": rust_stats,
                "semantic_query_router": dict(self._router_stats),
                "session_lineage_count": len(self._session_lineage),
                "session_checkpoint_count": sum(len(items) for items in self._session_checkpoints.values()),
                "trust_root_path": str(self._trust_root_path),
                "quarantined_node_count": sum(1 for item in self._node_lineage.values() if item.get("quarantined")),
            }


__all__ = ["MemoryCatalog", "MemoryItem", "privacy_filter", "source_hash"]
