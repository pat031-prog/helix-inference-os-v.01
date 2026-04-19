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
    derive_ephemeral_keypair,
    enforce_retrieval_signatures,
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
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent,
                metadata={
                    **payload,
                    "record_kind": "observation",
                    "index_content": clean_content,
                    "content_available": True,
                    "audit_status": "verified",
                },
            )

            if session_id:
                self._session_heads[session_id] = node.hash

            self._observations[obs_id] = payload

        return {
            "observation_id": obs_id,
            "project": project,
            "agent_id": agent_id,
            "session_id": session_id,
            "source_hash": digest,
            "content": clean_content,
            "summary": clean_summary,
            "tags": tag_list,
        }

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
            receipt = self._build_receipt(
                item=item,
                node_hash=node.hash,
                parent_hash=parent,
                llm_call_id=llm_call_id,
            )
            self._insert_rust_indexed(
                content_dump=content_dump,
                parent_hash=parent,
                metadata={
                    **item.to_dict(),
                    "record_kind": "memory",
                    "index_content": clean_content,
                    "content_available": True,
                    "audit_status": "verified",
                },
            )

            if session_id:
                self._session_heads[session_id] = node.hash

            self._memories[mem_id] = item
            self._memory_node_hashes[mem_id] = node.hash
            self._memory_receipts[mem_id] = receipt
            self._index_router_item_unlocked(item)
            
        return item

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
                if item.session_id:
                    self._session_heads[item.session_id] = node.hash
                self._memories[item.memory_id] = item
                self._memory_node_hashes[item.memory_id] = node.hash
                self._memory_receipts[item.memory_id] = self._build_receipt(
                    item=item,
                    node_hash=node.hash,
                    parent_hash=parent,
                    llm_call_id=llm_call_id,
                )
                self._index_router_item_unlocked(item)

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
            return dict(receipt) if receipt else None

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
            return payload

    def search(
        self,
        *,
        project: str,
        agent_id: str | None,
        query: str,
        limit: int = 5,
        memory_types: Iterable[str] | None = None,
        exclude_memory_ids: Iterable[str] | None = None,
        route_query: bool | None = None,
        signature_enforcement: str | None = None,
        rerank_mode: str | None = None,
    ) -> list[dict[str, Any]]:
        project = _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")
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
                limit=limit,
                type_filter=type_filter,
                exclude_ids=exclude_ids,
                routed=routed,
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
                query=effective_query,
                limit=effective_limit,
                type_filter=type_filter,
                exclude_ids=exclude_ids,
                routed=routed,
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
            tokens=effective_tokens,
            limit=effective_limit,
            type_filter=type_filter,
            exclude_ids=exclude_ids,
            routed=routed,
        )
        return self._finalize_search_results(
            results,
            query=effective_query,
            limit=limit,
            signature_enforcement=signature_enforcement,
            rerank_mode=selected_rerank_mode,
        )

    def _resolve_signature_enforcement(self, mode: str | None) -> str:
        # Default is strict: only receipts with signature_verified=true are returned.
        # warn/permissive must be opted into explicitly, either via the `mode` argument
        # or the HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT env var. This is the retrieval-side
        # half of the "strict signed retrieval end-to-end" contract.
        selected = str(mode or os.environ.get("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "strict")).lower()
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
    ) -> dict[str, Any]:
        return {
            "node_hash": str(node_hash),
            "parent_hash": parent_hash,
            "memory_id": item.memory_id,
            "project": item.project,
            "agent_id": item.agent_id,
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
    ) -> dict[str, Any]:
        mode = os.environ.get("HELIX_RECEIPT_SIGNING_MODE", "off").lower()
        signer_id = os.environ.get("HELIX_RECEIPT_SIGNER_ID") or item.agent_id
        payload = self._receipt_payload(
            item=item,
            node_hash=node_hash,
            parent_hash=parent_hash,
            signer_id=signer_id,
            llm_call_id=llm_call_id,
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
                "llm_call_id": None,
                "issued_at_utc": _utc_now(),
                "signer_id": payload.get("agent_id"),
                "receipt_payload_version": "helix-memory-receipt-payload-v1",
            }
            receipt = attach_verification(unsigned_legacy_receipt(legacy_payload))
        payload["signed_receipt"] = receipt
        payload["receipt"] = receipt
        payload["signature_verified"] = bool(receipt.get("signature_verified"))
        payload["key_provenance"] = receipt.get("key_provenance")
        payload["attestation_status"] = "verified" if (receipt.get("attestation") or {}).get("verified") else "none"
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
        query: str,
        limit: int,
        type_filter: set[str],
        exclude_ids: set[str],
        routed: RoutedQuery,
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
                if item.project != project:
                    continue
                if agent_filter is not None and item.agent_id != agent_filter:
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
                self._attach_router_payload(payload, routed)
                results.append(payload)
                if len(results) >= int(limit):
                    break
        return results

    def _search_recent_fallback(
        self,
        *,
        project: str,
        agent_filter: str | None,
        limit: int,
        type_filter: set[str],
        exclude_ids: set[str],
        routed: RoutedQuery,
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
                    if type_filter and item.memory_type not in type_filter:
                        continue
                    if exclude_ids and mem_id in exclude_ids:
                        continue
                    candidates.append(item)
                candidates.sort(key=lambda item: ((item.importance * item.decay_score), item.last_access_ms), reverse=True)
            results = []
            for item in candidates[: int(limit)]:
                updated_item = MemoryItem(**{**item.to_dict(), "last_access_ms": now})
                self._memories[item.memory_id] = updated_item
                payload = updated_item.to_dict()
                payload["score"] = float(item.importance) * float(item.decay_score)
                payload["matched_terms"] = []
                payload["search_backend"] = "semantic_router_recent_fallback"
                self._attach_router_payload(payload, routed)
                results.append(payload)
            return results

    def _search_python_scan(
        self,
        *,
        project: str,
        agent_filter: str | None,
        tokens: list[str],
        limit: int,
        type_filter: set[str],
        exclude_ids: set[str],
        routed: RoutedQuery,
    ) -> list[dict[str, Any]]:
        """BM25-compatible Python fallback. Ranking matches Rust engine."""
        import math

        query_terms = list(dict.fromkeys(tokens))

        with self._lock:
            # Rust uses global corpus DF/avgdl, then applies filters at scoring time.
            global_field_counts: dict[str, tuple[dict[str, int], dict[str, int], dict[str, int], int]] = {}
            df: dict[str, int] = {}
            for mem_id, item in self._memories.items():
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
                scored.append((score, item))

            scored.sort(key=lambda x: (-x[0], -x[1].created_ms, x[1].source_hash))

            now = _now_ms()
            results = []
            for score, item in scored[: int(limit)]:
                updated_item = MemoryItem(**{**item.to_dict(), "last_access_ms": now})
                self._memories[item.memory_id] = updated_item
                payload = updated_item.to_dict()
                payload["score"] = float(score)
                payload["search_backend"] = "python_bm25_fallback"
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
        for node in chain:
            expected_payload = node.content.encode("utf-8")
            if node.parent_hash:
                expected_payload += node.parent_hash.encode("utf-8")
            expected = hashlib.sha256(expected_payload).hexdigest()
            if expected != node.hash and not str(node.content).startswith("[GC_TOMBSTONE:"):
                failed_at = node.hash
                break
        return {
            "status": "failed" if failed_at else "verified",
            "leaf_hash": str(leaf_hash),
            "chain_len": len(chain),
            "failed_at": failed_at,
            "backend": "python_dag",
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
        limit: int = 20,
        exclude_memory_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        proj_filter = None if project is None else _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")
        exclude_ids = {str(item) for item in (exclude_memory_ids or [])}
        
        with self._lock:
            candidates = []
            for mem_id, item in self._memories.items():
                if proj_filter is not None and item.project != proj_filter:
                    continue
                if agent_filter is not None and item.agent_id != agent_filter:
                    continue
                if exclude_ids and mem_id in exclude_ids:
                    continue
                candidates.append(item)
                
            candidates.sort(key=lambda x: ((x.importance * x.decay_score), x.last_access_ms), reverse=True)
            return [item.to_dict() for item in candidates[:int(limit)]]

    def graph(self, *, project: str | None = None, agent_id: str | None = None, limit: int = 50) -> dict[str, Any]:
        proj_filter = None if project is None else _safe_scope(project, field="project")
        agent_filter = None if agent_id is None else _safe_scope(agent_id, field="agent_id")
        
        with self._lock:
            items = []
            for item in self._memories.values():
                if proj_filter is not None and item.project != proj_filter: continue
                if agent_filter is not None and item.agent_id != agent_filter: continue
                items.append(item)
                
            obs = []
            for item in self._observations.values():
                if proj_filter is not None and item["project"] != proj_filter: continue
                if agent_filter is not None and item["agent_id"] != agent_filter: continue
                obs.append(item)
                
            items.sort(key=lambda x: x.last_access_ms, reverse=True)
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
            
            nodes = [{"id": f"session:{sid}", "kind": "session", "label": sid} for sid in sorted(session_ids)]
            for item in items:
                nodes.append({
                    "id": f"memory:{item.memory_id}", "kind": "memory", "label": item.summary,
                    "memory_type": item.memory_type, "importance": item.importance,
                    "decay_score": item.decay_score, "agent_id": item.agent_id
                })
            for raw in obs:
                nodes.append({
                    "id": f"observation:{raw['observation_id']}", "kind": "observation", "label": raw["summary"],
                    "observation_type": raw["observation_type"], "importance": raw["importance"],
                    "agent_id": raw["agent_id"]
                })
            
            edges = [{"source": f"session:{ln['session_id']}", "target": f"memory:{ln['memory_id']}", "relation": ln["relation"]} for ln in valid_links]
            for raw in obs:
                if raw["session_id"]:
                    edges.append({"source": f"session:{raw['session_id']}", "target": f"observation:{raw['observation_id']}", "relation": "observed"})
                    
            return {
                "nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges),
                "project": project, "agent_id": agent_id
            }

    def build_context(
        self,
        *,
        project: str,
        agent_id: str | None,
        query: str | None = None,
        budget_tokens: int = 2000,
        mode: str = "search",
        limit: int = 5,
        exclude_memory_ids: Iterable[str] | None = None,
        signature_enforcement: str | None = None,
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
                query=query,
                limit=limit,
                exclude_memory_ids=exclude_ids,
                signature_enforcement=signature_enforcement,
            )
        else:
            items = self.list_memories(project=project, agent_id=agent_id, limit=limit, exclude_memory_ids=exclude_ids)
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
                "busy_timeout_ms": self.busy_timeout_ms,
                "dag_node_count": len(self.dag._nodes),
                "memory_backend": "in_memory_dag",
                "search_backend": "rust_bm25" if self._rust_index is not None else "python_bm25_fallback",
                "rust_index_available": self._rust_index is not None,
                "rust_index_error": self._rust_index_error,
                "rust_index_stats": rust_stats,
                "semantic_query_router": dict(self._router_stats),
            }


__all__ = ["MemoryCatalog", "MemoryItem", "privacy_filter", "source_hash"]
