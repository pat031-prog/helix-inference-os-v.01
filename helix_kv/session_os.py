from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def token_hash(token_ids: Iterable[int]) -> str:
    payload = ",".join(str(int(item)) for item in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prefix_block_hashes(token_ids: Iterable[int], *, block_size: int = 64) -> list[str]:
    values = [int(item) for item in token_ids]
    if not values:
        return []
    return [token_hash(values[index : index + int(block_size)]) for index in range(0, len(values), int(block_size))]


def common_prefix_length(left: Iterable[int], right: Iterable[int]) -> int:
    count = 0
    for a_item, b_item in zip(left, right):
        if int(a_item) != int(b_item):
            break
        count += 1
    return count


@dataclass(frozen=True)
class CatalogSession:
    session_id: str
    model_id: str
    agent_id: str
    model_ref: str
    arch: str
    path: Path
    token_count: int
    token_hash: str | None
    prefix_block_hashes: list[str]
    session_bytes: int
    codec: str
    audit_status: str | None
    session_hash: str | None
    merkle_root: str | None
    parent_session_id: str | None
    created_ms: float
    last_access_ms: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CatalogSession":
        block_hashes = json.loads(row["prefix_block_hashes"] or "[]")
        return cls(
            session_id=str(row["session_id"]),
            model_id=str(row["model_id"]),
            agent_id=str(row["agent_id"]),
            model_ref=str(row["model_ref"]),
            arch=str(row["arch"]),
            path=Path(str(row["path"])),
            token_count=int(row["token_count"] or 0),
            token_hash=row["token_hash"],
            prefix_block_hashes=[str(item) for item in block_hashes],
            session_bytes=int(row["session_bytes"] or 0),
            codec=str(row["codec"]),
            audit_status=row["audit_status"],
            session_hash=row["session_hash"],
            merkle_root=row["merkle_root"],
            parent_session_id=row["parent_session_id"],
            created_ms=float(row["created_ms"] or 0.0),
            last_access_ms=float(row["last_access_ms"] or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "model_ref": self.model_ref,
            "arch": self.arch,
            "path": str(self.path),
            "token_count": self.token_count,
            "token_hash": self.token_hash,
            "prefix_block_hashes": list(self.prefix_block_hashes),
            "session_bytes": self.session_bytes,
            "codec": self.codec,
            "audit_status": self.audit_status,
            "session_hash": self.session_hash,
            "merkle_root": self.merkle_root,
            "parent_session_id": self.parent_session_id,
            "created_ms": self.created_ms,
            "last_access_ms": self.last_access_ms,
        }


class SessionCatalog:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.root = self.db_path.parent.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @classmethod
    def open(cls, path: str | Path) -> "SessionCatalog":
        return cls(path)

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                model_ref TEXT NOT NULL,
                arch TEXT NOT NULL,
                path TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                token_hash TEXT,
                prefix_block_hashes TEXT NOT NULL,
                session_bytes INTEGER NOT NULL,
                codec TEXT NOT NULL,
                audit_status TEXT,
                session_hash TEXT,
                merkle_root TEXT,
                parent_session_id TEXT,
                created_ms REAL NOT NULL,
                last_access_ms REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_latest ON sessions(model_id, agent_id, created_ms)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_hash ON sessions(model_id, agent_id, token_hash)")
        self._conn.commit()

    def _resolve_session_path(self, session_path: str | Path) -> Path:
        raw = Path(session_path)
        if ".." in raw.parts:
            raise ValueError("session path traversal is not allowed")
        resolved = (self.root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("session path must stay inside the catalog root") from exc
        return resolved

    def record_session(
        self,
        *,
        session_id: str,
        model_id: str,
        agent_id: str,
        model_ref: str,
        arch: str,
        path: str | Path,
        token_ids: Iterable[int] | None = None,
        session_bytes: int = 0,
        codec: str = "rust-hlx-buffered-flat",
        audit_status: str | None = None,
        session_hash: str | None = None,
        merkle_root: str | None = None,
        parent_session_id: str | None = None,
        created_ms: float | None = None,
    ) -> CatalogSession:
        session_path = self._resolve_session_path(path)
        values = [int(item) for item in (token_ids or [])]
        now = time.time() * 1000.0
        created = float(created_ms if created_ms is not None else now)
        token_digest = token_hash(values) if values else None
        block_hashes = prefix_block_hashes(values)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                session_id, model_id, agent_id, model_ref, arch, path, token_count, token_hash,
                prefix_block_hashes, session_bytes, codec, audit_status, session_hash, merkle_root,
                parent_session_id, created_ms, last_access_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(session_id),
                str(model_id),
                str(agent_id),
                str(model_ref),
                str(arch),
                str(session_path),
                len(values),
                token_digest,
                json.dumps(block_hashes, separators=(",", ":")),
                int(session_bytes),
                str(codec),
                audit_status,
                session_hash,
                merkle_root,
                parent_session_id,
                created,
                now,
            ),
        )
        self._conn.commit()
        session = self.get_session(str(session_id))
        if session is None:
            raise RuntimeError("failed to record session")
        return session

    def get_session(self, session_id: str) -> CatalogSession | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (str(session_id),)).fetchone()
        return None if row is None else CatalogSession.from_row(row)

    def find_latest(self, model_id: str, agent_id: str) -> CatalogSession | None:
        row = self._conn.execute(
            """
            SELECT * FROM sessions
            WHERE model_id = ? AND agent_id = ?
            ORDER BY created_ms DESC
            LIMIT 1
            """,
            (str(model_id), str(agent_id)),
        ).fetchone()
        if row is None:
            return None
        session = CatalogSession.from_row(row)
        self.touch(session.session_id)
        return session

    def list_sessions(self, *, model_id: str | None = None, agent_id: str | None = None) -> list[CatalogSession]:
        clauses: list[str] = []
        params: list[str] = []
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(str(model_id))
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(str(agent_id))
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(f"SELECT * FROM sessions {where} ORDER BY created_ms DESC", params).fetchall()
        return [CatalogSession.from_row(row) for row in rows]

    def touch(self, session_id: str) -> None:
        self._conn.execute("UPDATE sessions SET last_access_ms = ? WHERE session_id = ?", (time.time() * 1000.0, str(session_id)))
        self._conn.commit()

    def parent_chain(self, session_id: str) -> list[CatalogSession]:
        chain: list[CatalogSession] = []
        seen: set[str] = set()
        current = self.get_session(session_id)
        while current is not None and current.session_id not in seen:
            chain.append(current)
            seen.add(current.session_id)
            if not current.parent_session_id:
                break
            current = self.get_session(current.parent_session_id)
        return chain

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute("SELECT COUNT(*) AS count, COALESCE(SUM(session_bytes), 0) AS bytes FROM sessions").fetchone()
        return {"session_count": int(row["count"] or 0), "session_bytes": int(row["bytes"] or 0)}


@dataclass(frozen=True)
class PrefixMatch:
    status: str
    session: CatalogSession | None
    prefix_match_tokens: int
    requested_tokens: int
    new_tokens_computed: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": None if self.session is None else self.session.session_id,
            "model_id": None if self.session is None else self.session.model_id,
            "agent_id": None if self.session is None else self.session.agent_id,
            "prefix_match_tokens": self.prefix_match_tokens,
            "requested_tokens": self.requested_tokens,
            "new_tokens_computed": self.new_tokens_computed,
            "reason": self.reason,
        }


class PrefixResolver:
    def __init__(self, catalog: SessionCatalog) -> None:
        self.catalog = catalog

    def find_best_prefix(
        self,
        *,
        model_id: str,
        agent_id: str,
        token_ids: Iterable[int],
        arch: str = "transformer",
    ) -> PrefixMatch:
        requested = [int(item) for item in token_ids]
        if "mamba" in str(arch).lower() or "hybrid" in str(arch).lower():
            best: tuple[int, CatalogSession] | None = None
            saw_partial = False
            for session in self.catalog.list_sessions(model_id=str(model_id), agent_id=str(agent_id)):
                meta_path = session.path / "session.json"
                stored_ids: list[int] = []
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        stored_ids = [int(item) for item in meta.get("session_token_ids") or []]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        stored_ids = []
                if not stored_ids:
                    continue
                matched = common_prefix_length(stored_ids, requested)
                if matched == len(stored_ids):
                    if best is None or matched > best[0]:
                        best = (matched, session)
                elif matched > 0:
                    saw_partial = True
            if best is not None:
                matched, session = best
                return PrefixMatch(
                    status="hybrid_checkpoint_v0",
                    session=session,
                    prefix_match_tokens=matched,
                    requested_tokens=len(requested),
                    new_tokens_computed=max(0, len(requested) - matched),
                    reason="exact hybrid checkpoint prefix restored; arbitrary Mamba slicing is not used",
                )
            return PrefixMatch(
                status="unsupported_partial_hybrid_prefix" if saw_partial else "unsupported_hybrid_v0",
                session=None,
                prefix_match_tokens=0,
                requested_tokens=len(requested),
                new_tokens_computed=len(requested),
                reason="hybrid prefix v0 requires an exact saved checkpoint boundary",
            )
        best: tuple[int, CatalogSession] | None = None
        for session in self.catalog.list_sessions(model_id=str(model_id), agent_id=str(agent_id)):
            if "mamba" in session.arch.lower() or "hybrid" in session.arch.lower():
                continue
            meta_path = session.path / "session.json"
            stored_ids: list[int] = []
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    stored_ids = [int(item) for item in meta.get("session_token_ids") or []]
                except (json.JSONDecodeError, TypeError, ValueError):
                    stored_ids = []
            if not stored_ids and session.token_count and session.token_hash == token_hash(requested[: session.token_count]):
                stored_ids = requested[: session.token_count]
            matched = common_prefix_length(stored_ids, requested)
            if matched <= 0:
                continue
            if best is None or matched > best[0]:
                best = (matched, session)
        if best is None:
            return PrefixMatch(
                status="miss",
                session=None,
                prefix_match_tokens=0,
                requested_tokens=len(requested),
                new_tokens_computed=len(requested),
                reason="no compatible prefix session",
            )
        matched, session = best
        return PrefixMatch(
            status="hit",
            session=session,
            prefix_match_tokens=matched,
            requested_tokens=len(requested),
            new_tokens_computed=max(0, len(requested) - matched),
        )


@dataclass(frozen=True)
class RouteDecision:
    selected_model_id: str
    estimated_cost_ms: float
    active_model_reused: bool
    model_swapped: bool
    session_restored: bool
    audit_status_used: str | None
    prefix_match_tokens: int
    candidate_models: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model_id": self.selected_model_id,
            "estimated_cost_ms": self.estimated_cost_ms,
            "active_model_reused": self.active_model_reused,
            "model_swapped": self.model_swapped,
            "session_restored": self.session_restored,
            "audit_status_used": self.audit_status_used,
            "prefix_match_tokens": self.prefix_match_tokens,
            "candidate_models": _json_ready(self.candidate_models),
        }


class SessionScheduler:
    def __init__(self, catalog: SessionCatalog, *, audit_requires_verified: bool = False) -> None:
        self.catalog = catalog
        self.audit_requires_verified = bool(audit_requires_verified)

    def route(
        self,
        task: dict[str, Any],
        registry: dict[str, dict[str, Any]],
        lifecycle: Any,
    ) -> RouteDecision:
        requested = str(task.get("model_id") or "")
        candidates = [requested] if requested else []
        if not requested:
            for model_id, item in registry.items():
                if model_id not in candidates and task.get("capability") in (item.get("capabilities") or []):
                    candidates.append(model_id)
        if not candidates:
            candidates = list(registry)
        active_model_id = getattr(lifecycle, "active_model_id", None)
        token_ids = [int(item) for item in task.get("token_ids") or []]
        expected_decode = int(task.get("expected_decode_tokens") or task.get("max_new_tokens") or 1)
        scored: list[dict[str, Any]] = []
        resolver = PrefixResolver(self.catalog)
        for model_id in candidates:
            info = registry.get(model_id) or {}
            session = self.catalog.find_latest(model_id, str(task.get("agent_id") or "default"))
            arch = str(info.get("arch") or (session.arch if session else "transformer"))
            prefix = resolver.find_best_prefix(model_id=model_id, agent_id=str(task.get("agent_id") or "default"), token_ids=token_ids, arch=arch)
            missing_tokens = prefix.new_tokens_computed if prefix.status == "hit" else len(token_ids)
            load_cost = 0.0 if active_model_id == model_id else float(info.get("load_time_estimate_ms") or 0.0)
            restore_cost = float(task.get("session_restore_ms") or 0.0) if session else 0.0
            prefill_cost = missing_tokens * float(task.get("prefill_ms_per_token") or info.get("prefill_ms_per_token") or 1.0)
            decode_cost = expected_decode * float(task.get("decode_ms_per_token") or info.get("decode_ms_per_token") or 1.0)
            audit_penalty = 0.0
            if self.audit_requires_verified and session and session.audit_status != "verified":
                audit_penalty = float(task.get("audit_penalty_ms") or 10_000.0)
            memory_pressure = float(task.get("memory_pressure_penalty_ms") or 0.0) if info.get("ram_bytes_estimate") else 0.0
            total = load_cost + restore_cost + prefill_cost + decode_cost + audit_penalty + memory_pressure
            scored.append(
                {
                    "model_id": model_id,
                    "estimated_cost_ms": total,
                    "active_model_reused": active_model_id == model_id,
                    "model_swapped": active_model_id != model_id,
                    "session_restored": session is not None,
                    "audit_status_used": None if session is None else session.audit_status,
                    "prefix_match_tokens": prefix.prefix_match_tokens,
                    "prefix_status": prefix.status,
                    "missing_tokens": missing_tokens,
                    "load_cost_ms": load_cost,
                    "restore_cost_ms": restore_cost,
                    "prefill_cost_ms": prefill_cost,
                    "decode_cost_ms": decode_cost,
                    "audit_penalty_ms": audit_penalty,
                }
            )
        selected = min(scored, key=lambda item: (float(item["estimated_cost_ms"]), 0 if item["active_model_reused"] else 1))
        return RouteDecision(
            selected_model_id=str(selected["model_id"]),
            estimated_cost_ms=float(selected["estimated_cost_ms"]),
            active_model_reused=bool(selected["active_model_reused"]),
            model_swapped=bool(selected["model_swapped"]),
            session_restored=bool(selected["session_restored"]),
            audit_status_used=selected.get("audit_status_used"),
            prefix_match_tokens=int(selected["prefix_match_tokens"]),
            candidate_models=scored,
        )
