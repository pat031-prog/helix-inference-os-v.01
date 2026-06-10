from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any

DAG_HASH_PROFILE_LEGACY = "helix-merkle-dag-v1-legacy-concat-sha256"
DAG_HASH_PROFILE_V2 = "helix-merkle-dag-v2-domain-length-sha256"
DAG_HASH_V2_DOMAIN = b"HLX-DAG-V2"


@dataclass(frozen=True)
class MerkleNode:
    content: str
    hash: str
    parent_hash: str | None
    timestamp: float
    depth: int
    hash_profile: str = DAG_HASH_PROFILE_V2


class MerkleDAG:
    """In-memory Directed Acyclic Graph structure built via cryptographic chaining."""

    def __init__(self) -> None:
        self._nodes: dict[str, MerkleNode] = {}
        # Used a standard Lock instead of RLock to explicitly guard against 
        # re-entrant deadlocks per Red Team audit
        self._lock = threading.Lock()

    @staticmethod
    def _compute_hash_legacy(content: str, parent_hash: str | None) -> str:
        payload = content.encode("utf-8")
        if parent_hash:
            payload += parent_hash.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _compute_hash_v2(content: str, parent_hash: str | None) -> str:
        content_bytes = content.encode("utf-8")
        parent_bytes = parent_hash.encode("utf-8") if parent_hash is not None else b""
        payload = b"".join(
            [
                DAG_HASH_V2_DOMAIN,
                len(content_bytes).to_bytes(8, "big"),
                content_bytes,
                b"\x01" if parent_hash is not None else b"\x00",
                len(parent_bytes).to_bytes(8, "big"),
                parent_bytes,
            ]
        )
        return hashlib.sha256(payload).hexdigest()

    def _compute_hash(
        self,
        content: str,
        parent_hash: str | None,
        *,
        hash_profile: str = DAG_HASH_PROFILE_V2,
    ) -> str:
        if hash_profile == DAG_HASH_PROFILE_LEGACY:
            return self._compute_hash_legacy(content, parent_hash)
        if hash_profile != DAG_HASH_PROFILE_V2:
            raise ValueError(f"unsupported MerkleDAG hash profile: {hash_profile}")
        return self._compute_hash_v2(content, parent_hash)

    def _select_hash_profile(
        self,
        content: str,
        parent_hash: str | None,
        expected_hash: str | None,
    ) -> tuple[str, str]:
        v2_hash = self._compute_hash_v2(content, parent_hash)
        if not expected_hash or expected_hash == v2_hash:
            return v2_hash, DAG_HASH_PROFILE_V2
        legacy_hash = self._compute_hash_legacy(content, parent_hash)
        if expected_hash == legacy_hash:
            return legacy_hash, DAG_HASH_PROFILE_LEGACY
        return v2_hash, DAG_HASH_PROFILE_V2

    def hash_matches(self, node: MerkleNode) -> bool:
        profile = node.hash_profile or DAG_HASH_PROFILE_LEGACY
        if profile == DAG_HASH_PROFILE_LEGACY:
            return node.hash == self._compute_hash_legacy(node.content, node.parent_hash)
        if profile == DAG_HASH_PROFILE_V2:
            return node.hash == self._compute_hash_v2(node.content, node.parent_hash)
        return False

    def _insert_unlocked(
        self,
        content: str,
        parent_hash: str | None = None,
        *,
        expected_hash: str | None = None,
    ) -> MerkleNode:
        """Insert without acquiring self._lock. Caller MUST hold its own lock."""
        node_hash, hash_profile = self._select_hash_profile(content, parent_hash, expected_hash)
        now = time.time() * 1000.0

        if node_hash in self._nodes:
            return self._nodes[node_hash]

        depth = 0
        if parent_hash:
            if parent_hash not in self._nodes:
                raise ValueError(f"parent_hash {parent_hash} not found in MerkleDAG")
            depth = self._nodes[parent_hash].depth + 1

        node = MerkleNode(
            content=content,
            hash=node_hash,
            parent_hash=parent_hash,
            timestamp=now,
            depth=depth,
            hash_profile=hash_profile,
        )
        self._nodes[node_hash] = node
        return node

    def insert(self, content: str, parent_hash: str | None = None) -> MerkleNode:
        with self._lock:
            return self._insert_unlocked(content, parent_hash)

    def lookup(self, hash_hex: str) -> MerkleNode | None:
        with self._lock:
            return self._nodes.get(hash_hex)

    def audit_chain(self, leaf_hash: str, max_depth: int = 10000) -> list[MerkleNode]:
        """Traverses parent links from leaf to root ensuring tamper-evident chains."""
        chain = []
        current_hash: str | None = leaf_hash

        # Traverse quickly under lock to prevent concurrent tears
        with self._lock:
            while current_hash and len(chain) < max_depth:
                node = self._nodes.get(current_hash)
                if not node:
                    break
                chain.append(node)
                current_hash = node.parent_hash

        if len(chain) == max_depth and current_hash:
            raise RuntimeError(f"audit_chain exceeded max_depth={max_depth}. Possible cycle injected.")

        return chain

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                hash_id: {
                    "content": node.content,
                    "parent_hash": node.parent_hash,
                    "timestamp": node.timestamp,
                    "depth": node.depth,
                    "hash_profile": node.hash_profile,
                }
                for hash_id, node in self._nodes.items()
            }
