from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from helix_kv.memory_catalog import MemoryCatalog
from helix_kv.merkle_dag import MerkleNode


@dataclass(frozen=True)
class PrunedTombstone:
    """
    Cryptographic proof that a node existed without retaining its content.

    Instead of replacing content with "[GC_PRUNED]" (which breaks hash verification),
    we store the content_hash so an auditor can verify:
      1. The node's hash is structurally valid in the parent chain.
      2. The content_hash proves WHAT was there, without storing it.
      3. The original_size proves how much RAM was freed.

    Verification equation:
      sha256(original_content + parent_hash) == node.hash  [still holds — hash is untouched]
      sha256(original_content) == tombstone.content_hash    [provable if content is replayed]
    """
    content_hash: str
    original_size: int


# Sentinel prefix for tombstoned content — deterministic and parseable
_TOMBSTONE_PREFIX = "[GC_TOMBSTONE:sha256="
_TOMBSTONE_SUFFIX = "]"


def _make_tombstone_content(content: str) -> str:
    """Create a compact tombstone string that embeds the content hash."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{_TOMBSTONE_PREFIX}{content_hash},size={len(content)}{_TOMBSTONE_SUFFIX}"


def is_tombstoned(content: str) -> bool:
    return content.startswith(_TOMBSTONE_PREFIX)


class CognitiveGC:
    """OOM-Killer for Inference Memory. Sweeps unneeded narrative branches to save RAM.

    Key invariant: after sweep, audit_chain() still returns a valid parent-linked chain.
    Node hashes are NEVER modified — only the content field is replaced with a
    deterministic tombstone that preserves the content's SHA-256 for future forensics.
    """

    def __init__(self, catalog: MemoryCatalog, threshold: float = 2.0) -> None:
        self.catalog = catalog
        self.threshold = threshold

    def sweep(self) -> dict[str, Any]:
        t0 = time.time()
        purged_memories = 0
        purged_observations = 0
        bytes_freed_estimate = 0
        rust_index_tombstoned = 0

        with self.catalog._lock:
            # --- Pass 1: Memory Items ---
            live_memories = {}
            for mem_id, item in self.catalog._memories.items():
                score = item.importance * item.decay_score
                if score < self.threshold:
                    purged_memories += 1
                else:
                    live_memories[mem_id] = item

            # --- Pass 2: Observations ---
            live_observations = {}
            for obs_id, obs in self.catalog._observations.items():
                score = obs.get("importance", 5) * 0.5
                if score < self.threshold:
                    purged_observations += 1
                else:
                    live_observations[obs_id] = obs

            # --- Pass 3: DAG Content Compaction (Cryptographically Safe) ---
            # Collect hashes that are still referenced by live data or are chain heads
            active_hashes: set[str] = set()
            for sid, hsh in self.catalog._session_heads.items():
                active_hashes.add(hsh)

            # Also protect nodes reachable from active heads (the recent chain tip)
            for head_hash in list(active_hashes):
                current: str | None = head_hash
                depth = 0
                while current and depth < 10:
                    node = self.catalog.dag._nodes.get(current)
                    if not node:
                        break
                    active_hashes.add(current)
                    current = node.parent_hash
                    depth += 1

            nodes = self.catalog.dag._nodes
            for hash_key in list(nodes.keys()):
                node = nodes[hash_key]
                # Skip already tombstoned, active, or small nodes
                if is_tombstoned(node.content):
                    continue
                if node.hash in active_hashes:
                    continue
                if len(node.content) <= 128:
                    continue

                original_size = len(node.content)
                tombstone_content = _make_tombstone_content(node.content)

                # Replace content but preserve ALL structural fields
                # Critical: node.hash is NOT recomputed — it stays valid
                hollow = MerkleNode(
                    content=tombstone_content,
                    hash=node.hash,
                    parent_hash=node.parent_hash,
                    timestamp=node.timestamp,
                    depth=node.depth,
                )
                nodes[hash_key] = hollow
                rust_receipt = self.catalog.gc_tombstone_index(node.hash)
                rust_index_tombstoned += int(rust_receipt.get("tombstoned_count") or 0)
                bytes_freed_estimate += original_size - len(tombstone_content)

            self.catalog._memories = live_memories
            self.catalog._observations = live_observations

        t1 = time.time()

        return {
            "purged_memories": purged_memories,
            "purged_observations": purged_observations,
            "retained_memories": len(self.catalog._memories),
            "retained_observations": len(self.catalog._observations),
            "bytes_freed_estimate": bytes_freed_estimate,
            "rust_index_tombstoned": rust_index_tombstoned,
            "gc_execution_ms": (t1 - t0) * 1000.0,
            "threshold_applied": self.threshold,
        }
