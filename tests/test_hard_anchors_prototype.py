"""
test_hard_anchors_prototype.py

Standalone prototype to empirically validate the Hard-Anchors (Semantic DNA)
architecture concept before modifying the production MerkleDAG.

Simulates 5000 rounds of memory insertion.
Compares context build times between:
- Legacy Naive (Loading full narrative context)
- Hard Anchors (Loading only the identity lane / raw anchors)
"""
import time
import hashlib
from typing import List, Dict, Any

class PrototypeNode:
    def __init__(self, narrative_payload: str, raw_anchor: str):
        self.narrative_payload = narrative_payload
        self.raw_anchor = raw_anchor
        # In a real DAG, we'd hash the contents and link to parent
        self.node_hash = hashlib.sha256(f"{narrative_payload}{raw_anchor}".encode()).hexdigest()

class HardAnchorsMemoryCatalog:
    def __init__(self):
        self.nodes: List[PrototypeNode] = []

    def insert(self, narrative: str, anchor: str):
        self.nodes.append(PrototypeNode(narrative, anchor))

    def build_context_legacy(self, depth: int) -> str:
        """Simulates legacy full-history replay or naive copy."""
        # Load the full narrative for the last N nodes
        selected = self.nodes[-depth:]

        # Simulate the 'cost' of loading heavy narrative strings
        # (in SQLite/Rust this involves I/O and deserialization of large strings)
        context = []
        for node in selected:
            # We touch the narrative payload to simulate loading it
            payload = node.narrative_payload
            context.append(f"<legacy_memory>{payload}</legacy_memory>")

        return "\n".join(context)

    def build_context_hard_anchors(self, depth: int) -> str:
        """Simulates the new architecture using the Identity Lane."""
        # Load ONLY the lightweight anchors
        selected = self.nodes[-depth:]

        context = []
        for node in selected:
            # We ONLY touch the raw_anchor (64 bytes)
            # The narrative payload remains 'lazy loaded' / untouched in memory
            anchor = node.raw_anchor
            context.append(f"<hard_anchor>{anchor}</hard_anchor>")

        return "\n".join(context)

def run_experiment():
    catalog = HardAnchorsMemoryCatalog()

    print("[*] Generating 5000 heavy nodes (Simulating 4KB narrative per node)...")
    heavy_narrative = "A" * 4000  # 4KB string to simulate deep thoughts/summaries

    for i in range(5000):
        anchor = hashlib.sha256(str(i).encode()).hexdigest()
        catalog.insert(heavy_narrative + str(i), anchor)

    print("[*] Testing Legacy Naive Context Build (Depth=5000)...")
    t0 = time.perf_counter()
    # Build context 100 times to get stable average
    for _ in range(100):
        _ = catalog.build_context_legacy(5000)
    t1 = time.perf_counter()
    legacy_ms = ((t1 - t0) / 100) * 1000

    print("[*] Testing Hard Anchors Context Build (Depth=5000)...")
    t0 = time.perf_counter()
    # Build context 100 times to get stable average
    for _ in range(100):
        _ = catalog.build_context_hard_anchors(5000)
    t1 = time.perf_counter()
    anchors_ms = ((t1 - t0) / 100) * 1000

    speedup = legacy_ms / max(anchors_ms, 0.001)

    print(f"\n--- EMPIRICAL RESULTS ---")
    print(f"Legacy Latency:       {legacy_ms:.2f} ms")
    print(f"Hard Anchors Latency: {anchors_ms:.2f} ms")
    print(f"Speedup:              {speedup:.1f}x")

    # Validation against the claim:
    if anchors_ms <= 0.4:
        print("[SUCCESS] Hard Anchors Latency is sub-0.4ms!")
    else:
        print("[FAIL] Hard Anchors Latency exceeded baseline.")

    if speedup >= 9.0:
        print("[SUCCESS] Speedup is >= 9.0x!")
    else:
        print("[FAIL] Speedup is insufficient.")

if __name__ == "__main__":
    run_experiment()
