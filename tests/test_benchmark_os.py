"""
HeliX-Bench Suite — Pure Python benchmarks. Zero LLM dependencies.

Measures:
  1. IOPS: Inserts per second into MerkleDAG under 500 concurrent threads
  2. TTFC: Time to First Context traversing 100k tombstoned nodes
  3. GC Sweep: RAM freed in MB after GC compacts the DAG

Usage:
  pytest tests/test_benchmark_os.py -v -s
  python -m pytest tests/test_benchmark_os.py --tb=short -s
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time
import tracemalloc

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_kv.merkle_dag import MerkleDAG
from helix_kv.memory_catalog import MemoryCatalog
from helix_kv.memory_gc import CognitiveGC


@pytest.fixture(autouse=True)
def _disable_rust_index_for_python_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELIX_MEMORY_RUST_INDEX", "0")
    MemoryCatalog._REGISTRY.clear()  # noqa: SLF001


# ---------------------------------------------------------------------------
# 1. IOPS — MerkleDAG insert throughput under 500 threads
# ---------------------------------------------------------------------------

class TestDAGIOPS:
    """Measures insert throughput under heavy thread contention."""

    THREAD_COUNT = 500
    INSERTS_PER_THREAD = 20  # 10,000 total inserts

    def test_iops_500_threads(self) -> None:
        dag = MerkleDAG()

        # Plant a root so all threads can attach
        root = dag.insert("benchmark-root")
        errors: list[str] = []
        barrier = threading.Barrier(self.THREAD_COUNT)

        def worker(thread_id: int) -> None:
            try:
                barrier.wait(timeout=30)
                parent = root.hash
                for i in range(self.INSERTS_PER_THREAD):
                    node = dag.insert(f"t{thread_id}-op{i}", parent_hash=parent)
                    parent = node.hash
            except Exception as exc:
                errors.append(f"thread-{thread_id}: {exc}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(self.THREAD_COUNT)]

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        elapsed = time.perf_counter() - t0

        total_ops = self.THREAD_COUNT * self.INSERTS_PER_THREAD
        # Actual unique inserts (content is unique per thread+op, so all should land)
        actual_nodes = len(dag._nodes) - 1  # minus root

        iops = actual_nodes / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"  IOPS BENCHMARK")
        print(f"  Threads:        {self.THREAD_COUNT}")
        print(f"  Target ops:     {total_ops}")
        print(f"  Actual inserts: {actual_nodes}")
        print(f"  Wall time:      {elapsed:.3f}s")
        print(f"  IOPS:           {iops:,.0f}")
        print(f"  Errors:         {len(errors)}")
        print(f"{'='*60}")

        assert not errors, f"Thread errors: {errors[:5]}"
        assert actual_nodes == total_ops, f"Expected {total_ops}, got {actual_nodes}"
        assert iops > 1000, f"IOPS {iops:.0f} is below minimum threshold of 1000"

    def test_iops_contention_fairness(self) -> None:
        """Verify no thread starvation — each thread must complete all inserts."""
        dag = MerkleDAG()
        root = dag.insert("fairness-root")

        completed = [0] * self.THREAD_COUNT
        barrier = threading.Barrier(self.THREAD_COUNT)

        def worker(tid: int) -> None:
            barrier.wait(timeout=30)
            parent = root.hash
            for i in range(self.INSERTS_PER_THREAD):
                node = dag.insert(f"fair-t{tid}-op{i}", parent_hash=parent)
                parent = node.hash
                completed[tid] += 1

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(self.THREAD_COUNT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        starved = [tid for tid, count in enumerate(completed) if count < self.INSERTS_PER_THREAD]
        assert not starved, f"Threads starved (incomplete inserts): {starved[:10]}"


# ---------------------------------------------------------------------------
# 2. TTFC — Time to First Context over 100k dead nodes
# ---------------------------------------------------------------------------

class TestTTFC:
    """Time to First Context: how fast can audit_chain() traverse a deep chain."""

    CHAIN_LENGTH = 100_000

    def test_ttfc_100k_chain(self) -> None:
        dag = MerkleDAG()

        # Build a 100k deep linear chain
        print(f"\n  Building {self.CHAIN_LENGTH:,} node chain...", end=" ", flush=True)
        t_build_start = time.perf_counter()
        parent: str | None = None
        leaf_hash = ""
        for i in range(self.CHAIN_LENGTH):
            node = dag.insert(f"thought-{i}", parent_hash=parent)
            parent = node.hash
            leaf_hash = node.hash
        t_build = time.perf_counter() - t_build_start
        print(f"done in {t_build:.2f}s")

        # Measure audit traversal
        t0 = time.perf_counter()
        chain = dag.audit_chain(leaf_hash, max_depth=self.CHAIN_LENGTH + 1)
        ttfc = time.perf_counter() - t0

        print(f"\n{'='*60}")
        print(f"  TTFC BENCHMARK")
        print(f"  Chain depth:    {self.CHAIN_LENGTH:,}")
        print(f"  Nodes returned: {len(chain):,}")
        print(f"  Traversal time: {ttfc*1000:.2f}ms")
        print(f"  Nodes/sec:      {len(chain)/ttfc:,.0f}")
        print(f"{'='*60}")

        assert len(chain) == self.CHAIN_LENGTH
        # Chain should be traversable in under 2 seconds even on slow hardware
        assert ttfc < 2.0, f"TTFC {ttfc:.3f}s exceeds 2s budget"

    def test_ttfc_with_tombstoned_nodes(self) -> None:
        """TTFC after GC has tombstoned most nodes — should be same or faster."""
        catalog = MemoryCatalog(":memory:")
        session_id = "ttfc-bench"

        # Insert 10k observations to build a chain
        node_count = 10_000
        for i in range(node_count):
            catalog.observe(
                project="bench",
                agent_id="ttfc",
                content=f"Deep thought observation number {i} with padding " + ("x" * 200),
                importance=1,  # Low importance — will be GC'd
                session_id=session_id,
            )

        leaf = catalog._session_heads[session_id]

        # GC sweep to tombstone content
        gc_obj = CognitiveGC(catalog, threshold=2.0)
        gc_result = gc_obj.sweep()

        # Now traverse the tombstoned chain
        t0 = time.perf_counter()
        chain = catalog.dag.audit_chain(leaf, max_depth=node_count + 1)
        ttfc = time.perf_counter() - t0

        print(f"\n{'='*60}")
        print(f"  TTFC POST-GC BENCHMARK")
        print(f"  Chain depth:      {len(chain):,}")
        print(f"  Traversal time:   {ttfc*1000:.2f}ms")
        print(f"  GC purged obs:    {gc_result['purged_observations']}")
        print(f"  Bytes freed est:  {gc_result.get('bytes_freed_estimate', 'N/A')}")
        print(f"{'='*60}")

        assert len(chain) == node_count
        assert ttfc < 1.0


# ---------------------------------------------------------------------------
# 3. GC Sweep — RAM measurement in MB
# ---------------------------------------------------------------------------

class TestGCSweepRAM:
    """Measures actual Python RSS/tracemalloc delta from GC sweep."""

    NODE_COUNT = 50_000
    CONTENT_SIZE = 512  # bytes per content string

    def test_gc_ram_freed(self) -> None:
        # Force clean state
        gc.collect()
        tracemalloc.start()
        snapshot_before_insert = tracemalloc.take_snapshot()

        catalog = MemoryCatalog(":memory:")
        padding = "A" * self.CONTENT_SIZE

        for i in range(self.NODE_COUNT):
            catalog.observe(
                project="gc-ram",
                agent_id="bench",
                content=f"obs-{i}: {padding}",
                importance=1,  # Low — will be swept
                session_id="gc-ram-session",
            )

        snapshot_after_insert = tracemalloc.take_snapshot()
        mem_after_insert = sum(stat.size for stat in snapshot_after_insert.statistics("filename"))

        # Run GC
        gc_obj = CognitiveGC(catalog, threshold=2.0)
        t0 = time.perf_counter()
        result = gc_obj.sweep()
        gc_time = time.perf_counter() - t0

        # Force Python GC to actually free unreferenced objects
        gc.collect()

        snapshot_after_gc = tracemalloc.take_snapshot()
        mem_after_gc = sum(stat.size for stat in snapshot_after_gc.statistics("filename"))

        freed_mb = (mem_after_insert - mem_after_gc) / (1024 * 1024)
        peak_mb = mem_after_insert / (1024 * 1024)
        final_mb = mem_after_gc / (1024 * 1024)

        tracemalloc.stop()

        print(f"\n{'='*60}")
        print(f"  GC SWEEP RAM BENCHMARK")
        print(f"  Nodes inserted:   {self.NODE_COUNT:,}")
        print(f"  Content size:     {self.CONTENT_SIZE} bytes/node")
        print(f"  Peak RAM:         {peak_mb:.2f} MB")
        print(f"  Post-GC RAM:      {final_mb:.2f} MB")
        print(f"  RAM freed:        {freed_mb:.2f} MB")
        print(f"  GC wall time:     {gc_time*1000:.2f}ms")
        print(f"  Purged obs:       {result['purged_observations']}")
        print(f"  Purged memories:  {result['purged_memories']}")
        print(f"  Bytes freed est:  {result.get('bytes_freed_estimate', 0) / (1024*1024):.2f} MB")
        print(f"{'='*60}")

        # GC should free meaningful RAM
        assert result["purged_observations"] > 0
        assert gc_time < 15.0, f"GC took {gc_time:.2f}s — too slow for {self.NODE_COUNT} nodes"

    def test_gc_preserves_dag_integrity(self) -> None:
        """After GC sweep, audit_chain must still return valid linked chain."""
        catalog = MemoryCatalog(":memory:")
        session_id = "integrity-test"

        for i in range(1000):
            catalog.observe(
                project="integrity",
                agent_id="bench",
                content=f"thought-{i}: " + ("Z" * 300),
                importance=1,
                session_id=session_id,
            )

        leaf = catalog._session_heads[session_id]

        # Verify chain before GC
        chain_before = catalog.dag.audit_chain(leaf, max_depth=1100)
        assert len(chain_before) == 1000

        # GC sweep
        gc_obj = CognitiveGC(catalog, threshold=2.0)
        gc_obj.sweep()

        # Verify chain after GC — must be identical length
        chain_after = catalog.dag.audit_chain(leaf, max_depth=1100)
        assert len(chain_after) == 1000, f"Chain broke: {len(chain_after)} != 1000"

        # Verify hash linkage is intact
        for i in range(len(chain_after) - 1):
            child = chain_after[i]
            parent = chain_after[i + 1]
            assert child.parent_hash == parent.hash, (
                f"Broken link at depth {i}: child.parent_hash={child.parent_hash} != parent.hash={parent.hash}"
            )
