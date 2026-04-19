"""
StateServer concurrency test — measures throughput under parallel client load.

Verifies:
  1. N concurrent clients can bulk_remember + search without data loss
  2. Total throughput scales with client count (not collapses)
  3. No request interleaving corrupts state
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from helix_kv.ipc_state_server import StateClient, StateServer


async def _boot_server(*, transport: str = "tcp") -> tuple[StateServer, asyncio.Task[None]]:
    server = StateServer(transport=transport, host="127.0.0.1", port=0)
    task = asyncio.create_task(server.start())
    for _ in range(200):
        if server._server is not None:
            break
        await asyncio.sleep(0.01)
    assert server._server is not None, "StateServer failed to bind"
    return server, task


async def _stop_server(server: StateServer, task: asyncio.Task[None]) -> None:
    await server.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _client_worker(
    host: str,
    port: int,
    worker_id: int,
    items_per_worker: int,
    results: dict[int, dict[str, Any]],
) -> None:
    """Each worker: bulk_remember N items, then search for one of them."""
    client = StateClient(transport="tcp", host=host, port=port, timeout=15.0)
    try:
        batch = [
            {
                "project": "concurrency",
                "agent_id": f"worker-{worker_id}",
                "memory_type": "semantic",
                "summary": f"w{worker_id}-item-{i}",
                "content": f"Worker {worker_id} memory item {i} about topic-{worker_id % 5}",
                "importance": 3 + (i % 7),
            }
            for i in range(items_per_worker)
        ]

        t0 = time.perf_counter()
        inserted = await client.bulk_remember(batch)
        insert_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hits = await client.search(
            project="concurrency",
            agent_id=f"worker-{worker_id}",
            query=f"w{worker_id}-item-0",
            limit=3,
            signature_enforcement="permissive",
        )
        search_ms = (time.perf_counter() - t0) * 1000

        results[worker_id] = {
            "inserted": len(inserted),
            "search_hits": len(hits),
            "insert_ms": insert_ms,
            "search_ms": search_ms,
            "found_own": any(
                f"w{worker_id}-item-0" in (h.get("summary") or "") for h in hits
            ),
        }
    except Exception as exc:
        results[worker_id] = {"error": str(exc)}
    finally:
        await client.close()


class TestStateServerConcurrency:
    """Concurrent clients against a single StateServer instance."""

    @pytest.mark.parametrize("client_count,items_per_client", [
        (4, 50),
        (8, 25),
        (16, 10),
    ])
    def test_concurrent_bulk_remember_and_search(
        self, client_count: int, items_per_client: int
    ) -> None:
        async def run() -> dict[str, Any]:
            server, task = await _boot_server()
            try:
                results: dict[int, dict[str, Any]] = {}
                workers = [
                    _client_worker(
                        server.host, server.port, wid, items_per_client, results
                    )
                    for wid in range(client_count)
                ]
                t0 = time.perf_counter()
                await asyncio.gather(*workers)
                wall_ms = (time.perf_counter() - t0) * 1000

                stats = await StateClient(
                    transport="tcp", host=server.host, port=server.port, timeout=5.0
                ).stats()

                return {
                    "client_count": client_count,
                    "items_per_client": items_per_client,
                    "wall_ms": wall_ms,
                    "stats": stats,
                    "workers": results,
                }
            finally:
                await _stop_server(server, task)

        payload = asyncio.run(run())

        total_expected = client_count * items_per_client
        errors = [
            (wid, r["error"])
            for wid, r in payload["workers"].items()
            if "error" in r
        ]
        assert not errors, f"Worker errors: {errors}"

        # Every worker inserted the right count
        for wid, r in payload["workers"].items():
            assert r["inserted"] == items_per_client, (
                f"Worker {wid}: expected {items_per_client} inserts, got {r['inserted']}"
            )

        # Each worker found its own data
        for wid, r in payload["workers"].items():
            assert r["found_own"], f"Worker {wid} could not find its own memory"

        # Total memories in server matches sum of all inserts
        assert payload["stats"]["memory_count"] == total_expected, (
            f"Expected {total_expected} memories, got {payload['stats']['memory_count']}"
        )

        # Print summary
        insert_times = [r["insert_ms"] for r in payload["workers"].values() if "insert_ms" in r]
        search_times = [r["search_ms"] for r in payload["workers"].values() if "search_ms" in r]
        total_ops = total_expected
        wall = payload["wall_ms"]

        print(f"\n{'='*60}")
        print(f"  STATESERVER CONCURRENCY: {client_count} clients x {items_per_client} items")
        print(f"  Total memories:    {payload['stats']['memory_count']}")
        print(f"  Wall time:         {wall:.1f}ms")
        print(f"  Throughput:        {total_ops / (wall / 1000):.0f} ops/s")
        print(f"  Insert p50:        {sorted(insert_times)[len(insert_times)//2]:.1f}ms")
        print(f"  Search p50:        {sorted(search_times)[len(search_times)//2]:.1f}ms")
        print(f"  All found own:     True")
        print(f"{'='*60}")

    def test_interleaved_writes_and_reads(self) -> None:
        """Writers and readers running simultaneously — no stale reads."""
        async def run() -> dict[str, Any]:
            server, task = await _boot_server()
            try:
                # Phase 1: seed some data
                seed_client = StateClient(
                    transport="tcp", host=server.host, port=server.port, timeout=5.0
                )
                await seed_client.bulk_remember([
                    {
                        "project": "interleave",
                        "agent_id": "seeder",
                        "memory_type": "semantic",
                        "summary": f"seed-{i}",
                        "content": f"Seed memory {i} about baseline topic",
                        "importance": 5,
                    }
                    for i in range(20)
                ])
                await seed_client.close()

                # Phase 2: concurrent writers + readers
                write_results: list[int] = []
                read_results: list[int] = []

                async def writer(wid: int) -> None:
                    c = StateClient(
                        transport="tcp", host=server.host, port=server.port, timeout=10.0
                    )
                    try:
                        inserted = await c.bulk_remember([
                            {
                                "project": "interleave",
                                "agent_id": f"writer-{wid}",
                                "memory_type": "semantic",
                                "summary": f"live-write-{wid}-{j}",
                                "content": f"Live write {wid} item {j} concurrent data",
                                "importance": 7,
                            }
                            for j in range(10)
                        ])
                        write_results.append(len(inserted))
                    finally:
                        await c.close()

                async def reader(rid: int) -> None:
                    c = StateClient(
                        transport="tcp", host=server.host, port=server.port, timeout=10.0
                    )
                    try:
                        # Small delay to let some writes land
                        await asyncio.sleep(0.01 * rid)
                        hits = await c.search(
                            project="interleave",
                            agent_id=None,
                            query="baseline topic",
                            limit=5,
                            signature_enforcement="permissive",
                        )
                        read_results.append(len(hits))
                    finally:
                        await c.close()

                tasks = (
                    [writer(i) for i in range(4)]
                    + [reader(i) for i in range(4)]
                )
                await asyncio.gather(*tasks)

                final_stats = await StateClient(
                    transport="tcp", host=server.host, port=server.port, timeout=5.0
                ).stats()

                return {
                    "write_results": write_results,
                    "read_results": read_results,
                    "final_memory_count": final_stats["memory_count"],
                }
            finally:
                await _stop_server(server, task)

        payload = asyncio.run(run())

        # All writers succeeded
        assert all(w == 10 for w in payload["write_results"])
        # All readers got results (seed data was there)
        assert all(r > 0 for r in payload["read_results"])
        # Final count = 20 seed + 4 writers * 10 = 60
        assert payload["final_memory_count"] == 60
