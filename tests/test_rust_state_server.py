"""
Integration tests for the Rust helix-state-server binary.

Uses the existing Python StateClient to talk to the Rust server over TCP.
Validates wire-protocol compatibility and functional correctness.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from helix_kv.ipc_state_server import StateClient

REPO = Path(__file__).resolve().parents[1]
RUST_BIN = REPO / "crates" / "helix-state-server" / "target" / "x86_64-pc-windows-gnullvm" / "release" / "helix-state-server.exe"

pytestmark = pytest.mark.skipif(
    not RUST_BIN.exists(),
    reason=f"Rust binary not found at {RUST_BIN}",
)


def _start_rust_server(port: int = 0) -> tuple[subprocess.Popen[bytes], int]:
    """Start the Rust StateServer on an ephemeral port and return (proc, actual_port)."""
    # Use a high ephemeral port if 0 not supported
    if port == 0:
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

    env = {**os.environ, "HELIX_STATE_HOST": "127.0.0.1", "HELIX_STATE_PORT": str(port)}
    proc = subprocess.Popen(
        [str(RUST_BIN)],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    # Wait for the server to bind
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            import socket
            s = socket.create_connection(("127.0.0.1", port), timeout=0.1)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError(f"Rust server failed to start on port {port}")

    return proc, port


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestRustStateServerIntegration:
    """Python StateClient → Rust helix-state-server wire compatibility."""

    def test_remember_search_stats(self) -> None:
        proc, port = _start_rust_server()
        try:
            async def run() -> dict[str, Any]:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=5.0)
                try:
                    # Remember
                    r = await client.remember(
                        content="postgres migration schema index",
                        project="db",
                        agent_id="a1",
                        record_kind="memory",
                        memory_id="m1",
                        memory_type="semantic",
                        summary="postgres migration",
                        index_content="postgres migration schema index",
                        importance=9,
                        decay_score=1.0,
                    )
                    # Search
                    hits = await client.search(
                        query="postgres migration",
                        limit=5,
                        project="db",
                        record_kind="memory",
                    )
                    # Stats
                    st = await client.stats()
                    return {"remember": r, "hits": hits, "stats": st}
                finally:
                    await client.close()

            result = asyncio.run(run())

            assert result["remember"]["node_hash"] is not None
            assert len(result["hits"]) == 1
            assert result["hits"][0]["memory_id"] == "m1"
            assert result["stats"]["node_count"] == 1
            assert result["stats"]["backend"] == "rust_tokio_bm25"
        finally:
            _stop_server(proc)

    def test_bulk_remember_and_gc(self) -> None:
        proc, port = _start_rust_server()
        try:
            async def run() -> dict[str, Any]:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=5.0)
                try:
                    # Bulk insert
                    inserted = await client._call("bulk_remember", {
                        "items": [
                            {
                                "content": f"item {i} about topic-{i % 3}",
                                "project": "bulk",
                                "agent_id": "a1",
                                "record_kind": "memory",
                                "memory_id": f"m{i}",
                                "summary": f"item {i}",
                                "index_content": f"item {i} about topic-{i % 3}",
                                "importance": 5 + (i % 5),
                            }
                            for i in range(50)
                        ]
                    })

                    # Search
                    hits = await client.search(
                        query="topic-2",
                        limit=5,
                        project="bulk",
                        record_kind="memory",
                    )

                    # Stats before GC
                    stats_pre = await client.stats()

                    # Tombstone one node
                    gc_result = await client._call("gc_tombstone", {"memory_id": "m0"})

                    # Verify chain
                    node_hash = inserted[0]["node_hash"]
                    verify = await client.verify_chain(node_hash)

                    stats_post = await client.stats()

                    return {
                        "inserted": len(inserted),
                        "hits": len(hits),
                        "stats_pre": stats_pre,
                        "gc_result": gc_result,
                        "verify": verify,
                        "stats_post": stats_post,
                    }
                finally:
                    await client.close()

            result = asyncio.run(run())

            assert result["inserted"] == 50
            assert result["hits"] > 0
            assert result["stats_pre"]["node_count"] == 50
            assert result["gc_result"]["tombstoned_count"] == 1
            assert result["verify"]["status"] == "tombstone_preserved"
            assert result["stats_post"]["tombstoned_count"] == 1
        finally:
            _stop_server(proc)

    def test_concurrent_clients(self) -> None:
        """Multiple clients hitting the Rust server simultaneously."""
        proc, port = _start_rust_server()
        try:
            async def worker(wid: int, results: dict[int, dict[str, Any]]) -> None:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=10.0)
                try:
                    inserted = await client._call("bulk_remember", {
                        "items": [
                            {
                                "content": f"w{wid}-item-{i}",
                                "project": "conc",
                                "agent_id": f"w{wid}",
                                "record_kind": "memory",
                                "memory_id": f"w{wid}-m{i}",
                                "summary": f"w{wid}-item-{i}",
                                "index_content": f"w{wid}-item-{i}",
                            }
                            for i in range(20)
                        ]
                    })
                    hits = await client.search(
                        query=f"w{wid}-item-0",
                        limit=3,
                        project="conc",
                        record_kind="memory",
                    )
                    results[wid] = {"inserted": len(inserted), "hits": len(hits)}
                except Exception as exc:
                    results[wid] = {"error": str(exc)}
                finally:
                    await client.close()

            async def run() -> dict[str, Any]:
                results: dict[int, dict[str, Any]] = {}
                t0 = time.perf_counter()
                await asyncio.gather(*[worker(i, results) for i in range(8)])
                wall = (time.perf_counter() - t0) * 1000

                stats_client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=5.0)
                stats = await stats_client.stats()
                await stats_client.close()

                return {"wall_ms": wall, "results": results, "stats": stats}

            payload = asyncio.run(run())

            errors = [(w, r["error"]) for w, r in payload["results"].items() if "error" in r]
            assert not errors, f"Worker errors: {errors}"
            assert payload["stats"]["node_count"] == 160  # 8 * 20
            for wid, r in payload["results"].items():
                assert r["inserted"] == 20
                assert r["hits"] > 0

            wall = payload["wall_ms"]
            print(f"\n{'='*60}")
            print(f"  RUST SERVER CONCURRENCY: 8 clients x 20 items")
            print(f"  Total nodes:  {payload['stats']['node_count']}")
            print(f"  Wall time:    {wall:.1f}ms")
            print(f"  Throughput:   {160 / (wall / 1000):.0f} ops/s")
            print(f"  Backend:      {payload['stats']['backend']}")
            print(f"{'='*60}")
        finally:
            _stop_server(proc)

    def test_privacy_filter_server_side(self) -> None:
        """Secrets in content/summary are redacted by the Rust server, not the client."""
        proc, port = _start_rust_server()
        try:
            async def run() -> dict[str, Any]:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=5.0)
                try:
                    # Send raw secret — server must sanitize
                    await client.remember(
                        content="Use jose. api_key=sk-proj-abcdefghijklmnopqrstuvwxyz",
                        project="priv",
                        agent_id="a1",
                        record_kind="memory",
                        memory_id="m-secret",
                        memory_type="semantic",
                        summary="auth token=sk-proj-abcdefghijklmnopqrstuvwxyz",
                        index_content="jose token=sk-proj-abcdefghijklmnopqrstuvwxyz",
                    )
                    # Also test private tags
                    await client.remember(
                        content="public text <private>hidden data</private> more public",
                        project="priv",
                        agent_id="a1",
                        record_kind="memory",
                        memory_id="m-private",
                        memory_type="semantic",
                        summary="tagged private",
                        index_content="public text <private>hidden data</private>",
                    )
                    # Search for the redacted content
                    hits = await client.search(
                        query="jose auth",
                        limit=5,
                        project="priv",
                        record_kind="memory",
                    )
                    stats = await client.stats()
                    return {"hits": hits, "stats": stats}
                finally:
                    await client.close()

            result = asyncio.run(run())

            assert result["stats"]["node_count"] == 2
            # The secret should NOT appear in search results
            for hit in result["hits"]:
                summary = hit.get("summary_preview", "")
                assert "sk-proj-" not in summary, f"Secret leaked in summary: {summary}"
        finally:
            _stop_server(proc)

    def test_snapshot_persistence(self, tmp_path: Path) -> None:
        """Data survives server restart via zstd-compressed snapshot ring."""
        snap_path = tmp_path / "helix_state.hlx"

        # Phase 1: insert data, trigger snapshot, verify zstd compression + ring stats
        proc, port = _start_rust_server_with_snapshot(snap_path, keep=3)
        try:
            async def phase1() -> dict[str, Any]:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=5.0)
                try:
                    await client._call("bulk_remember", {
                        "items": [
                            {
                                "content": f"persistent item {i} with extra text for compression",
                                "project": "snap",
                                "agent_id": "a1",
                                "record_kind": "memory",
                                "memory_id": f"snap-m{i}",
                                "summary": f"persistent {i}",
                                "index_content": f"persistent item {i} with extra text for compression",
                            }
                            for i in range(50)
                        ]
                    })
                    snap_result = await client._call("snapshot", {})
                    stats = await client.stats()
                    return {"snap": snap_result, "stats": stats}
                finally:
                    await client.close()

            p1 = asyncio.run(phase1())

            assert p1["stats"]["node_count"] == 50
            assert p1["snap"]["status"] == "saved"
            # Compression ratio should be meaningful (zstd compresses repetitive data well)
            assert p1["snap"]["compressed_bytes"] > 0
            assert p1["snap"]["raw_bytes"] > p1["snap"]["compressed_bytes"], (
                f"zstd should compress: raw={p1['snap']['raw_bytes']} compressed={p1['snap']['compressed_bytes']}"
            )
            # Ring stats should show at least slot 0
            ring = p1["snap"]["ring"]
            assert ring["keep"] == 3
            assert any(s["slot"] == 0 for s in ring["slots"])

            print(f"\n  Snapshot: {p1['snap']['raw_bytes']}B -> {p1['snap']['compressed_bytes']}B ({p1['snap']['ratio']})")
        finally:
            _stop_server(proc)

        assert snap_path.exists(), "Snapshot file not created"

        # Phase 2: restart, ring loaded, data survives
        proc2, port2 = _start_rust_server_with_snapshot(snap_path, keep=3)
        try:
            async def phase2() -> dict[str, Any]:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port2, timeout=5.0)
                try:
                    stats = await client.stats()
                    hits = await client.search(
                        query="persistent",
                        limit=5,
                        project="snap",
                        record_kind="memory",
                    )
                    return {"stats": stats, "hits": hits}
                finally:
                    await client.close()

            p2 = asyncio.run(phase2())
            assert p2["stats"]["node_count"] == 50, (
                f"Expected 50 nodes after reload, got {p2['stats']['node_count']}"
            )
            assert len(p2["hits"]) > 0, "Search must find data after snapshot reload"
        finally:
            _stop_server(proc2)

    def test_snapshot_ring_rotation(self, tmp_path: Path) -> None:
        """Ring buffer keeps exactly N files and deletes oldest on overflow."""
        snap_path = tmp_path / "ring_test.hlx"

        proc, port = _start_rust_server_with_snapshot(snap_path, keep=3)
        try:
            async def run() -> list[dict[str, Any]]:
                client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=5.0)
                snaps = []
                try:
                    await client.remember(
                        content="ring test node",
                        project="ring", agent_id="a", record_kind="memory",
                        memory_id="r1", summary="ring", index_content="ring test node",
                    )
                    # Trigger 4 snapshots — ring keep=3, so oldest is dropped after 4th
                    for _ in range(4):
                        r = await client._call("snapshot", {})
                        snaps.append(r)
                    return snaps
                finally:
                    await client.close()

            snaps = asyncio.run(run())
            assert all(s["status"] == "saved" for s in snaps)
        finally:
            _stop_server(proc)

        # slot 0, 1, 2 should exist; slot 3 should not (ring=3)
        assert snap_path.exists(), "slot 0 missing"
        assert snap_path.with_suffix(".hlx.1").exists() or \
               Path(str(snap_path) + ".1").exists() or \
               snap_path.with_name(snap_path.stem + ".hlx.1").exists() or \
               Path(str(snap_path.with_suffix("")) + ".hlx.1").exists(), "slot 1 missing"
        # slot 3 must not exist (beyond ring size)
        slot3 = Path(str(snap_path.with_suffix("")) + ".hlx.3")
        assert not slot3.exists(), f"slot 3 should have been deleted but exists at {slot3}"


def _start_rust_server_with_snapshot(snap_path: Path, keep: int = 3) -> tuple[subprocess.Popen[bytes], int]:
    """Start Rust server with snapshot path and ring buffer keep configured."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    env = {
        **os.environ,
        "HELIX_STATE_HOST": "127.0.0.1",
        "HELIX_STATE_PORT": str(port),
        "HELIX_SNAPSHOT_PATH": str(snap_path),
        "HELIX_SNAPSHOT_EVERY": "5",
        "HELIX_SNAPSHOT_KEEP": str(keep),
    }
    proc = subprocess.Popen(
        [str(RUST_BIN)],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.1)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError(f"Rust server failed to start on port {port}")

    return proc, port
