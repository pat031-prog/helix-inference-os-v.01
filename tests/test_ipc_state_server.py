from __future__ import annotations

import asyncio
from typing import Any

from helix_kv.ipc_state_server import StateClient, StateServer


async def _run_tcp_bulk_remember_smoke() -> dict[str, Any]:
    server = StateServer(transport="tcp", host="127.0.0.1", port=0, offload_blocking=True)
    server_task = asyncio.create_task(server.start())
    try:
        for _ in range(100):
            if server._server is not None:  # noqa: SLF001 - test waits for bound ephemeral port.
                break
            await asyncio.sleep(0.01)
        assert server._server is not None  # noqa: SLF001

        client = StateClient(transport="tcp", host=server.host, port=server.port, timeout=5.0)
        try:
            inserted = await client.bulk_remember(
                [
                    {
                        "project": "ipc",
                        "agent_id": "agent-a",
                        "memory_type": "semantic",
                        "summary": "alpha state server batch",
                        "content": "alpha state server batch memory",
                        "importance": 7,
                    },
                    {
                        "project": "ipc",
                        "agent_id": "agent-a",
                        "memory_type": "semantic",
                        "summary": "beta state server batch",
                        "content": "beta state server batch memory",
                        "importance": 9,
                    },
                ]
            )
            hits = await client.search(project="ipc", agent_id="agent-a", query="beta", limit=3)
            stats = await client.stats()
        finally:
            await client.close()

        return {"inserted": inserted, "hits": hits, "stats": stats}
    finally:
        await server.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def test_state_server_bulk_remember_tcp_smoke() -> None:
    payload = asyncio.run(_run_tcp_bulk_remember_smoke())

    assert len(payload["inserted"]) == 2
    assert payload["inserted"][1]["summary"] == "beta state server batch"
    assert payload["hits"][0]["memory_id"] == payload["inserted"][1]["memory_id"]
    assert payload["stats"]["memory_count"] == 2
    assert payload["stats"]["state_server_offload_blocking"] is True
