"""
HeliX IPC State Server — Proceso dedicado que posee el MerkleDAG canónico.

Arquitectura:
  - Un solo proceso Python posee el MerkleDAG + MemoryCatalog en RAM.
  - Workers de Gunicorn/K8s se conectan vía Unix Domain Socket (o TCP loopback).
  - Protocolo: líneas JSON delimitadas por newline (JSON-RPC simplificado).
  - asyncio mantiene el I/O; parseo/dispatch/encoding bloqueante puede offloadearse a threads.

Esto elimina V1 (singleton in-process) y V2 (threading.Lock inútil cross-process).

Producción: reemplazar por el crate Rust helix-merkle-ipc con tokio + UDS.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from helix_kv.memory_catalog import MemoryCatalog
from helix_kv.memory_gc import CognitiveGC


class StateServer:
    """Single-owner async server for the canonical MerkleDAG state."""

    def __init__(
        self,
        socket_path: str = "/tmp/helix_state.sock",
        gc_threshold: float = 2.0,
        *,
        transport: str = "auto",
        host: str = "127.0.0.1",
        port: int = 8765,
        offload_blocking: bool | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.transport = self._resolve_transport(transport)
        self.host = host
        self.port = int(port)
        if offload_blocking is None:
            offload_blocking = os.environ.get("HELIX_STATE_OFFLOAD_BLOCKING", "0").lower() in {"1", "true", "on", "yes"}
        self.offload_blocking = bool(offload_blocking)
        self.catalog = MemoryCatalog(":memory:")
        self.gc = CognitiveGC(self.catalog, threshold=gc_threshold)
        self._server: asyncio.Server | None = None

    @staticmethod
    def _resolve_transport(transport: str) -> str:
        value = str(transport or "auto").lower()
        if value == "auto":
            return "tcp" if os.name == "nt" else "uds"
        if value not in {"tcp", "uds"}:
            raise ValueError("transport must be auto, tcp, or uds")
        return value

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername") or "uds-client"
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    request = await self._maybe_to_thread(self._decode_request, line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    writer.write(await self._encode_response({"error": "invalid_json"}))
                    await writer.drain()
                    continue

                method = request.get("method", "")
                params = request.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                req_id = request.get("id")

                result = await self._maybe_to_thread(self._dispatch, method, params)
                writer.write(await self._encode_response({"id": req_id, "result": result}))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def _maybe_to_thread(self, func: Any, *args: Any) -> Any:
        if self.offload_blocking:
            return await asyncio.to_thread(func, *args)
        return func(*args)

    @staticmethod
    def _decode_request(line: bytes) -> dict[str, Any]:
        request = json.loads(line.decode("utf-8"))
        return request if isinstance(request, dict) else {}

    async def _encode_response(self, response: dict[str, Any]) -> bytes:
        payload = await self._maybe_to_thread(json.dumps, response)
        return (payload + "\n").encode("utf-8")

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        try:
            if method == "observe":
                return self.catalog.observe(**params)
            elif method == "remember":
                item = self.catalog.remember(**params)
                payload = item.to_dict()
                payload["node_hash"] = self.catalog.get_memory_node_hash(item.memory_id)
                payload["signed_receipt"] = self.catalog.get_memory_receipt(item.memory_id)
                return payload
            elif method == "bulk_remember":
                items = params.get("items", [])
                if not isinstance(items, list):
                    return {"error": "bulk_remember requires params.items list"}
                payloads = []
                for item in self.catalog.bulk_remember(items):
                    payload = item.to_dict()
                    payload["node_hash"] = self.catalog.get_memory_node_hash(item.memory_id)
                    payload["signed_receipt"] = self.catalog.get_memory_receipt(item.memory_id)
                    payloads.append(payload)
                return payloads
            elif method == "search":
                return self.catalog.search(**params)
            elif method == "get_memory":
                item = self.catalog.get_memory(params["memory_id"])
                return item.to_dict() if item else None
            elif method == "list_memories":
                return self.catalog.list_memories(**params)
            elif method == "audit_chain":
                chain = self.catalog.dag.audit_chain(params["leaf_hash"], params.get("max_depth", 10000))
                return [{"hash": n.hash, "parent_hash": n.parent_hash, "depth": n.depth, "content_len": len(n.content)} for n in chain]
            elif method == "verify_chain":
                return self.catalog.verify_chain(params["leaf_hash"], params.get("policy"))
            elif method == "gc_sweep":
                return self.gc.sweep()
            elif method == "stats":
                stats = self.catalog.stats()
                stats["state_server_offload_blocking"] = self.offload_blocking
                stats["state_server_transport"] = self.transport
                return stats
            elif method == "dag_snapshot":
                return self.catalog.dag.to_dict()
            else:
                return {"error": f"unknown_method: {method}"}
        except Exception as exc:
            return {"error": str(exc)}

    async def start(self) -> None:
        if self.transport == "uds":
            sock_path = Path(self.socket_path)
            sock_path.unlink(missing_ok=True)
            self._server = await asyncio.start_unix_server(
                self._handle_client, path=self.socket_path
            )
            endpoint = self.socket_path
        else:
            self._server = await asyncio.start_server(
                self._handle_client, host=self.host, port=self.port
            )
            sockets = self._server.sockets or []
            if sockets:
                bound = sockets[0].getsockname()
                self.host = str(bound[0])
                self.port = int(bound[1])
            endpoint = f"{self.host}:{self.port}"
        print(f"[helix-state] Listening on {self.transport}://{endpoint}", file=sys.stderr)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.stop()))
            except NotImplementedError:
                # Windows' default event loop does not support POSIX signal handlers.
                pass

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self.transport == "uds":
            Path(self.socket_path).unlink(missing_ok=True)


class StateClient:
    """Lightweight async client for Gunicorn workers / K8s pods."""

    def __init__(
        self,
        socket_path: str = "/tmp/helix_state.sock",
        *,
        transport: str = "auto",
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout: float = 10.0,
    ) -> None:
        self.socket_path = socket_path
        self.transport = StateServer._resolve_transport(transport)
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._req_id = 0

    @classmethod
    def from_env(cls) -> "StateClient":
        return cls(
            socket_path=os.environ.get("HELIX_STATE_SOCKET", "/tmp/helix_state.sock"),
            transport=os.environ.get("HELIX_STATE_TRANSPORT", "auto"),
            host=os.environ.get("HELIX_STATE_HOST", "127.0.0.1"),
            port=int(os.environ.get("HELIX_STATE_PORT", "8765")),
            timeout=float(os.environ.get("HELIX_STATE_TIMEOUT", "10")),
        )

    async def connect(self) -> None:
        if self.transport == "uds":
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=self.timeout,
            )
        else:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout,
            )

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self._writer or not self._reader:
            await self.connect()
        self._req_id += 1
        request = {"id": self._req_id, "method": method, "params": params or {}}
        self._writer.write((json.dumps(request) + "\n").encode("utf-8"))  # type: ignore[union-attr]
        await asyncio.wait_for(self._writer.drain(), timeout=self.timeout)  # type: ignore[union-attr]
        line = await asyncio.wait_for(self._reader.readline(), timeout=self.timeout)  # type: ignore[union-attr]
        if not line:
            raise RuntimeError("helix state server closed the connection")
        resp = json.loads(line.decode("utf-8"))
        if isinstance(resp.get("result"), dict) and "error" in resp["result"]:
            raise RuntimeError(resp["result"]["error"])
        return resp.get("result")

    async def observe(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("observe", kwargs)

    async def remember(self, **kwargs: Any) -> dict[str, Any]:
        return await self._call("remember", kwargs)

    async def bulk_remember(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self._call("bulk_remember", {"items": items})

    async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call("search", kwargs)

    async def gc_sweep(self) -> dict[str, Any]:
        return await self._call("gc_sweep")

    async def stats(self) -> dict[str, Any]:
        return await self._call("stats")

    async def audit_chain(self, leaf_hash: str, max_depth: int = 10000) -> list[dict[str, Any]]:
        return await self._call("audit_chain", {"leaf_hash": leaf_hash, "max_depth": max_depth})

    async def verify_chain(self, leaf_hash: str, policy: str | None = None) -> dict[str, Any]:
        return await self._call("verify_chain", {"leaf_hash": leaf_hash, "policy": policy})


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="HeliX State Server")
    parser.add_argument("--socket", default="/tmp/helix_state.sock")
    parser.add_argument("--transport", default="auto", choices=["auto", "tcp", "uds"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gc-threshold", type=float, default=2.0)
    parser.add_argument("--offload-blocking", action="store_true", default=None)
    args = parser.parse_args()
    server = StateServer(
        socket_path=args.socket,
        gc_threshold=args.gc_threshold,
        transport=args.transport,
        host=args.host,
        port=args.port,
        offload_blocking=args.offload_blocking,
    )
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
