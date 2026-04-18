from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helix_kv.ipc_state_server import StateClient  # noqa: E402
from helix_proto.osint_oracle import (  # noqa: E402
    DEFAULT_GITHUB_REPOS,
    PROJECT,
    SignalIngestor,
    SignalNormalizer,
    SourceFetcher,
    build_alerts,
    build_oracle_artifact,
    fixture_feed_items,
    write_json,
)


RUST_BIN = REPO / "crates" / "helix-state-server" / "target" / "x86_64-pc-windows-gnullvm" / "release" / "helix-state-server.exe"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server() -> tuple[subprocess.Popen[bytes], int]:
    if not RUST_BIN.exists():
        raise FileNotFoundError(f"Rust state server binary not found: {RUST_BIN}")
    port = _free_port()
    env = {**os.environ, "HELIX_STATE_HOST": "127.0.0.1", "HELIX_STATE_PORT": str(port), "HELIX_STATE_OFFLOAD_BLOCKING": "1"}
    proc = subprocess.Popen([str(RUST_BIN)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return proc, port
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(f"State server failed to start on :{port}")


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def _collect_items(args: argparse.Namespace) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    if args.fixture_only:
        items = fixture_feed_items()
        return items, [], [{"source": "fixture", "items": len(items), "errors": 0}]
    fetcher = SourceFetcher()
    source_errors: list[str] = []
    source_rows: list[dict[str, Any]] = []
    all_items: list[Any] = []
    for repo in DEFAULT_GITHUB_REPOS:
        items, errors = await fetcher.fetch_github_atom(repo, max_items=args.max_items_per_source)
        all_items.extend(items)
        source_errors.extend(errors)
        source_rows.append({"source": f"github:{repo}", "items": len(items), "errors": len(errors)})
    hn_items, hn_errors = await fetcher.fetch_hackernews(max_items=args.max_items_per_source)
    all_items.extend(hn_items)
    source_errors.extend(hn_errors)
    source_rows.append({"source": "hackernews:top", "items": len(hn_items), "errors": len(hn_errors)})
    for url in args.rss_url:
        rss_items, rss_errors = await fetcher.fetch_rss(url, max_items=args.max_items_per_source)
        all_items.extend(rss_items)
        source_errors.extend(rss_errors)
        source_rows.append({"source": f"rss:{url}", "items": len(rss_items), "errors": len(rss_errors)})
    return all_items, source_errors, source_rows


async def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    mode = "fixture" if args.fixture_only else "live"
    proc, port = _start_server()
    try:
        raw_items, source_errors, source_rows = await _collect_items(args)
        signals = SignalNormalizer().normalize_many(raw_items)
        if args.live and args.lookback_hours > 0:
            cutoff_ms = int((time.time() - (args.lookback_hours * 60 * 60)) * 1000)
            signals = [signal for signal in signals if _signal_time_ms(signal) >= cutoff_ms]
        client = StateClient(transport="tcp", host="127.0.0.1", port=port, timeout=30.0)
        try:
            ingest = await SignalIngestor(client).ingest(signals)
            alerts, correlation_payload = await build_alerts(
                client,
                signals,
                min_independent_sources=args.min_independent_sources,
                correlation_window_hours=args.correlation_window_hours,
                llm_mode=args.llm_mode,
                model=args.model,
            )
            stats = await client.stats()
        finally:
            await client.close()
        artifact = build_oracle_artifact(
            mode=mode,
            profile=args.profile,
            source_errors=source_errors,
            signals=signals,
            ingest_receipt=ingest,
            alerts=alerts,
            correlation_payload=correlation_payload,
            duration_s=time.perf_counter() - started,
            llm_mode=args.llm_mode,
        )
        artifact["state_server_stats"] = {
            "node_count": stats.get("node_count"),
            "search_backend": stats.get("search_backend"),
            "semantic_query_router": stats.get("semantic_query_router"),
        }
        out_dir = Path(args.output_dir)
        write_json(out_dir / "local-zero-day-osint-oracle.json", artifact)
        write_json(
            out_dir / "local-zero-day-osint-oracle-sources.json",
            {
                "artifact": "local-zero-day-osint-oracle-sources",
                "mode": mode,
                "profile": args.profile,
                "source_rows": source_rows,
                "source_errors": source_errors,
                "project": PROJECT,
                "signals_seen": len(signals),
                "generated_ms": int(time.time() * 1000),
            },
        )
        print("=" * 64)
        print("  HELIX ZERO-DAY OSINT ORACLE")
        print("=" * 64)
        print(f"  Mode:             {mode}")
        print(f"  Signals seen:     {len(signals)}")
        print(f"  Signals stored:   {artifact['signals_ingested']}")
        print(f"  Alerts:           {artifact['alerts_count']}")
        print(f"  Search p50:       {artifact['search_ms_p50']}ms")
        print(f"  Source errors:    {len(source_errors)}")
        for alert in artifact["alerts"]:
            print(f"  ALERT {alert['alert_id']}: {alert['claim_level']} {alert['component']} score={alert['confidence_score']}")
            print(f"    {alert['synthesis'][:180]}")
        print("=" * 64)
        return artifact
    finally:
        _stop_server(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HeliX Zero-Day OSINT Oracle v0.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture-only", action="store_true", help="Use deterministic offline fixtures.")
    mode.add_argument("--live", action="store_true", help="Fetch public live sources.")
    parser.add_argument("--profile", default="public-light")
    parser.add_argument("--lookback-hours", type=float, default=72.0)
    parser.add_argument("--max-items-per-source", type=int, default=50)
    parser.add_argument("--min-independent-sources", type=int, default=2)
    parser.add_argument("--correlation-window-hours", type=float, default=24.0)
    parser.add_argument("--llm-mode", choices=["synthetic", "deepinfra"], default="synthetic")
    parser.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    parser.add_argument("--rss-url", action="append", default=[])
    parser.add_argument("--output-dir", default="verification")
    return parser.parse_args()


def _signal_time_ms(signal: Any) -> int:
    if getattr(signal, "published_at", ""):
        try:
            return int(datetime.fromisoformat(signal.published_at).astimezone(timezone.utc).timestamp() * 1000)
        except ValueError:
            pass
    return int(getattr(signal, "observed_at_ms", 0))


if __name__ == "__main__":
    args = parse_args()
    if not args.fixture_only and not args.live:
        args.fixture_only = True
    asyncio.run(run(args))
