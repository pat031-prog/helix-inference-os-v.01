from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402


TERMS = [
    "postgres",
    "migration",
    "scheduler",
    "merkle",
    "tombstone",
    "audit",
    "prefix",
    "restore",
    "qwen",
    "zamba",
    "sigstore",
    "runbook",
]

GENERIC_QUERIES = [
    "agent memory",
    "helix retrieval",
    "indexed dag",
    "synthetic memory",
]

NARROWABLE_QUERIES = [
    "agent postgres memory",
    "helix merkle retrieval",
    "memory qwen session",
    "zamba agent context",
]

SELECTIVE_QUERIES = [
    "rare_00042 postgres",
    "rare_00137 merkle",
    "rare_00256 prefix",
    "rare_00777 zamba",
]


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }


def _make_items(start: int, count: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i in range(start, start + count):
        a = TERMS[i % len(TERMS)]
        b = TERMS[(i * 7 + 3) % len(TERMS)]
        rare = f"rare_{i % 1000:05d}"
        items.append(
            {
                "project": "router-guard",
                "agent_id": f"agent-{i % 8}",
                "memory_id": f"router-mem-{i}",
                "memory_type": "semantic" if i % 3 else "episodic",
                "summary": f"{a} {b} {rare}",
                "content": (
                    f"Agent memory node {i} records HeliX retrieval, indexed DAG, {a}, {b}, and {rare}. "
                    "This corpus stresses generic LLM queries against WAND/BM25."
                ),
                "importance": 1 + (i % 10),
                "tags": [a, rare],
            }
        )
    return items


def _bench(
    catalog: MemoryCatalog,
    *,
    queries: list[str],
    repeats: int,
    route_query: bool,
    limit: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    router_actions: dict[str, int] = {}
    anchor_samples: list[list[str]] = []
    top_hits: list[str | None] = []
    expanded = [queries[i % len(queries)] for i in range(repeats)]
    for query in expanded:
        t0 = time.perf_counter()
        hits = catalog.search(
            project="router-guard",
            agent_id=None,
            query=query,
            limit=limit,
            route_query=route_query,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        top_hits.append(hits[0].get("memory_id") if hits else None)
        if hits:
            route = hits[0].get("semantic_router") or {"action": "pass_through", "anchor_terms": []}
            action = str(route.get("action") or "unknown")
            router_actions[action] = router_actions.get(action, 0) + 1
            anchors = route.get("anchor_terms") or []
            if anchors and len(anchor_samples) < 8:
                anchor_samples.append([str(item) for item in anchors])
    return {
        "queries": queries,
        "repeats": repeats,
        "route_query": route_query,
        "latency": _percentiles(latencies),
        "router_actions": router_actions,
        "anchor_samples": anchor_samples,
        "top_hit_sample": top_hits[:8],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = MemoryCatalog.open(":semantic-router-guard:")
    inserted = 0
    t0 = time.perf_counter()
    while inserted < int(args.nodes):
        count = min(int(args.chunk_size), int(args.nodes) - inserted)
        catalog.bulk_remember(_make_items(inserted, count))
        inserted += count
    insert_ms = (time.perf_counter() - t0) * 1000.0

    raw_generic = _bench(catalog, queries=GENERIC_QUERIES, repeats=args.queries, route_query=False, limit=args.limit)
    routed_generic = _bench(catalog, queries=GENERIC_QUERIES, repeats=args.queries, route_query=True, limit=args.limit)
    raw_narrowable = _bench(catalog, queries=NARROWABLE_QUERIES, repeats=args.queries, route_query=False, limit=args.limit)
    routed_narrowable = _bench(catalog, queries=NARROWABLE_QUERIES, repeats=args.queries, route_query=True, limit=args.limit)
    selective = _bench(catalog, queries=SELECTIVE_QUERIES, repeats=args.queries, route_query=True, limit=args.limit)
    raw_p50 = raw_generic["latency"]["p50_ms"] or 0.0
    routed_p50 = routed_generic["latency"]["p50_ms"] or 0.0
    raw_narrowable_p50 = raw_narrowable["latency"]["p50_ms"] or 0.0
    routed_narrowable_p50 = routed_narrowable["latency"]["p50_ms"] or 0.0

    payload = {
        "artifact": "local-semantic-router-wand-guard",
        "status": "completed",
        "node_count": inserted,
        "insert_ms": insert_ms,
        "insert_ops_per_sec": inserted / (insert_ms / 1000.0) if insert_ms > 0 else None,
        "raw_generic": raw_generic,
        "routed_generic": routed_generic,
        "raw_narrowable": raw_narrowable,
        "routed_narrowable": routed_narrowable,
        "selective_control": selective,
        "generic_p50_speedup": raw_p50 / routed_p50 if routed_p50 > 0 else None,
        "narrowable_p50_speedup": raw_narrowable_p50 / routed_narrowable_p50 if routed_narrowable_p50 > 0 else None,
        "catalog_stats": catalog.stats(),
        "claim_boundary": (
            "This guard is a deterministic Python control-plane router. It rewrites broad LLM retrieval "
            "queries into high-information anchors already present in the scoped corpus, or falls back to "
            "recent important memories when no safe anchors exist. It does not claim semantic embedding search."
        ),
        "generated_ms": int(time.time() * 1000),
    }
    out = Path(args.output_dir) / "local-semantic-router-wand-guard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the semantic query router guard before Rust WAND search.")
    parser.add_argument("--nodes", type=int, default=25_000)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", default="verification")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
