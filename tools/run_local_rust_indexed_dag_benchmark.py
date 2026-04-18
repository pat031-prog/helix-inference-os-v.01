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

from helix_kv.memory_catalog import MemoryCatalog, _tokenize

try:
    from _helix_merkle_dag import RustIndexedMerkleDAG
except Exception:  # noqa: BLE001
    RustIndexedMerkleDAG = None  # type: ignore[assignment]


TERMS = [
    "postgres",
    "migration",
    "scheduler",
    "merkle",
    "tombstone",
    "audit",
    "agent",
    "memory",
    "prefix",
    "restore",
    "qwen",
    "zamba",
]

QUERIES = [
    "postgres migration",
    "merkle audit",
    "agent memory",
    "prefix restore",
    "scheduler qwen",
    "zamba tombstone",
]

GENERIC_QUERIES = [
    "agent memory",
    "helix retrieval",
    "indexed dag",
    "synthetic memory",
]

SELECTIVE_QUERIES = [
    "rare_00042 postgres",
    "rare_00137 merkle",
    "rare_00256 prefix",
    "rare_00777 zamba",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p95_ms": None}
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))]
    return {"p50_ms": p50, "p95_ms": p95}


def _populate_catalog(catalog: MemoryCatalog, *, node_count: int) -> float:
    t0 = time.perf_counter()
    for i in range(node_count):
        a = TERMS[i % len(TERMS)]
        b = TERMS[(i * 7 + 3) % len(TERMS)]
        c = TERMS[(i * 11 + 5) % len(TERMS)]
        catalog.remember(
            project="bench",
            agent_id=f"agent-{i % 8}",
            session_id=f"session-{i % 32}",
            memory_type="semantic" if i % 3 else "episodic",
            summary=f"{a} {b} memory {i} rare_{i % 1000:05d}",
            content=(
                f"Node {i} records {a}, {b}, {c}, and rare_{i % 1000:05d} for HeliX retrieval. "
                f"This synthetic memory exists to benchmark indexed DAG search."
            ),
            importance=1 + (i % 10),
            decay_score=1.0 - ((i % 5) * 0.05),
        )
    return (time.perf_counter() - t0) * 1000.0


def _make_batch_records(*, node_count: int) -> list[dict[str, Any]]:
    records = []
    for i in range(node_count):
        a = TERMS[i % len(TERMS)]
        b = TERMS[(i * 7 + 3) % len(TERMS)]
        c = TERMS[(i * 11 + 5) % len(TERMS)]
        content = (
            f"Node {i} records {a}, {b}, {c}, and rare_{i % 1000:05d} for HeliX retrieval. "
            f"This synthetic memory exists to benchmark indexed DAG search."
        )
        records.append(
            {
                "content": content,
                "metadata": {
                    "project": "bench",
                    "agent_id": f"agent-{i % 8}",
                    "record_kind": "memory",
                    "memory_id": f"batch-mem-{i}",
                    "memory_type": "semantic" if i % 3 else "episodic",
                    "summary": f"{a} {b} memory {i} rare_{i % 1000:05d}",
                    "index_content": content,
                    "importance": 1 + (i % 10),
                    "decay_score": 1.0 - ((i % 5) * 0.05),
                },
            }
        )
    return records


def _bench_rust_direct_batch_insert(*, node_count: int) -> dict[str, Any]:
    if RustIndexedMerkleDAG is None:
        return {
            "available": False,
            "node_count": node_count,
            "insert_total_ms": None,
            "insert_ops_per_sec": None,
            "skip_reason": "rust_extension_not_importable",
        }
    idx = RustIndexedMerkleDAG()
    records_json = json.dumps(_make_batch_records(node_count=node_count), separators=(",", ":"))
    t0 = time.perf_counter()
    inserted = idx.insert_indexed_batch(records_json)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "available": True,
        "node_count": node_count,
        "inserted_count": len(inserted),
        "insert_total_ms": elapsed_ms,
        "insert_ops_per_sec": (node_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else None,
        "stats": idx.stats(),
    }


def _bench_python_scan(catalog: MemoryCatalog, *, query: str, limit: int) -> tuple[float, list[dict[str, Any]]]:
    tokens = _tokenize(query)
    t0 = time.perf_counter()
    hits = catalog._search_python_scan(  # noqa: SLF001 - benchmark intentionally compares fallback path.
        project="bench",
        agent_filter=None,
        tokens=tokens,
        limit=limit,
        type_filter=set(),
        exclude_ids=set(),
    )
    return (time.perf_counter() - t0) * 1000.0, hits


def _bench_catalog_search(catalog: MemoryCatalog, *, query: str, limit: int) -> tuple[float, list[dict[str, Any]]]:
    t0 = time.perf_counter()
    hits = catalog.search(project="bench", agent_id=None, query=query, limit=limit)
    return (time.perf_counter() - t0) * 1000.0, hits


def _bench_query_set(
    catalog: MemoryCatalog,
    *,
    query_set_name: str,
    queries: list[str],
    query_count: int,
    limit: int,
) -> dict[str, Any]:
    expanded = [queries[i % len(queries)] for i in range(query_count)]
    python_times: list[float] = []
    rust_times: list[float] = []
    same_top_hit = 0
    comparable_queries = 0
    for query in expanded:
        py_ms, py_hits = _bench_python_scan(catalog, query=query, limit=limit)
        rust_ms, rust_hits = _bench_catalog_search(catalog, query=query, limit=limit)
        python_times.append(py_ms)
        rust_times.append(rust_ms)
        if py_hits and rust_hits:
            comparable_queries += 1
            if py_hits[0].get("memory_id") == rust_hits[0].get("memory_id"):
                same_top_hit += 1
    python_p = _percentiles(python_times)
    rust_p = _percentiles(rust_times)
    return {
        "query_set": query_set_name,
        "query_count": query_count,
        "example_queries": queries,
        "python_scan": python_p,
        "catalog_search": rust_p,
        "speedup_p50": (
            python_p["p50_ms"] / rust_p["p50_ms"]
            if python_p["p50_ms"] and rust_p["p50_ms"]
            else None
        ),
        "same_top_hit_count": same_top_hit,
        "comparable_query_count": comparable_queries,
        "same_top_hit_rate": (same_top_hit / comparable_queries) if comparable_queries else None,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    node_counts = [int(item.strip()) for item in str(args.nodes).split(",") if item.strip()]
    query_count = int(args.queries)
    limit = int(args.limit)
    rows: list[dict[str, Any]] = []

    for node_count in node_counts:
        MemoryCatalog._REGISTRY.clear()  # noqa: SLF001 - isolate benchmark runs.
        catalog = MemoryCatalog.open(output_dir / f"_rust-indexed-dag-{node_count}.sqlite")
        insert_ms = _populate_catalog(catalog, node_count=node_count)
        stats = catalog.stats()
        query_sets = [
            _bench_query_set(catalog, query_set_name="mixed", queries=QUERIES, query_count=query_count, limit=limit),
            _bench_query_set(catalog, query_set_name="generic", queries=GENERIC_QUERIES, query_count=query_count, limit=limit),
            _bench_query_set(catalog, query_set_name="selective", queries=SELECTIVE_QUERIES, query_count=query_count, limit=limit),
        ]
        mixed = query_sets[0]
        rows.append(
            {
                "node_count": node_count,
                "query_count": query_count,
                "insert_total_ms": insert_ms,
                "insert_ops_per_sec": (node_count / (insert_ms / 1000.0)) if insert_ms > 0 else None,
                "rust_direct_batch_insert": _bench_rust_direct_batch_insert(node_count=node_count),
                "search_backend": stats.get("search_backend"),
                "rust_index_available": stats.get("rust_index_available"),
                "rust_index_error": stats.get("rust_index_error"),
                "rust_index_stats": stats.get("rust_index_stats"),
                "python_scan": mixed["python_scan"],
                "catalog_search": mixed["catalog_search"],
                "speedup_p50": mixed["speedup_p50"],
                "same_top_hit_count": mixed["same_top_hit_count"],
                "comparable_query_count": mixed["comparable_query_count"],
                "same_top_hit_rate": mixed["same_top_hit_rate"],
                "query_sets": query_sets,
                "projection": False,
            }
        )
        catalog.close()

    payload = {
        "title": "HeliX Rust Indexed MerkleDAG Benchmark",
        "benchmark_kind": "rust-indexed-merkle-dag-wand-bm25-v2",
        "status": "completed",
        "rows": rows,
        "claim_boundary": (
            "Measured rows compare Python scan against the active catalog search backend. "
            "Query sets are separated because generic high-document-frequency terms can dominate postings. "
            "The Rust backend now uses dynamic WAND upper bounds instead of a fixed 50k posting cap. "
            "No industrial-scale claim is made unless projection=false and the node_count was actually measured."
        ),
    }
    _write_json(output_dir / "local-rust-indexed-dag-benchmark.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Rust indexed MerkleDAG search against Python scan.")
    parser.add_argument("--nodes", default="10000,100000")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", default="verification")
    return parser


def main() -> None:
    payload = run_benchmark(build_parser().parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
