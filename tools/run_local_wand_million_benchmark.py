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

try:
    from _helix_merkle_dag import RustIndexedMerkleDAG
except Exception as exc:  # noqa: BLE001
    RustIndexedMerkleDAG = None  # type: ignore[assignment]
    RUST_IMPORT_ERROR = str(exc)
else:
    RUST_IMPORT_ERROR = None


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

QUERY_SETS = {
    "selective": [
        "rare_00042 postgres",
        "rare_00137 merkle",
        "rare_00256 prefix",
        "rare_00777 zamba",
    ],
    "mixed": [
        "postgres migration",
        "merkle audit",
        "prefix restore",
        "scheduler qwen",
        "zamba tombstone",
    ],
    "generic": [
        "agent memory",
        "helix retrieval",
        "indexed dag",
        "synthetic memory",
    ],
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_batch(start: int, count: int) -> str:
    records: list[dict[str, Any]] = []
    for i in range(start, start + count):
        a = TERMS[i % len(TERMS)]
        b = TERMS[(i * 7 + 3) % len(TERMS)]
        c = TERMS[(i * 11 + 5) % len(TERMS)]
        rare = f"rare_{i % 1000:05d}"
        content = (
            f"Node {i} records {a}, {b}, {c}, and {rare} for HeliX WAND recovery. "
            "This synthetic memory exists to stress the indexed Merkle DAG without Python scan."
        )
        records.append(
            {
                "content": content,
                "metadata": {
                    "project": "wand-million",
                    "agent_id": f"agent-{i % 16}",
                    "record_kind": "memory",
                    "memory_id": f"wand-mem-{i}",
                    "memory_type": "semantic" if i % 3 else "episodic",
                    "summary": f"{a} {b} memory {i} {rare}",
                    "index_content": content,
                    "importance": 1 + (i % 10),
                    "decay_score": 1.0 - ((i % 5) * 0.05),
                },
            }
        )
    return json.dumps(records, separators=(",", ":"))


def _bench_queries(
    idx: Any,
    *,
    query_count: int,
    limit: int,
) -> list[dict[str, Any]]:
    filters = json.dumps({"project": "wand-million", "record_kind": "memory"})
    rows: list[dict[str, Any]] = []
    for name, queries in QUERY_SETS.items():
        times: list[float] = []
        hit_counts: list[int] = []
        top_hits: list[str | None] = []
        expanded = [queries[i % len(queries)] for i in range(query_count)]
        for query in expanded:
            t0 = time.perf_counter()
            hits = idx.search(query, limit, filters)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)
            hit_counts.append(len(hits))
            top_hits.append(hits[0].get("memory_id") if hits else None)
        rows.append(
            {
                "query_set": name,
                "query_count": query_count,
                "example_queries": queries,
                "latency": _percentiles(times),
                "hit_count_min": min(hit_counts) if hit_counts else 0,
                "hit_count_max": max(hit_counts) if hit_counts else 0,
                "top_hit_sample": top_hits[:8],
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if RustIndexedMerkleDAG is None:
        payload = {
            "artifact": "local-wand-million-benchmark",
            "status": "skipped",
            "skip_reason": "rust_extension_not_importable",
            "rust_import_error": RUST_IMPORT_ERROR,
        }
        _write_json(Path(args.output_dir) / "local-wand-million-benchmark.json", payload)
        return payload

    target_nodes = int(args.nodes)
    chunk_size = int(args.chunk_size)
    query_count = int(args.queries)
    limit = int(args.limit)
    pending_checkpoints = sorted({
        int(item.strip())
        for item in str(args.checkpoints).split(",")
        if item.strip()
    } | {target_nodes})

    idx = RustIndexedMerkleDAG()
    inserted = 0
    insert_chunks: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    while inserted < target_nodes:
        next_checkpoint = pending_checkpoints[0] if pending_checkpoints else target_nodes
        count = min(chunk_size, target_nodes - inserted, next_checkpoint - inserted)
        batch_json = _make_batch(inserted, count)
        t0 = time.perf_counter()
        nodes = idx.insert_indexed_batch(batch_json)
        chunk_ms = (time.perf_counter() - t0) * 1000.0
        inserted += len(nodes)
        insert_chunks.append(
            {
                "inserted_total": inserted,
                "chunk_count": len(nodes),
                "chunk_ms": chunk_ms,
                "chunk_ops_per_sec": (len(nodes) / (chunk_ms / 1000.0)) if chunk_ms > 0 else None,
            }
        )

        while pending_checkpoints and inserted >= pending_checkpoints[0]:
            checkpoint = pending_checkpoints.pop(0)
            search_rows = _bench_queries(idx, query_count=query_count, limit=limit)
            stats = idx.stats()
            checkpoint_rows.append(
                {
                    "node_count": checkpoint,
                    "actual_inserted": inserted,
                    "search": search_rows,
                    "rust_index_stats": stats,
                    "projection": False,
                }
            )

    total_ms = (time.perf_counter() - wall_start) * 1000.0
    payload = {
        "artifact": "local-wand-million-benchmark",
        "benchmark_kind": "rust-indexed-merkle-dag-wand-million-v1",
        "status": "completed",
        "target_nodes": target_nodes,
        "inserted_nodes": inserted,
        "chunk_size": chunk_size,
        "query_count_per_set": query_count,
        "limit": limit,
        "total_wall_ms": total_ms,
        "overall_insert_ops_per_sec": (inserted / (total_ms / 1000.0)) if total_ms > 0 else None,
        "insert_chunks_sample": insert_chunks[:3] + insert_chunks[-3:],
        "checkpoint_rows": checkpoint_rows,
        "claim_boundary": (
            "This benchmark intentionally avoids Python scan at 1M nodes. It measures direct Rust "
            "WAND/BM25 search over the indexed Merkle DAG. projection=false rows are measured."
        ),
        "generated_ms": int(time.time() * 1000),
    }
    _write_json(Path(args.output_dir) / "local-wand-million-benchmark.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stress Rust WAND/BM25 search at 1M DAG nodes.")
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--checkpoints", default="10000,100000,1000000")
    parser.add_argument("--output-dir", default="verification")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
