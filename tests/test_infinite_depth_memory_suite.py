from __future__ import annotations

import json
import shutil
import uuid
from argparse import Namespace
from pathlib import Path

from tools.run_infinite_depth_memory_suite_v1 import (
    _case_claim_boundary_detector,
    _case_empty_retrieval_fast_path,
    run_baseline,
    run_suite,
)


def _write_legacy(path: Path, *, nodes: int) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark": "Infinite Asymmetric Loop (Context Depth Stress)",
                "memory_nodes": nodes,
                "insertion_time_ms": 12.34,
                "build_context_5000_depth_ms": 0.0,
                "rag_relevant_hits": 0,
                "context_tokens_packed": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _workspace() -> Path:
    workspace = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / "infinite-depth" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _args(workspace: Path, *, depth: int = 64, case: str = "all") -> Namespace:
    legacy = workspace / "legacy.json"
    _write_legacy(legacy, nodes=max(depth, 5000))
    return Namespace(
        case=case,
        output_dir=str(workspace / "out"),
        run_id="pytest-infinite-depth",
        legacy_telemetry=str(legacy),
        depth=depth,
        small_depth=16,
        mid_depth=32,
        repeats=1,
        budget_tokens=400,
        limit=3,
        max_empty_query_ms=1000.0,
        max_bounded_context_ms=1000.0,
        max_audit_chain_ms=1000.0,
        baseline_min_speedup=0.0,
        baseline_runs=1,
    )


def test_claim_boundary_detector_rejects_literal_infinite_and_zero() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace)
        artifact = _case_claim_boundary_detector(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        assert artifact["status"] == "completed"
        assert artifact["score"]["gates"]["rejects_literal_infinite"] is True
        assert artifact["score"]["gates"]["rejects_literal_zero_latency"] is True
        assert "literal physical zero latency" in artifact["result"]["rejected_claims"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_empty_retrieval_fast_path_returns_no_context() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace, depth=48)
        artifact = _case_empty_retrieval_fast_path(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        assert artifact["status"] == "completed"
        assert artifact["result"]["context_memory_ids"] == []
        assert artifact["result"]["context_tokens"] == 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_infinite_depth_suite_small_run_writes_transcripts() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace, depth=64)
        artifact = run_suite(args)

        assert artifact["status"] == "completed"
        assert artifact["case_count"] == 6
        assert Path(artifact["artifact_path"]).exists()
        assert Path(artifact["transcript_exports"]["jsonl_path"]).exists()
        assert Path(artifact["transcript_exports"]["md_path"]).exists()
        for case in artifact["cases"]:
            exports = case["transcript_exports"]
            assert Path(exports["jsonl_path"]).exists()
            assert Path(exports["md_path"]).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_infinite_depth_baseline_calibration_writes_metric_summary() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace, depth=32)
        args.baseline_runs = 2
        artifact = run_baseline(args)

        assert artifact["status"] == "completed"
        assert artifact["baseline_runs"] == 2
        assert artifact["metrics"]["bounded_context_median_ms"]["p95"] >= 0.0
        assert artifact["suggested_thresholds"]["max_bounded_context_ms"] >= artifact["metrics"]["bounded_context_median_ms"]["p95"]
        assert Path(artifact["artifact_path"]).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
