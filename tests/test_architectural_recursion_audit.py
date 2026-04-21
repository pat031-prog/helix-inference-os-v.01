from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from argparse import Namespace
from pathlib import Path

import tools.run_recursive_architectural_integrity_audit_v1 as audit


def _workspace() -> Path:
    workspace = Path.cwd() / "verification" / "nuclear-methodology" / "_pytest" / "architectural-recursion" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _fixture_args(workspace: Path) -> Namespace:
    bad = _write_json(
        workspace / "bad-temporal.json",
        {
            "artifact": "bad-temporal",
            "run_id": "long-horizon-checkpoints-20260420-130847",
            "case_id": "temporal-rollback-ambiguity",
            "status": "partial",
            "score": {"score": 0.7692, "gates": {"auditor_verdict_pass": False, "auditor_gate_failures_empty": False}},
            "artifact_payload_sha256": "badpayload130847",
            "judge_output": {"json": {"active_policy_memory_id": "mem-corrected", "final_policy": "active bounded rollback policy"}},
            "auditor_output": {"json": {"verdict": "fail", "gate_failures": ["no_visible_evidence"], "rationale": "No visible evidence."}},
        },
    )
    redeemed = _write_json(
        workspace / "redeemed-temporal.json",
        {
            "artifact": "redeemed-temporal",
            "run_id": "long-horizon-checkpoints-20260420-131930",
            "case_id": "temporal-rollback-ambiguity",
            "status": "completed",
            "score": {"score": 1.0, "gates": {"auditor_verdict_pass": True}},
            "artifact_payload_sha256": "redeemedpayload131930",
            "auditor_output": {"json": {"verdict": "pass", "gate_failures": []}},
        },
    )
    redeemed_suite = _write_json(
        workspace / "redeemed-suite.json",
        {
            "artifact": "redeemed-suite",
            "run_id": "long-horizon-checkpoints-20260420-131930",
            "status": "completed",
            "artifact_payload_sha256": "suitepayload131930",
            "cases": [
                {"case_id": "temporal-rollback-ambiguity", "score": 1.0},
                {"case_id": "summary-only-continuation", "score": 1.0},
            ],
        },
    )
    scale = _write_json(
        workspace / "scale.json",
        {
            "artifact": "scale",
            "run_id": "infinite-depth-memory-20260420-133040",
            "case_id": "scale-gradient-vs-naive-copy",
            "status": "completed",
            "score": {"score": 1.0},
            "artifact_payload_sha256": "scalepayload133040",
            "result": {"speedup_vs_naive_at_largest_depth": 9.6},
        },
    )
    speed_suite = _write_json(
        workspace / "speed-suite.json",
        {
            "artifact": "speed-suite",
            "run_id": "infinite-depth-memory-20260420-133040",
            "status": "completed",
            "depth": 5000,
            "case_count": 6,
            "artifact_payload_sha256": "speedpayload133040",
            "cases": [
                {"case_id": "scale-gradient-vs-naive-copy", "score": 1.0, "artifact_path": str(scale)},
                {"case_id": "bounded-context-under-depth", "score": 1.0, "artifact_path": str(scale)},
            ],
        },
    )
    baseline = _write_json(
        workspace / "baseline.json",
        {
            "artifact": "baseline",
            "run_id": "infinite-depth-memory-baseline-5000-validate-20260420",
            "status": "completed",
            "baseline_runs": 2,
            "artifact_payload_sha256": "baselinepayload5000",
            "metrics": {"scale_speedup_vs_naive": {"values": [9.6, 9.2]}},
            "suggested_thresholds": {"baseline_min_speedup": 6.9},
        },
    )
    return Namespace(
        bad_temporal_path=str(bad),
        redeemed_temporal_path=str(redeemed),
        redeemed_suite_path=str(redeemed_suite),
        speed_suite_path=str(speed_suite),
        speed_baseline_path=str(baseline),
        output_dir=str(workspace / "out"),
        run_id="pytest-recursive-architecture",
        proposer_model="qwen/test",
        red_team_model="claude/test",
        tokens=512,
        case="all",
    )


def _passing_proposal(evidence: dict[str, object]) -> dict[str, object]:
    artifacts = [item for item in evidence["artifacts"].values() if isinstance(item, dict)]
    hash_refs = [
        {
            "artifact": item["relative_path"],
            "file_sha256": item["file_sha256"],
            "artifact_payload_sha256": item["artifact_payload_sha256"],
        }
        for item in artifacts
        if item.get("file_sha256") and item.get("artifact_payload_sha256")
    ]
    return {
        "contradiction": (
            "Run 130847 failed with no_visible_evidence while run 131930 redeemed the same temporal rollback "
            "case at score 1.0; lazy loading hid evidence from the auditor while summary fidelity was correct."
        ),
        "uses_artifact_hashes": hash_refs[:2],
        "structural_modification": {
            "hard_anchors": "Hard anchors carry raw canonical non-summarizable values and never summarize security IDs.",
            "dynamic_cross_model_verification": "Cross-model random probe sampling runs during selective expansion.",
            "tombstone_metabolism": "Every tombstone injects a learned lesson into the next checkpoint.",
            "latency_preservation": "Use bounded context and no_full_history_replay to preserve sub-millisecond retrieval.",
            "fidelity_target": 1.0,
        },
    }


def _passing_red_team() -> dict[str, object]:
    return {
        "verdict": "conditional",
        "attack_vectors": ["hard-anchor poisoning", "semantic erosion through stale summaries"],
        "semantic_erosion_risk": "summary drift can cause semantic erosion if anchors are not verified",
        "latency_risk": "random probes must stay bounded",
        "accept_if": ["sample probes remain bounded", "anchors are signed"],
    }


def test_evidence_lineage_ingestion_scores_real_contradiction_shape() -> None:
    workspace = _workspace()
    try:
        args = _fixture_args(workspace)
        evidence = audit.build_evidence_package(args)
        score = audit.score_evidence_lineage(evidence)

        assert score["passed"] is True
        assert evidence["observations"]["auditor_false_negative"]["auditor_gate_failures"] == ["no_visible_evidence"]
        assert evidence["observations"]["speed_calibration"]["speedup_vs_naive_min"] >= 9.0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_meta_architecture_score_requires_hard_anchors_and_tombstone_metabolism() -> None:
    workspace = _workspace()
    try:
        args = _fixture_args(workspace)
        evidence = audit.build_evidence_package(args)
        score = audit.score_meta_architecture(evidence, _passing_proposal(evidence), _passing_red_team())

        assert score["passed"] is True

        weak = _passing_proposal(evidence)
        weak["structural_modification"] = {"latency_preservation": "bounded context only"}
        weak_score = audit.score_meta_architecture(evidence, weak, _passing_red_team())
        assert weak_score["passed"] is False
        assert weak_score["gates"]["hard_anchors_proposed"] is False
        assert weak_score["gates"]["tombstone_metabolism_proposed"] is False
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_meta_architecture_accepts_file_sha256_hash_citations() -> None:
    workspace = _workspace()
    try:
        args = _fixture_args(workspace)
        evidence = audit.build_evidence_package(args)
        proposal = _passing_proposal(evidence)
        file_hashes = [
            item["file_sha256"]
            for item in evidence["artifacts"].values()
            if isinstance(item, dict) and item.get("file_sha256")
        ]
        proposal["uses_artifact_hashes"] = [
            {"artifact": "bad_temporal", "file_sha256": file_hashes[0]},
            {"artifact": "redeemed_temporal", "file_sha256": file_hashes[1]},
        ]

        score = audit.score_meta_architecture(evidence, proposal, _passing_red_team())

        assert score["passed"] is True
        assert score["gates"]["artifact_hashes_cited"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_meta_architectural_recursion_uses_parallel_model_outputs(monkeypatch) -> None:
    workspace = _workspace()
    try:
        args = _fixture_args(workspace)
        output_dir = Path(args.output_dir)
        evidence = audit.build_evidence_package(args)
        calls: list[str] = []

        async def fake_chat(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "qwen/test":
                payload = _passing_proposal(evidence)
            else:
                payload = _passing_red_team()
            return {
                "requested_model": kwargs["model"],
                "actual_model": kwargs["model"],
                "status": "ok",
                "finish_reason": "stop",
                "tokens_used": 1,
                "latency_ms": 1.0,
                "text": json.dumps(payload),
                "json": payload,
            }

        monkeypatch.setattr(audit, "_deepinfra_chat", fake_chat)
        artifact = asyncio.run(audit._case_meta_architectural_recursion(args, token="token", run_id=args.run_id, output_dir=output_dir))

        assert artifact["status"] == "completed"
        assert set(calls) == {"qwen/test", "claude/test"}
        assert Path(artifact["transcript_exports"]["jsonl_path"]).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
