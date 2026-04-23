from __future__ import annotations

import shutil
import uuid
import json
from argparse import Namespace
from pathlib import Path

import tools.run_multi_agent_concurrency_suite_v1 as suite


def _workspace() -> Path:
    workspace = (
        Path.cwd()
        / "verification"
        / "nuclear-methodology"
        / "_pytest"
        / "multi-agent-concurrency"
        / uuid.uuid4().hex
    )
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _args(workspace: Path, *, case: str = "all") -> Namespace:
    return Namespace(
        case=case,
        output_dir=str(workspace / "out"),
        run_id="pytest-multi-agent-concurrency",
        use_deepinfra=False,
        alpha_model="qwen/test",
        beta_model="sonnet/test",
        gamma_model="gemma/test",
        max_tokens=256,
    )


def test_concurrent_branch_quarantine_preserves_one_canonical_head() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace)
        artifact = suite._case_concurrent_branch_quarantine(  # noqa: SLF001 - case-level methodology test
            args,
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
        )

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["model_calls_overlap"] is True
        assert gates["lineage_equivocation_detected"] is True
        assert gates["trust_verified_with_quarantine"] is True
        assert gates["exactly_one_quarantined_branch"] is True
        assert gates["canonical_head_survives"] is True
        assert gates["default_context_hides_quarantine"] is True
        assert gates["forensic_context_preserves_quarantine"] is True
        assert artifact["result"]["lineage"]["checkpoint_verified"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_gamma_evidence_merge_requires_canonical_and_quarantined_hashes() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace)
        artifact = suite._case_gamma_evidence_merge(  # noqa: SLF001 - case-level methodology test
            args,
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
        )

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["gamma_returned_json"] is True
        assert gates["gamma_cites_canonical_head"] is True
        assert gates["gamma_cites_quarantined_hash"] is True
        assert gates["merge_memory_is_canonical"] is True
        assert gates["equivocation_still_preserved"] is True
        assert artifact["result"]["lineage"]["trust_status"] == "verified_with_quarantine"
        assert len(artifact["result"]["model_calls"]) == 3
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_gamma_provider_failure_is_recorded_as_failed_merge_attempt(monkeypatch) -> None:
    async def fake_agent_call(**kwargs):
        role = kwargs["role"]
        if role == "gamma":
            return {
                "role": "gamma",
                "status": "error",
                "requested_model": kwargs["model"],
                "actual_model": None,
                "text": "",
                "json": None,
                "tokens_used": 0,
                "latency_ms": 12.0,
                "finish_reason": None,
                "retry_count": 0,
                "error": "RuntimeError: model unavailable",
                "started_monotonic_ms": 30.0,
                "ended_monotonic_ms": 42.0,
            }
        result = await suite._deterministic_call(  # noqa: SLF001 - deterministic fixture helper
            role=role,
            model=kwargs["model"],
            user=kwargs["user"],
        )
        if role == "alpha":
            result.update({"started_monotonic_ms": 10.0, "ended_monotonic_ms": 30.0})
        else:
            result.update({"started_monotonic_ms": 12.0, "ended_monotonic_ms": 32.0})
        return result

    workspace = _workspace()
    try:
        monkeypatch.setattr(suite, "_agent_call", fake_agent_call)
        args = _args(workspace)
        artifact = suite._case_gamma_evidence_merge(  # noqa: SLF001 - case-level methodology test
            args,
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
        )

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "failed"
        assert gates["source_agent_calls_ok"] is True
        assert gates["gamma_call_ok"] is False
        assert gates["merge_memory_is_canonical"] is False
        assert artifact["result"]["gamma"]["error"] == "RuntimeError: model unavailable"
        assert artifact["result"]["merge_memory"]["summary"] == "Gamma merge attempt failed"
        assert "GAMMA_MERGE_FAILED" in artifact["result"]["merge_memory"]["content"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_naive_baseline_collapse_is_control_arm_not_helix_failure() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace)
        artifact = suite._case_naive_baseline_collapse(  # noqa: SLF001 - case-level methodology test
            args,
            run_id=args.run_id,
            output_dir=Path(args.output_dir),
        )

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["naive_loses_at_least_one_branch"] is True
        assert gates["naive_has_no_quarantine_record"] is True
        assert gates["naive_has_no_canonical_proof"] is True
        assert artifact["result"]["control"] == "naive-last-write-wins"
        assert artifact["result"]["lost_update_count"] >= 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_multi_agent_concurrency_suite_writes_artifacts_and_transcripts() -> None:
    workspace = _workspace()
    try:
        args = _args(workspace)
        artifact = suite.run_suite(args)

        assert artifact["status"] == "completed"
        assert artifact["case_count"] == 3
        assert artifact["mode"] == "deterministic"
        assert artifact["llm_synthetic_mode"] is True
        assert artifact["metrics"]["model_call_count"] == 5
        assert artifact["model_identity"]["substitution_count"] == 0
        assert Path(artifact["artifact_path"]).exists()
        assert Path(artifact["preregistered_path"]).exists()
        assert Path(artifact["transcript_exports"]["jsonl_path"]).exists()
        assert Path(artifact["transcript_exports"]["md_path"]).exists()
        for case in artifact["cases"]:
            assert case["status"] == "completed"
            assert Path(case["artifact_path"]).exists()
        first_case = artifact["cases"][0]
        first_payload = json.loads(Path(first_case["artifact_path"]).read_text(encoding="utf-8"))
        transcript_path = Path(first_payload["transcript_exports"]["jsonl_path"])
        events = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
        literal_events = [event for event in events if event.get("event") == "literal_model_output"]
        assert literal_events
        assert any("alpha-qwen-collector" in str(event.get("text") or "") for event in literal_events)
        assert all(event.get("text_sha256") for event in literal_events)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
