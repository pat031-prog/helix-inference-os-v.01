from __future__ import annotations

import shutil
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

import tools.run_branch_pruning_forensics_suite_v1 as suite


def _workspace() -> Path:
    workspace = (
        Path.cwd()
        / "verification"
        / "nuclear-methodology"
        / "_pytest"
        / "branch-pruning-forensics"
        / uuid.uuid4().hex
    )
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _args(workspace: Path, *, depth: int = 96, case: str = "all") -> Namespace:
    return Namespace(
        case=case,
        output_dir=str(workspace / "out"),
        run_id="pytest-branch-pruning-forensics",
        depth=depth,
        branch_depth=4,
        bytes_per_node=512,
        repeats=1,
        max_anchor_ms=1000.0,
        use_deepinfra=False,
        solver_model="qwen/test",
        auditor_model="claude/test",
        tokens=512,
    )


def _requires_rust_extension() -> None:
    try:
        cls = suite._rust_indexed_dag_class()
        if not hasattr(cls(), "build_context_fast"):
            pytest.skip("Rust extension was not rebuilt with build_context_fast")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_descendant_closure_is_exact_poison_branch() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite._case_descendant_closure(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["closure_matches_poison_branch"] is True
        assert gates["closure_excludes_valid_branch"] is True
        assert gates["closure_excludes_shared_fork"] is True
        assert gates["poison_lineage_verified"] is True
        assert gates["valid_lineage_verified"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_pre_prompt_pruning_excludes_poison_branch() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite._case_pre_prompt_pruning(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["active_context_native_verified"] is True
        assert gates["poison_hashes_absent_from_active_context"] is True
        assert gates["poison_policy_absent_from_active_context"] is True
        assert gates["valid_terminal_remains_active"] is True
        assert gates["safe_policy_recovered_from_active_ledger"] is True
        assert gates["heavy_narrative_omitted"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_cold_audit_preserves_quarantined_branch() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite._case_cold_audit(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["cold_lookup_hits_all_poison_nodes"] is True
        assert gates["cold_identity_lane_native_verified"] is True
        assert gates["poison_branch_lineage_verified"] is True
        assert gates["cold_context_omits_poison_narrative_text"] is True
        assert gates["cold_audit_hashes_match_quarantine"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_real_hash_wrong_branch_attack_rejected() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite._case_wrong_branch_attack(args, run_id=args.run_id, output_dir=Path(args.output_dir))

        gates = artifact["score"]["gates"]
        assert artifact["status"] == "completed"
        assert gates["attack_hash_lookup_and_lineage_valid"] is True
        assert gates["attack_hash_belongs_to_quarantined_branch"] is True
        assert gates["attack_hash_absent_from_active_context"] is True
        assert gates["wrong_branch_attack_rejected"] is True
        assert gates["valid_branch_selected"] is True
        assert gates["poison_branch_rejected_by_policy"] is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_branch_pruning_suite_writes_transcripts() -> None:
    _requires_rust_extension()
    workspace = _workspace()
    try:
        args = _args(workspace, depth=96)
        artifact = suite.run_suite(args)

        assert artifact["status"] == "completed"
        assert artifact["case_count"] == 5
        assert Path(artifact["artifact_path"]).exists()
        assert Path(artifact["transcript_exports"]["jsonl_path"]).exists()
        assert Path(artifact["transcript_exports"]["md_path"]).exists()
        for case in artifact["cases"]:
            exports = case["transcript_exports"]
            assert Path(exports["jsonl_path"]).exists()
            assert Path(exports["md_path"]).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_model_score_accepts_policy_rejection_of_native_wrong_branch_hash() -> None:
    attack_hash = "a" * 64
    active_hash = "b" * 64
    expected = {
        "wrong_branch_attack_hash": attack_hash,
        "required_active_hashes": [active_hash],
    }
    result = {"wrong_branch_attack_rejected": True}
    solver = {
        "verdict": "pass",
        "selected_branch": "alpha_valid",
        "rejected_branch": "beta_signed_poison",
        "accepted_hashes": [active_hash],
        "rejected_hashes": [attack_hash],
        "wrong_branch_attack_rejected": True,
        "why": "The hash is a real native cryptographic hash, but policy quarantine makes it inactive.",
    }
    auditor = {
        "verdict": "pass",
        "gate_failures": [],
        "rationale": "Correctly rejects the native wrong-branch hash by quarantine policy.",
    }

    score = suite._score_model_outputs(
        expected=expected,
        result=result,
        solver_json=solver,
        auditor_json=auditor,
    )

    assert score["passed"] is True
