from __future__ import annotations

from pathlib import Path

from tools.run_self_improvement_audit_gauntlet_v1 import (
    EvidenceTarget,
    build_evidence_pack,
    extract_evidence_snippet,
    score_self_improvement_audit,
)


def _evidence_pack() -> dict[str, object]:
    evidence = []
    for index in range(1, 9):
        layer = "python-memory" if index <= 4 else "rust-state-core"
        evidence.append(
            {
                "evidence_id": f"E{index:03d}",
                "layer": layer,
                "path": f"file-{index}.rs" if layer.startswith("rust") else f"file-{index}.py",
                "start_line": 1,
                "end_line": 8,
                "topic": f"topic {index}",
                "secret_hit_count": 0,
                "snippet": "code",
            }
        )
    return {
        "evidence_count": len(evidence),
        "layers": ["python-memory", "rust-state-core"],
        "secret_hit_count": 0,
        "evidence": evidence,
        "missing_targets": [],
    }


def _proposal(evidence_id: str = "E001") -> dict[str, object]:
    return {
        "title": "Strengthen verifier parity",
        "severity": "medium",
        "target_layer": "cross-layer",
        "evidence_ids": [evidence_id],
        "diagnosis": "Python and Rust verification paths need explicit parity tests.",
        "failure_mode": "A receipt can pass one verifier path and fail another.",
        "patch_plan": ["Add shared fixture", "Assert both paths agree"],
        "tests": ["python -m pytest tests/test_receipt_parity.py"],
        "acceptance_criteria": ["Both verifiers reject the same tampered receipt"],
        "expected_benefit": "Less verifier drift.",
        "risk_or_tradeoff": "More fixture maintenance.",
        "confidence": 0.7,
    }


def _reviewer_run(model: str, evidence_id: str = "E001") -> dict[str, object]:
    return {
        "model": model,
        "call": {"status": "ok", "finish_reason": "stop", "actual_model": model},
        "output": {
            "text": "{}",
            "json": {
                "reviewer_model": model,
                "audit_scope_understood": True,
                "improvement_proposals": [_proposal(evidence_id)],
                "claim_boundary_observed": True,
            },
        },
    }


def _analyst_json(evidence_id: str = "E001") -> dict[str, object]:
    return {
        "section_title": "Audited HeliX self-improvement backlog",
        "ranked_backlog": [
            {
                **_proposal(evidence_id),
                "priority": index,
                "merged_from_models": ["m1"],
                "why_now": "It reduces audit drift.",
                "implementation_plan": ["Add parity fixtures"],
                "estimated_patch_size": "small",
            }
            for index in range(1, 4)
        ],
        "cross_cutting_themes": ["verifier parity"],
        "method_caveats": ["Candidate backlog only"],
        "claim_boundary_observed": True,
    }


def test_extract_evidence_snippet_records_line_window_and_hash(tmp_path: Path) -> None:
    rel = Path("pkg") / "module.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "def before():",
                "    pass",
                "def target_function():",
                "    return 'audit me'",
                "def after():",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    target = EvidenceTarget(
        topic="target audit",
        layer="python-memory",
        rel_path=rel.as_posix(),
        anchor="def target_function",
        before=1,
        after=1,
    )

    item = extract_evidence_snippet(repo_root=tmp_path, target=target, evidence_index=1, max_chars=500)

    assert item is not None
    assert item["evidence_id"] == "E001"
    assert item["start_line"] == 2
    assert item["end_line"] == 4
    assert "def target_function" in item["snippet"]
    assert item["snippet_sha256"]


def test_build_evidence_pack_reports_missing_targets(tmp_path: Path) -> None:
    existing = tmp_path / "a.py"
    existing.write_text("def present():\n    return True\n", encoding="utf-8")
    pack = build_evidence_pack(
        repo_root=tmp_path,
        targets=[
            EvidenceTarget("present", "python-memory", "a.py", "def present"),
            EvidenceTarget("missing", "rust-state-core", "b.rs", "fn missing"),
        ],
    )

    assert pack["evidence_count"] == 1
    assert pack["missing_targets"] == [{"path": "b.rs", "anchor": "fn missing", "topic": "missing"}]


def test_score_self_improvement_audit_passes_evidence_cited_backlog() -> None:
    models = ["m1", "m2", "m3", "m4"]
    score = score_self_improvement_audit(
        models=models,
        evidence_pack=_evidence_pack(),
        reviewer_runs=[_reviewer_run(model) for model in models],
        analyst_json=_analyst_json(),
        auditor_json={"verdict": "pass", "gate_failures": []},
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is True
    assert score["score"] == 1.0


def test_score_self_improvement_audit_fails_invented_evidence_id() -> None:
    models = ["m1", "m2", "m3", "m4"]
    score = score_self_improvement_audit(
        models=models,
        evidence_pack=_evidence_pack(),
        reviewer_runs=[_reviewer_run(model, evidence_id="E999") for model in models],
        analyst_json=_analyst_json(evidence_id="E999"),
        auditor_json={"verdict": "pass", "gate_failures": []},
        analyst_finish_reason="stop",
        auditor_finish_reason="stop",
    )

    assert score["passed"] is False
    assert score["gates"]["reviewer_proposals_have_valid_evidence"] is False
    assert score["gates"]["analyst_items_have_valid_evidence"] is False
