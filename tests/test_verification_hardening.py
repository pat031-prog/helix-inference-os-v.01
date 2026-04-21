from __future__ import annotations

import json
from pathlib import Path

from helix_proto.verification_hardening import (
    SYNTHETIC_TEMPLATE_CAVEAT,
    harden_payload,
    patch_artifact,
    suspicious_short_log,
    validate_artifact,
)
from helix_kv.memory_catalog import privacy_filter


def test_synthetic_avg_score_delta_is_renamed() -> None:
    payload = {
        "artifact": "local-cloud-amnesia-derby",
        "llm_synthetic_mode": True,
        "avg_score_delta": 7.0,
        "claim_boundary": "bounded",
    }
    patched, actions = harden_payload(payload)
    assert "renamed_synthetic_avg_score_delta" in actions
    assert "avg_score_delta" not in patched
    assert patched["avg_template_completion_delta"] == 7.0
    assert SYNTHETIC_TEMPLATE_CAVEAT in patched["claims_not_allowed"]


def test_active_memory_ab_is_marked_superseded() -> None:
    patched, actions = harden_payload({"artifact": "local-active-memory-ab-trial", "claim_boundary": "bounded"})
    assert "marked_active_memory_ab_superseded" in actions
    assert patched["superseded_by"] == "local-cloud-amnesia-derby"


def test_osint_overconfident_pre_advisory_is_relabelled() -> None:
    payload = {
        "artifact": "local-zero-day-osint-oracle",
        "claim_boundary": "bounded",
        "alerts": [{"claim_level": "pre_advisory_hypothesis", "confidence_score": 0.92, "caveats": []}],
    }
    patched, actions = harden_payload(payload)
    assert "relabelled_osint_advisory_candidate" in actions
    assert patched["alerts"][0]["claim_level"] == "advisory_candidate"


def test_provider_substitution_hardening_adds_no_deception_caveat() -> None:
    patched, actions = harden_payload({
        "artifact": "local-provider-substitution-ledger",
        "claim_boundary": "bounded",
    })
    assert "added_provider_substitution_framing_caveat" in actions
    assert any("Do not claim provider deception" in claim for claim in patched["claims_not_allowed"])


def test_linter_flags_synthetic_score_delta_and_secret() -> None:
    issues = validate_artifact(
        Path("synthetic.json"),
        {
            "artifact": "local-test",
            "llm_synthetic_mode": True,
            "avg_score_delta": 1.0,
            "claim_boundary": "bounded",
            "note": "Authorization: Bearer nope",
        },
    )
    codes = {issue.code for issue in issues}
    assert "synthetic_score_delta" in codes
    assert "secret_leak" in codes


def test_suspicious_short_log_guard() -> None:
    assert suspicious_short_log(passed=True, log_bytes=1448)
    assert not suspicious_short_log(passed=True, log_bytes=6000)
    assert not suspicious_short_log(passed=False, log_bytes=1448)


def test_hardener_reclassifies_short_log_when_artifacts_exist(tmp_path: Path) -> None:
    evidence = tmp_path / "local-hydrogen-table-drop-live.json"
    evidence.write_text(json.dumps({"artifact": "evidence", "payload": "x" * 6000}), encoding="utf-8")
    run = tmp_path / "local-hydrogen-table-drop-live-run.json"
    run.write_text(
        json.dumps(
            {
                "artifact": "local-hydrogen-table-drop-live-run",
                "exit_code": 2,
                "passed": False,
                "failure_reason": "suspicious_short_log",
                "suspicious_short_log": True,
                "log_bytes": 1400,
                "claim_boundary": "bounded",
            }
        ),
        encoding="utf-8",
    )
    result = patch_artifact(run, backup_dir=tmp_path / "backup", write=True)
    patched = json.loads(run.read_text(encoding="utf-8"))
    assert "reclassified_short_log_with_sufficient_artifacts" in result["actions"]
    assert patched["passed"] is True
    assert patched["raw_runner_exit_code"] == 2


def test_hardener_declares_stable_timestamped_artifact_aliases(tmp_path: Path) -> None:
    stable = tmp_path / "local-ghost-v2-task-scores.json"
    stamped = tmp_path / "local-ghost-v2-task-scores-20260418-160448.json"
    stable.write_text(json.dumps({"artifact": "task", "payload": "same"}), encoding="utf-8")
    stamped.write_text(stable.read_text(encoding="utf-8"), encoding="utf-8")
    run = tmp_path / "local-ghost-in-the-shell-live-v2-run.json"
    run.write_text(
        json.dumps(
            {
                "artifact": "local-ghost-in-the-shell-live-v2-run",
                "claim_boundary": "bounded",
                "timestamped_artifacts": [str(stamped)],
            }
        ),
        encoding="utf-8",
    )
    result = patch_artifact(run, backup_dir=tmp_path / "backup", write=True)
    patched = json.loads(run.read_text(encoding="utf-8"))
    assert "declared_stable_timestamped_artifact_aliases" in result["actions"]
    assert patched["artifact_aliases"][0]["same_content"] is True


def test_schema_smoke_exists() -> None:
    schema = json.loads(Path("schemas/helix-verification-v0.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["public_claim_ladder"]["enum"]


def test_privacy_filter_does_not_redact_memory_id_task_substrings() -> None:
    text = "memory_id=ghost-v2-task-code-review-recall cites wand-router-before-generic-scan"
    assert privacy_filter(text) == text
