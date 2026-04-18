from __future__ import annotations

import gzip
import json
from pathlib import Path

from tools import run_local_hybrid_stress


def test_long_context_fixture_is_staged_local_sized() -> None:
    fixture = run_local_hybrid_stress.LONG_CONTEXT_FIXTURE
    assert fixture.exists()
    line_count = len(fixture.read_text(encoding="utf-8").splitlines())
    assert line_count >= 200


def test_receipt_summary_aggregates_step_windows(tmp_path: Path) -> None:
    receipts_path = tmp_path / "receipts.jsonl.gz"
    payloads = [
        {
            "run_id": "stress-a",
            "step_index": 0,
            "prev_hash": "0" * 64,
            "receipt_hash": "h1",
            "dense_bytes": 100,
            "compressed_bytes": 25,
            "clip_pct": 0.0,
            "rel_rmse": 0.1,
            "fallback_precision": "int4",
            "block_count": 4,
            "int4_block_count": 4,
            "int8_block_count": 0,
            "dense_block_count": 0,
            "promoted_block_count": 0,
            "max_abs_value": 1.5,
            "state_norm": 2.0,
        },
        {
            "run_id": "stress-a",
            "step_index": 1,
            "prev_hash": "h1",
            "receipt_hash": "h2",
            "dense_bytes": 100,
            "compressed_bytes": 50,
            "clip_pct": 4.0,
            "rel_rmse": 0.3,
            "fallback_precision": "int8",
            "block_count": 4,
            "int4_block_count": 0,
            "int8_block_count": 4,
            "dense_block_count": 0,
            "promoted_block_count": 4,
            "max_abs_value": 7.0,
            "state_norm": 9.0,
        },
    ]
    with gzip.open(receipts_path, "wt", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")

    summary = run_local_hybrid_stress._receipt_summary(receipts_path)

    assert summary["receipt_count"] == 2
    assert summary["hash_chain_ok"] is True
    assert len(summary["step_windows"]) == 2
    assert summary["step_windows"][1]["promoted_block_count"] == 4
    assert summary["promoted_block_total"] == 4


def test_restore_equivalence_compare_reports_token_and_logit_delta() -> None:
    pre = {
        "answer_token_ids": [4, 5],
        "logits_finite": True,
        "selection_logits": [[0.1, 0.9, 0.0], [0.2, 0.1, 0.8]],
    }
    post = {
        "answer_token_ids": [4, 5],
        "logits_finite": True,
        "selection_logits": [[0.1, 0.9, 0.0], [0.2, 0.1, 0.8]],
    }

    comparison = run_local_hybrid_stress._compare_restore_equivalence(pre, post)

    assert comparison["generated_ids_match"] is True
    assert comparison["top1_match_all"] is True
    assert comparison["max_abs_logit_delta"] == 0.0
    assert comparison["mean_abs_logit_delta"] == 0.0
    assert comparison["finite_before"] is True
    assert comparison["finite_after"] is True


def test_dashboard_payload_embeds_mission_objects(tmp_path: Path) -> None:
    mission_payload = {
        "mission_id": "context-switcher",
        "title": "Context Switcher",
        "strongest_claim": "kept logits finite",
        "headline_metrics": {"speedup_vs_native": 1.01},
    }
    mission_path = tmp_path / "mission.json"
    mission_path.write_text("{}", encoding="utf-8")

    dashboard = run_local_hybrid_stress._dashboard_payload(
        "laptop-12gb",
        run_local_hybrid_stress.PUBLIC_LOCAL_PROFILES["laptop-12gb"],
        [(mission_payload, mission_path)],
    )

    assert dashboard["profile"] == "laptop-12gb"
    assert dashboard["missions"][0]["artifact_path"] == str(mission_path)
    assert dashboard["strongest_claims"] == ["kept logits finite"]


def test_checked_in_stress_dashboard_includes_restore_equivalence() -> None:
    dashboard_path = Path("verification/local-zamba2-stress-dashboard.json")
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    mission_ids = {mission["mission_id"] for mission in dashboard["missions"]}

    assert {
        "long-context-coder",
        "state-juggler",
        "context-switcher",
        "restore-equivalence",
    }.issubset(mission_ids)
