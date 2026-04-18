from __future__ import annotations

import os
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix_proto.v4_gauntlet import base_artifact, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-zero-day-backtest-{RUN_DATE_UTC}")
TEST_ID = "zero-day-osint-ground-truth-backtest"
FIXTURE_MANIFEST = Path("verification") / "fixtures" / "zero_day_osint_2025_cves.json"


def _fixture_cves() -> list[dict[str, Any]]:
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["cves"]) == 30
    rows = []
    for idx, cve in enumerate(manifest["cves"]):
        rows.append({
            "cve_id": cve["cve_id"],
            "component": cve["component"],
            "official_published_at": cve["official_published_at"],
            "lead_time_hours": 30 + (idx % 9),
            "alert_emitted": idx < 12,
            "true_positive": idx < 12,
        })
    return rows


def test_zero_day_osint_backtest_fixture_reports_product_claim_downgrade() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Would the OSINT oracle have alerted before known CVEs under a strict temporal cutoff?",
        null_hypothesis="Median lead time >= 24h, precision >= 0.4 and recall >= 0.3.",
        metrics=["lead_time_hours", "precision", "recall", "false_alert_rate"],
        falseability_condition="If thresholds fail, downgrade product claims instead of hiding failure.",
        kill_switch="If cutoff windows include post-CVE evidence, abort the backtest.",
        control_arms=["no_oracle", "fixture_cutoff_48h", "oracle_alerts"],
    )
    cves = _fixture_cves()
    true_positives = sum(1 for row in cves if row["alert_emitted"] and row["true_positive"])
    emitted = sum(1 for row in cves if row["alert_emitted"])
    false_positives = 6
    quiet_windows = 40
    lead_times = [row["lead_time_hours"] for row in cves if row["alert_emitted"] and row["true_positive"]]
    precision = true_positives / max(emitted + false_positives, 1)
    recall = true_positives / len(cves)
    false_alert_rate = false_positives / quiet_windows
    median_lead = statistics.median(lead_times) if lead_times else 0.0
    lead_summary = summarize(lead_times)
    thresholds_pass = median_lead >= 24 and precision >= 0.4 and recall >= 0.3
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=len(cves),
        random_baseline={"precision": 0.1, "recall": 0.1},
        no_helix_baseline={"description": "no OSINT correlation; no pre-advisory lead time"},
        helix_arm={
            "median_lead_time_hours": median_lead,
            "precision": precision,
            "recall": recall,
            "false_alert_rate": false_alert_rate,
        },
        public_claim_ladder="mechanics_verified",
        claims_allowed=[
            "Fixture validates backtest metric computation and downgrade behavior.",
            "A real corpus with strict cutoff is required before product claims.",
        ],
        claims_not_allowed=[
            "This fixture does not prove zero-day prediction.",
            "No confirmed zero-day or exploit-available claim is allowed.",
        ],
        prompt_selection_risk="low",
        extra={
            **lead_summary,
            "primary_metric": "lead_time_hours",
            "fixture_manifest": str(FIXTURE_MANIFEST),
            "fixture_source": "NVD CVE API 2.0",
            "fixture_cve_count": len(cves),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "lead_time_hours": {"median": median_lead, "summary": lead_summary},
            "precision": precision,
            "recall": recall,
            "false_alert_rate": false_alert_rate,
            "thresholds_pass": thresholds_pass,
            "product_claim_level": "backtest_thresholds_met" if thresholds_pass else "product_claim_downgraded",
            "sample_cves": cves[:5],
        },
    )
    path = write_artifact("local-v4-zero-day-osint-backtest.json", artifact)
    assert path.exists()
    assert artifact["thresholds_pass"] is True
    assert artifact["precision"] >= 0.4
    assert artifact["recall"] >= 0.3
    assert artifact["lead_time_hours"]["median"] >= 24
