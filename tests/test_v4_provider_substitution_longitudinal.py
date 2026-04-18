from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix_proto.v4_gauntlet import base_artifact, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-provider-substitution-{RUN_DATE_UTC}")
TEST_ID = "provider-substitution-longitudinal"
LONGITUDINAL_DIR = Path("verification") / "provider-substitution-longitudinal"


FIXTURE_DAYS = [
    {
        "date": "2026-04-16",
        "provider": "deepinfra",
        "rows": [
            ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct", "a1"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "b1"),
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-14B", "c1"),
        ],
    },
    {
        "date": "2026-04-17",
        "provider": "deepinfra",
        "rows": [
            ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct", "a2"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "b2"),
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-14B", "c2"),
        ],
    },
    {
        "date": "2026-04-18",
        "provider": "deepinfra",
        "rows": [
            ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct", "a3"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "b3"),
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-14B", "c3"),
        ],
    },
]


def _analyze(days: list[dict[str, Any]]) -> dict[str, Any]:
    mapping_counts: dict[str, Counter[str]] = defaultdict(Counter)
    digest_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for day in days:
        for requested, actual, digest in day["rows"]:
            key = f"{day['provider']}|{requested}"
            mapping_counts[key][actual] += 1
            digest_counts[f"{key}|{actual}"][digest] += 1
    matrix = []
    churn_values = []
    for key, counter in sorted(mapping_counts.items()):
        provider, requested = key.split("|", 1)
        total = sum(counter.values())
        top_actual, top_count = counter.most_common(1)[0]
        churn = 1.0 - (top_count / max(total, 1))
        churn_values.append(churn)
        matrix.append({
            "provider": provider,
            "requested_model": requested,
            "actual_model_counts": dict(counter),
            "dominant_actual_model": top_actual,
            "dominant_consistency_rate": top_count / max(total, 1),
            "substitution_churn_rate": churn,
        })
    digest_stability = []
    for key, counter in sorted(digest_counts.items()):
        total = sum(counter.values())
        digest_stability.append({
            "key": key,
            "distinct_output_digest_count": len(counter),
            "byte_identical_rate": max(counter.values()) / max(total, 1),
        })
    return {
        "day_count": len({day["date"] for day in days}),
        "requested_actual_matrix": matrix,
        "substitution_churn_summary": summarize(churn_values),
        "output_digest_stability": digest_stability,
        "longitudinal_ready": len({day["date"] for day in days}) >= 14,
    }


def test_provider_substitution_longitudinal_fixture_has_claim_ladder() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Are provider model substitutions stable or noisy over time?",
        null_hypothesis="For each provider/requested model, actual_model is consistent on >= 90% of days.",
        metrics=["requested_actual_matrix", "substitution_churn_rate", "output_digest_stability"],
        falseability_condition="If consistency < 90%, substitution is dynamic rather than fixed aliasing.",
        kill_switch="If requested/actual model fields are absent, abort the longitudinal claim.",
        control_arms=["temperature_0_same_prompt", "provider_served_actual_model"],
    )
    analysis = _analyze(FIXTURE_DAYS)
    ladder = "longitudinal" if analysis["longitudinal_ready"] else "empirically_observed"
    churn_summary = analysis["substitution_churn_summary"]
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=analysis["day_count"],
        random_baseline={"description": "random provider mapping has no stable requested->actual matrix"},
        no_helix_baseline={"description": "no audit ledger means requested/actual substitutions are not independently preserved"},
        helix_arm=analysis,
        public_claim_ladder=ladder,
        claims_allowed=[
            "Fixture validates the longitudinal artifact shape and churn metrics.",
            "A 14-day series is required before claiming longitudinal substitution rates.",
        ],
        claims_not_allowed=["This 3-day fixture is not a 14-day industry substitution rate."],
        prompt_selection_risk="low",
        extra={**churn_summary, "primary_metric": "substitution_churn_rate", **analysis},
    )
    LONGITUDINAL_DIR.mkdir(parents=True, exist_ok=True)
    path = write_artifact("provider-substitution-longitudinal-fixture.json", artifact, verification_dir=LONGITUDINAL_DIR)
    assert path.exists()
    assert artifact["public_claim_ladder"] == "empirically_observed"
    assert all(row["dominant_consistency_rate"] >= 0.9 for row in artifact["requested_actual_matrix"])
