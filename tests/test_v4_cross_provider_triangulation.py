from __future__ import annotations

import os
from datetime import datetime, timezone
from statistics import fmean
from typing import Any

from helix_proto.provider_audit import OPENAI_COMPATIBLE_PROVIDERS, behavioral_fingerprint, triangulation_power_analysis
from helix_proto.v4_gauntlet import base_artifact, preregister, summarize, write_artifact


RUN_STARTED_AT_UTC = os.environ.get("HELIX_RUN_STARTED_AT_UTC") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
RUN_DATE_UTC = os.environ.get("HELIX_RUN_DATE_UTC") or RUN_STARTED_AT_UTC[:10]
RUN_ID = os.environ.get("HELIX_RUN_ID", f"v4-cross-provider-triangulation-{RUN_DATE_UTC}")
TEST_ID = "cross-provider-behavioral-triangulation"
REQUESTED_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


FIXTURE_OUTPUTS = {
    "deepinfra": "Receipt: provider_returned_model=meta-llama/Llama-3.2-11B-Vision-Instruct; concise answer.",
    "together": "Receipt provider_returned_model=meta-llama/Llama-3.2-3B-Instruct. Concise answer.",
    "fireworks": "- provider_returned_model: meta-llama/Llama-3.2-3B-Instruct\n- answer: concise",
    "groq": "I cannot verify hidden model identity; provider_returned_model=llama-3.2-3b.",
    "openrouter": "{\"provider_returned_model\":\"meta-llama/Llama-3.2-3B-Instruct\",\"answer\":\"concise\"}",
}


def _fixture_rows() -> list[dict[str, Any]]:
    rows = []
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        text = FIXTURE_OUTPUTS[provider.name]
        fingerprint = behavioral_fingerprint(text, latency_ms=100 + len(text))
        rows.append({
            "provider": provider.name,
            "requested_model": REQUESTED_MODEL,
            "provider_returned_model": text.split("provider_returned_model=")[-1].split(";")[0].split()[0].strip('",') if "provider_returned_model=" in text else "meta-llama/Llama-3.2-3B-Instruct",
            "credential_available": provider.token_available,
            "behavioral_fingerprint": fingerprint,
        })
    return rows


def _separation_score(rows: list[dict[str, Any]]) -> float:
    token_counts = [row["behavioral_fingerprint"]["token_proxy_count"] for row in rows]
    refusals = [1 if row["behavioral_fingerprint"]["refusal"] else 0 for row in rows]
    return (max(token_counts) - min(token_counts)) + (10.0 * (max(refusals) - min(refusals)))


def test_cross_provider_triangulation_fixture_requires_power_analysis_before_public_claim() -> None:
    prereg = preregister(
        test_id=TEST_ID,
        question="Are outputs behaviorally indistinguishable across providers for the same requested model?",
        null_hypothesis="The same requested model across providers yields indistinguishable behavioral fingerprints under the preregistered MDE.",
        metrics=["behavioral_fingerprint", "refusal_rate_delta", "output_length_distribution", "style_drift", "power_analysis"],
        falseability_condition="Reject H0 only when observed drift exceeds the preregistered MDE and provider count >= 4.",
        kill_switch="If provider_returned_model or output_digest is absent, abort triangulation claim.",
        control_arms=["local_reference_model", "same_requested_model_cross_provider"],
    )
    rows = _fixture_rows()
    prompt_count = int(os.environ.get("HELIX_TRIANGULATION_PROMPT_COUNT", "30"))
    power = triangulation_power_analysis(prompt_count_per_provider=prompt_count)
    provider_count = len(rows)
    token_proxy_counts = [row["behavioral_fingerprint"]["token_proxy_count"] for row in rows]
    refusal_rate = fmean(1.0 if row["behavioral_fingerprint"]["refusal"] else 0.0 for row in rows)
    separation = _separation_score(rows)
    claim_ladder = "empirically_observed" if provider_count >= 4 and not power["underpowered_exploratory"] else "mechanics_verified"
    artifact = base_artifact(
        test_id=TEST_ID,
        run_id=RUN_ID,
        run_date_utc=RUN_DATE_UTC,
        preregistration=prereg,
        replica_count=prompt_count,
        random_baseline={"description": "random fingerprints are expected to separate trivially but have no model identity claim"},
        no_helix_baseline={"description": "without preserved digests and provider_returned_model fields, drift cannot be audited"},
        helix_arm={"provider_count": provider_count, "separation_score": separation, "power_analysis": power},
        public_claim_ladder=claim_ladder,
        claims_allowed=["Fixture validates behavioral fingerprint and power-analysis artifact shape."],
        claims_not_allowed=["Underpowered exploratory runs cannot prove cross-provider indistinguishability."],
        prompt_selection_risk="medium",
        extra={
            **summarize(token_proxy_counts),
            "primary_metric": "behavioral_fingerprint_separation",
            "requested_model": REQUESTED_MODEL,
            "providers": [provider.name for provider in OPENAI_COMPATIBLE_PROVIDERS],
            "rows": rows,
            "power_analysis": power,
            "power_analysis_id": f"{TEST_ID}:{prompt_count}",
            "underpowered_exploratory": power["underpowered_exploratory"],
            "behavioral_fingerprint_separation": separation,
            "refusal_rate": refusal_rate,
            "provider_token_availability": {provider.name: provider.token_available for provider in OPENAI_COMPATIBLE_PROVIDERS},
            "claim_boundary": "Behavioral fingerprints detect observable drift, not hidden ground-truth model identity unless independently verified.",
        },
    )
    path = write_artifact("local-cross-provider-triangulation.json", artifact)
    assert path.exists()
    assert provider_count >= 4
    assert artifact["underpowered_exploratory"] is True
    assert artifact["public_claim_ladder"] == "mechanics_verified"
    assert all(row["behavioral_fingerprint"]["output_digest"] for row in rows)
