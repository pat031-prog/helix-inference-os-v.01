from __future__ import annotations

import json

import tools.run_cloud_provider_substitution_longitudinal_v1 as longitudinal
from tools.run_agent_run_transparency_gauntlet_v1 import verify_standalone_bundle


def test_longitudinal_suite_records_substitution_and_verifies_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "unit-test-token")
    counter = {"value": 0}

    def fake_deepinfra_chat_sync(**kwargs):
        counter["value"] += 1
        model = kwargs["model"]
        actual = f"{model}:served" if model == "anthropic/claude-sonnet-4-6" else model
        text = json.dumps(
            {
                "strongest_signal": "signed local evidence chain",
                "provider_identity_risk": "actual model can differ from requested",
                "output_drift_risk": f"digest can change across calls {counter['value']}",
                "missing_witness": "no external log",
                "next_test": "repeat and compare STHs",
            },
            sort_keys=True,
        )
        return {
            "status": "ok",
            "requested_model": model,
            "actual_model": actual,
            "provider_mismatch": actual != model,
            "text": text,
            "text_digest": f"sha256:{longitudinal._sha256_text(text)}",
            "tokens_used": 31,
            "latency_ms": 11,
            "retry_count": 0,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(longitudinal, "_deepinfra_chat_sync", fake_deepinfra_chat_sync)

    artifact = longitudinal.run_longitudinal_suite(
        run_id="unit-cloud-provider-longitudinal",
        output_dir=tmp_path,
        models=["Qwen/Qwen3-235B-A22B-Instruct-2507", "anthropic/claude-sonnet-4-6"],
        rounds=2,
        max_tokens=64,
    )

    assert artifact["status"] == "completed"
    assert artifact["score"] == 1.0
    assert artifact["findings"]["provider_substitution_detected"] is True
    assert artifact["findings"]["output_drift_detected"] is True
    assert artifact["gates"]["checkpoint_consistency_verified"] is True
    assert artifact["gates"]["standalone_verifier_bundle_passes"] is True
    assert len(artifact["cloud_calls"]) == 4
    assert verify_standalone_bundle(artifact["standalone_bundle"])["ok"] is True
