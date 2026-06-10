from __future__ import annotations

import json

import tools.run_cloud_response_contract_stress_v1 as contract_stress
from tools.run_agent_run_transparency_gauntlet_v1 import verify_standalone_bundle


def test_contract_stress_records_malformed_outputs_without_failing_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "unit-test-token")

    def fake_deepinfra_chat_sync(**kwargs):
        model = kwargs["model"]
        user = kwargs["user"]
        if model == "Qwen/Qwen3-235B-A22B-Instruct-2507" and "strongest_signal" in user:
            text = "to\n\nbe\n\na\n\nmalformed\n\nresponse"
        elif "semantic_authority" in user:
            text = json.dumps(
                {
                    "verdict": "valid signature is not semantic authority",
                    "semantic_authority": False,
                    "invalid_claims": ["model truth", "global non-equivocation"],
                    "next_test": "add external witness",
                },
                sort_keys=True,
            )
        else:
            text = json.dumps(
                {
                    "strongest_signal": "signed local provenance",
                    "provider_identity_risk": "actual model may differ",
                    "output_drift_risk": "digests can change",
                    "missing_witness": "external witness absent",
                    "next_test": "repeat panel",
                },
                sort_keys=True,
            )
        return {
            "status": "ok",
            "requested_model": model,
            "actual_model": model,
            "provider_mismatch": False,
            "text": text,
            "text_digest": f"sha256:{contract_stress._sha256_text(text)}",
            "tokens_used": 17,
            "latency_ms": 5,
            "retry_count": 0,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(contract_stress, "_deepinfra_chat_sync", fake_deepinfra_chat_sync)

    artifact = contract_stress.run_contract_stress_suite(
        run_id="unit-contract-stress",
        output_dir=tmp_path,
        models=["Qwen/Qwen3-235B-A22B-Instruct-2507", "anthropic/claude-sonnet-4-6"],
        contracts=["minimal_json", "adversarial_boundary"],
        rounds=1,
        max_tokens=64,
    )

    assert artifact["status"] == "completed"
    assert artifact["score"] == 1.0
    assert artifact["findings"]["contract_violation_count"] == 1
    assert artifact["findings"]["malformed_json_count"] == 1
    assert artifact["gates"]["contract_violations_auditable"] is True
    assert verify_standalone_bundle(artifact["standalone_bundle"])["ok"] is True
