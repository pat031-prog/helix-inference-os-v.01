from __future__ import annotations

import json

import tools.run_agent_run_transparency_gauntlet_v1 as gauntlet
from tools.run_agent_run_transparency_gauntlet_v1 import (
    AgentRunTransparencyLog,
    build_agent_run_attestation,
    build_consistency_path,
    build_inclusion_proof,
    ct_leaf_hash,
    derive_ephemeral_keypair,
    detect_split_views,
    merkle_root_from_leaf_hashes,
    run_cloud_deepinfra_transparency,
    run_local_gauntlet,
    run_real_memory_transparency,
    verify_consistency_proof,
    verify_consistency_path,
    verify_inclusion_proof,
    verify_standalone_bundle,
)
from tools.verify_agent_run_bundle import main as verify_bundle_main


def _log(run_id: str = "unit-run") -> AgentRunTransparencyLog:
    keypair = derive_ephemeral_keypair(f"unit:{run_id}")
    log = AgentRunTransparencyLog(tree_id=f"tree:{run_id}", run_id=run_id, keypair=keypair)
    log.append("task_capsule", {"goal": "test"})
    log.append("model_call", {"requested_model": "m-a", "actual_model": "m-b", "provider_mismatch": True})
    log.append("patch", {"patch_digest": "sha256:patch"})
    return log


def test_inclusion_proof_accepts_original_event_and_rejects_tamper() -> None:
    log = _log()
    sth = log.signed_tree_head(label="final")
    event = log.events[2].to_leaf_payload()
    proof = log.inclusion_proof_for_event("evt-0003")

    ok = verify_inclusion_proof(event, proof, sth)
    tampered = {**event, "payload": {**event["payload"], "patch_digest": "sha256:evil"}}
    bad = verify_inclusion_proof(tampered, proof, sth)

    assert ok["ok"] is True
    assert bad["ok"] is False
    assert bad["reason"] == "leaf_hash_mismatch"


def test_consistency_proof_accepts_append_only_growth() -> None:
    log = _log()
    old_sth = log.signed_tree_head(size=2, label="old")
    final_sth = log.signed_tree_head(label="final")
    proof = log.consistency_proof(old_size=2)

    result = verify_consistency_proof(old_sth, final_sth, proof)

    assert result["ok"] is True
    assert result["old_root_hash"] == old_sth["root_hash"]
    assert result["new_root_hash"] == final_sth["root_hash"]
    assert proof["proof_profile"] == "rfc9162-consistency-proof-v0"
    assert "old_leaf_hashes" not in proof
    assert "append_leaf_hashes" not in proof


def test_rfc9162_consistency_path_handles_odd_non_power_sizes() -> None:
    leaves = [ct_leaf_hash({"i": index}) for index in range(7)]
    path = build_consistency_path(leaves, old_size=5, new_size=7)
    old_root = merkle_root_from_leaf_hashes(leaves[:5])
    new_root = merkle_root_from_leaf_hashes(leaves)

    result = verify_consistency_path(
        old_size=5,
        new_size=7,
        old_root_hash=old_root,
        new_root_hash=new_root,
        consistency_path=path,
    )
    tampered = verify_consistency_path(
        old_size=5,
        new_size=7,
        old_root_hash=old_root,
        new_root_hash=new_root,
        consistency_path=[("0" if item[0] != "0" else "1") + item[1:] for item in path],
    )

    assert result["ok"] is True
    assert tampered["ok"] is False


def test_reforged_history_fails_against_old_sth() -> None:
    old_log = _log("reforge")
    old_sth = old_log.signed_tree_head(size=2, label="old")

    forged = _log("reforge")
    forged.events[1] = forged.events[1].__class__(
        event_id=forged.events[1].event_id,
        event_type=forged.events[1].event_type,
        payload={"requested_model": "m-a", "actual_model": "m-c", "provider_mismatch": True},
        log_index=forged.events[1].log_index,
        appended_at_utc=forged.events[1].appended_at_utc,
    )
    forged_sth = forged.signed_tree_head(label="forged")
    forged_proof = forged.consistency_proof(old_size=2)

    result = verify_consistency_proof(old_sth, forged_sth, forged_proof)

    assert result["ok"] is False
    assert result["reason"] == "old_root_mismatch"


def test_split_view_detection_flags_same_size_different_root() -> None:
    a = _log("split")
    b = _log("split")
    b.events[1] = b.events[1].__class__(
        event_id=b.events[1].event_id,
        event_type=b.events[1].event_type,
        payload={"requested_model": "m-a", "actual_model": "m-z", "provider_mismatch": True},
        log_index=b.events[1].log_index,
        appended_at_utc=b.events[1].appended_at_utc,
    )

    conflicts = detect_split_views([a.signed_tree_head(label="a"), b.signed_tree_head(label="b")])

    assert len(conflicts) == 1
    assert conflicts[0]["tree_size"] == 3


def test_standalone_bundle_verifies_without_log_object() -> None:
    log = _log()
    old_sth = log.signed_tree_head(size=2, label="old")
    sth = log.signed_tree_head(label="final")
    event = log.events[2].to_leaf_payload()
    proof = log.inclusion_proof_for_event("evt-0003")
    consistency = log.consistency_proof(old_size=2)
    attestation = build_agent_run_attestation(
        run_id="unit-run",
        subject_event=event,
        inclusion_proof=proof,
        sth=sth,
        previous_sth=old_sth,
        consistency_proof=consistency,
        model_audit={"requested_model": "m-a", "actual_model": "m-b", "provider_mismatch": True},
        memory_summary={"admitted": ["m1"], "quarantined": []},
        checks={"unit_tests": True, "claim_boundary_present": True},
    )
    bundle = {
        "event": event,
        "inclusion_proof": proof,
        "previous_sth": old_sth,
        "consistency_proof": consistency,
        "sth": sth,
        "attestation": attestation,
    }

    result = verify_standalone_bundle(bundle)
    tampered = {
        **bundle,
        "event": {**event, "payload": {**event["payload"], "patch_digest": "sha256:evil"}},
    }
    proof_tampered = {
        **bundle,
        "consistency_proof": {
            **consistency,
            "consistency_path": [("0" if item[0] != "0" else "1") + item[1:] for item in consistency["consistency_path"]],
        },
    }

    assert result["ok"] is True
    assert result["bundle_sha256"]
    assert verify_standalone_bundle(tampered)["ok"] is False
    assert verify_standalone_bundle(proof_tampered)["ok"] is False


def test_standalone_bundle_verifier_cli_accepts_bundle_file(tmp_path) -> None:
    artifact = run_local_gauntlet(run_id="unit-agent-run-cli")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(artifact["standalone_bundle"]), encoding="utf-8")

    assert verify_bundle_main([str(bundle_path)]) == 0


def test_local_gauntlet_completes_all_gates() -> None:
    artifact = run_local_gauntlet(run_id="unit-agent-run-transparency")

    assert artifact["status"] == "completed"
    assert artifact["score"] == 1.0
    assert all(artifact["gates"].values())
    assert artifact["attestation"]["_type"] == "https://in-toto.io/Statement/v1"


def test_real_memory_transparency_run_completes_and_verifies_bundle(tmp_path) -> None:
    artifact = run_real_memory_transparency(run_id="unit-real-memory-transparency", output_dir=tmp_path)

    assert artifact["status"] == "completed"
    assert artifact["score"] == 1.0
    assert artifact["catalog"]["dag_coverage"]["status"] == "verified"
    assert artifact["gates"]["real_memory_hash_profile_v2"] is True
    assert verify_standalone_bundle(artifact["standalone_bundle"])["ok"] is True


def test_cloud_deepinfra_transparency_run_uses_model_metadata_without_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "unit-test-token")

    def fake_deepinfra_chat_sync(**kwargs):
        model = kwargs["model"]
        text = json.dumps(
            {
                "strongest_signal": f"{model} can inspect provider evidence.",
                "weakest_assumption": "No external witness is present.",
                "verification_gap": "Provider identity is observed, not globally proven.",
                "next_experiment": "Compare STH consistency across repeated cloud panels.",
            },
            sort_keys=True,
        )
        return {
            "status": "ok",
            "requested_model": model,
            "actual_model": f"{model}:served",
            "provider_mismatch": True,
            "text": text,
            "text_digest": f"sha256:{gauntlet._sha256_text(text)}",
            "tokens_used": 42,
            "latency_ms": 7,
            "retry_count": 0,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(gauntlet, "_deepinfra_chat_sync", fake_deepinfra_chat_sync)

    artifact = run_cloud_deepinfra_transparency(
        run_id="unit-cloud-deepinfra-transparency",
        output_dir=tmp_path,
        models=["owner/model-a", "owner/model-b"],
        max_tokens=32,
    )

    assert artifact["status"] == "completed"
    assert artifact["score"] == 1.0
    assert artifact["gates"]["cloud_calls_completed"] is True
    assert artifact["gates"]["provider_mismatch_auditable"] is True
    assert artifact["attestation"]["predicate"]["model"]["provider_mismatch"] is True
    assert len(artifact["attestation"]["predicate"]["model"]["calls"]) == 2
    assert verify_standalone_bundle(artifact["standalone_bundle"])["ok"] is True


def test_merkle_root_uses_domain_separated_leaf_hashes() -> None:
    leaves = [ct_leaf_hash({"i": index}) for index in range(3)]
    root = merkle_root_from_leaf_hashes(leaves)
    proof = build_inclusion_proof(leaves, 1)

    assert root == proof["root_hash"]
    assert leaves[1] == proof["leaf_hash"]
    assert root != merkle_root_from_leaf_hashes([leaf.replace("0", "1", 1) for leaf in leaves])
