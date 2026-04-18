from __future__ import annotations

import pytest

from helix_proto.api import HelixRuntime
from helix_proto.research_artifacts import load_research_artifact, research_artifact_manifest


def test_research_artifact_manifest_includes_frontier_and_canonical_entries() -> None:
    manifest = research_artifact_manifest()
    names = {item["name"] for item in manifest}

    assert "hybrid-memory-frontier-summary.json" in names
    assert "remote-transformers-gpu-summary.json" in names

    frontier = next(item for item in manifest if item["name"] == "hybrid-memory-frontier-summary.json")
    assert frontier["category"] == "canonical evidence"
    assert "frontier" in frontier["tags"]
    assert frontier["headline_metrics"]["combined_runtime_cache_ratio_vs_native"] is not None


def test_load_research_artifact_rejects_path_traversal() -> None:
    with pytest.raises(FileNotFoundError):
        load_research_artifact("../../README.md")


def test_runtime_research_artifact_wraps_payload() -> None:
    runtime = HelixRuntime(root=None)

    artifact = runtime.research_artifact("hybrid-memory-frontier-summary.json")

    assert artifact["name"] == "hybrid-memory-frontier-summary.json"
    assert artifact["title"] == "Hybrid memory frontier summary"
    assert artifact["payload"]["benchmark_kind"] == "hybrid-memory-frontier-summary-v1"


def test_gemma_attempts_manifest_uses_models_shape() -> None:
    manifest = research_artifact_manifest()

    gemma = next(item for item in manifest if item["name"] == "local-gemma-attempts.json")

    assert "google/gemma-4-E4B-it" in gemma["model_refs"]
    assert gemma["headline_metrics"]["attempt_count"] >= 1


def test_claims_matrix_manifest_has_caveated_public_claims() -> None:
    manifest = research_artifact_manifest()
    names = {item["name"] for item in manifest}

    assert "helix-claims-matrix.json" in names

    claims_meta = next(item for item in manifest if item["name"] == "helix-claims-matrix.json")
    assert claims_meta["category"] == "canonical evidence"
    assert "claims" in claims_meta["tags"]
    assert claims_meta["headline_metrics"]["verified_claim_count"] >= 1

    payload = load_research_artifact("helix-claims-matrix.json")
    for claim in payload["claims"]:
        assert claim["status"] in {"verified", "promising", "blocked"}
        assert claim["public_wording"]
        assert claim["evidence_artifacts"]
        assert claim["caveat"]


def test_agent_memory_evidence_artifacts_are_manifested() -> None:
    manifest = research_artifact_manifest()
    names = {item["name"] for item in manifest}

    assert "local-ttft-cold-warm-summary.json" in names
    assert "local-agent-capacity-budget.json" in names
    assert "agent-memory-comparison-summary.json" in names

    comparison = next(item for item in manifest if item["name"] == "agent-memory-comparison-summary.json")
    assert comparison["category"] == "canonical evidence"
    assert "agent-memory" in comparison["tags"]


def test_session_os_reliability_artifacts_are_manifested() -> None:
    manifest = research_artifact_manifest()
    by_name = {item["name"]: item for item in manifest}

    for name in [
        "local-memory-catalog-concurrency.json",
        "local-memory-decay-selection.json",
        "local-hlx-layer-chaos.json",
        "local-rust-python-layer-slice-soak.json",
        "local-airllm-real-smoke.json",
    ]:
        assert name in by_name

    concurrency = by_name["local-memory-catalog-concurrency.json"]
    assert concurrency["category"] == "canonical evidence"
    assert "reliability" in concurrency["tags"]
    assert concurrency["headline_metrics"]["lost_observations"] == 0

    airllm = by_name["local-airllm-real-smoke.json"]
    assert "airllm" in airllm["tags"]
    assert airllm["headline_metrics"]["real_airllm_injection_supported"] is False


def test_inference_os_blueprint_artifacts_are_manifested() -> None:
    manifest = research_artifact_manifest()
    by_name = {item["name"]: item for item in manifest}

    for name in [
        "local-inference-os-architecture-summary.json",
        "local-blueprint-stack-catalog.json",
        "local-blueprint-meta-microsite-demo.json",
        "local-blueprint-meta-microsite-real-cached.json",
        "local-blueprint-frontend-factory-smoke.json",
        "local-blueprint-framework-showcase.json",
    ]:
        assert name in by_name

    meta = by_name["local-blueprint-meta-microsite-demo.json"]
    assert meta["category"] == "canonical evidence"
    assert "blueprint" in meta["tags"]
    assert "inference-os" in meta["tags"]
    assert meta["headline_metrics"]["quality_status"] in {"passed", None}
