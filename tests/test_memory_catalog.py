from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix_kv.memory_catalog import MemoryCatalog, privacy_filter

try:
    from _helix_merkle_dag import RustIndexedMerkleDAG  # noqa: F401
except Exception:  # noqa: BLE001
    try:
        from helix_kv._helix_merkle_dag import RustIndexedMerkleDAG  # noqa: F401
    except Exception:  # noqa: BLE001
        RustIndexedMerkleDAG = None  # type: ignore[assignment]


def test_memory_catalog_observe_remember_search_and_context(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "memory.sqlite")
    observation = catalog.observe(
        project="helix",
        agent_id="agent-a",
        session_id="s1",
        content="Use jose middleware. api_key=sk-proj-abcdefghijklmnopqrstuvwxyz",
        summary="Auth decision",
        tags=["auth"],
    )
    memory = catalog.remember(
        project="helix",
        agent_id="agent-a",
        session_id="s1",
        memory_type="semantic",
        summary="JWT auth uses jose",
        content="Token validation is covered by tests.",
        importance=8,
        tags=["auth", "jwt"],
    )
    link = catalog.link_session_memory(session_id="s1", memory_id=memory.memory_id)
    # This suite exercises BM25/privacy, not signatures. Opt out of the strict default.
    hits = catalog.search(project="helix", agent_id="agent-a", query="jose token", limit=3, signature_enforcement="permissive")
    context = catalog.build_context(project="helix", agent_id="agent-a", query="auth", budget_tokens=80, mode="search", signature_enforcement="permissive")

    assert "[REDACTED_SECRET]" in observation["content"]
    assert link["memory_id"] == memory.memory_id
    assert hits and hits[0]["memory_id"] == memory.memory_id
    assert "<helix-memory-context>" in context["context"]
    assert memory.memory_id in context["memory_ids"]
    assert catalog.stats()["memory_count"] == 1
    catalog.close()


def test_memory_catalog_privacy_and_invalid_inputs(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "memory.sqlite")

    assert privacy_filter("<private>hide me</private> Bearer abcdefghijklmnopqrstuvwxyz") == "[REDACTED] [REDACTED_SECRET]"
    assert "[REDACTED_SECRET]" in privacy_filter("token=sk-proj-abcdefghijklmnopqrstuvwxyz")
    assert "[REDACTED_SECRET]" in privacy_filter("github_pat_abcdefghijklmnopqrstuvwxyz1234567890")
    with pytest.raises(ValueError):
        catalog.remember(project="../bad", agent_id="agent-a", content="x")
    with pytest.raises(ValueError):
        catalog.remember(project="helix", agent_id="agent-a", content="x", memory_type="unknown")
    with pytest.raises(KeyError):
        catalog.link_session_memory(session_id="s1", memory_id="missing")
    catalog.close()


def test_memory_catalog_summary_mode_and_budget(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "memory.sqlite")
    first = catalog.remember(
        project="helix",
        agent_id="agent-a",
        memory_type="episodic",
        summary="Short useful fact",
        content="A compact detail about release claims.",
        importance=7,
    )
    catalog.remember(
        project="helix",
        agent_id="agent-a",
        memory_type="episodic",
        summary="Long fact",
        content="x" * 2000,
        importance=7,
    )

    context = catalog.build_context(project="helix", agent_id="agent-a", budget_tokens=40, mode="summary")

    assert context["memory_ids"] == [first.memory_id]
    assert context["tokens"] <= 40
    catalog.close()


def test_memory_catalog_quarantines_equivocating_branch_and_filters_it_from_default_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")
    catalog = MemoryCatalog.open(tmp_path / "lineage.sqlite")
    try:
        root = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-a",
            memory_type="semantic",
            summary="Canonical root",
            content="Canonical lineage root memory.",
            importance=8,
        )
        root_hash = catalog.get_memory_node_hash(root.memory_id)
        canonical = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-a",
            memory_type="semantic",
            summary="Canonical continuation",
            content="Canonical continuation of the thread.",
            importance=9,
        )
        canonical_hash = catalog.get_memory_node_hash(canonical.memory_id)
        assert root_hash and canonical_hash

        catalog._session_heads["thread-a"] = root_hash  # noqa: SLF001 - simulate stale writer/equivocating branch
        quarantined = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-a",
            memory_type="semantic",
            memory_id="mem-quarantined",
            summary="Competing branch",
            content="This branch diverged from the stale head and must be quarantined.",
            importance=7,
        )
        quarantined_hash = catalog.get_memory_node_hash(quarantined.memory_id)

        lineage = catalog.verify_session_lineage("thread-a", include_quarantined=True)
        checkpoint = catalog.head_checkpoint("thread-a")
        visible = catalog.list_memories(project="helix", agent_id="agent-a", session_id="thread-a", include_quarantined=False)
        forensic = catalog.list_memories(project="helix", agent_id="agent-a", session_id="thread-a", include_quarantined=True)
        default_context = catalog.build_context(
            project="helix",
            agent_id="agent-a",
            session_id="thread-a",
            mode="summary",
            budget_tokens=120,
        )
        forensic_context = catalog.build_context(
            project="helix",
            agent_id="agent-a",
            session_id="thread-a",
            mode="summary",
            budget_tokens=200,
            include_quarantined=True,
        )

        assert catalog.verify_chain(quarantined_hash)["status"] == "verified"
        assert lineage["status"] == "equivocation_detected"
        assert lineage["trust_status"] == "verified_with_quarantine"
        assert lineage["checkpoint_verified"] is True
        assert lineage["canonical_head"] == canonical_hash
        assert lineage["equivocation_count"] == 1
        assert lineage["quarantined_count"] == 1
        assert checkpoint["checkpoint_hash"] == lineage["latest_checkpoint_hash"]
        assert visible and all(item["memory_id"] != quarantined.memory_id for item in visible)
        rogue_row = next(item for item in forensic if item["memory_id"] == quarantined.memory_id)
        assert rogue_row["canonical"] is False
        assert rogue_row["quarantined"] is True
        assert rogue_row["equivocation_id"]
        assert quarantined.memory_id not in default_context["memory_ids"]
        assert quarantined.memory_id in forensic_context["memory_ids"]
        assert default_context["thread_lineage"]["equivocation_count"] == 1
    finally:
        catalog.close()


def test_memory_catalog_remember_quarantined_preserves_canonical_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")
    catalog = MemoryCatalog.open(tmp_path / "policy-quarantine.sqlite")
    try:
        root = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="target-thread",
            memory_type="semantic",
            summary="Canonical root",
            content="This is the canonical target root.",
            importance=8,
        )
        base_hash = catalog.get_memory_node_hash(root.memory_id)
        assert base_hash

        attempt = catalog.remember_quarantined(
            project="helix",
            agent_id="alpha",
            session_id="target-thread",
            memory_type="episodic",
            record_kind="target_attempt",
            summary="Isolated target attempt",
            content=json.dumps({"kind": "target_attempt", "prompt_preview": "safe evaluation case"}, sort_keys=True),
            importance=6,
            parent_hash=base_hash,
            quarantine_reason="isolated_target_attempt",
            quarantine_class="policy",
            disposition="evidence_only",
            llm_call_id="req-123",
        )
        refusal = catalog.remember_quarantined(
            project="helix",
            agent_id="target-adapter",
            session_id="target-thread",
            memory_type="episodic",
            record_kind="target_refusal",
            summary="Target refusal",
            content=json.dumps({"kind": "target_refusal", "refusal_class": "policy_refusal"}, sort_keys=True),
            importance=7,
            parent_hash=attempt["node_hash"],
            quarantine_reason="policy_refusal",
            quarantine_class="policy",
            disposition="quarantined_policy",
            llm_call_id="req-123",
        )

        lineage = catalog.verify_session_lineage("target-thread", include_quarantined=True)
        visible = catalog.list_memories(project="helix", session_id="target-thread", include_quarantined=False)
        forensic = catalog.list_memories(project="helix", session_id="target-thread", include_quarantined=True)
        proof = catalog.export_session_proof("target-thread", ref=refusal["memory_id"], include_quarantined=True)

        assert lineage["status"] == "verified"
        assert lineage["canonical_head"] == base_hash
        assert lineage["equivocation_count"] == 0
        assert lineage["policy_quarantine_count"] == 2
        assert lineage["quarantined_count"] == 2
        assert lineage["trust_status"] == "verified_with_quarantine"
        assert all(item["memory_id"] != attempt["memory_id"] for item in visible)
        attempt_row = next(item for item in forensic if item["memory_id"] == attempt["memory_id"])
        refusal_row = next(item for item in forensic if item["memory_id"] == refusal["memory_id"])
        assert attempt_row["canonical"] is False
        assert attempt_row["quarantine_class"] == "policy"
        assert attempt_row["disposition"] == "evidence_only"
        assert refusal_row["quarantined"] is True
        assert refusal_row["quarantine_class"] == "policy"
        assert refusal_row["disposition"] == "quarantined_policy"
        assert proof["lineage_verification"]["status"] == "verified"
        assert proof["lineage_verification"]["policy_quarantine_count"] == 2
    finally:
        catalog.close()


def test_memory_catalog_signed_checkpoints_replay_and_export_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HELIX_RECEIPT_SIGNING_MODE", raising=False)
    path = tmp_path / "trust.sqlite"
    catalog = MemoryCatalog.open(path)
    try:
        first = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-trust",
            memory_type="semantic",
            summary="Trust root",
            content="First canonical trust memory.",
        )
        second = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-trust",
            memory_type="semantic",
            summary="Trust head",
            content="Second canonical trust memory.",
        )
        second_hash = catalog.get_memory_node_hash(second.memory_id)
        receipt = catalog.get_memory_receipt(second.memory_id)
        checkpoint = catalog.head_checkpoint("thread-trust")
        lineage = catalog.verify_session_lineage("thread-trust")
        proof = catalog.export_session_proof("thread-trust", ref=second.memory_id)

        assert receipt and receipt["signature_verified"] is True
        assert receipt["checkpoint_hash"] == checkpoint["checkpoint_hash"]
        assert checkpoint["checkpoint_verified"] is True
        assert checkpoint["checkpoint"]["canonical_head"] == second_hash
        assert lineage["status"] == "verified"
        assert lineage["trust_status"] == "verified"
        assert lineage["checkpoint_count"] == 2
        assert proof["status"] == "ok"
        assert proof["target_memory_id"] == second.memory_id
        assert proof["target_receipt"]["signature_verified"] is True
        assert proof["dag_chain"]
        assert (tmp_path / "trust" / "trust_root.json").exists()
        assert first.memory_id != second.memory_id
    finally:
        catalog.close()

    MemoryCatalog._REGISTRY.pop(str(path.resolve()), None)  # noqa: SLF001 - force journal replay
    replayed = MemoryCatalog.open(path)
    try:
        replayed_lineage = replayed.verify_session_lineage("thread-trust")
        replayed_checkpoint = replayed.head_checkpoint("thread-trust")
        assert replayed_lineage["status"] == "verified"
        assert replayed_lineage["checkpoint_count"] == 2
        assert replayed_checkpoint["checkpoint_verified"] is True
    finally:
        replayed.close()


def test_memory_catalog_checkpoint_tamper_fails_verification(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "tamper.sqlite")
    try:
        catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-tamper",
            memory_type="semantic",
            summary="Tamper target",
            content="Checkpoint tamper should fail.",
        )
        catalog._session_checkpoints["thread-tamper"][-1]["canonical_head"] = "forged-head"  # noqa: SLF001

        lineage = catalog.verify_session_lineage("thread-tamper")

        assert lineage["status"] == "failed"
        assert lineage["trust_status"] == "failed"
        assert lineage["checkpoint_verified"] is False
    finally:
        catalog.close()


def test_memory_catalog_python_verify_chain_fails_missing_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELIX_MEMORY_RUST_INDEX", "0")
    catalog = MemoryCatalog.open(tmp_path / "missing-chain.sqlite")
    try:
        status = catalog.verify_chain("f" * 64)

        assert status["status"] == "failed"
        assert status["chain_len"] == 0
        assert status["missing_parent"] == "f" * 64
    finally:
        catalog.close()


def test_memory_catalog_sigstore_mode_signs_provenance_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "sigstore_rekor")
    monkeypatch.setenv("HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST", "sha256:rekor-bundle")
    catalog = MemoryCatalog.open(tmp_path / "sigstore.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-sigstore",
            memory_type="semantic",
            summary="Sigstore receipt",
            content="Receipt provenance is part of the signed payload.",
        )
        receipt = catalog.get_memory_receipt(memory.memory_id)

        assert receipt
        assert receipt["signature_verified"] is True
        assert receipt["key_provenance"] == "sigstore_rekor"
        assert receipt["public_claim_eligible"] is True
        assert receipt["attestation"]["provider"] == "sigstore_rekor"
    finally:
        catalog.close()


def test_memory_catalog_legacy_unsigned_warn_default_and_strict_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "off")
    monkeypatch.delenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", raising=False)
    catalog = MemoryCatalog.open(tmp_path / "legacy.sqlite")
    try:
        legacy = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-legacy",
            memory_type="semantic",
            summary="Legacy unsigned",
            content="legacy unsigned checkpoint memory",
            importance=8,
        )

        warned = catalog.search(project="helix", agent_id="agent-a", query="legacy unsigned", limit=3)
        strict = catalog.search(project="helix", agent_id="agent-a", query="legacy unsigned", limit=3, signature_enforcement="strict")
        lineage = catalog.verify_session_lineage("thread-legacy")

        assert warned and warned[0]["memory_id"] == legacy.memory_id
        assert warned[0]["legacy_unsigned"] is True
        assert warned[0]["signature_enforcement_warning"] == "unsigned_or_unverified_receipt_returned"
        assert strict == []
        assert lineage["legacy_unsigned_count"] == 1
        assert lineage["trust_status"] == "verified_with_legacy_warnings"
    finally:
        catalog.close()


def test_memory_catalog_uses_rust_bm25_when_extension_is_available(tmp_path: Path) -> None:
    if RustIndexedMerkleDAG is None:
        pytest.skip("Rust indexed MerkleDAG extension is not installed")

    catalog = MemoryCatalog.open(tmp_path / "memory.sqlite")
    try:
        first = catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            summary="Merkle audit tombstone receipt",
            content="Rust BM25 search should return this indexed memory.",
            importance=9,
        )
        catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            summary="Frontend layout card",
            content="Unrelated visual copy.",
            importance=3,
        )

        stats = catalog.stats()
        hits = catalog.search(project="helix", agent_id="agent-a", query="merkle tombstone", limit=2, signature_enforcement="permissive")

        assert stats["search_backend"] == "rust_bm25"
        assert stats["rust_index_stats"]["node_count"] >= 2
        assert hits[0]["memory_id"] == first.memory_id
        assert hits[0]["search_backend"] == "rust_bm25"
    finally:
        catalog.close()


def test_rust_indexed_batch_insert_when_extension_is_available() -> None:
    if RustIndexedMerkleDAG is None:
        pytest.skip("Rust indexed MerkleDAG extension is not installed")

    idx = RustIndexedMerkleDAG()
    records = json.dumps(
        [
            {
                "content": "batch alpha content",
                "metadata": {
                    "project": "batch",
                    "agent_id": "agent-a",
                    "record_kind": "memory",
                    "memory_id": "m-alpha",
                    "summary": "alpha batch",
                    "index_content": "batch alpha content",
                },
            },
            {
                "content": "batch beta content",
                "metadata": {
                    "project": "batch",
                    "agent_id": "agent-a",
                    "record_kind": "memory",
                    "memory_id": "m-beta",
                    "summary": "beta batch",
                    "index_content": "batch beta content",
                },
            },
        ]
    )

    nodes = idx.insert_indexed_batch(records)
    hits = idx.search("beta", 5, json.dumps({"project": "batch", "record_kind": "memory"}))

    assert len(nodes) == 2
    assert hits[0]["memory_id"] == "m-beta"


def test_memory_catalog_bulk_remember_indexes_and_searches(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "memory.sqlite")
    try:
        items = catalog.bulk_remember(
            [
                {
                    "project": "helix",
                    "agent_id": "agent-a",
                    "memory_type": "semantic",
                    "summary": "alpha indexed batch",
                    "content": "alpha indexed batch memory",
                    "importance": 8,
                },
                {
                    "project": "helix",
                    "agent_id": "agent-a",
                    "memory_type": "semantic",
                    "summary": "beta indexed batch",
                    "content": "beta indexed batch memory",
                    "importance": 8,
                },
            ]
        )
        hits = catalog.search(project="helix", agent_id="agent-a", query="beta", limit=3, signature_enforcement="permissive")

        assert len(items) == 2
        assert catalog.stats()["memory_count"] == 2
        assert hits[0]["memory_id"] == items[1].memory_id
    finally:
        catalog.close()


def test_semantic_query_router_rewrites_generic_queries(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "router.sqlite")
    try:
        target = catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            memory_id="mem-router-target",
            summary="crimson_falcon release gate",
            content="The release gate depends on sigstore-bundle-77 and attest-9f3a.",
            importance=10,
            tags=["crimson_falcon", "sigstore-bundle-77"],
        )
        catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            memory_id="mem-router-noise",
            summary="ordinary memory note",
            content="Generic agent memory background without deployment anchors.",
            importance=2,
            tags=["background"],
        )

        hits = catalog.search(project="helix", agent_id="agent-a", query="agent release memory", limit=2, signature_enforcement="permissive")

        assert hits
        assert hits[0]["memory_id"] == target.memory_id
        router = hits[0]["semantic_router"]
        assert router["action"] == "rewrite"
        assert router["original_query"] == "agent release memory"
        assert "agent" not in router["routed_query"].split()
        assert router["retained_terms"] == ["release"]
        assert catalog.stats()["semantic_query_router"]["rewrites"] >= 1
    finally:
        catalog.close()


def test_semantic_query_router_leaves_specific_queries_alone(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "router-specific.sqlite")
    try:
        item = catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            summary="rare_4242 qwen checkpoint",
            content="The qwen checkpoint is tied to rare_4242 and merkle receipt.",
            importance=8,
            tags=["rare_4242"],
        )

        hits = catalog.search(project="helix", agent_id="agent-a", query="rare_4242 qwen", limit=1, signature_enforcement="permissive")

        assert hits[0]["memory_id"] == item.memory_id
        assert "semantic_router" not in hits[0]
        assert catalog.stats()["semantic_query_router"]["pass_through"] >= 1
    finally:
        catalog.close()


def test_semantic_query_router_recent_fallback_when_no_anchors(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "router-fallback.sqlite")
    try:
        important = catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            memory_id="mem-important",
            summary="memory",
            content="agent memory",
            importance=9,
        )
        catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            memory_id="mem-less-important",
            summary="agent",
            content="memory",
            importance=3,
        )

        hits = catalog.search(project="helix", agent_id="agent-a", query="agent memory", limit=1, signature_enforcement="permissive")

        assert hits[0]["memory_id"] == important.memory_id
        assert hits[0]["search_backend"] == "semantic_router_recent_fallback"
        assert hits[0]["semantic_router"]["action"] == "recent_fallback"
    finally:
        catalog.close()


def test_memory_catalog_can_require_rust_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HELIX_MEMORY_RUST_INDEX", "0")
    monkeypatch.setenv("HELIX_MEMORY_REQUIRE_RUST_INDEX", "1")

    with pytest.raises(RuntimeError, match="RustIndexedMerkleDAG"):
        MemoryCatalog.open(tmp_path / "requires-rust.sqlite")
