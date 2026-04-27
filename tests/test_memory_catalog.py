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


def test_memory_catalog_sigstore_mode_marks_simulated_when_bundle_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default sigstore_rekor path takes the bundle digest on trust and must
    mark the attestation simulated_only=True so verifiers downgrade public
    claims, even though the local Ed25519 signature is valid."""
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "sigstore_rekor")
    monkeypatch.setenv("HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST", "sha256:rekor-bundle")
    monkeypatch.delenv("HELIX_SIGSTORE_REKOR_BUNDLE_VERIFIED", raising=False)
    catalog = MemoryCatalog.open(tmp_path / "sigstore-sim.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-sigstore-sim",
            memory_type="semantic",
            summary="Sigstore simulated receipt",
            content="Receipt provenance is signed but bundle is not Rekor-verified.",
        )
        receipt = catalog.get_memory_receipt(memory.memory_id)

        assert receipt
        assert receipt["signature_verified"] is True
        assert receipt["key_provenance"] == "sigstore_rekor"
        assert receipt["attestation"]["provider"] == "sigstore_rekor"
        assert receipt["attestation"]["simulated_only"] is True
        assert receipt["public_claim_eligible"] is False
    finally:
        catalog.close()


def test_memory_catalog_sigstore_mode_public_claim_only_when_bundle_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When HELIX_SIGSTORE_REKOR_BUNDLE_VERIFIED=1 the operator asserts the
    bundle was verified out-of-band against the live Rekor log; only then is
    the receipt eligible for public claims."""
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "sigstore_rekor")
    monkeypatch.setenv("HELIX_SIGSTORE_REKOR_BUNDLE_DIGEST", "sha256:rekor-bundle-verified")
    monkeypatch.setenv("HELIX_SIGSTORE_REKOR_BUNDLE_VERIFIED", "1")
    catalog = MemoryCatalog.open(tmp_path / "sigstore-verified.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-sigstore-verified",
            memory_type="semantic",
            summary="Sigstore verified receipt",
            content="Receipt provenance signed and bundle verified out-of-band.",
        )
        receipt = catalog.get_memory_receipt(memory.memory_id)

        assert receipt
        assert receipt["signature_verified"] is True
        assert receipt["key_provenance"] == "sigstore_rekor"
        assert receipt["attestation"]["provider"] == "sigstore_rekor"
        assert receipt["attestation"]["simulated_only"] is False
        assert receipt["public_claim_eligible"] is True
    finally:
        catalog.close()


def test_memory_catalog_local_signing_key_file_permissions_restricted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local signing key file must be owner-only on POSIX. On Windows os.chmod
    only flips the read-only bit so we skip the strict-mode assertion."""
    import sys
    monkeypatch.delenv("HELIX_RECEIPT_SIGNING_MODE", raising=False)
    catalog = MemoryCatalog.open(tmp_path / "key-perms.sqlite")
    try:
        # remember() forces the local signing key to be materialized to disk
        catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-key-perms",
            memory_type="semantic",
            summary="Materialize the local signing key.",
            content="Triggers _local_signing_key_unlocked which writes the key file.",
        )
        key_path = catalog._local_signing_key_path
        assert key_path.exists()
        if sys.platform != "win32":
            mode = key_path.stat().st_mode & 0o777
            assert mode == 0o600, f"signing key mode is {oct(mode)}, expected 0o600"
            parent_mode = key_path.parent.stat().st_mode & 0o777
            assert parent_mode == 0o700, f"trust dir mode is {oct(parent_mode)}, expected 0o700"
    finally:
        catalog.close()


def test_memory_catalog_checkpoint_chain_detects_dropped_intermediate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Threat model: an attacker has the local signing key (e.g. via filesystem
    access). They drop checkpoint #1 from a 3-checkpoint chain, re-link
    previous_checkpoint_hash on the survivor and re-sign so the hash chain
    looks valid again. The signed checkpoint_index still encodes the original
    position (2), exposing the gap to the verifier."""
    monkeypatch.delenv("HELIX_RECEIPT_SIGNING_MODE", raising=False)
    catalog = MemoryCatalog.open(tmp_path / "cp-chain.sqlite")
    try:
        for index in range(3):
            catalog.remember(
                project="helix",
                agent_id="agent-a",
                session_id="thread-cp-chain",
                memory_type="semantic",
                summary=f"checkpoint anchor {index}",
                content=f"checkpoint sequence body {index}",
                importance=5 + index,
            )
        checkpoints = catalog._session_checkpoints["thread-cp-chain"]
        assert len(checkpoints) >= 3
        for index, checkpoint in enumerate(checkpoints):
            assert checkpoint.get("checkpoint_index") == index

        # Attacker drops checkpoint #1. To keep the prev-hash chain valid,
        # they re-link checkpoint #2 onto checkpoint #0 and re-sign with the
        # leaked local key — but they keep the original signed checkpoint_index
        # (=2) on the survivor (changing it would also expose the drop via
        # the upstream chain).
        from helix_proto.signed_receipts import canonical_payload_sha256, sign_receipt_payload

        key = catalog._local_signing_key_unlocked()
        before = checkpoints[0]
        survivor_body = catalog._checkpoint_hash_body(checkpoints[2])
        survivor_body["previous_checkpoint_hash"] = before["checkpoint_hash"]
        new_hash = canonical_payload_sha256(survivor_body)
        re_signed = sign_receipt_payload(
            {**survivor_body, "checkpoint_hash": new_hash},
            private_key_b64=str(key["private_key"]),
            public_key_b64=str(key["public_key"]),
            signer_id="helix-local-checkpoint",
            key_provenance="local_self_signed",
            attestation=None,
        )
        catalog._session_checkpoints["thread-cp-chain"] = [before, re_signed]

        status = catalog._checkpoint_chain_status_unlocked("thread-cp-chain")
        assert status["checkpoint_chain_verified"] is False
        assert status["checkpoint_error"] == "checkpoint_index_gap"
        assert status["expected_checkpoint_index"] == 1
        assert status["observed_checkpoint_index"] == 2
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


# ─── Consolidacion: journal hash-chain + DAG coverage + bundle export ───────


def _fresh_catalog(path: Path) -> MemoryCatalog:
    """Defeat MemoryCatalog._REGISTRY caching between sub-tests of the same fn."""
    resolved = str(path.resolve())
    MemoryCatalog._REGISTRY.pop(resolved, None)
    return MemoryCatalog.open(path)


def test_memory_catalog_journal_hash_chain_appends_seq_and_prev_hash(
    tmp_path: Path,
) -> None:
    """Every fresh append must carry journal_seq + prev_journal_sha256, and
    the prev hash on entry N must equal SHA-256 of the raw bytes of entry N-1."""
    import hashlib as _hashlib
    catalog = _fresh_catalog(tmp_path / "chain.sqlite")
    try:
        for index in range(3):
            catalog.remember(
                project="helix",
                agent_id="agent-a",
                session_id="thread-chain",
                memory_type="semantic",
                summary=f"chain entry {index}",
                content=f"chain body {index}",
            )
        journal_path = catalog._journal_path
        with journal_path.open("rb") as handle:
            lines = [line for line in handle if line.strip()]
        assert lines, "journal must be non-empty"
        last_sha = None
        last_seq = 0
        for line in lines:
            entry = json.loads(line.decode("utf-8"))
            assert isinstance(entry.get("journal_seq"), int)
            assert entry["journal_seq"] == last_seq + 1
            assert entry.get("prev_journal_sha256") == last_sha
            last_seq = entry["journal_seq"]
            last_sha = _hashlib.sha256(line).hexdigest()
    finally:
        catalog.close()


def test_memory_catalog_verify_journal_detects_intermediate_tamper(
    tmp_path: Path,
) -> None:
    """Edit a line in the middle of the journal. verify_journal must surface
    a prev_journal_sha256_mismatch on the line after the tampered one."""
    catalog = _fresh_catalog(tmp_path / "tamper.sqlite")
    try:
        for index in range(3):
            catalog.remember(
                project="helix",
                agent_id="agent-a",
                session_id="thread-tamper",
                memory_type="semantic",
                summary=f"tamper anchor {index}",
                content=f"tamper body {index}",
            )
        journal_path = catalog._journal_path
        # Mutate line 2 (index 1) by appending a benign field BEFORE the closing brace.
        # The byte-exact line changes -> next line's prev_journal_sha256 no longer matches.
        with journal_path.open("rb") as handle:
            lines = handle.readlines()
        edited = json.loads(lines[1].decode("utf-8"))
        edited["__tamper__"] = "x"
        lines[1] = (json.dumps(edited, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with journal_path.open("wb") as handle:
            handle.writelines(lines)

        # verify_journal does NOT need the catalog state — it re-reads the file.
        report = catalog.verify_journal()
        assert report["status"] == "tampered"
        codes = {error["error"] for error in report["errors"]}
        assert "prev_journal_sha256_mismatch" in codes
    finally:
        catalog.close()


def test_memory_catalog_verify_journal_handles_missing_and_no_chain(
    tmp_path: Path,
) -> None:
    catalog = _fresh_catalog(tmp_path / "empty.sqlite")
    try:
        report = catalog.verify_journal()
        assert report["status"] == "missing"
        # Now force a single legacy line (pre-v2, no journal_seq) — must report no_chain.
        catalog._journal_path.parent.mkdir(parents=True, exist_ok=True)
        catalog._journal_path.write_text(
            json.dumps({"journal_version": "helix-memory-journal-v1", "op": "noop"}) + "\n",
            encoding="utf-8",
        )
        report = catalog.verify_journal()
        assert report["status"] == "no_chain"
        assert report["legacy_lines"] == 1
        assert report["signed_lines"] == 0
    finally:
        catalog.close()


def test_memory_catalog_verify_dag_coverage_detects_python_drift(
    tmp_path: Path,
) -> None:
    """Drop a node from the in-memory Python DAG after a remember(). The
    journal still records the node_hash; verify_dag_coverage must surface
    the drift."""
    catalog = _fresh_catalog(tmp_path / "coverage.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-cov",
            memory_type="semantic",
            summary="coverage anchor",
            content="coverage body",
        )
        node_hash = catalog.get_memory_node_hash(memory.memory_id)
        assert node_hash and node_hash in catalog.dag._nodes

        # Healthy baseline first.
        clean = catalog.verify_dag_coverage()
        assert clean["status"] == "verified"
        assert clean["checked"] >= 1
        assert not clean["python_missing"]

        # Drop the node from Python DAG only — Rust may or may not still have it
        # depending on env (we just assert Python drift is raised).
        catalog.dag._nodes.pop(node_hash)
        drifted = catalog.verify_dag_coverage()
        assert drifted["status"] == "drift_detected"
        assert any(item["node_hash"] == node_hash for item in drifted["python_missing"])
    finally:
        catalog.close()


def test_memory_catalog_export_full_session_bundle_is_self_contained(
    tmp_path: Path,
) -> None:
    """The bundle must include: signed proof, signed checkpoints, journal slice
    for that session, journal hash-chain integrity, DAG coverage, signing
    public key. An auditor should be able to verify everything offline."""
    from helix_proto.signed_receipts import verify_signed_receipt

    catalog = _fresh_catalog(tmp_path / "bundle.sqlite")
    try:
        for index in range(2):
            catalog.remember(
                project="helix",
                agent_id="agent-a",
                session_id="thread-bundle",
                memory_type="semantic",
                summary=f"bundle anchor {index}",
                content=f"bundle body {index}",
            )
        bundle = catalog.export_full_session_bundle("thread-bundle")
        assert bundle["bundle_version"] == "helix-session-bundle-v1"
        assert bundle["session_id"] == "thread-bundle"
        assert bundle["session_proof"]["status"] == "ok"
        assert bundle["journal_integrity"]["status"] == "verified"
        assert bundle["dag_coverage"]["status"] == "verified"
        assert bundle["signer"]["public_key"]
        assert bundle["signer"]["key_id"]
        # Journal slice scoped to this session_id only.
        assert bundle["journal_entries"]
        for entry in bundle["journal_entries"]:
            payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
            assert str(payload.get("session_id") or entry.get("session_id") or "") == "thread-bundle"
        # Receipt embedded in the proof must verify with the signer key offline.
        receipt = bundle["session_proof"].get("target_receipt")
        if receipt:
            assert receipt.get("public_key") == bundle["signer"]["public_key"]
            verified = verify_signed_receipt(dict(receipt))
            assert verified["signature_verified"] is True
        # At least one checkpoint and it round-trips through verify.
        assert bundle["checkpoints"]
        for checkpoint in bundle["checkpoints"]:
            verification = catalog._verify_checkpoint_unlocked(checkpoint)
            assert verification["checkpoint_verified"] is True
    finally:
        catalog.close()


def test_memory_catalog_export_proof_cli_writes_bundle_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The python -m helix_kv.export_proof entrypoint must write a valid JSON
    bundle to --out and exit 0 on a healthy catalog."""
    from helix_kv import export_proof

    db_path = tmp_path / "cli.sqlite"
    catalog = _fresh_catalog(db_path)
    try:
        catalog.remember(
            project="helix",
            agent_id="agent-a",
            session_id="thread-cli",
            memory_type="semantic",
            summary="cli anchor",
            content="cli body",
        )
    finally:
        catalog.close()
    # Drop the registry entry so the CLI re-opens cleanly.
    MemoryCatalog._REGISTRY.pop(str(db_path.resolve()), None)

    out_path = tmp_path / "out" / "bundle.json"
    rc = export_proof.main([
        str(db_path),
        "thread-cli",
        "--out",
        str(out_path),
        "--fail-on-drift",
    ])
    assert rc == 0
    assert out_path.exists()
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["session_id"] == "thread-cli"
    assert bundle["session_proof"]["status"] == "ok"
    assert bundle["journal_integrity"]["status"] == "verified"
