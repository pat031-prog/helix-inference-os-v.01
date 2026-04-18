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
    hits = catalog.search(project="helix", agent_id="agent-a", query="jose token", limit=3)
    context = catalog.build_context(project="helix", agent_id="agent-a", query="auth", budget_tokens=80, mode="search")

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
        hits = catalog.search(project="helix", agent_id="agent-a", query="merkle tombstone", limit=2)

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
        hits = catalog.search(project="helix", agent_id="agent-a", query="beta", limit=3)

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

        hits = catalog.search(project="helix", agent_id="agent-a", query="agent release memory", limit=2)

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

        hits = catalog.search(project="helix", agent_id="agent-a", query="rare_4242 qwen", limit=1)

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

        hits = catalog.search(project="helix", agent_id="agent-a", query="agent memory", limit=1)

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
