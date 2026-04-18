from __future__ import annotations

from pathlib import Path

import pytest

from helix_kv.session_os import PrefixResolver, SessionCatalog, SessionScheduler, common_prefix_length, token_hash


def test_catalog_records_lists_and_filters_sessions(tmp_path: Path) -> None:
    catalog = SessionCatalog.open(tmp_path / "catalog.sqlite")
    session_path = tmp_path / "sessions" / "qwen" / "v0001"
    session_path.mkdir(parents=True)

    recorded = catalog.record_session(
        session_id="s1",
        model_id="qwen",
        agent_id="agent-a",
        model_ref="Qwen/Qwen2.5-1.5B-Instruct",
        arch="transformer",
        path=session_path,
        token_ids=[1, 2, 3],
        session_bytes=123,
        audit_status="pending",
        session_hash="abc",
    )

    assert recorded.token_hash == token_hash([1, 2, 3])
    assert catalog.find_latest("qwen", "agent-a") is not None
    assert catalog.find_latest("qwen", "missing") is None
    assert len(catalog.list_sessions(model_id="qwen")) == 1
    assert catalog.stats()["session_count"] == 1
    catalog.close()


def test_catalog_rejects_path_traversal(tmp_path: Path) -> None:
    catalog = SessionCatalog.open(tmp_path / "catalog.sqlite")

    with pytest.raises(ValueError):
        catalog.record_session(
            session_id="bad",
            model_id="qwen",
            agent_id="agent-a",
            model_ref="qwen",
            arch="transformer",
            path="../outside",
        )
    catalog.close()


def test_prefix_resolver_finds_best_transformer_prefix(tmp_path: Path) -> None:
    catalog = SessionCatalog.open(tmp_path / "catalog.sqlite")
    session_path = tmp_path / "sessions" / "gpt2" / "v0001"
    session_path.mkdir(parents=True)
    (session_path / "session.json").write_text('{"session_token_ids":[10,20,30,40]}', encoding="utf-8")
    catalog.record_session(
        session_id="gpt2-a-v1",
        model_id="gpt2",
        agent_id="agent-a",
        model_ref="gpt2",
        arch="transformer",
        path=session_path,
        token_ids=[10, 20, 30, 40],
    )

    match = PrefixResolver(catalog).find_best_prefix(
        model_id="gpt2",
        agent_id="agent-a",
        token_ids=[10, 20, 30, 99, 100],
        arch="transformer",
    )

    assert common_prefix_length([1, 2], [1, 3]) == 1
    assert match.status == "hit"
    assert match.prefix_match_tokens == 3
    assert match.new_tokens_computed == 2
    catalog.close()


def test_prefix_resolver_marks_hybrid_unsupported(tmp_path: Path) -> None:
    catalog = SessionCatalog.open(tmp_path / "catalog.sqlite")

    match = PrefixResolver(catalog).find_best_prefix(
        model_id="zamba",
        agent_id="agent-a",
        token_ids=[1, 2, 3],
        arch="hybrid-mamba-transformer",
    )

    assert match.status == "unsupported_hybrid_v0"
    assert match.new_tokens_computed == 3
    catalog.close()


def test_prefix_resolver_allows_exact_hybrid_checkpoint(tmp_path: Path) -> None:
    catalog = SessionCatalog.open(tmp_path / "catalog.sqlite")
    session_path = tmp_path / "sessions" / "zamba" / "v0001"
    session_path.mkdir(parents=True)
    (session_path / "session.json").write_text('{"session_token_ids":[10,20,30]}', encoding="utf-8")
    catalog.record_session(
        session_id="zamba-a-v1",
        model_id="zamba",
        agent_id="agent-a",
        model_ref="Zyphra/Zamba2-1.2B-Instruct-v2",
        arch="hybrid-mamba-transformer",
        path=session_path,
        token_ids=[10, 20, 30],
    )

    exact = PrefixResolver(catalog).find_best_prefix(
        model_id="zamba",
        agent_id="agent-a",
        token_ids=[10, 20, 30, 40],
        arch="hybrid-mamba-transformer",
    )
    partial = PrefixResolver(catalog).find_best_prefix(
        model_id="zamba",
        agent_id="agent-a",
        token_ids=[10, 99, 30, 40],
        arch="hybrid-mamba-transformer",
    )

    assert exact.status == "hybrid_checkpoint_v0"
    assert exact.prefix_match_tokens == 3
    assert exact.new_tokens_computed == 1
    assert partial.status == "unsupported_partial_hybrid_prefix"
    catalog.close()


def test_scheduler_prefers_active_model_when_cost_is_lower(tmp_path: Path) -> None:
    catalog = SessionCatalog.open(tmp_path / "catalog.sqlite")
    session_path = tmp_path / "sessions" / "gpt2" / "v0001"
    session_path.mkdir(parents=True)
    catalog.record_session(
        session_id="gpt2-a-v1",
        model_id="gpt2",
        agent_id="agent-a",
        model_ref="gpt2",
        arch="transformer",
        path=session_path,
        token_ids=[1, 2, 3],
        audit_status="verified",
    )

    class Lifecycle:
        active_model_id = "gpt2"

    decision = SessionScheduler(catalog).route(
        {"capability": "drafting", "agent_id": "agent-a", "token_ids": [1, 2, 3, 4], "prefill_ms_per_token": 1.0},
        {
            "gpt2": {"arch": "transformer", "capabilities": ["drafting"], "load_time_estimate_ms": 0},
            "qwen": {"arch": "transformer", "capabilities": ["drafting"], "load_time_estimate_ms": 1000},
        },
        Lifecycle(),
    )

    assert decision.selected_model_id == "gpt2"
    assert decision.model_swapped is False
    assert decision.candidate_models
    catalog.close()
