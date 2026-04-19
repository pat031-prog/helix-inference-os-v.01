from __future__ import annotations

from tools import (
    run_local_agent_framework_showcase,
    run_local_hmem_wiring_smoke,
    run_local_hybrid_prefix_checkpoint,
    run_local_openai_smoke,
    run_local_prefix_reuse,
    run_local_session_branching,
    run_local_segmented_sessions,
    run_local_session_catalog_smoke,
    run_local_session_os_demo,
)


def test_catalog_smoke_writes_expected_shape(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_session_catalog_smoke.run_catalog_smoke(
        run_local_session_catalog_smoke.build_parser().parse_args(["--output-dir", str(tmp_path)])
    )

    assert payload["status"] == "completed"
    assert payload["prefix_match"]["status"] == "hit"
    assert payload["traversal_rejected"] is True
    assert (tmp_path / "local-session-catalog-smoke.json").exists()


def test_segmented_sessions_smoke_writes_chain(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_segmented_sessions.run_segmented_sessions(
        run_local_segmented_sessions.build_parser().parse_args(["--output-dir", str(tmp_path)])
    )

    assert payload["status"] == "completed"
    assert payload["segment_chain_length"] == 3
    assert payload["verify_chain_status"] == "verified"
    assert payload["rewrite_avoided_bytes"] >= 0


def test_prefix_reuse_parser_defaults_are_budgeted() -> None:
    args = run_local_prefix_reuse.build_parser().parse_args([])

    assert args.models == "gpt2,qwen"
    assert args.codec == "rust-hlx-buffered-flat"
    assert args.audit_policy == "deferred"
    assert args.repeats == 3
    assert args.include_native_dense is True


def test_session_os_demo_parser_defaults_are_short() -> None:
    args = run_local_session_os_demo.build_parser().parse_args([])

    assert args.max_new_tokens == 12
    assert args.prompt_tokens == 192
    assert args.codec == "rust-hlx-buffered-flat"
    assert args.audit_policy == "deferred"


def test_openai_smoke_writes_openai_shaped_artifact(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_openai_smoke.run_openai_smoke(
        run_local_openai_smoke.build_parser().parse_args(["--output-dir", str(tmp_path)])
    )

    assert payload["status"] == "completed"
    assert payload["object"] == "chat.completion"
    assert payload["session_recorded"] is True


def test_session_branching_writes_three_agent_branches(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_session_branching.run_session_branching(
        run_local_session_branching.build_parser().parse_args(["--output-dir", str(tmp_path)])
    )

    assert payload["status"] == "completed"
    assert payload["branch_count"] == 3
    assert payload["verify_chain_status"] == "verified"
    assert payload["rewrite_avoided_bytes_estimate"] >= 0


def test_agent_framework_showcase_openai_mock_completes(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_agent_framework_showcase.run_agent_framework_showcase(
        run_local_agent_framework_showcase.build_parser().parse_args(
            ["--client", "openai", "--mock-server", "--output-dir", str(tmp_path)]
        )
    )

    assert payload["status"] == "completed"
    assert payload["client"] == "openai"
    assert payload["agent_count"] == 2
    assert payload["server_mode"] == "mock-openai-compatible-contract"


def test_agent_framework_showcase_langchain_skips_when_missing(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_agent_framework_showcase.run_agent_framework_showcase(
        run_local_agent_framework_showcase.build_parser().parse_args(
            ["--client", "langchain", "--mock-server", "--output-dir", str(tmp_path)]
        )
    )

    assert payload["status"] in {"completed", "skipped_dependency_missing"}


def test_agent_framework_showcase_crewai_skips_when_missing(tmp_path) -> None:  # noqa: ANN001
    payload = run_local_agent_framework_showcase.run_agent_framework_showcase(
        run_local_agent_framework_showcase.build_parser().parse_args(
            ["--client", "crewai", "--mock-server", "--output-dir", str(tmp_path)]
        )
    )

    assert payload["status"] in {"completed", "skipped_dependency_missing"}


def test_hybrid_prefix_checkpoint_parser_defaults_to_zamba() -> None:
    args = run_local_hybrid_prefix_checkpoint.build_parser().parse_args([])

    assert args.model == "zamba"
    assert args.prefix_tokens == 64
    assert args.followup_tokens == 8


def test_hmem_wiring_smoke_records_agent_memory_and_hybrid_search(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # hmem smoke inserts unsigned_legacy memories; opt out of strict default.
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")
    payload = run_local_hmem_wiring_smoke.run_hmem_wiring_smoke(
        run_local_hmem_wiring_smoke.build_parser().parse_args(["--output-dir", str(tmp_path)])
    )

    assert payload["status"] == "completed"
    assert payload["acceptance"]["auto_tool_observe"] is True
    assert payload["acceptance"]["startup_context_injected"] is True
    assert payload["acceptance"]["hybrid_search_returns_memory_and_knowledge"] is True
    assert payload["acceptance"]["memory_graph_populated"] is True
    assert payload["acceptance"]["compress_fn_hook_used"] is True
