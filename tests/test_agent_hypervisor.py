from __future__ import annotations

from tools import run_local_agent_hypervisor


def test_hypervisor_parser_defaults_to_gpt2_and_five_agents() -> None:
    args = run_local_agent_hypervisor.build_parser().parse_args([])

    assert args.model == "gpt2"
    assert args.agents == 5
    assert args.rounds == 2
    assert args.timeslice_tokens == 1
    assert args.codec == "rust-hlx"
    assert args.audit_policy == "blocking"


def test_pr_war_room_scenario_has_five_distinct_roles() -> None:
    agents = run_local_agent_hypervisor._scenario_agents("pr-war-room-long", 5)

    assert len(agents) == 5
    assert {agent["role"] for agent in agents} == {
        "bug_hunter",
        "perf_engineer",
        "benchmark_scientist",
        "claims_editor",
        "release_captain",
    }
    assert all("prompt" in agent and "Task:" in agent["prompt"] for agent in agents)
    assert all("handoffs" in agent["prompt"] for agent in agents)


def test_pr_war_room_long_uses_external_handoff_transcript() -> None:
    transcript = [{"role": "bug_hunter", "handoff_summary": "The diff risks stale cache metadata."}]
    agent = run_local_agent_hypervisor._scenario_agents("pr-war-room-long", 2)[1]

    prompt = run_local_agent_hypervisor._turn_prompt_for_agent("pr-war-room-long", agent, transcript)

    assert "Shared PR war-room transcript" in prompt
    assert "bug_hunter" in prompt
    assert "stale cache metadata" in prompt
    assert len(run_local_agent_hypervisor._transcript_hash(transcript)) == 64


def test_hybrid_cameo_caps_agents_to_two() -> None:
    agents = run_local_agent_hypervisor._scenario_agents("hybrid-cameo", 5)

    assert len(agents) == 2
    assert [agent["role"] for agent in agents] == ["hybrid_integrity", "hybrid_claims"]


def test_deferred_war_room_artifact_name_is_distinct() -> None:
    assert (
        run_local_agent_hypervisor._artifact_name("pr-war-room-long", "qwen", audit_policy="deferred")
        == "local-agent-hypervisor-pr-war-room-deferred.json"
    )
    assert (
        run_local_agent_hypervisor._artifact_name("pr-war-room-long", "qwen", audit_policy="blocking")
        == "local-agent-hypervisor-pr-war-room-long.json"
    )
