from __future__ import annotations

import io
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto import agent as helix_agent
from helix_proto import helix_cli
from helix_proto.memory import append_memory_event


def _test_root() -> Path:
    return Path.cwd() / "verification" / "cli-sessions" / "_test" / uuid.uuid4().hex


def _write_suite_fixture(base: Path, suite_id: str = "hard-anchor-utility") -> tuple[Path, Path, dict[str, Path]]:
    evidence_root = base / "repo" / "verification"
    suite_dir = evidence_root / "nuclear-methodology" / suite_id
    case_dir = suite_dir / "exact-anchor-recovery-under-lossy-summary"
    case_dir.mkdir(parents=True)
    preregistered = suite_dir / "PREREGISTERED.md"
    preregistered.write_text("# preregistered\n- hard anchor recovery\n", encoding="utf-8")
    transcript = case_dir / f"local-{suite_id}-case-{suite_id}-20260421-120000-transcript.jsonl"
    transcript.write_text(
        '{"event":"proposer","model":"qwen","content":"hard anchor recovered"}\n'
        '{"event":"auditor","model":"sonnet","content":"lineage verified"}\n',
        encoding="utf-8",
    )
    transcript_md = case_dir / f"local-{suite_id}-case-{suite_id}-20260421-120000-transcript.md"
    transcript_md.write_text("## Transcript\nhard anchor recovered\n", encoding="utf-8")
    artifact = suite_dir / f"local-{suite_id}-suite-{suite_id}-20260421-120000.json"
    artifact.write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "run_id": f"{suite_id}-20260421-120000",
                "status": "completed",
                "case_count": 1,
                "score": 1.0,
                "cases": [{"case_id": "exact-anchor-recovery-under-lossy-summary", "status": "completed", "score": 1.0}],
                "transcript_exports": {"jsonl_path": str(transcript), "md_path": str(transcript_md)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = suite_dir / f"local-{suite_id}-suite-20260421-120000-run.json"
    manifest.write_text(
        json.dumps({"run_id": f"{suite_id}-20260421-120000", "artifact_path": str(artifact)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return evidence_root, suite_dir, {
        "artifact": artifact,
        "manifest": manifest,
        "transcript": transcript,
        "transcript_md": transcript_md,
        "preregistered": preregistered,
    }


def test_provider_registry_includes_cloud_local_and_openai_compatible() -> None:
    assert "deepinfra" in helix_cli.PROVIDERS
    assert "gemini" in helix_cli.PROVIDERS
    assert "ollama" in helix_cli.PROVIDERS
    assert "llamacpp" in helix_cli.PROVIDERS
    assert "local" in helix_cli.PROVIDERS
    assert helix_cli.PROVIDERS["deepinfra"].token_env == "DEEPINFRA_API_TOKEN"
    assert helix_cli.PROVIDERS["gemini"].token_env == "GEMINI_API_KEY"
    assert helix_cli.PROVIDERS["gemini"].kind == "gemini"
    assert helix_cli.PROVIDERS["ollama"].requires_token is False


def test_redact_value_removes_sensitive_keys_and_secret_values() -> None:
    payload = {
        "Authorization": "Bearer abc123",
        "nested": {"text": "token abc123 should not leak", "safe": "ok"},
    }
    redacted = helix_cli.redact_value(payload, secrets=["abc123"])
    assert redacted["Authorization"] == helix_cli.REDACTED
    assert redacted["nested"]["text"] == f"token {helix_cli.REDACTED} should not leak"
    assert redacted["nested"]["safe"] == "ok"


def test_doctor_report_lists_registered_suite_scripts() -> None:
    report = helix_cli.doctor_report(probe_local=False)
    suites = {item["suite_id"]: item for item in report["suites"]}
    assert "policy-rag-legal-debate" in suites
    assert "branch-pruning-forensics" in suites
    assert "multi-agent-concurrency" in suites
    assert suites["policy-rag-legal-debate"]["script_exists"] is True
    assert suites["multi-agent-concurrency"]["script_exists"] is True


def test_cert_dry_run_adds_deepinfra_flag_for_optional_cloud_suite() -> None:
    report = helix_cli.run_cert_suite(
        "policy-rag-legal-debate",
        python_executable="python",
        provider_name="deepinfra",
        prompt_token=False,
        dry_run=True,
        extra_args=["--tokens", "64"],
    )
    command = report["command"]
    assert report["dry_run"] is True
    assert "--use-deepinfra" in command
    assert "--tokens" in command
    assert "64" in command


def test_cert_dry_run_adds_deepinfra_flag_for_multi_agent_concurrency_suite() -> None:
    report = helix_cli.run_cert_suite(
        "multi-agent-concurrency",
        python_executable="python",
        provider_name="deepinfra",
        prompt_token=False,
        dry_run=True,
        extra_args=["--max-tokens", "64"],
    )
    command = report["command"]
    assert report["dry_run"] is True
    assert "--use-deepinfra" in command
    assert "--max-tokens" in command
    assert "64" in command


def test_parser_accepts_cert_remainder_after_separator() -> None:
    args = helix_cli.parse_args([
        "cert",
        "run",
        "infinite-depth-memory",
        "--dry-run",
        "--",
        "--depth",
        "128",
    ])
    assert args.suite == "infinite-depth-memory"
    assert helix_cli._strip_remainder(args.extra_args) == ["--depth", "128"]


def test_no_args_enters_interactive_mode_without_required_subcommand() -> None:
    args = helix_cli.parse_args([])
    assert args.command is None


def test_unknown_provider_prompt_input_becomes_pending_chat(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "hola")
    provider, pending = helix_cli._choose_provider("deepinfra")
    assert provider == "deepinfra"
    assert pending == "hola"


def test_natural_language_routes_known_suite() -> None:
    routed = helix_cli._route_natural_language("corre la suite de polizas con deepinfra")
    assert routed == "/cert policy-rag-legal-debate"


def test_pasted_suite_analysis_does_not_route_to_cert() -> None:
    pasted = """quiero data de esto
Suite                     Run ID           Estado
branch-pruning-forensics  20260421-120000  completed
{
  "suite_id": "branch-pruning-forensics",
  "exit_code": 1,
  "stderr": "RuntimeError: RustIndexedMerkleDAG was not rebuilt with build_context_fast"
}
"""
    assert helix_cli._looks_like_pasted_suite_evidence(pasted) is True
    assert helix_cli._is_pasted_suite_analysis_request(pasted) is True
    assert helix_cli._route_natural_language(pasted) is None
    assert helix_cli._is_suite_evidence_request(pasted) is True


def test_natural_language_verify_hint_opens_suite_catalog() -> None:
    assert helix_cli._route_natural_language("bueno /verify") == "/suites"


def test_default_workspace_root_prefers_repo_workspace_over_config_override(monkeypatch) -> None:
    monkeypatch.setattr(helix_cli, "_load_config", lambda: {"workspace_root": "C:/tmp/elsewhere"})
    monkeypatch.setenv("HELIX_WORKSPACE_ROOT", "C:/tmp/from-env")
    assert helix_cli._default_workspace_root() == Path.cwd().resolve() / "workspace"


def test_clean_assistant_text_prefers_helix_output_tag() -> None:
    raw = "<think>private plan</think>\n<helix_output>Respuesta limpia.</helix_output>"
    assert helix_cli._clean_assistant_text(raw) == "Respuesta limpia."


def test_clean_assistant_text_suppresses_tool_protocol_residue() -> None:
    raw = "<tool_call>\n1 Input received\n2 Analysis: noisy"
    assert helix_cli._clean_assistant_text(raw) == ""


def test_clean_assistant_text_suppresses_thinking_process_only_output() -> None:
    raw = """Thinking Process:

1 Analyze the Request:
   - User asks: "que es helix?"
   - System Instructions: answer directly.
"""
    assert helix_cli._clean_assistant_text(raw) == ""


def test_clean_assistant_text_removes_planning_and_dedupes_final_answer() -> None:
    raw = """Plan:
1 Analyze the user request
Let's write the response.

La respuesta final.

No emojis? Checked.

La respuesta final.
"""
    assert helix_cli._clean_assistant_text(raw) == "La respuesta final."


def test_auto_router_selects_code_model_for_repo_work() -> None:
    route = helix_cli.route_model_for_task(
        "arregla este bug de pytest en el repo y armame un patch",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["profile"] == "code"
    assert "Coder" in route["model"]


def test_auto_router_selects_sonnet_for_high_stakes_audit() -> None:
    route = helix_cli.route_model_for_task(
        "audita la evidencia legal y los claims forenses de esta suite compleja",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["profile"] == "sonnet"


def test_auto_router_selects_research_model_for_benchmark_lookup() -> None:
    route = helix_cli.route_model_for_task(
        "necesito que me busques info sobre benchmark de claude mythos",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["intent"] == "research"
    assert route["profile"] == "qwen-big"
    assert route["model"] == "Qwen/Qwen3.5-122B-A10B"
    assert route["blueprint"] == "balanced"
    assert route["fallback_chain"] == ["qwen-122b", "default", "chat"]


def test_auto_router_selects_gemma_for_reasoning_in_balanced_blueprint() -> None:
    route = helix_cli.route_model_for_task(
        "analiza los tradeoffs y desglosa la hipotesis paso a paso",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["intent"] == "reasoning"
    assert route["profile"] == "reasoning"
    assert route["model"] == "google/gemma-4-31B"


def test_auto_router_explore_mode_uses_creative_helix_for_nontechnical_helix_prompt() -> None:
    route = helix_cli.route_model_for_task(
        "exploremos helix desde ghost in the shell y sus influencias culturales",
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="explore",
    )
    assert route["interaction_mode"] == "explore"
    assert route["intent"] == "creative_helix"
    assert route["mode_policy"]["name"] == "explore"
    assert route["grounding_plan"] == "helix-only"
    assert "creative/cultural synthesis" in route["mode_reason"]


def test_auto_router_technical_mode_biases_helix_prompts_toward_core_grounding() -> None:
    route = helix_cli.route_model_for_task(
        "explicame el canonical head de helix y como se relaciona con receipts y hashes",
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="technical",
    )
    assert route["interaction_mode"] == "technical"
    assert route["intent"] in {"helix_self", "audit", "reasoning"}
    assert route["mode_policy"]["name"] == "technical"
    assert route["tone_contract"]
    assert "code, audit, repo" in route["mode_reason"]


def test_router_prefers_gemini_url_context_for_technical_urls_when_available(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    route = helix_cli.route_model_for_task(
        (
            "compará https://ai.google.dev/gemini-api/docs/url-context con "
            "https://ai.google.dev/gemini-api/docs/function-calling y resumilo"
        ),
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="explore",
    )
    assert route["provider"] == "gemini"
    assert route["profile"] == "gemini-pro"
    assert route["capability_requirements"]["url_context"] is True
    assert route["native_tool_plan"]["mode"] == "gemini-native"
    assert route["grounding_plan"] == "gemini-native"
    assert len(route["native_tool_plan"]["url_context_urls"]) == 2
    assert route["url_refs"][0].startswith("https://ai.google.dev/")


def test_router_prioritizes_local_paths_over_gemini_url_context(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    route = helix_cli.route_model_for_task(
        "compará src/helix_proto con https://ai.google.dev/gemini-api/docs/url-context y proponé un patch",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["provider"] == "deepinfra"
    assert route["profile"] == "code"
    assert route["native_tool_plan"]["mode"] == "helix-only"
    assert any("file.inspect" in item for item in route["why_not"])


def test_auto_router_can_use_current_legacy_blueprint() -> None:
    route = helix_cli.route_model_for_task(
        "investiga este benchmark raro y sintetizalo",
        provider_name="deepinfra",
        policy="current",
    )
    assert route["blueprint"] == "current"
    assert route["profile"] == "legacy-research"
    assert route["model"] == "MiniMaxAI/MiniMax-M2.5"


def test_auto_router_handles_model_control_for_mistral_and_sonnet() -> None:
    mistral = helix_cli.route_model_for_task(
        "tenes algun modelo de mistral para responderme?",
        provider_name="deepinfra",
        policy="balanced",
    )
    sonnet = helix_cli.route_model_for_task(
        "quiero que me responda claude sonnet",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert mistral["intent"] == "model_control"
    assert mistral["profile"] == "mistral"
    assert "mistralai/" in mistral["model"]
    assert sonnet["intent"] == "model_control"
    assert sonnet["profile"] == "sonnet"

    gemini = helix_cli.route_model_for_task(
        "quiero que me responda gemini pro",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert gemini["intent"] == "model_control"
    assert gemini["provider"] == "gemini"
    assert gemini["profile"] == "gemini-pro"
    assert gemini["model"] == helix_cli.GEMINI_MODEL_PROFILES["gemini-pro"].model_id


def test_auto_router_does_not_treat_model_research_as_model_control() -> None:
    route = helix_cli.route_model_for_task(
        "quiero info de modelos nuevos para agentes de codigo",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["intent"] in {"research", "agentic_code"}
    assert route["intent"] != "model_control"
    assert route["profile"] in {"qwen-big", "code"}
    assert route["intent_scores"]


def test_auto_router_selects_agentic_coding_for_codex_like_repo_work() -> None:
    route = helix_cli.route_model_for_task(
        "actua como codex, lee el repo, arregla el bug y proponeme el patch",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["intent"] == "agentic_code"
    assert route["profile"] == "code"
    assert "Coder" in route["model"]
    assert route["fallback_chain"] == ["devstral", "qwen-big", "chat"]


def test_model_alias_resolution() -> None:
    assert helix_cli.resolve_model_alias("mistral").startswith("mistralai/")
    assert helix_cli.resolve_model_alias("sonnet") == helix_cli.DEEPINFRA_MODEL_PROFILES["sonnet"].model_id
    assert helix_cli.resolve_model_alias("qwen") == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id
    assert helix_cli.resolve_model_alias("qwen-122b") == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-122b"].model_id
    assert helix_cli.resolve_model_alias("gemma") == helix_cli.DEEPINFRA_MODEL_PROFILES["gemma"].model_id
    assert helix_cli.resolve_model_alias("gemini-pro") == helix_cli.GEMINI_MODEL_PROFILES["gemini-pro"].model_id
    assert helix_cli.resolve_model_alias("gemini-pro-tools") == helix_cli.GEMINI_MODEL_PROFILES["gemini-pro-tools"].model_id
    assert helix_cli.resolve_model_alias("gemini flash") == helix_cli.GEMINI_MODEL_PROFILES["gemini-flash"].model_id
    assert helix_cli.resolve_model_alias("gemini-3.1-flash-lite-preview") == helix_cli.GEMINI_MODEL_PROFILES["gemini-lite"].model_id
    assert helix_cli.resolve_model_alias("gemini-2.5-pro") == "gemini-2.5-pro"
    assert helix_cli.resolve_model_alias("gemini 2.5 flash") == "gemini-2.5-flash"
    assert helix_cli.resolve_model_alias("gemini-2.5-flash-lite") == "gemini-2.5-flash-lite"
    assert helix_cli.resolve_model_alias("nvidia-code") == helix_cli.NVIDIA_MODEL_PROFILES["nvidia-code"].model_id
    assert helix_cli.resolve_model_alias("nvidia research") == helix_cli.NVIDIA_MODEL_PROFILES["nvidia-research"].model_id
    assert helix_cli.resolve_model_alias("magistral") == helix_cli.NVIDIA_MODEL_PROFILES["nvidia-chat"].model_id
    assert helix_cli.resolve_model_alias("gliner pii") == helix_cli.NVIDIA_MODEL_PROFILES["nvidia-pii"].model_id
    assert helix_cli.resolve_model_alias("llama-vision") == helix_cli.DEEPINFRA_MODEL_PROFILES["llama-vision"].model_id
    assert helix_cli.resolve_model_alias("auto") == "auto"


def test_models_payload_exposes_capabilities_and_provider_constraints() -> None:
    payload = helix_cli.models_payload()
    gemini_profile = next(item for item in payload["gemini_model_profiles"] if item["alias"] == "gemini-pro")
    gemini_provider = next(item for item in payload["providers"] if item["name"] == "gemini")
    nvidia_profile = next(item for item in payload["nvidia_model_profiles"] if item["alias"] == "nvidia-code")
    nvidia_provider = next(item for item in payload["providers"] if item["name"] == "nvidia")
    assert gemini_profile["supports_url_context"] is True
    assert "docs_synthesis" in gemini_profile["preferred_workloads"]
    assert "url_context" in gemini_provider["native_capabilities"]
    assert gemini_provider["native_constraints"]
    assert nvidia_profile["model_id"] == "qwen/qwen3-coder-480b-a35b-instruct"
    assert nvidia_provider["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert nvidia_provider["token_env"] == "NVIDIA_API_KEY"
    assert any("NVIDIA Build free-endpoint" in item for item in nvidia_provider["native_constraints"])
    assert any(item["name"] == "technical" for item in payload["interaction_modes"])


def test_router_blueprints_report_lists_current_and_hybrid_presets() -> None:
    blueprints = {item["name"]: item for item in helix_cli.router_blueprints_report()}
    assert "balanced" in blueprints
    assert "current" in blueprints
    assert "qwen-heavy" in blueprints
    assert "qwen-gemma-mistral" in blueprints
    assert "nvidia-build" in blueprints
    assert blueprints["balanced"]["reasoning_alias"] == "reasoning"
    assert blueprints["balanced"]["research_alias"] == "qwen-big"
    assert blueprints["qwen-heavy"]["default_alias"] == "qwen-big"
    assert blueprints["current"]["research_alias"] == "legacy-research"
    assert blueprints["nvidia-build"]["code_alias"] == "nvidia-code"


def test_rich_theme_registers_panel_style() -> None:
    if not helix_cli._HAS_UI:
        return
    from rich.console import Console
    from rich.panel import Panel

    console = Console(theme=helix_cli._rich_theme("cyberpunk"), width=80, record=True)
    console.print(Panel("ok", border_style="panel"))
    assert "ok" in console.export_text()


def test_spinner_messages_include_spanish_terminal_phrases() -> None:
    assert "pensando..." in helix_cli._THINKING_MESSAGES
    assert "conspirando con el DAG..." in helix_cli._THINKING_MESSAGES
    assert "mucho laburo..." in helix_cli._THINKING_MESSAGES
    assert "separando humo de señal..." in helix_cli._THINKING_MESSAGES
    assert "consultando al oráculo barato..." in helix_cli._THINKING_MESSAGES
    assert "abriendo tablero de misión..." in helix_cli._THINKING_MESSAGE_PHASES["task"]


def test_spanish_utf8_strings_round_trip() -> None:
    text = "niño, señal, español, oráculo, determinística"
    encoded = text.encode("utf-8")
    assert encoded.decode("utf-8") == text


def test_theme_aliases_include_command_center_options() -> None:
    assert "industrial-brutalist" in helix_cli._THEME_PALETTES
    assert "industrial-neon" in helix_cli._THEME_PALETTES
    assert "cyberpunk-gray" in helix_cli._THEME_PALETTES
    assert "brown-console" in helix_cli._THEME_PALETTES
    assert helix_cli.DEFAULT_THEME == "industrial-brutalist"


def test_theme_aliases_resolve_to_canonical_palettes() -> None:
    assert helix_cli._theme_palette("cyberpunk-gray")["theme_name"] == "industrial-neon"
    assert helix_cli._theme_palette("cyberpunk")["theme_name"] == "industrial-neon"
    assert helix_cli._theme_palette("brown")["theme_name"] == "brown-console"


def test_theme_list_command_prints_canonical_theme_report(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/theme list") is True
    output = capsys.readouterr().out
    assert "industrial-brutalist" in output
    assert "industrial-neon" in output


def test_model_use_command_persists_until_auto() -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/model use sonnet") is True
    assert session.model == helix_cli.DEEPINFRA_MODEL_PROFILES["sonnet"].model_id
    assert helix_cli._handle_interactive_command(session, "/model auto") is True
    assert session.model == "auto"


def test_model_use_bioinformatics_alias_maps_to_qwen_research_profile() -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/model use bioinformatics") is True
    assert session.provider_name == "deepinfra"
    assert session.model == helix_cli.DEEPINFRA_MODEL_PROFILES["bioinformatics"].model_id


def test_task_engine_command_sets_sticky_opencode(monkeypatch, capsys) -> None:
    saved: dict[str, str] = {}
    monkeypatch.setattr(helix_cli, "_load_config", lambda: {})
    monkeypatch.setattr(helix_cli, "_save_config", lambda config: saved.update(config) or Path("config.json"))
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/task engine opencode") is True
    assert session.task_engine == "opencode"
    assert saved["task_engine"] == "opencode"
    assert "task_engine=opencode" in capsys.readouterr().out


def test_task_engine_opencode_uses_rust_core_sandbox(monkeypatch) -> None:
    task_root = _test_root() / "repo"
    task_root.mkdir(parents=True)
    patch = (
        "diff --git a/src/example.py b/src/example.py\n"
        "new file mode 100644\n"
        "index 0000000..257cc56\n"
        "--- /dev/null\n"
        "+++ b/src/example.py\n"
        "@@ -0,0 +1 @@\n"
        "+value = 1\n"
    )

    def fake_route(text: str, **kwargs: object) -> dict[str, object]:
        return {"status": "ok", "path": "agentic", "routing_ms": 1.0, "rust_core_ms": 1.0}

    def fake_opencode_run(**kwargs: object) -> dict[str, object]:
        return {
            "status": "passed",
            "engine": "opencode",
            "run_id": "fake-run",
            "artifact_path": "verification/opencode-agent/fake-run/artifact.json",
            "patch_path": "verification/opencode-agent/fake-run/patch.diff",
            "sandbox_root": ".helix/opencode-runs/fake-run/worktree",
            "changed_files": ["src/example.py"],
            "patch": patch,
            "patch_sha256": "abc123",
            "trust_card_path": "verification/opencode-agent/fake-run/trust_card.json",
            "task_capsule_path": "verification/opencode-agent/fake-run/task_capsule.json",
            "trust_card": {
                "kind": "helix-trust-card-v1",
                "status": "passed",
                "engine": "opencode",
                "assurance": "quick",
                "run_id": "fake-run",
                "changed_files": ["src/example.py"],
                "checks_passed": [{"id": "sandbox_provenance", "status": "passed"}],
                "patch": {"sha256": "abc123"},
                "artifact_paths": {"artifact": "verification/opencode-agent/fake-run/artifact.json"},
                "claim_boundary": "local provenance only",
            },
            "opencode_trace": {"exit_code": 0, "latency_ms": 12.0},
            "rust_core_ms": 15.0,
        }

    monkeypatch.setattr(helix_cli.helix_cli_core, "route", fake_route)
    monkeypatch.setattr(helix_cli.helix_cli_core, "opencode_run", fake_opencode_run)
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "verify_capsule",
        lambda **kwargs: {"status": "passed", "checks": [{"id": "patch_integrity", "status": "passed"}]},
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=task_root,
    )
    result = session.task("implementa un fix", engine_override="opencode", assurance="balanced")
    assert result["engine"] == "opencode"
    assert result["patch_available"] is True
    assert session.last_patch == patch
    assert result["artifact_path"].endswith("artifact.json")
    assert result["assurance"] == "balanced"
    assert result["assurance_followup"]["effective"] == "balanced"
    assert result["trust_card"]["kind"] == "helix-trust-card-v1"
    assert session.last_trust_card["run_id"] == "fake-run"


def test_opencode_no_patch_is_partial_not_passed(monkeypatch) -> None:
    task_root = _test_root() / "repo"
    task_root.mkdir(parents=True)

    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "route",
        lambda text, **kwargs: {"status": "ok", "path": "agentic", "routing_ms": 1.0, "rust_core_ms": 1.0},
    )
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: {
            "status": "passed",
            "engine": "opencode",
            "run_id": "no-patch",
            "artifact_path": "verification/opencode-agent/no-patch/artifact.json",
            "patch_path": "verification/opencode-agent/no-patch/patch.diff",
            "trust_card_path": "verification/opencode-agent/no-patch/trust_card.json",
            "task_capsule_path": "verification/opencode-agent/no-patch/task_capsule.json",
            "changed_files": [],
            "patch": "",
            "patch_sha256": "empty",
            "opencode_trace": {"exit_code": 0, "latency_ms": 12.0},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=task_root,
    )

    result = session.task("revisa el routing y proponeme un patch", engine_override="opencode")

    assert result["status"] == "partial"
    assert result["patch_available"] is False
    assert "partial" in result["final"]


def test_core_wrapper_decodes_rust_output_as_utf8(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout='{"status":"ok","text":"caf\u00e9"}', stderr="")

    monkeypatch.setattr(helix_cli.helix_cli_core, "rust_core_binary", lambda: Path("fake-helix-cli-core.exe"))
    monkeypatch.setattr(helix_cli.helix_cli_core.subprocess, "run", fake_run)
    payload = helix_cli.helix_cli_core._run_core(["latency-report"])
    assert payload["status"] == "ok"
    assert payload["text"] == "café"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_flow_profiles_list_reports_commercial_modes(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/flow list") is True
    payload = json.loads(capsys.readouterr().out)
    profile_ids = {item["id"] for item in payload["profiles"]}
    assert {"web", "web-recursive", "patch-safe", "doc-grounded", "privacy-swarm"} <= profile_ids


def test_flow_run_web_recursive_wraps_opencode_task(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(helix_cli.helix_cli_core, "route", lambda *args, **kwargs: {"status": "ok", "path": "agentic", "routing_ms": 1.0})

    def fake_opencode_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "passed",
            "engine": "opencode",
            "run_id": "flow-run",
            "artifact_path": "verification/opencode-agent/flow-run/artifact.json",
            "patch_path": "verification/opencode-agent/flow-run/patch.diff",
            "sandbox_root": ".helix/opencode-runs/flow-run/worktree",
            "changed_files": ["web/helix-recursive-site/index.html"],
            "patch": "diff --git a/web/helix-recursive-site/index.html b/web/helix-recursive-site/index.html\n",
            "patch_sha256": "abc123",
            "trust_card": {
                "kind": "helix-trust-card-v1",
                "status": "passed",
                "engine": "opencode",
                "assurance": "balanced",
                "run_id": "flow-run",
                "changed_files": ["web/helix-recursive-site/index.html"],
                "checks_passed": [{"id": "patch_integrity", "status": "passed"}],
                "patch": {"sha256": "abc123"},
                "artifact_paths": {"artifact": "verification/opencode-agent/flow-run/artifact.json"},
            },
            "opencode_trace": {"exit_code": 0, "latency_ms": 12.0},
        }

    monkeypatch.setattr(helix_cli.helix_cli_core, "opencode_run", fake_opencode_run)
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "verify_capsule",
        lambda **kwargs: {"status": "passed", "checks": [{"id": "patch_integrity", "status": "passed"}]},
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=_test_root() / "repo",
    )
    assert helix_cli._handle_interactive_command(session, "/flow run web-recursive Crear sitio sobre HeliX") is True
    output = json.loads(capsys.readouterr().out)
    assert output["flow"]["id"] == "web-recursive"
    assert output["assurance"] == "balanced"
    assert output["trust_card"]["flow_profile"] == "web-recursive"
    assert "HeliX Flow Profile: web-recursive" in str(captured["goal"])
    assert "Crear sitio sobre HeliX" in str(captured["goal"])
    assert session.last_task_result["flow"]["id"] == "web-recursive"


def test_work_source_collector_extracts_text_file_anchors() -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    source = root / "paper.md"
    source.write_text("# Paper\n\nHeliX work runtime should preserve source anchors for generated output.\n", encoding="utf-8")
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    plan, sources = session._work_plan('analiza "paper.md" y armame un reporte en docs/report.md')

    assert plan["work_intent"] == "source_to_document"
    assert plan["flow_profile"] == "doc-grounded"
    assert plan["output_target"] == "docs/report.md"
    assert sources["sources"][0]["status"] == "ok"
    assert sources["anchors"][0]["source_ref"].endswith("paper.md")
    assert "HeliX work runtime" in sources["anchors"][0]["text"]


def test_work_path_refs_do_not_capture_leading_verbs() -> None:
    refs = helix_cli._extract_work_path_refs("analiza README.md y armame una pagina web en web/helix-work-demo/")
    assert "README.md" in refs
    assert "analiza README.md" not in refs
    assert "web/helix-work-demo" in refs


def test_work_run_source_to_web_uses_helix_first_and_records_metadata(monkeypatch) -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    (root / "brief.md").write_text("HeliX combines source anchors, sandbox patches, and trust cards.\n", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("doc/web work should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    result = session.work('analiza "brief.md" y armame una pagina web en web/demo/')

    assert result["mode"] == "work"
    assert result["engine"] == "helix-internal-generator"
    assert result["work_plan"]["flow_profile"] == "web-recursive"
    assert result["work_plan"]["needs_opencode"] is False
    assert result["trust_card"]["subject_type"] == "work"
    assert result["trust_card"]["sources"][0]["path"].endswith("brief.md")
    assert result["changed_files"] == ["web/demo/index.html"]
    assert result["patch_available"] is True
    assert result["trust_card"]["checks_passed"][5]["id"] == "apply_check"
    assert result["trust_card"]["checks_passed"][5]["status"] == "passed"
    assert Path(result["work_artifact_paths"]["plan"]).exists()
    assert Path(result["work_artifact_paths"]["sources"]).exists()
    assert session.last_work_result["run_id"] == result["run_id"]


def test_work_analysis_only_does_not_call_opencode(monkeypatch) -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    (root / "notes.md").write_text("HeliX can answer from inspected notes without generating files.\n", encoding="utf-8")

    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("opencode should not run for analysis-only work")),
    )
    monkeypatch.setattr(
        helix_cli,
        "run_chat_with_failover",
        lambda **kwargs: {"text": "<helix_output>Respuesta grounded.</helix_output>", "latency_ms": 1.0},
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    result = session.work('analiza "notes.md"')

    assert result["status"] == "completed"
    assert result["engine"] == "helix-planner"
    assert result["final"] == "Respuesta grounded."
    assert result["trust_card"]["checks_passed"][0]["id"] == "source_collection"


def test_last_work_persists_across_session_restart(monkeypatch) -> None:
    root = _test_root() / "repo"
    workspace = _test_root() / "workspace"
    root.mkdir(parents=True)
    (root / "README.md").write_text("Persistent work results should survive shell restarts.\n", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("doc/web work should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )
    first = session.work("analiza README.md y armame una pagina web en web/persisted/")

    restarted = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    assert restarted.last_work_result["run_id"] == first["run_id"]
    assert restarted.last_patch and "web/persisted/index.html" in restarted.last_patch
    assert restarted.work_history(limit=1)[0]["run_id"] == first["run_id"]


def test_work_pdf_export_writes_and_verifies_output(monkeypatch) -> None:
    root = _test_root() / "repo"
    output_dir = _test_root() / "Desktop"
    root.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pdf export should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    result = session.work(f'armame un pdf sobre postestructuralismo y dejalo en "{output_dir}"')

    output_file = Path(result["output_file"]["path"])
    assert result["status"] == "completed"
    assert result["engine"] == "helix-internal-exporter"
    assert output_file.exists()
    assert output_file.name == "postestructuralismo.pdf"
    payload = output_file.read_bytes()
    assert payload.startswith(b"%PDF-")
    decoded = payload.decode("latin-1", errors="replace")
    assert "Derrida" in decoded
    assert "Foucault" in decoded
    assert "Deleuze" in decoded
    assert not result["work_plan"]["source_refs"]
    assert result["trust_card"]["output_file"]["sha256"] == result["output_file"]["sha256"]
    assert any(check["id"] == "output_exists" and check["status"] == "passed" for check in result["trust_card"]["checks_passed"])
    assert any(check["id"] == "artifact_readback_after" and check["status"] == "passed" for check in result["trust_card"]["checks_passed"])
    assert result["artifact"]["readback"]["chars"] >= 1200
    assert result["artifact"]["readback"]["pages"] >= 1
    assert result["patch_available"] is False


def test_modify_last_pdf_reuses_previous_output_and_curates_content(monkeypatch) -> None:
    root = _test_root() / "repo"
    output_dir = _test_root() / "Desktop"
    root.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pdf export should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    first = session.work(f'armame un pdf sobre postestructuralismo y dejalo en "{output_dir}"')
    second = session.work("modifica ese y agregale realmente un contenido curado")

    assert second["status"] == "completed"
    assert second["output_file"]["path"] == first["output_file"]["path"]
    decoded = Path(second["output_file"]["path"]).read_bytes().decode("latin-1", errors="replace")
    assert "Postestructuralismo" in decoded
    assert "deconstruccion" in decoded
    assert second["work_plan"]["output_target"] == first["output_file"]["path"]
    assert second["artifact_before"]["readback"]["chars"] > 0
    assert second["artifact"]["readback"]["chars"] >= 2200
    assert second["artifact"]["sha256"] != second["artifact_before"]["sha256"]
    assert any(check["id"] == "artifact_readback_before" and check["status"] == "passed" for check in second["trust_card"]["checks_passed"])


def test_read_last_artifact_reports_pdf_pages_chars_preview(monkeypatch, capsys) -> None:
    root = _test_root() / "repo"
    output_dir = _test_root() / "Desktop"
    root.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(helix_cli.helix_cli_core, "opencode_run", lambda **kwargs: {})
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )
    session.work(f'armame un pdf sobre postestructuralismo y dejalo en "{output_dir}"')

    assert helix_cli._handle_interactive_command(session, "/read last") is True

    output = capsys.readouterr().out
    assert "HeliX Artifact" in output
    assert "kind: pdf" in output
    assert "chars=" in output
    assert "Postestructuralismo" in output
    assert session.last_artifact["artifact_state"]["artifact_kind"] == "pdf"
    assert session.last_artifact["artifact_state"]["readback_chars"] > 0


def test_pdf_writer_supports_multipage_without_truncating_sections() -> None:
    target = _test_root() / "multi" / "long.pdf"
    body = "\n".join(f"Linea {index}: contenido suficiente para ocupar varias paginas." for index in range(140))

    written = helix_cli._write_simple_pdf(target, title="Documento largo", body=body)
    text, pages, warnings = helix_cli._extract_pdf_text_with_optional_ocr(target)

    assert written["bytes"] > 0
    assert len(pages) >= 3
    assert "Linea 139" in text
    assert not warnings or all("OCR unavailable" not in item for item in warnings)


def test_web_artifact_inspector_reads_index_css_js_structure() -> None:
    root = _test_root() / "repo"
    web = root / "web" / "demo"
    web.mkdir(parents=True)
    (web / "index.html").write_text("<html><body><h1>HeliX Demo</h1><p>Artifact workbench.</p><script src='app.js'></script></body></html>", encoding="utf-8")
    (web / "style.css").write_text("body { color: black; }", encoding="utf-8")
    (web / "app.js").write_text("console.log('helix');", encoding="utf-8")
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    artifact = helix_cli._inspect_work_artifact(session, "web/demo", last_action="inspect")

    assert artifact["kind"] == "web_directory"
    assert artifact["readback"]["chars"] > 0
    assert any(path.endswith("index.html") for path in artifact["web_files"])


def test_read_path_describes_markdown_docx_html_and_pdf() -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    (root / "note.md").write_text("# Nota\n\nContenido de prueba para HeliX artifact workbench.", encoding="utf-8")
    (root / "page.html").write_text("<html><body><h1>Pagina</h1><p>Contenido HTML.</p></body></html>", encoding="utf-8")
    helix_cli._write_simple_docx(root / "doc.docx", title="Documento", body="Texto DOCX verificable.")
    helix_cli._write_simple_pdf(root / "doc.pdf", title="Documento PDF", body="Texto PDF verificable.\n" * 100)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    md = helix_cli._inspect_work_artifact(session, "note.md")
    docx = helix_cli._inspect_work_artifact(session, "doc.docx")
    html = helix_cli._inspect_work_artifact(session, "page.html")
    pdf = helix_cli._inspect_work_artifact(session, "doc.pdf")

    assert md["kind"] == "markdown" and md["readback"]["chars"] > 0
    assert docx["kind"] == "docx" and docx["readback"]["chars"] > 0
    assert html["kind"] == "html" and html["readback"]["chars"] > 0
    assert pdf["kind"] == "pdf" and pdf["readback"]["chars"] > 0


def test_work_source_to_web_falls_back_when_opencode_returns_no_patch(monkeypatch) -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    (root / "README.md").write_text("HeliX turns source anchors into reviewable work artifacts.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli.helix_cli_core, "route", lambda *args, **kwargs: {"status": "ok", "path": "agentic", "routing_ms": 1.0})
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: {
            "status": "passed",
            "engine": "opencode",
            "run_id": "no-patch-run",
            "artifact_path": str(root / "verification" / "opencode-agent" / "no-patch-run" / "artifact.json"),
            "patch_path": str(root / "verification" / "opencode-agent" / "no-patch-run" / "patch.diff"),
            "trust_card_path": str(root / "verification" / "opencode-agent" / "no-patch-run" / "trust_card.json"),
            "changed_files": [],
            "patch": "",
            "opencode_trace": {"exit_code": 0, "latency_ms": 12.0},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )

    result = session.work("analiza README.md y armame una pagina web en web/helix-work-demo/", engine_override="opencode")

    assert result["fallback_used"] is True
    assert result["engine"] == "helix-internal-generator"
    assert "error" not in result
    assert result["changed_files"] == ["web/helix-work-demo/index.html"]
    assert result["patch_available"] is True
    assert session.last_patch and "web/helix-work-demo/index.html" in session.last_patch
    assert session.last_patch_sha256 == result["patch_sha256"]
    assert Path(result["work_artifact_paths"]["patch"]).exists()
    assert result["trust_card"]["checks_passed"][3]["id"] == "helix_generation"
    assert result["trust_card"]["checks_passed"][3]["status"] == "passed"


def test_trust_last_renders_compact_human_card(monkeypatch, capsys) -> None:
    monkeypatch.setattr(helix_cli.helix_cli_core, "route", lambda *args, **kwargs: {"status": "ok", "path": "agentic", "routing_ms": 1.0})
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: {
            "status": "passed",
            "engine": "opencode",
            "run_id": "fake-run",
            "artifact_path": "verification/opencode-agent/fake-run/artifact.json",
            "patch_path": "verification/opencode-agent/fake-run/patch.diff",
            "trust_card_path": "verification/opencode-agent/fake-run/trust_card.json",
            "sandbox_root": ".helix/opencode-runs/fake-run/worktree",
            "changed_files": ["src/example.py"],
            "patch": "diff --git a/src/example.py b/src/example.py\n",
            "patch_sha256": "abc123",
            "trust_card": {
                "kind": "helix-trust-card-v1",
                "status": "passed",
                "engine": "opencode",
                "assurance": "quick",
                "run_id": "fake-run",
                "changed_files": ["src/example.py"],
                "checks_passed": [{"id": "patch_integrity", "status": "passed"}],
                "patch": {"sha256": "abc123"},
                "artifact_paths": {"artifact": "verification/opencode-agent/fake-run/artifact.json"},
                "claim_boundary": "local provenance only",
            },
            "opencode_trace": {"exit_code": 0, "latency_ms": 12.0},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=_test_root() / "repo",
    )
    result = session.task("implementa un fix", engine_override="opencode")
    assert result["trust_card"]["run_id"] == "fake-run"
    assert helix_cli._handle_interactive_command(session, "/trust last") is True
    output = capsys.readouterr().out
    assert "HeliX Trust Card" in output
    assert "fake-run" in output
    assert "public_key" not in output


def test_work_aliases_last_trust_and_open_are_human(monkeypatch, capsys) -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    (root / "README.md").write_text("Alias commands should explain the last generated work.\n", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("doc/web work should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=root,
    )
    session.work("analiza README.md y armame una pagina web en web/alias-demo/")

    assert helix_cli._handle_interactive_command(session, "/last") is True
    last_output = capsys.readouterr().out
    assert "HeliX Work Result" in last_output
    assert "apply ready" in last_output

    assert helix_cli._handle_interactive_command(session, "/trust") is True
    trust_output = capsys.readouterr().out
    assert "HeliX Trust Card" in trust_output
    assert "public_key" not in trust_output

    assert helix_cli._handle_interactive_command(session, "/open last") is True
    open_output = capsys.readouterr().out
    assert "web\\alias-demo\\index.html" in open_output or "web/alias-demo/index.html" in open_output


def test_verify_last_uses_task_capsule_artifact(monkeypatch, capsys) -> None:
    captured: dict[str, Path] = {}

    def fake_verify_capsule(*, artifact_path: Path, **kwargs: object) -> dict[str, object]:
        captured["artifact_path"] = artifact_path
        return {
            "status": "passed",
            "kind": "helix-capsule-verification-v1",
            "trust_card": {"kind": "helix-trust-card-v1", "run_id": "fake-run", "status": "passed"},
        }

    monkeypatch.setattr(helix_cli.helix_cli_core, "verify_capsule", fake_verify_capsule)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    session.last_task_result = {"artifact_path": "verification/opencode-agent/fake-run/artifact.json"}
    assert helix_cli._handle_interactive_command(session, "/verify last --level quick") is True
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "helix-capsule-verification-v1"
    assert captured["artifact_path"].as_posix().endswith("artifact.json")
    assert session.last_trust_card["run_id"] == "fake-run"


def test_lab_profiles_and_run_are_cli_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "lab_profiles",
        lambda: {"status": "ok", "profiles": [{"id": "patch-safety"}]},
    )
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "lab_run",
        lambda **kwargs: {"status": "ok", "profile": kwargs["profile"], "checks": [{"id": "rust_core", "status": "passed"}]},
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/lab profiles") is True
    profiles = json.loads(capsys.readouterr().out)
    assert profiles["profiles"][0]["id"] == "patch-safety"
    assert helix_cli._handle_interactive_command(session, "/lab run patch-safety") is True
    run = json.loads(capsys.readouterr().out)
    assert run["profile"] == "patch-safety"


def test_models_compare_last_reports_task_roles(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    session.last_trust_card = {
        "kind": "helix-trust-card-v1",
        "models_used": {
            "planner_model": "planner-x",
            "coder_engine": "opencode",
            "critic_model": None,
            "verifier_model": "helix-rust-core",
        },
    }
    assert helix_cli._handle_interactive_command(session, "/models compare last") is True
    output = json.loads(capsys.readouterr().out)
    assert output["models_used"]["coder_engine"] == "opencode"
    assert "balanced" in output["next"]


def test_model_use_gemini_switches_provider_and_model(monkeypatch) -> None:
    monkeypatch.setattr(helix_cli, "_ensure_provider_token", lambda provider_name: None)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/model use gemini-pro") is True
    assert session.provider_name == "gemini"
    assert session.model == helix_cli.GEMINI_MODEL_PROFILES["gemini-pro"].model_id


def test_model_use_nvidia_switches_provider_and_model(monkeypatch) -> None:
    monkeypatch.setattr(helix_cli, "_ensure_provider_token", lambda provider_name: None)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/model use nvidia-code") is True
    assert session.provider_name == "nvidia"
    assert session.model == helix_cli.NVIDIA_MODEL_PROFILES["nvidia-code"].model_id


def test_missing_model_alias_error_falls_back_to_research_profile(monkeypatch, capsys) -> None:
    monkeypatch.setattr(helix_cli, "console", None)
    session = helix_cli.InteractiveSession(
        provider_name="local",
        model="bioinformatics",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    try:
        try:
            raise FileNotFoundError("model alias not found: bioinformatics")
        except FileNotFoundError as inner:
            raise RuntimeError(
                "all model attempts failed (local:bioinformatics: FileNotFoundError): model alias not found: bioinformatics"
            ) from inner
    except RuntimeError as exc:
        assert helix_cli._recover_missing_model_alias(session, exc) is True

    assert session.provider_name == "deepinfra"
    assert session.model == helix_cli.DEEPINFRA_MODEL_PROFILES["research"].model_id
    output = capsys.readouterr().out
    assert "Alias not found ('bioinformatics')" in output
    assert "default 'research' profile" in output


def test_key_save_accepts_explicit_gemini_provider(monkeypatch, capsys) -> None:
    saved = {}
    monkeypatch.setattr(helix_cli.getpass, "getpass", lambda prompt: "gemini-key")
    monkeypatch.setattr(
        helix_cli,
        "_save_config_token",
        lambda provider_name, token: saved.update({"provider": provider_name, "token": token}) or Path("config.json"),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/key save gemini") is True
    assert saved == {"provider": "gemini", "token": "gemini-key"}
    assert "token saved" in capsys.readouterr().out


def test_optional_gemini_token_prompt_can_save_key(monkeypatch, capsys) -> None:
    saved = {}
    configs = [{"tokens": {}}, {"tokens": {"gemini": "gemini-key"}}]
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(helix_cli, "_load_config", lambda: configs[-1])
    monkeypatch.setattr(helix_cli, "_config_token", lambda provider_name: None)
    monkeypatch.setattr(helix_cli, "_save_config_token", lambda provider_name, token: saved.update({"provider": provider_name, "token": token}) or Path("config.json"))
    monkeypatch.setattr(helix_cli.getpass, "getpass", lambda prompt: "gemini-key")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    updated = helix_cli._maybe_prompt_optional_provider_token("gemini", config=configs[0])
    assert saved == {"provider": "gemini", "token": "gemini-key"}
    assert updated["tokens"]["gemini"] == "gemini-key"
    assert "GEMINI_API_KEY saved" in capsys.readouterr().out


def test_optional_nvidia_token_prompt_can_save_key(monkeypatch, capsys) -> None:
    saved = {}
    configs = [{"tokens": {}}, {"tokens": {"nvidia": "nvidia-key"}}]
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(helix_cli, "_load_config", lambda: configs[-1])
    monkeypatch.setattr(helix_cli, "_config_token", lambda provider_name: None)
    monkeypatch.setattr(
        helix_cli,
        "_save_config_token",
        lambda provider_name, token: saved.update({"provider": provider_name, "token": token}) or Path("config.json"),
    )
    monkeypatch.setattr(helix_cli.getpass, "getpass", lambda prompt: "nvidia-key")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    updated = helix_cli._maybe_prompt_optional_provider_token("nvidia", config=configs[0])
    assert saved == {"provider": "nvidia", "token": "nvidia-key"}
    assert updated["tokens"]["nvidia"] == "nvidia-key"
    output = capsys.readouterr().out
    assert "NVIDIA_API_KEY saved" in output


def test_optional_gemini_token_prompt_can_be_skipped(monkeypatch, capsys) -> None:
    saved_config = {}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(helix_cli, "_config_token", lambda provider_name: None)
    monkeypatch.setattr(helix_cli, "_save_config", lambda config: saved_config.update(config) or Path("config.json"))
    monkeypatch.setattr("builtins.input", lambda prompt: "skip")

    updated = helix_cli._maybe_prompt_optional_provider_token("gemini", config={})
    assert updated["optional_token_prompts"]["gemini"] == "skip"
    assert saved_config["optional_token_prompts"]["gemini"] == "skip"
    assert "prompt disabled" in capsys.readouterr().out


def test_router_why_command_prints_scored_route(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    session.interaction_mode = "technical"
    assert helix_cli._handle_interactive_command(session, "/router why investiga modelos nuevos para codigo") is True
    output = capsys.readouterr().out
    assert '"intent_scores"' in output
    assert '"fallback_chain"' in output
    assert '"capability_requirements"' in output
    assert '"native_tool_plan"' in output
    assert '"interaction_mode": "technical"' in output
    assert '"grounding_plan"' in output


def test_extract_local_path_refs_supports_relative_directories_and_quoted_spaces() -> None:
    base = _test_root()
    folder = base / "space folder"
    folder.mkdir(parents=True)
    quoted_file = folder / "notes.txt"
    quoted_file.write_text("hola", encoding="utf-8")
    relative_with_spaces = str(quoted_file.relative_to(Path.cwd()))

    refs = helix_cli._extract_local_path_refs(f'lee "{relative_with_spaces}" y también src/helix_proto')

    assert relative_with_spaces in refs
    assert "src/helix_proto" in refs


def test_route_natural_language_reads_local_path_without_slash_command() -> None:
    routed = helix_cli._route_natural_language("lee src/helix_proto y resumilo")
    assert routed == "/work run lee src/helix_proto y resumilo"


def test_route_natural_language_pdf_output_routes_to_work_runtime() -> None:
    routed = helix_cli._route_natural_language('analiza "paper.pdf" y armame una pagina web en web/paper-demo/')
    assert routed == '/work run analiza "paper.pdf" y armame una pagina web en web/paper-demo/'


def test_interactive_helix_work_run_prefix_routes_to_work_runtime() -> None:
    routed = helix_cli._route_natural_language('helix work run "analiza README.md y armame una pagina web en web/demo/"')
    assert routed == '/work run "analiza README.md y armame una pagina web en web/demo/"'


def test_natural_followup_about_last_work_routes_to_work_last() -> None:
    assert helix_cli._route_natural_language("que hizo?") == "/work last"
    assert helix_cli._route_natural_language("donde quedo la ultima tarea?") == "/work last"


def test_workbench_router_exposes_lanes_and_followups() -> None:
    chat = helix_cli._classify_workbench_prompt("pensemos filosoficamente sobre HeliX")
    doc = helix_cli._classify_workbench_prompt("analiza README.md y armame un reporte")
    code = helix_cli._classify_workbench_prompt("revisa el routing y proponeme un patch chico")

    assert chat["lane"] == "conversation"
    assert doc["lane"] == "work_doc"
    assert doc["engine_selected"] == "helix"
    assert code["lane"] == "code_patch"
    assert code["engine_selected"] == "opencode"
    assert helix_cli._route_natural_language("aplicalo") == "/apply last"
    assert helix_cli._route_natural_language("abrilo") == "/open last"


def test_router_pdf_creation_routes_to_work_doc() -> None:
    route = helix_cli._classify_workbench_prompt("podes armarme un pdf sobre postestructuralismo?")

    assert route["lane"] == "work_doc"
    assert route["engine_selected"] == "helix"
    assert route["command"] == "/work run podes armarme un pdf sobre postestructuralismo?"
    assert helix_cli._route_natural_language("podes armarme un pdf sobre postestructuralismo?").startswith("/work run ")


def test_router_modify_last_output_routes_to_work_runtime() -> None:
    route = helix_cli._classify_workbench_prompt("modifica ese y agregale realmente un contenido curado")

    assert route["lane"] == "work_doc"
    assert route["route_reason"] == "modify_last_work"
    assert route["engine_selected"] == "helix"
    assert helix_cli._route_natural_language("modifica ese y agregale realmente un contenido curado").startswith("/work run ")
    assert helix_cli._route_natural_language("pero armalo bien porque el que esta no tiene info fijate").startswith("/work run ")
    assert helix_cli._route_natural_language("pero tiene que tener mejor contenido ademas de que tenga un gran formato").startswith("/work run ")
    assert helix_cli._route_natural_language("dije sobre land y ccru").startswith("/work run ")
    assert helix_cli._route_natural_language("nono nick land").startswith("/work run ")
    assert helix_cli._route_natural_language("que hay ahi?") == "/read last"
    assert helix_cli._route_natural_language("a ver") == "/read last"
    assert helix_cli._route_natural_language("mostrame eso") == "/read last"
    assert helix_cli._looks_like_work_confirmation("dale") is True


def test_router_noise_input_does_not_call_chat() -> None:
    route = helix_cli._classify_workbench_prompt("}")

    assert route["lane"] == "noop"
    assert route["route_reason"] == "noise_input"
    assert route["command"] == "/noop"
    assert helix_cli._route_natural_language("}") == "/noop"


def test_router_incomplete_document_request_asks_clarification() -> None:
    route = helix_cli._classify_workbench_prompt("quiero armar un documento de texto")

    assert route["lane"] == "clarify"
    assert route["route_reason"] == "incomplete_work_request"
    assert route["command"] == "/clarify work"
    assert helix_cli._route_natural_language("quiero armar un documento de texto") == "/clarify work"


def test_router_entonces_after_work_routes_last_work() -> None:
    assert helix_cli._route_natural_language("entonces?") == "/work last"
    route = helix_cli._classify_workbench_prompt("entonces?")
    assert route["lane"] == "work_status"
    assert route["route_reason"] == "last_work_followup"


def test_work_slug_preserves_compound_topic_with_y() -> None:
    assert helix_cli._work_slug_from_goal("quiero que me armes un buen pdf sobre nick land y el ccru") == "nick-land-ccru"
    pasted = (
        "quiero que me armes un buen pdf sobre nick land y el ccru\n"
        "pero tiene que tener mejor contenido ademas de que tenga un gran formato\n"
        "dije sobre land y ccru\n"
        "nono nick land"
    )
    assert helix_cli._work_slug_from_goal(pasted) == "nick-land-ccru"


def test_turn_plan_explains_work_chat_and_artifact_lanes() -> None:
    work_route = helix_cli._classify_workbench_prompt("armame un pdf sobre nick land y el ccru")
    read_route = helix_cli._classify_workbench_prompt("a ver")
    chat_route = helix_cli._classify_workbench_prompt("pensemos filosoficamente")

    work_plan = helix_cli._turn_plan_for_route("armame un pdf sobre nick land y el ccru", work_route)
    read_plan = helix_cli._turn_plan_for_route("a ver", read_route)
    chat_plan = helix_cli._turn_plan_for_route("pensemos filosoficamente", chat_route)

    assert work_plan["mode"] == "work"
    assert any("releer output" in step for step in work_plan["steps"])
    assert read_plan["mode"] == "artifact"
    assert read_plan["command"] == "/read last"
    assert chat_plan["mode"] == "chat"
    assert any("sin herramientas" in step for step in chat_plan["steps"])


def test_turn_controller_routes_chat_without_tools() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=workspace / "repo",
    )

    card = session.turn_controller("pensemos filosoficamente sobre hauntologia")

    assert card.lane == "conversation"
    assert card.requires_write is False
    assert card.requires_opencode is False
    assert card.fallback_command is None


def test_turn_controller_routes_pdf_generation_to_doc_generate() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=workspace / "repo",
    )

    card = session.turn_controller("quiero que me armes un buen pdf sobre Nick Land y el CCRU")

    assert card.lane == "doc_generate"
    assert card.requires_write is True
    assert card.requires_readback is True
    assert card.output_target.endswith(".pdf")
    assert card.fallback_command.startswith("/work run ")


def test_turn_controller_routes_a_ver_to_artifact_read() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=workspace / "repo",
    )
    session.last_artifact = {"path": str(workspace / "repo" / "docs" / "x.pdf")}

    card = session.turn_controller("a ver")

    assert card.lane == "artifact_read"
    assert card.requires_readback is True
    assert card.fallback_command == "/read last"


def test_turn_controller_routes_correction_to_artifact_modify() -> None:
    workspace = _test_root() / "workspace"
    artifact_path = workspace / "repo" / "docs" / "nick-land.pdf"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=workspace / "repo",
    )
    session.last_artifact = {"path": str(artifact_path)}

    card = session.turn_controller("nono nick land, agregale contenido curado")

    assert card.lane == "artifact_modify"
    assert card.requires_write is True
    assert card.output_target == str(artifact_path)
    assert str(artifact_path) in card.sources


def test_work_events_record_all_required_phases(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# Demo\n\nHeliX Workbench source text.\n", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("doc/web work should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
    )

    result = session.work("analiza README.md y armame una pagina web en web/demo")

    events = [item["event"] for item in result["progress_events"]]
    assert events[:3] == ["intent.detected", "source.collecting", "source.read"]
    assert "writer.started" in events
    assert "verify.apply_check" in events
    assert "trust.updated" in events
    assert result["latency_trace"]["work_event_count"] == len(result["progress_events"])
    assert "work_phase_ms" in result["latency_trace"]


def test_last_renders_progress_timeline(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# Demo\n\nHeliX Workbench source text.\n", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("doc/web work should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
    )
    session.work("analiza README.md y armame una pagina web en web/demo")

    assert helix_cli._handle_interactive_command(session, "/last") is True
    output = capsys.readouterr().out
    assert "- timeline:" in output
    assert "intent.detected" in output


def test_demo_wow_runs_fast_path_without_model_for_noise_and_status(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# HeliX\n\nFast workbench demo source.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli, "_git_apply_check", lambda task_root, patch: {"status": "passed", "ok": True})
    monkeypatch.setattr(helix_cli, "_demo_browser_verify", lambda site_index, run_dir, enabled=True: {"status": "skipped", "reason": "test"})
    monkeypatch.setattr(
        helix_cli,
        "run_chat_with_failover",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("demo wow fast path should not call a model")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
        evidence_root=repo / "verification",
    )

    result = session.demo_wow(browser=False)

    assert result["kind"] == "helix-demo-wow-run-v1"
    assert result["status"] == "passed"
    assert any(item["event"] == "intent.detected" for item in result["progress_events"])
    assert result["trust_card"]["engine"] == "helix-demo-orchestrator"


def test_demo_wow_generates_doc_web_artifact_and_readback(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# HeliX\n\nHeliX routes chat, work, code and trust with source anchors.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli, "_git_apply_check", lambda task_root, patch: {"status": "passed", "ok": True})
    monkeypatch.setattr(helix_cli, "_demo_browser_verify", lambda site_index, run_dir, enabled=True: {"status": "skipped", "reason": "test"})
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
        evidence_root=repo / "verification",
    )

    result = session.demo_wow(browser=False)
    paths = result["work_artifact_paths"]

    assert Path(paths["demo_run"]).exists()
    assert Path(paths["timeline"]).exists()
    assert Path(paths["trust_card"]).exists()
    assert Path(paths["site"]).exists()
    assert result["artifacts"]["site"]["exists"] is True
    assert result["artifacts"]["site"]["readback"]["chars"] > 600
    assert result["patch_available"] is True
    assert session.last_patch and "web/helix-wow-demo/index.html" in session.last_patch


def test_demo_wow_records_progress_events_and_claim_boundaries(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# HeliX\n\nTrust cards, anchors and patch gates.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli, "_git_apply_check", lambda task_root, patch: {"status": "passed", "ok": True})
    monkeypatch.setattr(helix_cli, "_demo_browser_verify", lambda site_index, run_dir, enabled=True: {"status": "skipped", "reason": "test"})
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
        evidence_root=repo / "verification",
    )

    result = session.demo_wow(browser=False)
    events = [item["event"] for item in result["progress_events"]]

    assert "demo.start" in events
    assert "source.read" in events
    assert "browser.verify" in events
    assert "trust.updated" in events
    assert events[-1] == "done"
    assert all(item.get("claim_boundary") for item in result["scenarios"])
    assert result["latency_trace"]["work_event_count"] == len(result["progress_events"])


def test_demo_wow_browser_verification_skips_cleanly_when_missing(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# HeliX\n\nBrowser optional.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli, "_git_apply_check", lambda task_root, patch: {"status": "passed", "ok": True})
    monkeypatch.setattr(helix_cli, "_agent_browser_status", lambda: {"available": False, "binary": None})
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
        evidence_root=repo / "verification",
    )

    result = session.demo_wow(browser=True)

    assert result["browser_verification"]["status"] == "skipped"
    assert "agent-browser" in result["browser_verification"]["reason"]
    assert Path(result["browser_verification"]["snapshot_path"]).exists()


def test_demo_wow_browser_verification_records_snapshot_when_available(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# HeliX\n\nBrowser proof.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli, "_git_apply_check", lambda task_root, patch: {"status": "passed", "ok": True})

    def fake_browser(site_index, run_dir, enabled=True):
        snapshot = run_dir / "browser_snapshot.txt"
        screenshot = run_dir / "screenshot.png"
        snapshot.write_text("body: HeliX Wow Demo", encoding="utf-8")
        screenshot.write_bytes(b"png")
        return {"status": "passed", "snapshot_path": str(snapshot), "screenshot_path": str(screenshot)}

    monkeypatch.setattr(helix_cli, "_demo_browser_verify", fake_browser)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
        evidence_root=repo / "verification",
    )

    result = session.demo_wow(browser=True)

    assert result["browser_verification"]["status"] == "passed"
    assert Path(result["browser_verification"]["snapshot_path"]).read_text(encoding="utf-8")
    assert any(check["id"] == "browser_verification" and check["status"] == "passed" for check in result["trust_card"]["checks_passed"])


def test_preflight_compact_hides_verbose_panel_for_chat(monkeypatch, capsys) -> None:
    monkeypatch.setattr(helix_cli, "console", None)
    route = helix_cli._classify_workbench_prompt("hola")
    card = helix_cli.IntentCard(
        lane="conversation",
        primary_goal="hola",
        correction_notes=[],
        sources=[],
        urls=[],
        output_target=None,
        requires_write=False,
        requires_model=True,
        requires_opencode=False,
        requires_readback=False,
        route_reason="default_conversation",
        confidence=0.74,
        fallback_command=None,
        route=route,
    )

    helix_cli._show_turn_plan("hola", route, intent_card=card, mode="compact")
    output = capsys.readouterr().out

    assert "TURN PREFLIGHT" not in output
    assert "chat listo" in output


def test_last_after_demo_shows_timeline_artifact_and_next_action(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# HeliX\n\nTimeline demo.\n", encoding="utf-8")
    monkeypatch.setattr(helix_cli, "_git_apply_check", lambda task_root, patch: {"status": "passed", "ok": True})
    monkeypatch.setattr(helix_cli, "_demo_browser_verify", lambda site_index, run_dir, enabled=True: {"status": "skipped", "reason": "test"})
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
        evidence_root=repo / "verification",
    )
    session.demo_wow(browser=False)

    assert helix_cli._handle_interactive_command(session, "/last") is True
    output = capsys.readouterr().out

    assert "HeliX Work Result" in output
    assert "- timeline:" in output
    assert "demo.start" in output
    assert "web/helix-wow-demo/index.html" in output


def test_demo_doctor_reports_rust_core_opencode_agent_browser_and_skills() -> None:
    report = helix_cli.demo_doctor_report()

    assert report["kind"] == "helix-demo-doctor-v1"
    assert "rust_core" in report
    assert "opencode" in report
    assert "agent_browser" in report
    assert "skills" in report


def test_prewrite_hook_can_block_external_write() -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
    )

    def block_hook(event_name, payload):
        if event_name == "PreWrite":
            return {"action": "block", "message": "blocked by test hook"}
        return {}

    session.internal_hooks.append(block_hook)
    result = session.work("armame un pdf sobre Nick Land en docs/nick-land.pdf")

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "blocked by test hook"
    assert any(item["event"] == "blocked" for item in result["progress_events"])


def test_hook_failure_becomes_warning_not_crash(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# Demo\n\nHeliX Workbench source text.\n", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli.helix_cli_core,
        "opencode_run",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("doc/web work should not call opencode by default")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=repo,
    )

    def bad_hook(event_name, payload):
        if event_name == "PreWrite":
            raise RuntimeError("boom")
        return {}

    session.internal_hooks.append(bad_hook)
    result = session.work("analiza README.md y armame una pagina web en web/demo")

    assert result["status"] in {"completed", "failed"}
    assert any(item["event"] == "hook.warning" and "boom" in item["message"] for item in result["progress_events"])


def test_chat_cannot_claim_file_creation() -> None:
    gate = helix_cli._conversation_gate(
        "He creado un PDF detallado y lo he guardado en C:\\Users\\Big Duck\\Desktop\\x.pdf."
    )

    assert gate["blocked_file_promise"] is True
    assert "requiere una tarea de Work Runtime" in gate["visible_text"]


def test_web_search_request_routes_to_web_research() -> None:
    route = helix_cli.route_model_for_task(
        "buscame en la web benchmarks actuales de modelos de codigo",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["intent"] == "web_research"
    assert route["profile"] == "qwen-big"
    assert "web_research" in route["signals"]


def test_with_command_uses_one_model_then_restores_auto(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["model"] = model
        return {
            "text": "<helix_output>Respuesta con Gemma.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 8},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/with gemma razona esta hipotesis") is True
    assert captured["model"] == helix_cli.DEEPINFRA_MODEL_PROFILES["gemma"].model_id
    assert session.model == "auto"
    assert "provider/model restored to deepinfra/auto" in capsys.readouterr().out


def test_with_command_can_use_gemini_once_then_restore(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["provider_name"] = provider_name
        captured["model"] = model
        return {
            "text": "<helix_output>Respuesta con Gemini.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 8},
        }

    monkeypatch.setattr(helix_cli, "_ensure_provider_token", lambda provider_name: None)
    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/with gemini-pro explicame esto") is True
    assert captured["provider_name"] == "gemini"
    assert captured["model"] == helix_cli.GEMINI_MODEL_PROFILES["gemini-pro"].model_id
    assert session.provider_name == "deepinfra"
    assert session.model == "auto"
    assert "provider/model restored to deepinfra/auto" in capsys.readouterr().out


def test_prompt_toolbar_markup_tracks_session_state() -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        router_policy="qwen-gemma-mistral",
    )
    session.theme_name = "industrial-neon"
    session.interaction_mode = "explore"
    session.blind_inference_enabled = True
    markup = helix_cli._prompt_toolbar_markup(session)
    assert "thread" in markup
    assert "provider" in markup
    assert "model" in markup
    assert "router" in markup
    assert "mode" in markup
    assert "blind" in markup
    assert "theme" in markup
    assert "explore" in markup
    assert "on" in markup
    assert "industrial-neon" in markup


def test_boot_banner_and_session_ribbon_export_text() -> None:
    if not helix_cli._HAS_UI:
        return
    from rich.console import Console

    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    console = Console(theme=helix_cli._rich_theme("industrial-brutalist"), width=120, record=True)
    session.blind_inference_enabled = True
    helix_cli._render_boot_banner(console)
    helix_cli._render_session_ribbon(console, session)
    output = console.export_text()
    assert "HeliX Inference OS" in output
    assert "SESSION BUS" in output
    assert "mode" in output
    assert "blind" in output
    assert session.run_id in output
    assert "task root" in output
    assert "helix-backend-repo" in output


def test_run_with_status_renders_spinner_without_crashing() -> None:
    if not helix_cli._HAS_UI:
        return
    from rich.console import Console

    console = Console(theme=helix_cli._rich_theme("industrial-brutalist"), width=96, record=True)
    result = helix_cli._run_with_status(console, lambda: {"text": "ok"}, phase="thinking")
    assert result == {"text": "ok"}


def test_agent_tool_call_parser_accepts_json_protocol() -> None:
    calls = helix_cli._parse_agent_tool_calls(
        '<tool_call>{"tool_calls":[{"tool":"search_text","arguments":{"query":"needle"}},'
        '{"name":"read_file","args":{"path":"README.md"}}]}</tool_call>'
    )
    assert calls == [
        {"tool": "search_text", "arguments": {"query": "needle"}, "id": None},
        {"tool": "read_file", "arguments": {"path": "README.md"}, "id": None},
    ]


def test_agent_tool_call_parser_accepts_fenced_json_list() -> None:
    calls = helix_cli._parse_agent_tool_calls(
        '```json\n[{"tool":"search_text","arguments":{"query":"needle"}}]\n```'
    )
    assert calls == [
        {"tool": "search_text", "arguments": {"query": "needle"}, "id": None},
    ]


def test_active_memory_query_uses_clean_goal_for_planner_step() -> None:
    query = helix_agent._active_memory_query(
        "revisá nuestra memoria sobre rizomas e hiperstición",
        "__planner__",
        {},
        [],
        [],
    )
    assert query == "revisá nuestra memoria sobre rizomas e hiperstición"
    assert "Tool: __planner__" not in query


def test_read_only_tools_block_path_escape_and_unsafe_commands() -> None:
    root = _test_root() / "repo"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("needle\n", encoding="utf-8")
    tools = helix_cli.ReadOnlyAgentTools(root=root)

    search = tools.call("search_text", {"query": "needle"})
    assert search["result"]["matches"][0]["path"] == "a.txt"

    escaped = tools.call("read_file", {"path": "..\\secret.txt"})
    assert escaped["result"]["status"] == "error"
    assert "escapes task root" in escaped["result"]["error"]

    blocked = tools.call("run_test", {"command": "cmd /c del a.txt"})
    assert blocked["result"]["status"] == "blocked"


def test_natural_language_repo_work_routes_to_work_runtime() -> None:
    routed = helix_cli._route_natural_language("fijate el repo y armame un patch para el bug")
    assert routed.startswith("/work run ")


def test_work_last_renders_human_summary(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        task_root=_test_root() / "repo",
    )
    session.last_work_plan = {"goal": "hacer demo", "output_target": "web/demo"}
    session.last_trust_card = {"sources": [{"path": "README.md"}], "changed_files": ["web/demo/index.html"]}
    session.last_work_result = {
        "status": "completed",
        "engine": "helix-internal-generator",
        "final": "Genero una propuesta aplicable.",
        "patch_available": True,
        "changed_files": ["web/demo/index.html"],
        "work_artifact_paths": {
            "patch": "verification/work-runtime/run/patch.diff",
            "trust_card": "verification/work-runtime/run/work_trust_card.json",
        },
    }

    assert helix_cli._handle_interactive_command(session, "/work last") is True
    output = capsys.readouterr().out
    assert "HeliX Work Result" in output
    assert "web/demo/index.html" in output
    assert "/apply last" in output


def test_render_task_result_handles_current_tool_event_shape() -> None:
    if not helix_cli._HAS_UI:
        return
    from rich.console import Console

    console = Console(theme=helix_cli._rich_theme("cyberpunk"), width=100, record=True)
    helix_cli._render_task_result(
        console,
        {
            "final": "ok",
            "selected_model": "mock-model",
            "route": {"intent": "task"},
            "patch_available": False,
            "tool_events": [
                {
                    "tool": "search_text",
                    "arguments": {"query": "needle"},
                    "result": {"status": "ok", "query": "needle", "matches": [{"path": "app.py"}]},
                }
            ],
        },
    )
    output = console.export_text()
    assert "search_text" in output
    assert "needle" in output


def test_render_task_result_handles_none_tool_payload() -> None:
    if not helix_cli._HAS_UI:
        return
    from rich.console import Console

    console = Console(theme=helix_cli._rich_theme("cyberpunk"), width=100, record=True)
    helix_cli._render_task_result(
        console,
        {
            "final": "ok",
            "selected_model": "mock-model",
            "route": {"intent": "task"},
            "patch_available": False,
            "tool_events": [
                {
                    "tool": "search_text",
                    "arguments": {"query": "needle"},
                    "result": None,
                }
            ],
        },
    )
    output = console.export_text()
    assert "search_text" in output
    assert "ok" in output


def test_render_task_result_handles_legacy_nested_tool_payload() -> None:
    if not helix_cli._HAS_UI:
        return
    from rich.console import Console

    console = Console(theme=helix_cli._rich_theme("cyberpunk"), width=100, record=True)
    helix_cli._render_task_result(
        console,
        {
            "final": "ok",
            "selected_model": "mock-model",
            "route": {"intent": "task"},
            "patch_available": False,
            "tool_events": [
                {
                    "result": {
                        "tool": "query_evidence",
                        "arguments": {"query": "needle"},
                        "result": {"status": "ok", "record_count": 3},
                    }
                }
            ],
        },
    )
    output = console.export_text()
    assert "query_evidence" in output
    assert "3 records" in output


def test_identity_question_is_detected_for_certified_evidence_injection() -> None:
    assert helix_cli._is_identity_question("que te hace especial?")
    assert helix_cli._is_identity_question("que hace especial a HeliX?")
    assert not helix_cli._is_identity_question("que hace especial a Qwen3.5-122B-A10B?")
    assert not helix_cli._is_identity_question("hola")


def test_helix_explanation_request_is_detected_from_context() -> None:
    history = [{"role": "user", "content": "estaba pensando en helix"}]
    assert helix_cli._is_helix_explanation_request("me gustaria que me ayudes a entenderlo", history) is True
    assert helix_cli._needs_certified_evidence("me gustaria que me ayudes a entenderlo", history=history) is False


def test_helix_auditability_request_is_detected_from_context() -> None:
    history = [{"role": "assistant", "content": "Si queres, seguimos hablando de HeliX."}]
    assert helix_cli._is_helix_auditability_request("que onda la auditabilidad y los hashes?", history) is True
    assert helix_cli._is_helix_explanation_request("que onda la auditabilidad y los hashes?", history) is True
    assert helix_cli._needs_certified_evidence("que onda la auditabilidad y los hashes?", history=history) is True


def test_helix_context_does_not_capture_clear_general_topic_shift() -> None:
    history = [{"role": "assistant", "content": "Si queres, seguimos hablando de HeliX."}]
    assert helix_cli._is_helix_explanation_request("hablame de argentina", history) is False
    assert helix_cli._is_helix_auditability_request("hablame de argentina", history) is False
    assert helix_cli._needs_certified_evidence("hablame de argentina", history=history) is False
    assert helix_cli._is_helix_explanation_request("que hace especial a Qwen3.5-122B-A10B?", history) is False
    assert helix_cli._needs_certified_evidence("que hace especial a Qwen3.5-122B-A10B?", history=history) is False


def test_explicit_helix_meta_task_detection_stays_narrow_for_memory_reviews() -> None:
    history = [{"role": "assistant", "content": "Si queres, seguimos hablando de HeliX."}]
    assert helix_cli._is_explicit_helix_meta_task_request("revisá nuestra memoria y resumime la conversación", history) is False
    assert helix_cli._is_explicit_helix_meta_task_request("como lo implementarias en la arquitectura?", history) is True
    assert helix_cli._is_explicit_helix_meta_task_request("cerrá la semántica de cabeza canónica y equivocation", history) is True


def test_helix_context_does_not_ground_pure_social_reactions() -> None:
    history = [{"role": "assistant", "content": "Si queres, seguimos hablando de HeliX."}]
    assert helix_cli._is_helix_explanation_request("la verdad es una locura esto en el buen sentido!", history) is False
    assert helix_cli._needs_certified_evidence(
        "la verdad es una locura esto en el buen sentido!",
        history=history,
    ) is False


def test_signature_followups_stay_grounded_in_helix_context() -> None:
    history = [{"role": "assistant", "content": "HeliX usa receipts y un Merkle DAG verificable."}]
    assert helix_cli._is_helix_auditability_request(
        "como te darias cuenta si una firma no es valida?",
        history,
    ) is True
    assert helix_cli._needs_certified_evidence(
        "como te darias cuenta si una firma no es valida?",
        history=history,
    ) is True


def test_interactive_record_writes_signed_memory_receipt() -> None:
    workspace = Path.cwd() / "verification" / "cli-sessions" / "_test" / uuid.uuid4().hex
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=16,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    event = session.record(role="user", content="hola", event_type="user_turn")
    memory = event["helix_memory"]
    assert memory["memory_id"].startswith("mem-")
    assert len(memory["node_hash"]) == 64
    assert memory["receipt"]["signature_verified"] is True
    assert memory["receipt"]["key_provenance"] == "local_self_signed"
    assert memory["receipt"]["signing_key_id"].startswith("ed25519-")
    assert memory["receipt"]["checkpoint_hash"]


def test_identity_question_injects_certified_evidence_pack(monkeypatch) -> None:
    workspace = Path.cwd() / "verification" / "cli-sessions" / "_test" / uuid.uuid4().hex
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, **kwargs):
        captured["provider_name"] = provider_name
        captured["model"] = model
        captured["prompt"] = prompt
        captured["system"] = system
        captured["kwargs"] = kwargs
        return {
            "text": "HeliX usa memoria Merkle-DAG certificada.",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    answer_obj = session.chat("que te hace especial?")
    answer = answer_obj["text"]
    system = captured["system"]
    assert "Merkle-DAG certificada" in answer
    assert "Certified HeliX evidence pack" in system
    assert '"latest_user_receipt"' in system
    assert '"signature_verified": true' in system
    assert '"key_provenance": "local_self_signed"' in system
    assert '"checkpoint_hash":' in system
    assert '"tombstone_boundary"' in system


def test_contextual_helix_followup_stays_lightweight_without_evidence_pack(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, **kwargs):
        captured["system"] = system
        return {
            "text": "HeliX firma memoria y receipts verificables.",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="estaba pensando en helix", event_type="user_turn")
    session.record(role="assistant", content="Dale, contame qué querés entender.", event_type="assistant_turn")

    session.chat("me gustaria que me ayudes a entenderlo")
    system = captured["system"]
    assert "Certified HeliX evidence pack" not in system
    assert "Thinking Process" in system
    assert session.events[-1]["metadata"]["lane"] == "conversation"


def test_chat_topic_shift_after_helix_context_answers_without_helix_grounding(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["model"] = model
        captured["system"] = system
        return {
            "text": "<helix_output>Argentina es un pais de America del Sur con una historia politica y cultural muy rica.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="estaba pensando en helix", event_type="user_turn")
    session.record(role="assistant", content="Dale, exploremos eso.", event_type="assistant_turn")

    result = session.chat("hablame de argentina")
    assert result["route"]["intent"] == "chat"
    assert captured["model"] == helix_cli.DEEPINFRA_MODEL_PROFILES["chat"].model_id
    assert '"claim": "This HeliX CLI session is backed by HeliX memory and evidence exports."' not in captured["system"]
    assert "Certified HeliX evidence pack" not in captured["system"]
    assert not (result["trace"].get("observations") or [])
    assert result["trace"]["mode"] == "lightweight_chat"


def test_openai_compatible_chat_uses_mocked_transport(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, *, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {
            "model": "mock-model-actual",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }

    monkeypatch.setattr(helix_cli, "_post_json", fake_post_json)
    result = helix_cli.run_chat(
        provider_name="ollama",
        model="mock-model",
        prompt="hello",
        prompt_token=False,
        max_tokens=8,
    )
    assert result["text"] == "ok"
    assert result["actual_model"] == "mock-model-actual"
    assert captured["url"].endswith("/chat/completions")
    assert json.loads(json.dumps(captured["payload"]))["model"] == "mock-model"


def test_openai_compatible_bad_request_retries_with_compact_payload(monkeypatch) -> None:
    calls = []

    def fake_post_json(url, payload, *, headers, timeout):
        calls.append(payload)
        if len(calls) == 1:
            raise helix_cli.error.HTTPError(
                url,
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"error":"context too large"}'),
            )
        return {
            "model": "mock-model-actual",
            "choices": [{"message": {"content": "ok compact"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }

    monkeypatch.setattr(helix_cli, "_post_json", fake_post_json)
    result = helix_cli.run_chat(
        provider_name="ollama",
        model="mock-model",
        prompt="hello" * 3000,
        system="system-context " * 4000,
        history=[
            {"role": "user", "content": "old user " * 2000},
            {"role": "assistant", "content": "old assistant " * 2000},
        ],
        prompt_token=False,
        max_tokens=8,
    )

    assert result["text"] == "ok compact"
    assert result["request_compacted_after_bad_request"] is True
    first_chars = sum(len(str(item.get("content") or "")) for item in calls[0]["messages"])
    second_chars = sum(len(str(item.get("content") or "")) for item in calls[1]["messages"])
    assert second_chars < first_chars
    assert "request compacted after provider Bad Request" in calls[1]["messages"][0]["content"]


def test_nvidia_openai_compatible_chat_uses_integrate_endpoint_and_json_mode(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, *, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {
            "model": "mistralai/magistral-small-2506",
            "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 5},
        }

    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-token")
    monkeypatch.setattr(helix_cli, "_post_json", fake_post_json)
    result = helix_cli.run_chat(
        provider_name="nvidia",
        model=helix_cli.NVIDIA_MODEL_PROFILES["nvidia-chat"].model_id,
        prompt="respond with json",
        prompt_token=False,
        max_tokens=8,
        native_request={"request_response_format": {"type": "json_object"}},
    )

    assert result["text"] == "{\"ok\": true}"
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer nvidia-test-token"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_gemini_chat_uses_generate_content_api(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, *, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {
            "modelVersion": "gemini-3.1-pro-preview",
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok gemini"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"totalTokenCount": 7},
        }

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-token")
    monkeypatch.setattr(helix_cli, "_post_json", fake_post_json)
    result = helix_cli.run_chat(
        provider_name="gemini",
        model="gemini-3.1-pro-preview",
        system="system guard",
        history=[{"role": "assistant", "content": "prev answer"}],
        prompt="hello",
        prompt_token=False,
        max_tokens=8,
    )
    assert result["text"] == "ok gemini"
    assert result["actual_model"] == "gemini-3.1-pro-preview"
    assert captured["url"].endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "gemini-test-token"
    assert captured["payload"]["systemInstruction"]["parts"][0]["text"] == "system guard"
    assert captured["payload"]["contents"][0]["role"] == "model"
    assert captured["payload"]["contents"][-1]["role"] == "user"


def test_run_chat_blind_inference_transforms_remote_and_rehydrates(monkeypatch) -> None:
    captured = {}

    def fake_token_for_provider(provider, prompt=True):
        return "deepinfra-test-token"

    def fake_openai(provider, model, messages, **kwargs):
        captured["messages"] = messages
        prompt = messages[-1]["content"]
        person = re.search(r"PERSON__T[A-F0-9]{4}__001", prompt)
        document = re.search(r"DOC_ID__T[A-F0-9]{4}__001", prompt)
        assert person is not None
        assert document is not None
        return {
            "provider": provider.name,
            "requested_model": model,
            "actual_model": model,
            "text": f"Analice {person.group(0)} y {document.group(0)}.",
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
            "latency_ms": 1.0,
            "raw": {},
        }

    monkeypatch.setattr(helix_cli, "_token_for_provider", fake_token_for_provider)
    monkeypatch.setattr(helix_cli, "_openai_compatible_chat", fake_openai)
    result = helix_cli.run_chat(
        provider_name="deepinfra",
        model="Qwen/Qwen3.5-122B-A10B",
        prompt="Compara a Juan Perez con DNI 12345678.",
        prompt_token=False,
        max_tokens=16,
        blind_inference={
            "enabled": True,
            "task_id": "blind-test-turn-1",
            "policy": {
                "enabled": True,
                        "scope": "cloud_proxy",
                        "placeholder_stability": "per_task",
                        "rules": [
                            {"name": "person-rule", "type": "PERSON", "values": ["Juan Perez"]},
                            {"name": "doc-rule", "type": "DOC_ID", "pattern": r"\b\d{8}\b"},
                        ],
                    },
                },
            )

    outbound = captured["messages"][-1]["content"]
    assert "Juan Perez" not in outbound
    assert "12345678" not in outbound
    assert result["text"] == "Analice Juan Perez y 12345678."
    assert result["blind_inference"]["enabled"] is True
    assert result["blind_inference"]["span_count"] == 2


def test_run_chat_blind_inference_bypasses_local_provider(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, *, headers, timeout):
        captured["payload"] = payload
        return {
            "model": "local-model",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }

    monkeypatch.setattr(helix_cli, "_post_json", fake_post_json)
    result = helix_cli.run_chat(
        provider_name="ollama",
        model="llama3.1",
        prompt="Compara a Juan Perez con DNI 12345678.",
        prompt_token=False,
        max_tokens=8,
        blind_inference={
            "enabled": True,
            "task_id": "blind-test-local",
            "policy": {
                "enabled": True,
                "scope": "cloud_proxy",
                "placeholder_stability": "per_task",
                "rules": [{"name": "person-rule", "type": "PERSON", "values": ["Juan Perez"]}],
            },
        },
    )

    assert "Juan Perez" in captured["payload"]["messages"][-1]["content"]
    assert result["blind_inference"]["bypassed"] is True
    assert result["blind_inference"]["provider_target"] == "ollama"


def test_memory_catalog_journal_replays_merkle_receipts(monkeypatch) -> None:
    db_path = _test_root() / "session-os" / "memory.sqlite"
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_MODE", "ephemeral_preregistered")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNER_ID", "pytest")
    monkeypatch.setenv("HELIX_RECEIPT_SIGNING_SEED", "pytest-journal-seed")

    catalog = MemoryCatalog.open(db_path)
    item = catalog.remember(
        project="helix-cli",
        agent_id="interactive",
        session_id="session-1",
        memory_type="semantic",
        summary="journal replay memory",
        content="certified journal replay content",
        importance=8,
    )
    node_hash = catalog.get_memory_node_hash(item.memory_id)
    receipt = catalog.get_memory_receipt(item.memory_id)
    assert node_hash
    assert receipt and receipt["signature_verified"] is True

    MemoryCatalog._REGISTRY.pop(str(db_path.resolve()), None)
    replayed = MemoryCatalog.open(db_path)
    assert replayed.get_memory(item.memory_id) is not None
    assert replayed.get_memory_node_hash(item.memory_id) == node_hash
    assert replayed.get_memory_receipt(item.memory_id)["signature_verified"] is True
    assert replayed.verify_chain(node_hash)["status"] == "verified"


def test_evidence_refresh_ingests_artifact_into_merkle_memory() -> None:
    base = _test_root()
    repo_root = base / "repo"
    evidence_root = repo_root / "verification"
    suite_dir = evidence_root / "nuclear-methodology" / "hard-anchor-utility"
    suite_dir.mkdir(parents=True)
    transcript = suite_dir / "local-hard-anchor-utility-suite-hard-anchor-utility-20260421-120000-transcripts.jsonl"
    transcript.write_text('{"event":"case","case_id":"exact-anchor"}\n', encoding="utf-8")
    artifact = suite_dir / "local-hard-anchor-utility-suite-hard-anchor-utility-20260421-120000.json"
    artifact.write_text(
        json.dumps(
            {
                "suite_id": "hard-anchor-utility",
                "run_id": "hard-anchor-utility-20260421-120000",
                "status": "completed",
                "case_count": 1,
                "cases": [{"case_id": "exact-anchor", "status": "completed", "score": 1.0}],
                "transcript_exports": {"jsonl_path": str(transcript)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = suite_dir / "local-hard-anchor-utility-suite-20260421-120000-run.json"
    manifest.write_text(
        json.dumps({"run_id": "hard-anchor-utility-20260421-120000", "artifact_path": str(artifact)}),
        encoding="utf-8",
    )

    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="helix-cli",
        agent_id="interactive",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=evidence_root,
    )
    pack = session.refresh_evidence("hard anchor", limit=4)
    records = pack["records"]
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] == "hard-anchor-utility-20260421-120000"
    assert record["memory_id"].startswith("mem-evidence-")
    assert len(record["node_hash"]) == 64
    assert record["signature_verified"] is True
    assert record["chain_status"] == "verified"

    context = session.memory_context("hard-anchor-utility exact-anchor")
    assert record["memory_id"] in context["memory_ids"]
    assert "hard-anchor-utility-20260421-120000" in context["context"]


def test_repository_evidence_pack_is_injected_for_verify_questions(monkeypatch) -> None:
    base = _test_root()
    repo_root = base / "repo"
    evidence_root = repo_root / "verification"
    suite_dir = evidence_root / "nuclear-methodology" / "branch-pruning-forensics"
    suite_dir.mkdir(parents=True)
    artifact = suite_dir / "local-branch-pruning-forensics-suite-branch-pruning-forensics-20260421-121000.json"
    artifact.write_text(
        json.dumps(
            {
                "suite_id": "branch-pruning-forensics",
                "run_id": "branch-pruning-forensics-20260421-121000",
                "status": "completed",
                "case_count": 1,
                "cases": [{"case_id": "cold-audit-branch-preservation", "status": "completed", "score": 1.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = suite_dir / "local-branch-pruning-forensics-suite-20260421-121000-run.json"
    manifest.write_text(
        json.dumps({"run_id": "branch-pruning-forensics-20260421-121000", "artifact_path": str(artifact)}),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, **kwargs):
        captured["system"] = system
        return {
            "text": "Evidencia cargada: branch-pruning-forensics-20260421-121000.",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="helix-cli",
        agent_id="interactive",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=evidence_root,
    )

    session.chat("contame de una corrida de /verify branch pruning")
    system = captured["system"]
    assert "Certified repository evidence pack" in system
    assert "branch-pruning-forensics-20260421-121000" in system
    assert "Do not invent dates, run IDs, hashes" in system


def test_suite_evidence_catalog_indexes_artifacts_manifests_and_transcripts() -> None:
    evidence_root, _suite_dir, paths = _write_suite_fixture(_test_root())
    catalog = helix_cli.SuiteEvidenceCatalog(evidence_root=evidence_root)

    listed = catalog.list_suites()
    assert listed["suite_count"] == 1
    assert listed["suites"][0]["suite_id"] == "hard-anchor-utility"
    assert listed["suites"][0]["counts"]["artifact"] == 1
    assert listed["suites"][0]["counts"]["manifest"] == 1
    assert listed["suites"][0]["counts"]["transcript_jsonl"] == 1

    latest = catalog.latest("hard-anchor-utility")
    assert latest["status"] == "ok"
    assert latest["artifact"]["run_id"] == "hard-anchor-utility-20260421-120000"
    assert latest["manifest"]["kind"] == "manifest"
    assert latest["transcripts"]

    transcripts = catalog.transcripts("hard-anchor-utility", query="exact-anchor")
    assert transcripts["transcript_count"] == 2

    search = catalog.search("lineage verified")
    assert search["result_count"] == 1
    assert search["results"][0]["kind"] == "transcript_jsonl"
    assert "lineage verified" in search["results"][0]["snippet"]

    read = catalog.read(str(paths["transcript"]))
    assert read["status"] == "ok"
    assert read["kind"] == "transcript_jsonl"
    assert "hard anchor recovered" in read["content"]


def test_suite_evidence_catalog_search_and_read_support_global_root_refs() -> None:
    evidence_root, suite_dir, paths = _write_suite_fixture(_test_root(), "cognitive-gauntlet")
    global_artifact = evidence_root / "local-ghost-in-the-shell-live.json"
    global_artifact.write_text(
        json.dumps(
            {
                "run_id": "local-ghost-in-the-shell-live",
                "status": "completed",
                "summary": "Transcript export for the ghost-in-the-shell live session.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    global_log = evidence_root / "local-ghost-in-the-shell-live.log"
    global_log.write_text("Ghost session resolved the contradiction through Merkle-DAG replay.\n", encoding="utf-8")
    misleading_suite_hit = suite_dir / "mentions-local-ghost.json"
    misleading_suite_hit.write_text(
        json.dumps({"note": "Comparison target local-ghost-in-the-shell-live appears in a suite note."}, ensure_ascii=False),
        encoding="utf-8",
    )

    catalog = helix_cli.SuiteEvidenceCatalog(evidence_root=evidence_root)
    search = catalog.search("local-ghost-in-the-shell-live", limit=5)
    assert search["status"] == "ok"
    assert search["results"]
    assert search["results"][0]["catalog_scope"] == "global"
    assert search["results"][0]["path"].endswith("local-ghost-in-the-shell-live.json")

    read = catalog.read("local-ghost-in-the-shell-live")
    assert read["status"] == "ok"
    assert read["path"].endswith("local-ghost-in-the-shell-live.json")
    assert "ghost-in-the-shell live session" in read["content"]

    dir_read = catalog.read("cognitive-gauntlet")
    assert dir_read["status"] == "ok"
    assert dir_read["type"] == "directory"
    assert any(item["name"] == paths["artifact"].name for item in dir_read["entries"])


def test_suite_commands_print_catalog_latest_and_transcripts(capsys) -> None:
    base = _test_root()
    evidence_root, _suite_dir, _paths = _write_suite_fixture(base)
    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=evidence_root,
    )

    assert helix_cli._handle_interactive_command(session, "/suites") is True
    output = capsys.readouterr().out
    assert "hard-anchor-utility" in output
    assert "json" in output

    assert helix_cli._handle_interactive_command(session, "/suite latest hard-anchor-utility") is True
    output = capsys.readouterr().out
    assert '"artifact"' in output
    assert "hard-anchor-utility-20260421-120000" in output

    assert helix_cli._handle_interactive_command(session, "/suite transcripts hard-anchor-utility exact-anchor") is True
    output = capsys.readouterr().out
    assert '"transcript_count": 2' in output


def test_models_tools_and_agents_commands_use_compact_output_by_default(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )

    assert helix_cli._handle_interactive_command(session, "/models") is True
    models_output = capsys.readouterr().out
    assert "qwen-big" in models_output
    assert "gemini-pro" in models_output
    assert "gemini" in models_output
    assert "Use /model use ALIAS" in models_output
    assert '"deepinfra_model_profiles"' not in models_output

    assert helix_cli._handle_interactive_command(session, "/tools") is True
    tools_output = capsys.readouterr().out
    assert "suite.latest" in tools_output
    assert "Use /tools blueprints" in tools_output

    assert helix_cli._handle_interactive_command(session, "/agents") is True
    agents_output = capsys.readouterr().out
    assert "suite-run-analyst" in agents_output
    assert "patch-planner" in agents_output


def test_chat_suite_questions_are_grounded_with_suite_tools(monkeypatch) -> None:
    base = _test_root()
    evidence_root, _suite_dir, _paths = _write_suite_fixture(base, "branch-pruning-forensics")
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["history"] = history
        return {
            "text": "<helix_output>La ultima corrida local disponible es branch-pruning-forensics-20260421-120000.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=evidence_root,
    )

    result = session.chat("contame la ultima corrida de branch pruning")
    assert result["route"]["intent"] == "suite_forensics"
    assert result["trace"]["observations"][0]["tool_name"] == "suite.latest"
    assert "branch-pruning-forensics-20260421-120000" in json.dumps(captured["history"], ensure_ascii=False)
    assert "branch-pruning-forensics-20260421-120000" in result["text"]


def test_chat_pasted_suite_failure_uses_search_not_cert_rerun(monkeypatch) -> None:
    base = _test_root()
    evidence_root, _suite_dir, _paths = _write_suite_fixture(base, "branch-pruning-forensics")
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["prompt"] = prompt
        captured["history"] = history
        return {
            "text": "<helix_output>La corrida pegada falló porque RustIndexedMerkleDAG no expone build_context_fast; no es un fallo del auditor sino del build/binding local.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 18},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=evidence_root,
    )
    pasted = """quiero data de esto
Suite                     Run ID           Estado
branch-pruning-forensics  20260421-120000  completed
{
  "suite_id": "branch-pruning-forensics",
  "exit_code": 1,
  "stderr": "RuntimeError: RustIndexedMerkleDAG was not rebuilt with build_context_fast"
}
"""
    result = session.chat(pasted)
    observations = result["trace"].get("observations") or []
    assert observations
    assert observations[0]["tool_name"] == "suite.search"
    assert "pasted suite output/logs" in captured["prompt"]
    assert "build_context_fast" in result["text"]


def test_chat_web_search_requests_call_web_tool_before_answering(monkeypatch) -> None:
    base = _test_root()
    base.mkdir(parents=True, exist_ok=True)
    workspace = base / "workspace"
    captured = {}

    def fake_web_search(query, *, limit=5, timeout=8.0):
        captured["web_query"] = query
        return {
            "status": "ok",
            "query": query,
            "result_count": 1,
            "results": [
                {
                    "title": "Claude benchmark source",
                    "url": "https://example.com/claude-benchmark",
                    "snippet": "Current benchmark details.",
                }
            ],
        }

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["history"] = history
        return {
            "text": "<helix_output>Fuente: https://example.com/claude-benchmark - benchmark actual localizado.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "web_search", fake_web_search)
    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("buscame en la web benchmarks actuales de claude mythos")
    assert result["route"]["intent"] == "web_research"
    assert captured["web_query"].startswith("buscame en la web")
    assert result["trace"]["observations"][0]["tool_name"] == "web.search"
    assert "https://example.com/claude-benchmark" in json.dumps(captured["history"], ensure_ascii=False)
    assert "https://example.com/claude-benchmark" in result["text"]


def test_memory_resolve_finds_exact_record_by_node_hash_prefix() -> None:
    base = _test_root()
    base.mkdir(parents=True, exist_ok=True)
    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=base / "empty-verification",
    )
    content = "Selección Puntual de las Mejores Transcripciones HeliX\ncontenido exacto certificado"
    event = session.record(role="assistant", content=content, event_type="assistant_turn")
    node_hash = str((event.get("helix_memory") or {}).get("node_hash") or "")

    resolved = session.memory_resolve(node_hash[:10])

    assert resolved["status"] == "ok"
    match = resolved["matches"][0]
    assert match["node_hash"] == node_hash
    assert match["content"] == content
    assert match["chain"]["status"] == "verified"


def test_chat_hash_recovery_uses_memory_resolve_without_model_reconstruction(monkeypatch) -> None:
    base = _test_root()
    workspace = base / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=base / "empty-verification",
    )
    target = "HeliX y la Hauntologia Rizomatica: texto real guardado, no reconstruido."
    event = session.record(role="assistant", content=target, event_type="assistant_turn")
    prefix = str((event.get("helix_memory") or {}).get("node_hash") or "")[:10]

    def fail_run_chat(*args, **kwargs):
        raise AssertionError("hash recovery must not ask the model to recreate exact content")

    monkeypatch.setattr(helix_cli, "run_chat", fail_run_chat)
    result = session.chat(f"quiero que recuperes completo este hash {prefix}")

    observations = result["trace"].get("observations") or []
    assert observations
    assert observations[0]["tool_name"] == "memory.resolve"
    assert observations[0]["arguments"]["ref"] == prefix
    assert target in result["text"]
    assert "sin reconstruirlo con el modelo" in result["text"]


def test_chat_bad_tool_arguments_are_observed_without_system_crash(monkeypatch) -> None:
    base = _test_root()
    calls = {"count": 0}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "text": '<tool_call>{"tool":"memory.resolve","arguments":{}}</tool_call>',
                "actual_model": model,
                "latency_ms": 1.0,
                "finish_reason": "stop",
                "usage": {"total_tokens": 10},
            }
        assert history
        assert "missing tool arguments: ref" in history[-1]["content"]
        return {
            "text": "<helix_output>No necesito resolver un hash para responder eso; puedo contestar directo.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model=helix_cli.DEEPINFRA_MODEL_PROFILES["chat"].model_id,
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.chat("contame algo interesante")

    observations = result["trace"].get("observations") or []
    assert calls["count"] >= 1
    assert observations
    assert observations[0]["tool_name"] == "memory.resolve"
    assert observations[0]["observation"]["result"]["status"] == "error"
    assert observations[0]["observation"]["result"]["error"] == "missing tool arguments: ref"
    assert "Task failed" not in result["text"]
    assert "memory.resolve" in result["text"]
    assert "sin `ref`" in result["text"]


def test_memory_resolve_falls_back_to_transcript_jsonl() -> None:
    base = _test_root()
    workspace = base / "workspace"
    transcript_dir = workspace / "transcripts"
    transcript_dir.mkdir(parents=True)
    node_hash = "5b71482b56abcdef1234567890abcdef1234567890abcdef1234567890abcd"
    stored_content = "Selección Puntual de las Mejores Transcripciones HeliX"
    (transcript_dir / "old-session.jsonl").write_text(
        json.dumps(
            {
                "event": "assistant_turn",
                "role": "assistant",
                "created_utc": "2026-04-21T20:00:00Z",
                "content": stored_content,
                "helix_memory": {"memory_id": "mem-old", "node_hash": node_hash},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=transcript_dir,
        evidence_root=base / "empty-verification",
    )

    resolved = session.memory_resolve("5b71482b56")

    assert resolved["status"] == "ok"
    assert resolved["matches"][0]["source"] == "transcript-jsonl"
    assert resolved["matches"][0]["content"] == stored_content


def test_file_inspect_reads_absolute_file_lists_directory_and_blocks_secrets() -> None:
    base = _test_root()
    workspace = base / "workspace"
    repo = base / "repo with spaces"
    verification = repo / "verification"
    verification.mkdir(parents=True)
    artifact = verification / "local-ghost-in-the-shell-live-20260418-093140-run.json"
    artifact.write_text('{"suite_id":"ghost","status":"completed"}\n', encoding="utf-8")
    secret = repo / ".env"
    secret.write_text("GEMINI_API_KEY=secret\n", encoding="utf-8")
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=verification,
        task_root=base,
    )

    file_result = session.file_inspect(str(artifact))
    assert file_result["status"] == "ok"
    assert file_result["type"] == "file"
    assert file_result["sha256"]
    assert '"suite_id":"ghost"' in file_result["content"]

    wrapped = str(artifact).replace("093140", "0\n93140")
    wrapped_result = session.file_inspect(wrapped)
    assert wrapped_result["status"] == "ok"
    assert wrapped_result["path"] == str(artifact)

    dir_result = session.file_inspect(str(verification))
    assert dir_result["status"] == "ok"
    assert dir_result["type"] == "directory"
    assert any(item["name"] == artifact.name for item in dir_result["entries"])

    blocked = session.file_inspect(str(secret))
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "environment secret file"


def test_chat_local_path_request_uses_file_inspect_before_answering(monkeypatch) -> None:
    base = _test_root()
    workspace = base / "workspace"
    repo = base / "repo with spaces"
    verification = repo / "verification"
    verification.mkdir(parents=True)
    artifact = verification / "local-ghost-v2-doppelganger-war-20260419-011343.json"
    artifact.write_text('{"suite_id":"doppelganger","verdict":"interesting"}\n', encoding="utf-8")
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["prompt"] = prompt
        captured["history"] = history
        return {
            "text": "<helix_output>Lei el archivo real: verdict=interesting.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=verification,
        task_root=base,
    )

    result = session.chat(f'hay unas que estan "{artifact}" que me interesan')

    observations = result["trace"].get("observations") or []
    assert observations
    assert observations[0]["tool_name"] == "file.inspect"
    assert observations[0]["arguments"]["path"] == str(artifact)
    assert "file.inspect observations" in captured["prompt"]
    assert "doppelganger" in json.dumps(captured["history"], ensure_ascii=False)
    assert "verdict=interesting" in result["text"]

    captured.clear()
    dir_result = session.chat(f"lee esta carpeta {verification}")
    dir_observations = dir_result["trace"].get("observations") or []
    assert dir_observations
    assert dir_observations[0]["tool_name"] == "file.inspect"
    assert dir_observations[0]["arguments"]["path"] == str(verification)
    dir_observation = dir_observations[0]["observation"]
    if isinstance(dir_observation.get("result"), dict):
        dir_observation = dir_observation["result"]
    assert dir_observation["type"] == "directory"
    assert artifact.name in json.dumps(captured["history"], ensure_ascii=False)


def test_chat_model_failover_uses_route_fallback_without_crashing(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    calls = []

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls.append(model)
        if model == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id:
            raise RuntimeError("provider overloaded")
        return {
            "text": "<helix_output>Respondido por fallback.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("investiga modelos nuevos de deepinfra actuales")
    assert result["text"] == "Respondido por fallback."
    assert calls[0] == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id
    assert calls[1] == helix_cli.DEEPINFRA_MODEL_PROFILES["default"].model_id
    latest = session.events[-1]
    assert latest["metadata"]["failover_used"] is True
    assert latest["metadata"]["failover_attempts"][0]["error_type"] == "RuntimeError"


def test_direct_web_command_prints_search_results(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        helix_cli,
        "web_search",
        lambda query, *, limit=5, timeout=8.0: {"status": "ok", "query": query, "results": [{"title": "T", "url": "https://example.com"}]},
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )
    assert helix_cli._handle_interactive_command(session, "/web claude benchmark") is True
    output = capsys.readouterr().out
    assert '"query": "claude benchmark"' in output
    assert "https://example.com" in output


def test_interactive_task_uses_read_only_tools_and_records_receipts(monkeypatch) -> None:
    base = _test_root()
    task_root = base / "repo"
    task_root.mkdir(parents=True)
    (task_root / "app.py").write_text("def answer():\n    return 'needle'\n", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "text": '<tool_call>{"tool":"search_text","arguments":{"query":"needle","path":"."}}</tool_call>',
                "actual_model": model,
                "latency_ms": 2.0,
                "finish_reason": "stop",
                "usage": {"total_tokens": 20},
            }
        assert history
        assert "HeliX read-only tool results" in history[-1]["content"]
        return {
            "text": "<helix_output>Encontré `needle` en `app.py`; no hace falta patch.</helix_output>",
            "actual_model": model,
            "latency_ms": 3.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 30},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        task_root=task_root,
    )
    result = session.task("fijate el repo y buscá needle", max_steps=3)
    assert result["status"] == "completed"
    assert result["mode"] == "read-only"
    assert result["tool_events"][0]["tool"] == "search_text"
    assert "app.py" in json.dumps(result["tool_events"], ensure_ascii=False)
    assert "Encontré" in result["final"]
    assert result["patch_available"] is False
    tool_result_events = [event for event in session.events if event["event"] == "task_tool_result"]
    assert tool_result_events
    assert tool_result_events[0]["helix_memory"]["receipt"]["signature_verified"] is True


def test_interactive_task_repairs_missing_query_tool_arguments(monkeypatch) -> None:
    base = _test_root()
    goal = "revisa nuestra memoria sobre local-ghost-in-the-shell-live"
    calls = {"count": 0}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "text": '<tool_call>{"tool":"helix.search","arguments":{}}</tool_call>',
                "actual_model": model,
                "latency_ms": 2.0,
                "finish_reason": "stop",
                "usage": {"total_tokens": 20},
            }
        return {
            "text": "<helix_output>Use la busqueda reparada y no crashee.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.task(goal, max_steps=2)

    assert result["status"] == "completed"
    assert result["tool_events"][0]["tool"] == "helix.search"
    assert result["tool_events"][0]["arguments"]["query"] == goal
    assert "no crashee" in result["final"]


def test_task_suite_prompt_starts_with_registered_cognitive_gauntlet(monkeypatch) -> None:
    base = _test_root()

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        assert history
        assert "suite.latest" in history[-1]["content"]
        return {
            "text": "<helix_output>Reporte basado en cognitive-gauntlet.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.task('Analiza la suite "cognitive-gauntlet"', max_steps=2)

    assert helix_cli._suite_from_text('suite "cognitive-gauntlet"') == "cognitive-gauntlet"
    assert result["tool_events"][0]["tool"] == "suite.latest"
    assert result["tool_events"][0]["arguments"]["suite_id"] == "cognitive-gauntlet"
    assert "cognitive-gauntlet" in result["final"]


def test_manual_gemini_pro_uses_explicit_fallback_chain(monkeypatch) -> None:
    base = _test_root()
    pro = helix_cli.resolve_model_alias("gemini-pro")
    pro_tools = helix_cli.resolve_model_alias("gemini-pro-tools")
    calls = []

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls.append((provider_name, model))
        if model == pro:
            raise RuntimeError("pro unavailable")
        assert model == pro_tools
        return {
            "text": "<helix_output>Respondido por fallback custom-tools.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="gemini",
        model=pro,
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.chat("hola")

    assert calls == [("gemini", pro), ("gemini", pro_tools)]
    assert result["text"] == "Respondido por fallback custom-tools."
    assert session.last_model_turns[-1]["failover_used"] is True
    assert session.last_model_turns[-1]["failover_attempts"][0]["model"] == pro
    latest = session.events[-1]
    assert latest["metadata"]["selected_model"] == pro
    assert latest["metadata"]["actual_model"] == pro_tools


def test_gemini_rate_limit_stops_same_provider_failover(monkeypatch) -> None:
    helix_cli._PROVIDER_COOLDOWNS.clear()
    calls = []

    def fake_run_chat(provider_name, model, **kwargs):
        calls.append((provider_name, model))
        raise helix_cli.ProviderRateLimitError(
            provider_name,
            model,
            "API Error (gemini): HTTP Error 429: Too Many Requests",
            retry_after_seconds=30,
        )

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)

    with pytest.raises(RuntimeError) as exc_info:
        helix_cli.run_chat_with_failover(
            provider_name="gemini",
            model="gemini-3-flash-preview",
            fallback_models=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            prompt="hola",
        )

    assert "HTTP Error 429" in str(exc_info.value)
    assert calls == [("gemini", "gemini-3-flash-preview")]
    assert helix_cli._provider_cooldown_status("gemini")["active"] is True
    helix_cli._PROVIDER_COOLDOWNS.clear()


def test_chat_gemini_rate_limit_returns_recovery_message(monkeypatch) -> None:
    helix_cli._PROVIDER_COOLDOWNS.clear()
    for env_name in ("DEEPINFRA_API_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(helix_cli, "_config_token", lambda provider_name: None)

    def fake_run_chat(provider_name, model, **kwargs):
        raise helix_cli.ProviderRateLimitError(
            provider_name,
            model,
            "API Error (gemini): HTTP Error 429: Too Many Requests",
            retry_after_seconds=45,
        )

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="gemini",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        evidence_root=_test_root() / "empty-verification",
    )

    result = session.chat("no sé contame que es helix")

    assert "Task failed" not in result["text"]
    assert "HTTP 429" in result["text"]
    assert "prefiero no inventar" in result["text"]
    helix_cli._PROVIDER_COOLDOWNS.clear()


def test_chat_bad_request_returns_recovery_message(monkeypatch) -> None:
    def fake_run_chat(provider_name, model, **kwargs):
        raise RuntimeError("API Error (deepinfra): HTTP Error 400: Bad Request")

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model=helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id,
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
        evidence_root=_test_root() / "empty-verification",
    )

    result = session.chat("yo fui quien creo helix")

    assert "Task failed" not in result["text"]
    assert "HTTP 400 Bad Request" in result["text"]
    assert "recompactar" in result["text"]


def test_gemini_chat_serializes_native_url_context_and_search_grounding(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, *, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "modelVersion": "gemini-2.5-flash",
            "candidates": [
                {
                    "content": {"parts": [{"text": "Respuesta con URL context."}]},
                    "finishReason": "STOP",
                    "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "https://ai.google.dev/"}]},
                }
            ],
            "usageMetadata": {"totalTokenCount": 42},
        }

    monkeypatch.setattr(helix_cli, "_post_json", fake_post_json)
    result = helix_cli._gemini_chat(
        helix_cli.PROVIDERS["gemini"],
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Compará estas docs"}],
        token="gemini-test-key",
        max_tokens=128,
        temperature=0.0,
        timeout=5.0,
        native_request={
            "url_context_urls": [
                "https://ai.google.dev/gemini-api/docs/url-context",
                "https://ai.google.dev/gemini-api/docs/function-calling",
            ],
            "enable_search_grounding": True,
        },
    )

    tools = captured["payload"]["tools"]
    assert {"url_context": {}} in tools
    assert {"google_search": {}} in tools
    assert result["text"] == "Respuesta con URL context."
    assert result["native_tool_metadata"]["url_context_metadata"]["urlMetadata"]


def test_gemini_native_request_blocks_url_context_with_function_calling() -> None:
    try:
        helix_cli._prepare_gemini_native_request(
            "gemini-2.5-flash",
            {
                "url_context_urls": ["https://ai.google.dev/gemini-api/docs/url-context"],
                "function_declarations": [
                    {"name": "lookup_weather", "description": "demo", "parameters": {"type": "object", "properties": {}}}
                ],
            },
        )
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("expected incompatible Gemini native request to raise ValueError")


def test_chat_fallback_summary_does_not_emit_raw_tool_json(monkeypatch) -> None:
    base = _test_root()

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": '<tool_call>{"tool":"suite.search","arguments":{"query":"local-ghost-in-the-shell-live","limit":10}}</tool_call>',
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.chat('Analiza la suite "cognitive-gauntlet" y local-ghost-in-the-shell-live')

    assert not result["text"].lstrip().startswith("{")
    assert "JSON crudo" in result["text"]
    assert "suite.search" in result["text"]


def test_chat_records_path_url_refs_and_native_plan(monkeypatch) -> None:
    base = _test_root()
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["provider_name"] = provider_name
        captured["model"] = model
        captured["native_request"] = kwargs.get("native_request")
        return {
            "text": "<helix_output>Resumen con grounding mixto.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.chat(
        "compará src/helix_proto con https://ai.google.dev/gemini-api/docs/url-context y resumilo"
    )

    assert result["text"] == "Resumen con grounding mixto."
    latest = session.events[-1]
    assert latest["metadata"]["path_refs"]
    assert latest["metadata"]["url_refs"] == ["https://ai.google.dev/gemini-api/docs/url-context"]
    assert latest["metadata"]["capability_requirements"]["url_context"] is True
    assert captured["native_request"]["mode"] == "helix-only"


def test_chat_uses_gemini_native_request_for_url_only_prompt(monkeypatch) -> None:
    base = _test_root()
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["provider_name"] = provider_name
        captured["model"] = model
        captured["native_request"] = kwargs.get("native_request")
        return {
            "text": "<helix_output>Resumen apoyado en URL Context.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        evidence_root=base / "empty-verification",
    )

    result = session.chat(
        "compará https://ai.google.dev/gemini-api/docs/url-context con https://ai.google.dev/gemini-api/docs/function-calling"
    )

    assert result["text"] == "Resumen apoyado en URL Context."
    assert captured["provider_name"] == "gemini"
    assert captured["native_request"]["mode"] == "gemini-native"
    assert len(captured["native_request"]["url_context_urls"]) == 2


def test_chat_records_blind_inference_metadata_when_enabled(monkeypatch) -> None:
    base = _test_root()
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["provider_name"] = provider_name
        captured["blind_inference"] = kwargs.get("blind_inference")
        return {
            "text": "<helix_output>Respuesta rehidratada.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
            "blind_inference": {
                "requested": True,
                "enabled": True,
                "bypassed": False,
                "provider_target": provider_name,
                "policy_id": "blindpolicy1234",
                "task_id": "blind-turn-abc",
                "span_count": 2,
                "sensitive_classes": ["DOC_ID", "PERSON"],
                "warnings": [{"detector": "email", "status": "suggested_not_redacted"}],
                "baseline_redaction_applied": False,
                "vault_present": True,
            },
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
    )
    session.blind_inference_enabled = True
    session.blind_inference_policy = helix_cli.BlindInferencePolicy.from_payload(
        {
            "enabled": True,
            "scope": "cloud_proxy",
            "placeholder_stability": "per_task",
            "rules": [{"name": "person-rule", "type": "PERSON", "values": ["Juan Perez"]}],
        }
    )

    result = session.chat("Analiza a Juan Perez.")

    assert result["text"] == "Respuesta rehidratada."
    assert captured["provider_name"] == "deepinfra"
    assert captured["blind_inference"]["enabled"] is True
    latest = session.events[-1]
    assert latest["metadata"]["blind_inference_enabled"] is True
    assert latest["metadata"]["blind_span_count"] == 2
    assert latest["metadata"]["blind_sensitive_classes"] == ["DOC_ID", "PERSON"]


def test_agent_suggest_command_records_suggest_mode(monkeypatch, capsys) -> None:
    base = _test_root()
    task_root = base / "repo"
    task_root.mkdir(parents=True)

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<helix_output>Plan seguro: leer archivos, proponer patch, correr tests sugeridos.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        task_root=task_root,
    )
    assert helix_cli._handle_interactive_command(session, "/agent suggest revisa el repo estilo codex") is True
    output = capsys.readouterr().out
    assert '"mode": "suggest"' in output
    assert session.last_task_result["mode"] == "suggest"
    assert session.last_task_result["route"]["intent"] == "agentic_code"


def test_style_command_changes_response_register_and_persists(monkeypatch, capsys) -> None:
    saved = {}

    monkeypatch.setattr(helix_cli, "_load_config", lambda: {})
    monkeypatch.setattr(helix_cli, "_save_config", lambda config: saved.update(config) or Path("config.json"))
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )

    assert helix_cli._handle_interactive_command(session, "/style interesante") is True
    assert session.response_style == "vivid"
    assert saved["response_style"] == "vivid"
    assert "response_style=vivid" in capsys.readouterr().out


def test_mode_command_changes_interaction_mode_and_persists(monkeypatch, capsys) -> None:
    saved = {}

    monkeypatch.setattr(helix_cli, "_load_config", lambda: {})
    monkeypatch.setattr(helix_cli, "_save_config", lambda config: saved.update(config) or Path("config.json"))
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )

    assert helix_cli._handle_interactive_command(session, "/mode technical") is True
    assert session.interaction_mode == "technical"
    assert saved["interaction_mode"] == "technical"
    assert "interaction_mode=technical" in capsys.readouterr().out


def test_mode_list_command_reports_profiles(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )

    assert helix_cli._handle_interactive_command(session, "/mode list") is True
    output = capsys.readouterr().out
    assert "balanced" in output
    assert "technical" in output
    assert "explore" in output


def test_blind_command_toggles_and_loads_policy(capsys) -> None:
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=_test_root() / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=_test_root() / "transcripts",
    )

    assert helix_cli._handle_interactive_command(session, "/blind on") is True
    assert session.blind_inference_enabled is True
    assert "blind_inference=on" in capsys.readouterr().out

    policy_json = json.dumps(
        {
            "rules": [
                {
                    "name": "recruit-name",
                    "type": "PERSON",
                    "values": ["Juan Perez"],
                }
            ]
        }
    )
    assert helix_cli._handle_interactive_command(session, f"/blind policy {policy_json}") is True
    assert len(session.blind_inference_policy.rules) == 1

    assert helix_cli._handle_interactive_command(session, "/blind status") is True
    output = capsys.readouterr().out
    assert "policy_id" in output
    assert "\"enabled\": true" in output.lower()

    assert helix_cli._handle_interactive_command(session, "/blind off") is True
    assert session.blind_inference_enabled is False
    assert "blind_inference=off" in capsys.readouterr().out


def test_explore_alias_runs_one_shot_without_changing_sticky_mode(monkeypatch) -> None:
    base = _test_root()

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<helix_output>Exploración creativa anclada.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 18},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
    )
    session.interaction_mode = "balanced"

    assert helix_cli._handle_interactive_command(session, "/explore helix y ghost in the shell") is True
    assert session.interaction_mode == "balanced"
    assert session.events[-1]["metadata"]["interaction_mode"] == "explore"


def test_agent_use_blueprint_selects_blueprint_model_and_records_allowed_tools(monkeypatch, capsys) -> None:
    base = _test_root()
    task_root = base / "repo"
    task_root.mkdir(parents=True)

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<helix_output>Analisis de suite basado en catalogo y transcripts locales.</helix_output>",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        task_root=task_root,
    )

    assert helix_cli._handle_interactive_command(
        session,
        "/agent use suite-run-analyst compara las ultimas corridas de hard-anchor",
    ) is True
    output = capsys.readouterr().out
    assert '"agent_blueprint": "suite-run-analyst"' in output
    assert session.last_task_result["agent_blueprint"] == "suite-run-analyst"
    assert session.last_task_result["route"]["intent"] == "agentic_blueprint"
    assert session.last_task_result["selected_model"] == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id
    task_start = next(event for event in session.events if event["event"] == "task_start")
    assert "suite.latest" in task_start["metadata"]["allowed_tools"]
    assert "suite.read" in task_start["metadata"]["allowed_tools"]


def test_interactive_task_forces_memory_search_before_accepting_preamble(monkeypatch) -> None:
    base = _test_root()
    workspace = base / "workspace"
    task_root = base / "repo"
    task_root.mkdir(parents=True)
    helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="older-thread",
        event_type="note",
        content=(
            "ConclusiÃ³n previa: tratamos a los LLMs como rizomas narrativos y "
            "la hipersticiÃ³n operaba como un bucle performativo."
        ),
        summary="ConclusiÃ³n previa sobre rizomas e hipersticiÃ³n",
        tags=["note"],
        promote=True,
    )
    calls = {"count": 0}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "text": "Voy a buscar en nuestra memoria sobre esos temas especÃ­ficos.",
                "actual_model": model,
                "latency_ms": 2.0,
                "finish_reason": "stop",
                "usage": {"total_tokens": 20},
            }
        assert history
        assert "HeliX read-only tool results" in history[-1]["content"]
        assert "rizomas" in history[-1]["content"].lower()
        return {
            "text": (
                "<helix_output>Concluimos que pensÃ¡bamos a los LLMs como rizomas "
                "narrativos y que la hipersticiÃ³n funcionaba como un bucle performativo."
                "</helix_output>"
            ),
            "actual_model": model,
            "latency_ms": 3.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 30},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        task_root=task_root,
    )
    result = session.task(
        "revisÃ¡ nuestra memoria y resumime quÃ© conclusiÃ³n sacamos sobre los LLMs como rizomas y la hipersticiÃ³n",
        max_steps=3,
    )
    assert calls["count"] == 2
    assert result["status"] == "completed"
    assert result["tool_events"]
    assert result["tool_events"][0]["tool"] == "helix.search"
    assert "rizomas" in result["final"].lower()


def test_interactive_task_suppresses_unparsed_tool_protocol(monkeypatch) -> None:
    base = _test_root()
    task_root = base / "repo"
    task_root.mkdir(parents=True)

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<tool_call>\n1 Input received\n2 Analysis: noisy",
            "actual_model": model,
            "latency_ms": 2.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        task_root=task_root,
    )
    result = session.task("fijate el repo y explicame el bug", max_steps=1, engine_override="helix")
    assert result["final"] == "[raw output suppressed: model returned only internal reasoning or tool protocol residue]"


def test_interactive_task_uses_extended_timeout_and_returns_structured_error(monkeypatch) -> None:
    base = _test_root()
    task_root = base / "repo"
    task_root.mkdir(parents=True)
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        raise RuntimeError("API Error (deepinfra): The read operation timed out")

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=base / "workspace",
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=base / "transcripts",
        task_root=task_root,
    )
    result = session.task("revisá nuestra memoria", max_steps=1)
    assert captured["timeout"] == helix_cli.AGENT_TASK_TIMEOUT_SECONDS
    assert result["status"] == "error"
    assert "timed out" in result["final"]
    assert result["patch_available"] is False
    assert session.last_task_result == result


def test_chat_runner_excludes_current_user_turn_from_initial_memory_context(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="older-thread",
        event_type="note",
        content="Hallazgo previo sobre rizomas y memoria.",
        summary="Hallazgo previo sobre rizomas y memoria.",
        tags=["note"],
        promote=True,
    )

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<helix_output>Hay memoria previa relevante.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    result = session.chat("encontraste algo sobre rizomas?")
    latest_user_memory_id = str((session.events[-2].get("helix_memory") or {}).get("memory_id") or "")
    initial_ids = set(result["trace"]["initial_memory_context"].get("memory_ids") or [])
    assert latest_user_memory_id
    assert latest_user_memory_id not in initial_ids


def test_chat_keeps_contextual_helix_explanation_in_conversation_lane(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="older-thread",
        event_type="note",
        content=(
            "HeliX permite memoria firmada, receipts verificables, búsqueda unificada, "
            "threads persistentes y evidencia certificada."
        ),
        summary="Capacidades verificadas de HeliX",
        tags=["note"],
        promote=True,
    )
    calls = {"count": 0}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        calls["count"] += 1
        return {
            "text": (
                "<helix_output>HeliX te permite persistir hilos, buscar memoria del workspace, "
                "verificar evidencia y dejar receipts firmados por turno.</helix_output>"
            ),
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 18},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="estaba pensando en helix", event_type="user_turn")
    session.record(role="assistant", content="Decime qué querés entender.", event_type="assistant_turn")
    result = session.chat("me gustaría que me ayudes a entenderlo")
    assert calls["count"] == 1
    observations = list(result["trace"].get("observations") or [])
    assert observations == []
    assert result["trace"]["mode"] == "lightweight_chat"
    assert session.events[-1]["metadata"]["lane"] == "conversation"


def test_chat_does_not_promote_contextual_helix_explanations_to_heavy_profile(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["model"] = model
        captured["system"] = system
        return {
            "text": "<helix_output>HeliX organiza memoria firmada, evidencia y tools sobre un thread persistente.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 16},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        router_policy="premium",
    )
    session.record(role="user", content="estaba pensando en helix", event_type="user_turn")
    session.record(role="assistant", content="Dale, exploremos eso.", event_type="assistant_turn")
    result = session.chat("me gustaria que me ayudes a entenderlo")
    assert captured["model"] == helix_cli.DEEPINFRA_MODEL_PROFILES["chat"].model_id
    assert result["route"]["profile"] == "chat"
    assert "Certified repository evidence pack" not in captured["system"]
    assert "HeliX architecture context pack" not in captured["system"]


def test_chat_explore_mode_keeps_creative_helix_without_architecture_pack(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["model"] = model
        captured["system"] = system
        return {
            "text": "<helix_output>Podemos pensar HeliX como un chasis cultural además de un runtime verificable.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 16},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        router_policy="premium",
    )
    session.interaction_mode = "explore"
    session.record(role="user", content="estaba pensando en helix", event_type="user_turn")
    session.record(role="assistant", content="Dale, exploremos eso.", event_type="assistant_turn")
    result = session.chat("me recuerda a ghost in the shell, exploralo")
    observations = list(result["trace"].get("observations") or [])
    assert result["route"]["interaction_mode"] == "explore"
    assert result["route"]["intent"] == "creative_helix"
    assert "HeliX architecture context pack" not in captured["system"]
    assert not any(item.get("tool_name") == "helix.architecture" for item in observations)


def test_chat_helix_auditability_requests_use_architecture_pack_and_audit_profile(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="verification-thread",
        event_type="evidence_ingest",
        content="Evidence artifact showing signature verification and chain status for a HeliX memory node.",
        summary="HeliX evidence with signature and chain verification",
        tags=["evidence", "verification"],
        promote=True,
    )
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["history"] = history
        captured["system"] = system
        return {
            "text": (
                "<helix_output>En HeliX la auditabilidad sale de receipts firmados, node hashes y verificaciones de firma/cadena "
                "sobre memorias y evidencia ingerida.</helix_output>"
            ),
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        router_policy="premium",
    )
    session.record(role="user", content="quiero que exploremos helix", event_type="user_turn")
    session.record(role="assistant", content="Dale, sigamos por ahi.", event_type="assistant_turn")
    result = session.chat("que onda la auditabilidad? los hashes y eso?")
    observations = list(result["trace"].get("observations") or [])
    assert observations
    assert observations[0]["tool_name"] == "helix.trust"
    assert captured["model"] == helix_cli.DEEPINFRA_MODEL_PROFILES["sonnet"].model_id
    assert "Do not narrate retrieval mechanics" in str(captured["prompt"])
    assert "HeliX read-only tool results" in captured["history"][-1]["content"]
    assert result["route"]["profile"] == "sonnet"
    assert result["route"]["interaction_mode"] == "balanced"
    assert "node hashes" in result["text"].lower()
    assert "HeliX architecture context pack" in captured["system"]


def test_interactive_session_reopens_last_active_thread(monkeypatch) -> None:
    workspace = _test_root() / "workspace"

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<helix_output>Seguimos en el mismo hilo.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    first = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    first.chat("hola")
    original_thread = first.thread_id
    assert original_thread

    second = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    assert second.thread_id == original_thread
    assert any(event["event"] == "thread_resume" for event in second.events)


def test_hmem_search_prioritizes_current_thread(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")
    hmem_payloads = [
        ("thread-alpha", "alpha memory from current thread"),
        ("thread-beta", "alpha memory from another thread"),
    ]
    for session_id, content in hmem_payloads:
        helix_cli.hmem.observe_event(
            root=workspace,
            project="test-project",
            agent_id="tester",
            session_id=session_id,
            event_type="note",
            content=content,
            summary=content,
            tags=["note"],
            promote=True,
        )

    result = helix_cli.hmem.search(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="thread-alpha",
        query="alpha memory",
        top_k=2,
        retrieval_scope="workspace",
    )
    assert result["results"][0]["thread_id"] == "thread-alpha"
    assert result["results"][0]["thread_match"] is True


def test_chat_default_uses_session_scope_memory(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")
    monkeypatch.setattr(helix_cli.InteractiveSession, "refresh_evidence", lambda self, query=None, limit=8: {"records": []})
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    current_thread = session.thread_id
    assert current_thread
    current = helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id=current_thread,
        event_type="note",
        content="isolated-token current thread answer",
        summary="isolated-token current",
        tags=["note"],
        promote=True,
    )
    other = helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="other-thread",
        event_type="note",
        content="isolated-token other thread contaminant",
        summary="isolated-token other",
        tags=["note"],
        promote=True,
    )

    context = session.memory_context("isolated-token", refresh_evidence_first=False)

    assert context["retrieval_scope"] == "session"
    assert context["context_policy"] == "thread_only"
    assert current["memory"]["memory_id"] in context["memory_ids"]
    assert other["memory"]["memory_id"] not in context["memory_ids"]


def test_session_scope_does_not_fallback_to_legacy_cross_thread() -> None:
    workspace = _test_root() / "workspace"
    append_memory_event("tester", kind="note", text="legacy-only-token should stay out", root=workspace)

    result = helix_cli.hmem.search(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="empty-thread",
        query="legacy-only-token",
        top_k=4,
        retrieval_scope="session",
    )

    assert result["source"] == "hmem"
    assert result["results"] == []


def test_thread_new_clean_starts_empty_context() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="old question", event_type="user_turn")
    session.record(role="assistant", content="old answer", event_type="assistant_turn")

    created = session.new_thread("clean slate")

    assert created["thread_id"] == session.thread_id
    assert session.recent_history() == []
    current = session.current_thread()
    assert current["context_policy"] == "thread_only"
    assert current["retrieval_scope"] == "session"
    assert current["branch_root_policy"] == "empty_history_thread_only_memory"


def test_branch_new_forks_from_last_completed_turn() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    parent_thread = session.thread_id
    session.record(role="user", content="question before fork", event_type="user_turn")
    assistant = session.record(role="assistant", content="answer before fork", event_type="assistant_turn")
    parent_turn_id = assistant["turn_id"]

    branch = session.branch_thread("alternate path")

    assert branch["thread_id"] == session.thread_id
    assert session.thread_id != parent_thread
    assert session.recent_history() == []
    info = session.current_thread()
    assert info["kind"] == "branch"
    assert info["parent_thread_id"] == parent_thread
    assert info["parent_turn_id"] == parent_turn_id
    assert info["parent_event_memory_id"] == assistant["helix_memory"]["memory_id"]


def test_thread_tree_renders_branch_hierarchy() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    parent_thread = session.thread_id
    session.record(role="assistant", content="fork point", event_type="assistant_turn")
    branch = session.branch_thread("visible branch")

    rendered = session.thread_tree_text()

    assert parent_thread in rendered
    assert branch["thread_id"] in rendered
    assert "parent=" in rendered
    assert "* " + branch["thread_id"] in rendered


def test_global_memory_search_is_explicit(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    current_thread = session.thread_id
    assert current_thread
    other = helix_cli.hmem.observe_event(
        root=workspace,
        project="test-project",
        agent_id="tester",
        session_id="other-thread",
        event_type="note",
        content="global-only-token lives elsewhere",
        summary="global-only-token other",
        tags=["note"],
        promote=True,
    )

    assert helix_cli._handle_interactive_command(session, "/memory global-only-token") is True
    scoped = json.loads(capsys.readouterr().out)
    assert scoped["retrieval_scope"] == "session"
    assert scoped["results"] == []

    assert helix_cli._handle_interactive_command(session, "/memory search --global global-only-token") is True
    global_result = json.loads(capsys.readouterr().out)
    assert global_result["retrieval_scope"] == "workspace"
    assert other["memory"]["memory_id"] in {item["memory_id"] for item in global_result["results"]}


def test_tool_registry_report_includes_unified_runtime_and_cli_tools() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    report = session.tool_registry_report()
    names = {item["name"] for item in report["tools"]}
    assert session.thread_id == report["thread_id"]
    assert "workspace.list_models" in names
    assert "helix.search" in names
    assert "helix.architecture" in names
    assert "helix.trust" in names
    assert "search_text" in names
    assert "evidence.refresh" in names
    assert "file.inspect" in names
    assert "suite.list" in names
    assert "suite.latest" in names
    assert "suite.read" in names
    assert "suite.transcripts" in names
    assert "web.search" in names
    assert "web.read" in names
    blueprints = {item["blueprint_id"] for item in report["agent_blueprints"]}
    assert "suite-run-analyst" in blueprints
    assert "patch-planner" in blueprints


def test_architecture_context_pack_tool_returns_lineage_excerpts_and_claim_boundaries() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="hola", event_type="user_turn")
    registry, _report = session._cli_extra_tool_registry()  # noqa: SLF001

    result = registry.call("helix.architecture", {"query": "auditabilidad merkle dag"})

    payload = result["result"]
    assert payload["kind"] == "helix-architecture-context-pack"
    assert payload["thread_lineage"]["thread_id"] == session.thread_id
    assert payload["claim_boundaries"]
    assert payload["interpretation_rules"]
    assert payload["module_map"]
    assert payload["excerpts"]
    assert any(item["path"] == "helix_kv/memory_catalog.py" for item in payload["excerpts"])
    assert any("Signed head checkpoints" in item for item in payload["verified_invariants"])
    assert any("global non-equivocation" in item for item in payload["verified_invariants"])
    assert any("verified_with_quarantine" in item for item in payload["claim_boundaries"])


def test_trust_command_and_tool_report_signed_checkpoint(capsys) -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="audit this head", event_type="user_turn")
    registry, _report = session._cli_extra_tool_registry()  # noqa: SLF001

    tool_result = registry.call("helix.trust", {"thread_id": session.thread_id})
    assert tool_result["result"]["lineage"]["checkpoint_verified"] is True
    assert tool_result["result"]["head_checkpoint"]["checkpoint_verified"] is True
    assert tool_result["result"]["interpretation_rules"]
    assert any("verified_with_quarantine" in item for item in tool_result["result"]["interpretation_rules"])

    assert helix_cli._handle_interactive_command(session, "/trust current") is True
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["kind"] == "helix-local-trust-summary"
    assert summary["checkpoint_verified"] is True
    assert summary["trust_root_active_key_id"]
    assert "proof" not in summary
    assert "public_key" not in output
    assert "canonical_payload_sha256" not in output

    assert helix_cli._handle_interactive_command(session, "/trust current json") is True
    raw_output = capsys.readouterr().out
    raw_report = json.loads(raw_output)
    assert raw_report["kind"] == "helix-local-trust-report"
    assert raw_report["proof"]
    assert raw_report["trust_root"]["active_key_id"]


def test_chat_does_not_inject_architecture_pack_for_normal_prompts(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["system"] = system
        return {
            "text": "<helix_output>Hola normal.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 8},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    monkeypatch.setattr(helix_cli.InteractiveSession, "refresh_evidence", lambda self, query=None, limit=8: (_ for _ in ()).throw(AssertionError("refresh_evidence should not run for lightweight chat")))
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("hola, como andas?")

    assert result["text"] == "Hola normal."
    assert "HeliX architecture context pack" not in captured["system"]
    assert "Certified HeliX evidence pack" not in captured["system"]
    assert "Certified repository evidence pack" not in captured["system"]
    assert "just answer the question" in captured["system"].lower()
    assert result["trace"]["mode"] == "lightweight_chat"
    assert session.events[-1]["metadata"]["fast_path"] == "lightweight_chat"


# ─── Subset 1+2+3: continuity routing, structured_output, vague follow-ups ──


def test_route_model_for_task_continuity_bumps_dominant_recent_intent() -> None:
    """Two of the last three turns were `code`. A vague-ish next prompt now
    gets a small +0.4 bump on `code`, biasing model selection toward the same
    lane instead of falling back to chat."""
    plain = helix_cli.route_model_for_task(
        "y eso?",
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="balanced",
    )
    with_trail = helix_cli.route_model_for_task(
        "y eso?",
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="balanced",
        recent_intents=["code", "agentic_code", "code"],
    )
    plain_code = float(plain.get("intent_scores", {}).get("code") or 0.0)
    trail_code = float(with_trail.get("intent_scores", {}).get("code") or 0.0)
    assert trail_code >= plain_code + 0.3
    assert any(signal.startswith("continuity:") for signal in with_trail.get("signals") or [])


def test_route_model_for_task_continuity_ignored_without_repeat() -> None:
    """A single non-chat intent in the trail is not enough to bump — needs
    at least 2/N support so a one-off does not override the current prompt."""
    base = helix_cli.route_model_for_task(
        "explica que es esto",
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="balanced",
    )
    with_one = helix_cli.route_model_for_task(
        "explica que es esto",
        provider_name="deepinfra",
        policy="balanced",
        interaction_mode="balanced",
        recent_intents=["code", "chat", "chat"],
    )
    # No continuity signal because only 1 occurrence of `code`.
    assert not any(signal.startswith("continuity:") for signal in with_one.get("signals") or [])
    # Scores stay the same shape as the no-trail baseline.
    assert (with_one.get("intent_scores", {}).get("code") or 0.0) == (base.get("intent_scores", {}).get("code") or 0.0)


def test_capability_requirements_activates_structured_output_for_json_signals() -> None:
    """Prompts that explicitly ask for JSON / schema / `/cert` flip
    structured_output=True; a plain chat prompt stays False."""
    plain = helix_cli._capability_requirements_for_prompt(
        "hola, como andas?",
        intent="chat",
        url_refs=[],
        path_refs=[],
    )
    json_prompt = helix_cli._capability_requirements_for_prompt(
        "devolveme la respuesta en formato JSON",
        intent="chat",
        url_refs=[],
        path_refs=[],
    )
    schema_prompt = helix_cli._capability_requirements_for_prompt(
        "necesito el output con schema validado",
        intent="reasoning",
        url_refs=[],
        path_refs=[],
    )
    agentic_prompt = helix_cli._capability_requirements_for_prompt(
        "arregla el bug en src/foo.py",
        intent="agentic_code",
        url_refs=[],
        path_refs=[],
    )
    assert plain["structured_output"] is False
    assert json_prompt["structured_output"] is True
    assert schema_prompt["structured_output"] is True
    # Tool-calling intents activate it implicitly so the planner gets a
    # well-formed tool_args payload.
    assert agentic_prompt["structured_output"] is True


def test_native_tool_plan_wires_response_format_for_deepinfra_when_supported() -> None:
    """When structured_output is on AND the selected DeepInfra profile
    advertises supports_structured_output, the plan must carry a
    `request_response_format` so `_openai_compatible_chat` can append it
    to the API payload."""
    route = {
        "provider": "deepinfra",
        "model": helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id,
        "intent": "agentic_code",
        "interaction_mode": "balanced",
    }
    plan = helix_cli._native_tool_plan_for_route(route, "arregla el bug en src/foo.py")
    assert plan["request_response_format"] == {"type": "json_object"}


def test_is_continuity_followup_detects_vague_inheritance_only_with_history() -> None:
    """`arreglalo` alone is not enough — it has to inherit from a recent
    agentic/technical intent. Without that history it stays a normal chat."""
    assert helix_cli._is_continuity_followup("arreglalo", None) is False
    assert helix_cli._is_continuity_followup("arreglalo", []) is False
    assert helix_cli._is_continuity_followup("arreglalo", ["chat", "chat"]) is False
    assert helix_cli._is_continuity_followup("arreglalo", ["agentic_code", "code"]) is True
    assert helix_cli._is_continuity_followup("y eso?", ["audit", "audit"]) is True
    assert helix_cli._is_continuity_followup("hacelo", ["research", "research"]) is False
    # Long prompts with explicit objects are not "vague follow-ups".
    assert helix_cli._is_continuity_followup(
        "arregla el bug que aparece cuando llamamos al endpoint",
        ["agentic_code"],
    ) is False


def test_route_natural_language_promotes_vague_followup_to_work_with_history() -> None:
    """Without recent_intents, `hacelo` is too vague and falls through. With
    an agentic trail it gets prefixed with `/work run` so Work Runtime owns it."""
    # `hacelo` is not in _looks_like_agent_task verbs, so without history it
    # falls through to None.
    assert helix_cli._route_natural_language("hacelo") is None
    routed = helix_cli._route_natural_language("hacelo", ["agentic_code", "code"])
    assert routed == "/work run hacelo"
    # `y eso?` is a vague follow-up too — only inherits with a trail.
    assert helix_cli._route_natural_language("y eso?") is None
    assert helix_cli._route_natural_language("y eso?", ["audit", "audit"]) == "/work run y eso?"


def test_route_natural_language_keeps_helix_continuation_in_chat() -> None:
    assert helix_cli._route_natural_language("continua", ["helix_self"]) is None
    assert helix_cli._route_natural_language("amplialo", ["reasoning"]) is None
    assert helix_cli._route_natural_language("y para que sirve?", ["helix_self"]) is None
    assert helix_cli._route_natural_language("como?", ["helix_self"]) is None
    assert helix_cli._route_natural_language("continua", ["agentic_code"]) == "/work run continua"


def test_lightweight_chat_skips_pre_model_memory_context(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["history"] = list(history or [])
        captured["max_tokens"] = kwargs.get("max_tokens")
        return {
            "text": "<helix_output>El otono en Buenos Aires suele ser suave y lindo para caminar.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 8},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "memory_context",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("memory_context should not run for zero-context lightweight chat")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("que es lo mejor del otono en buenos aires?")

    assert result["trace"]["mode"] == "lightweight_chat"
    assert result["trace"]["initial_memory_context"]["skipped"] is True
    assert result["trace"]["timing"]["max_tokens"] <= 384
    assert captured["max_tokens"] <= 384
    assert len(captured["history"]) <= 2
    assert session.events[-1]["metadata"]["memory_context_skipped"] is True
    assert session.status()["last_latency"]["fast_path"] is True
    assert session.status()["last_latency"]["max_tokens"] <= 384


def test_helix_self_chat_uses_lightweight_path_by_default(monkeypatch) -> None:
    workspace = _test_root() / "workspace"

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        return {
            "text": "<helix_output>La latencia viene de contexto pesado; lo vemos sin cargar memoria.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 8},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "refresh_evidence",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("refresh_evidence should be lazy for helix-self fast chat")),
    )
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "memory_context",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("memory_context should not run for helix-self fast chat")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("sigo teniendo problemas de latencia en helix")

    assert result["trace"]["mode"] == "lightweight_chat"
    assert session.events[-1]["metadata"]["latency_trace"]["path"] == "lightweight"


def test_basic_helix_definition_uses_local_fast_answer(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("basic helix intro should not call provider")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("que es helix ?")

    assert result["trace"]["mode"] == "lightweight_chat"
    assert result["trace"]["local_fast_answer"] is True
    assert "CLI/runtime" in result["text"]
    assert "Thinking Process" not in result["text"]


def test_helix_explanatory_followups_stay_conversation_not_agent_shell(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("helix explanatory followups should use local/conversation fast path")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    monkeypatch.setattr(session.runtime, "agent_runner", lambda: (_ for _ in ()).throw(AssertionError("AgentRunner should not run for helix explanatory followups")))

    first = session.chat("que es helix")
    second = session.chat("y para que sirve?")
    third = session.chat("como?")

    assert first["trace"]["mode"] == "lightweight_chat"
    assert second["trace"]["mode"] == "lightweight_chat"
    assert third["trace"]["mode"] == "lightweight_chat"
    assert "/trust current" in first["text"]
    assert "Sirve para" in second["text"]
    assert "Funciona como" in third["text"]
    assert session.events[-1]["metadata"]["lane"] == "conversation"


def test_accepting_helix_evidence_offer_escalates_to_deep() -> None:
    history = [
        {"role": "assistant", "content": "Puedo mostrar evidencia local si queres: /trust current o /evidence latest."},
    ]
    assert helix_cli._is_evidence_acceptance_request("dale", history) is True
    assert helix_cli._needs_certified_evidence("dale", history=history) is True
    assert helix_cli._should_use_lightweight_chat_path(
        "dale",
        route={"intent": "chat"},
        recent_history=history,
        helix_focus=False,
        helix_auditability=False,
        suite_focus=False,
        web_focus=False,
        hash_recovery_ref=None,
        file_path_ref=None,
        url_refs=[],
        latency_mode="fast",
    ) is False
    lane, reason = helix_cli._conversation_lane_for_turn(
        "dale",
        route={"intent": "chat"},
        use_lightweight_chat=False,
        file_path_ref=None,
        helix_auditability=False,
        suite_focus=False,
        web_focus=False,
        hash_recovery_ref=None,
        certified_evidence_required=True,
    )
    assert lane == "deep"
    assert "evidence" in reason


def test_conversation_default_handles_dense_philosophy_without_agentrunner(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["system"] = system
        return {
            "text": "<helix_output>Podemos pensarlo como una tension entre continuidad, agencia y criterio situado.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    monkeypatch.setattr(session.runtime, "agent_runner", lambda: (_ for _ in ()).throw(AssertionError("agent runner should not run for philosophy chat")))

    result = session.chat("pensemos filosoficamente la relacion entre memoria agencia y continuidad")

    assert result["trace"]["mode"] == "lightweight_chat"
    assert session.events[-1]["metadata"]["lane"] == "conversation"
    assert "continuidad" in result["text"]


def test_technical_dense_chat_does_not_load_evidence_without_explicit_signal(monkeypatch) -> None:
    workspace = _test_root() / "workspace"

    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "refresh_evidence",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("technical chat should not refresh evidence without explicit signal")),
    )
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "memory_context",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("technical chat should not load memory by default")),
    )
    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: {
            "text": "<helix_output>La diferencia tecnica central es separar contrato, politica y ejecucion.</helix_output>",
            "actual_model": kwargs.get("model"),
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("analicemos tecnicamente una arquitectura conversacional robusta", interaction_mode_override="technical")

    assert result["trace"]["mode"] == "lightweight_chat"
    assert session.events[-1]["metadata"]["lane"] == "conversation"


def test_thread_summary_is_injected_without_global_memory(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["system"] = system
        captured["history"] = list(history or [])
        return {
            "text": "<helix_output>Sigo el hilo anterior sobre memoria y agencia.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "memory_context",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("thread summary should not use semantic memory")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    for index in range(3):
        session.record(role="user", content=f"punto {index}: memoria como continuidad", event_type="user_turn")
        session.record(role="assistant", content=f"respuesta {index}: agencia situada", event_type="assistant_turn")

    result = session.chat("continuemos con esa linea")

    assert "Active thread summary" in captured["system"]
    assert result["trace"]["initial_memory_context"]["thread_summary_used"] is True
    assert session.events[-1]["metadata"]["thread_summary_used"] is True


def test_thread_summary_does_not_cross_branch_boundaries() -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="tema padre", event_type="user_turn")
    session.record(role="assistant", content="respuesta padre", event_type="assistant_turn")
    parent_summary = session.ensure_thread_summary(force=True)
    assert parent_summary["summary"]

    session.branch_thread("summary isolation")

    assert session.conversation_status()["thread_summary_present"] is False


def test_thinking_process_only_output_triggers_repair_or_local_fallback(monkeypatch) -> None:
    workspace = _test_root() / "workspace"

    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: {
            "text": "Thinking Process:\n1 Analyze the request\n2 Draft answer",
            "actual_model": kwargs.get("model"),
            "latency_ms": 1.0,
            "finish_reason": "length",
            "usage": {"total_tokens": 10},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )

    result = session.chat("hola, pensemos algo")

    gate = session.events[-1]["metadata"]["response_gate"]
    assert gate["suppressed_reasoning"] is True
    assert gate["local_fallback_used"] is True
    assert "razonamiento interno" in result["text"]


def test_read_local_document_routes_to_file_qa_and_uses_file_inspect(monkeypatch) -> None:
    base = _test_root()
    base.mkdir(parents=True, exist_ok=True)
    workspace = base / "workspace"
    doc = base / "notes.txt"
    doc.write_text("HeliX file QA reads this exact note.", encoding="utf-8")
    captured = {}

    def fake_run_chat(provider_name, model, prompt, system, history=None, **kwargs):
        captured["prompt"] = prompt
        captured["history"] = history
        return {
            "text": "<helix_output>El documento dice que file QA lee la nota exacta.</helix_output>",
            "actual_model": model,
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(helix_cli, "run_chat", fake_run_chat)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=base,
    )

    result = session.chat(f'lee "{doc}" y resumilo')

    assert result["trace"]["mode"] == "file_qa"
    assert result["trace"]["observations"][0]["tool_name"] == "file.inspect"
    assert session.events[-1]["metadata"]["lane"] == "file_qa"
    assert "file.inspect observations" in captured["prompt"]


def test_file_qa_blocks_sensitive_paths(monkeypatch) -> None:
    base = _test_root()
    base.mkdir(parents=True, exist_ok=True)
    workspace = base / "workspace"
    secret = base / ".env"
    secret.write_text("TOKEN=secret", encoding="utf-8")
    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("blocked file QA should not call provider")),
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        task_root=base,
    )

    result = session.chat(f'lee "{secret}"')

    assert result["trace"]["mode"] == "file_qa"
    assert "bloquea" in result["text"].lower()
    assert session.events[-1]["metadata"]["local_fallback_used"] is True


def test_task_not_invoked_for_philosophical_or_product_discussion() -> None:
    assert helix_cli._route_natural_language("compará conceptualmente agencia y memoria") is None
    assert helix_cli._route_natural_language("pensemos mejoras de producto para helix") is None


def test_latency_reports_lane_and_response_gate(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: {
            "text": "<helix_output>Hola conversacional.</helix_output>",
            "actual_model": kwargs.get("model"),
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 4},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.chat("hola normal")
    assert helix_cli._handle_interactive_command(session, "/latency") is True
    output = json.loads(capsys.readouterr().out)
    assert output["last_latency"]["lane"] == "conversation"
    assert "response_gate_ms" in output["last_latency"]


def test_conversation_status_reports_summary_and_last_route_reason(monkeypatch, capsys) -> None:
    workspace = _test_root() / "workspace"
    monkeypatch.setattr(
        helix_cli,
        "run_chat",
        lambda *args, **kwargs: {
            "text": "<helix_output>Seguimos.</helix_output>",
            "actual_model": kwargs.get("model"),
            "latency_ms": 1.0,
            "finish_reason": "stop",
            "usage": {"total_tokens": 4},
        },
    )
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=128,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.record(role="user", content="venimos pensando continuidad", event_type="user_turn")
    session.record(role="assistant", content="la continuidad queda como criterio conversacional", event_type="assistant_turn")
    session.ensure_thread_summary(force=True)
    session.chat("sigamos pensando")

    assert helix_cli._handle_interactive_command(session, "/conversation status") is True
    output = json.loads(capsys.readouterr().out)
    assert output["lane"] == "conversation"
    assert output["last_route_reason"]
    assert output["thread_summary_present"] is True


def test_helix_audit_request_uses_grounded_path(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    calls = {"memory": 0, "evidence": 0}

    class FakeRunner:
        def run(self, **kwargs):
            return {"final_answer": "<helix_output>auditado</helix_output>", "planner_attempts": [], "observations": []}

    def fake_memory(self, *args, **kwargs):
        calls["memory"] += 1
        return {"context": "", "memory_ids": [], "tokens": 0}

    def fake_evidence(self, *args, **kwargs):
        calls["evidence"] += 1
        return {"records": []}

    monkeypatch.setattr(helix_cli.InteractiveSession, "memory_context", fake_memory)
    monkeypatch.setattr(helix_cli.InteractiveSession, "refresh_evidence", fake_evidence)
    monkeypatch.setattr(helix_cli.InteractiveSession, "architecture_context_pack", lambda self, *args, **kwargs: None)
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    monkeypatch.setattr(session.runtime, "agent_runner", lambda: FakeRunner())
    monkeypatch.setattr(session, "_planner_callback_factory", lambda **kwargs: (lambda *_args, **_kwargs: None, [{"latency_ms": 2.0, "raw_text": "auditado"}]))

    result = session.chat("auditá helix con evidencia y hashes")

    assert result["trace"]["final_answer"] == "<helix_output>auditado</helix_output>"
    assert calls["memory"] == 1
    assert calls["evidence"] >= 1
    assert session.events[-1]["metadata"]["latency_trace"]["path"] == "grounded"


def test_thread_open_does_not_refresh_evidence(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    thread_id = session.thread_id
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "refresh_evidence",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("thread open should not refresh evidence")),
    )

    session.open_thread(thread_id)


def test_memory_context_does_not_refresh_by_default(monkeypatch) -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    monkeypatch.setattr(
        helix_cli.InteractiveSession,
        "refresh_evidence",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("memory_context should not refresh by default")),
    )
    monkeypatch.setattr(helix_cli.hmem, "build_context", lambda **kwargs: {"context": "", "memory_ids": [], "tokens": 0})

    context = session.memory_context("hola")

    assert context["context_policy"] == "thread_only"


def test_latency_command_renders_last_breakdown(capsys) -> None:
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
    )
    session.last_latency_trace = {"path": "lightweight", "total_turn_ms": 12.0, "dominant_phase": "provider_latency_ms"}

    assert helix_cli._handle_interactive_command(session, "/latency") is True

    output = json.loads(capsys.readouterr().out)
    assert output["latency_mode"] == "fast"
    assert output["last_latency"]["path"] == "lightweight"


def test_latency_mode_fast_balanced_deep_changes_routing_policy() -> None:
    route = {"intent": "helix_self"}

    assert helix_cli._should_use_lightweight_chat_path(
        "explicame helix",
        route=route,
        recent_history=[],
        helix_focus=True,
        helix_auditability=False,
        suite_focus=False,
        web_focus=False,
        hash_recovery_ref=None,
        file_path_ref=None,
        url_refs=[],
        latency_mode="fast",
    ) is True
    assert helix_cli._should_use_lightweight_chat_path(
        "explicame helix",
        route=route,
        recent_history=[],
        helix_focus=True,
        helix_auditability=False,
        suite_focus=False,
        web_focus=False,
        hash_recovery_ref=None,
        file_path_ref=None,
        url_refs=[],
        latency_mode="balanced",
    ) is False
    assert helix_cli._should_use_lightweight_chat_path(
        "hola",
        route={"intent": "chat"},
        recent_history=[],
        helix_focus=False,
        helix_auditability=False,
        suite_focus=False,
        web_focus=False,
        hash_recovery_ref=None,
        file_path_ref=None,
        url_refs=[],
        latency_mode="deep",
    ) is False


def test_rust_core_route_fast_path_under_budget() -> None:
    started = time.perf_counter()
    route = helix_cli.helix_cli_core.route("que pasa con HeliX y la latencia?", latency_mode="fast")
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert route["path"] == "lightweight"
    assert route["deep_required"] is False
    assert elapsed_ms < 250


def test_suite_list_uses_fast_index_without_deep_scan(monkeypatch) -> None:
    evidence_root, _suite_dir, _paths = _write_suite_fixture(_test_root())
    monkeypatch.setenv("HELIX_SUITE_CATALOG_MODE", "fast")
    catalog = helix_cli.SuiteEvidenceCatalog(evidence_root=evidence_root)
    refresh = catalog.refresh_index()
    assert refresh["status"] == "ok"

    def fail_deep_scan(*_args, **_kwargs):
        raise AssertionError("deep suite scan should not run when fast index is enabled")

    monkeypatch.setattr(catalog, "_iter_suite_files", fail_deep_scan)
    started = time.perf_counter()
    listed = catalog.list_suites()
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert listed["status"] == "ok"
    assert listed["suite_count"] == 1
    assert listed["suites"][0]["suite_id"] == "hard-anchor-utility"
    assert elapsed_ms < 500


def test_suite_search_uses_index_unless_deep_requested(monkeypatch) -> None:
    evidence_root, _suite_dir, _paths = _write_suite_fixture(_test_root())
    monkeypatch.setenv("HELIX_SUITE_CATALOG_MODE", "fast")
    catalog = helix_cli.SuiteEvidenceCatalog(evidence_root=evidence_root)
    catalog.refresh_index()

    def fail_body_scan(*_args, **_kwargs):
        raise AssertionError("body scan should only run with --deep")

    monkeypatch.setattr(catalog, "_search_record", fail_body_scan)
    result = catalog.search("lineage verified", limit=5)

    assert result["status"] == "ok"
    assert result["result_count"] >= 1
    assert result["results"][0]["snippet"]


def test_refresh_evidence_fast_skips_replay(monkeypatch) -> None:
    evidence_root, _suite_dir, _paths = _write_suite_fixture(_test_root())
    monkeypatch.setenv("HELIX_SUITE_CATALOG_MODE", "fast")
    workspace = _test_root() / "workspace"
    session = helix_cli.InteractiveSession(
        provider_name="deepinfra",
        model="auto",
        workspace_root=workspace,
        project="test-project",
        agent_id="tester",
        max_tokens=64,
        temperature=0.0,
        transcript_dir=workspace / "transcripts",
        evidence_root=evidence_root,
    )
    session.suite_catalog.refresh_index()

    def fail_deep_refresh(*_args, **_kwargs):
        raise AssertionError("deep evidence replay should require deep=True")

    monkeypatch.setattr(helix_cli, "refresh_evidence", fail_deep_refresh)
    pack = session.refresh_evidence("hard anchor", limit=4)

    assert pack["source"] == "helix-evidence-index-fast"
    assert pack["replay_skipped"] is True
    assert pack["record_count"] >= 1


def test_lightweight_chat_token_budget_keeps_simple_turns_short() -> None:
    assert helix_cli._lightweight_chat_token_budget("hola", 2048) == 220
    assert helix_cli._lightweight_chat_token_budget("que es lo mejor del otono en buenos aires?", 2048) == 384
    assert helix_cli._lightweight_chat_token_budget("explicame bien y con detalle el plan completo", 2048) == 700
    assert helix_cli._lightweight_chat_token_budget("hola", 128) == 128


def test_post_json_uses_persistent_http_session_when_available(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        status_code = 200
        reason = "OK"
        headers = {"content-type": "application/json"}
        content = b'{"ok": true}'

        def json(self):
            return {"ok": True}

    class FakeSession:
        def post(self, url, json, headers, timeout):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setattr(helix_cli, "_HTTP_SESSION", FakeSession())

    result = helix_cli._post_json(
        "https://example.test/v1/chat/completions",
        {"model": "m", "messages": []},
        headers={"Authorization": "Bearer test"},
        timeout=3.0,
    )

    assert result == {"ok": True}
    assert calls == [
        {
            "url": "https://example.test/v1/chat/completions",
            "json": {"model": "m", "messages": []},
            "headers": {"Authorization": "Bearer test"},
            "timeout": 3.0,
        }
    ]
