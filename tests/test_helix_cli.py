from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pytest

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto import agent as helix_agent
from helix_proto import helix_cli


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
    assert helix_cli.resolve_model_alias("llama-vision") == helix_cli.DEEPINFRA_MODEL_PROFILES["llama-vision"].model_id
    assert helix_cli.resolve_model_alias("auto") == "auto"


def test_models_payload_exposes_capabilities_and_provider_constraints() -> None:
    payload = helix_cli.models_payload()
    gemini_profile = next(item for item in payload["gemini_model_profiles"] if item["alias"] == "gemini-pro")
    gemini_provider = next(item for item in payload["providers"] if item["name"] == "gemini")
    assert gemini_profile["supports_url_context"] is True
    assert "docs_synthesis" in gemini_profile["preferred_workloads"]
    assert "url_context" in gemini_provider["native_capabilities"]
    assert gemini_provider["native_constraints"]
    assert any(item["name"] == "technical" for item in payload["interaction_modes"])


def test_router_blueprints_report_lists_current_and_hybrid_presets() -> None:
    blueprints = {item["name"]: item for item in helix_cli.router_blueprints_report()}
    assert "balanced" in blueprints
    assert "current" in blueprints
    assert "qwen-heavy" in blueprints
    assert "qwen-gemma-mistral" in blueprints
    assert blueprints["balanced"]["reasoning_alias"] == "reasoning"
    assert blueprints["balanced"]["research_alias"] == "qwen-big"
    assert blueprints["qwen-heavy"]["default_alias"] == "qwen-big"
    assert blueprints["current"]["research_alias"] == "legacy-research"


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
    assert helix_cli._route_natural_language("lee src/helix_proto y resumilo") == "/read src/helix_proto"


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
    markup = helix_cli._prompt_toolbar_markup(session)
    assert "thread" in markup
    assert "provider" in markup
    assert "model" in markup
    assert "router" in markup
    assert "mode" in markup
    assert "theme" in markup
    assert "explore" in markup
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
    helix_cli._render_boot_banner(console)
    helix_cli._render_session_ribbon(console, session)
    output = console.export_text()
    assert "HeliX Inference OS" in output
    assert "SESSION BUS" in output
    assert "mode" in output
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


def test_natural_language_repo_work_routes_to_task() -> None:
    routed = helix_cli._route_natural_language("fijate el repo y armame un patch para el bug")
    assert routed.startswith("/task ")


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
    assert helix_cli._needs_certified_evidence("me gustaria que me ayudes a entenderlo", history=history) is True


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


def test_contextual_helix_followup_injects_certified_evidence_pack(monkeypatch) -> None:
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
    assert "Certified HeliX evidence pack" in system
    assert '"claim": "This HeliX CLI session is backed by HeliX memory and evidence exports."' in system
    assert "Do not claim that HeliX captures 'trajectories of thought'" in system


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
    assert not (result["trace"].get("observations") or [])


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
    result = session.task("fijate el repo y explicame el bug", max_steps=1)
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


def test_chat_forces_helix_search_for_contextual_explanation_requests(monkeypatch) -> None:
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
        assert history
        assert "HeliX read-only tool results" in history[-1]["content"]
        assert "helix" in history[-1]["content"].lower()
        assert "memoria" in history[-1]["content"].lower() or "memory" in history[-1]["content"].lower()
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
    assert observations
    assert observations[0]["tool_name"] == "helix.architecture"
    assert "receipts firmados" in result["text"].lower()


def test_chat_promotes_contextual_helix_explanations_to_reasoning_profile(monkeypatch) -> None:
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
    assert captured["model"] == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-big"].model_id
    assert result["route"]["profile"] == "qwen-big"
    assert "Do not pad HeliX explanations with generic industry examples" in captured["system"]
    assert "HeliX architecture context pack" in captured["system"]


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
    assert "helix-local-trust-report" in output
    assert "checkpoint_verified" in output
    assert "active_key_id" in output


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
