from __future__ import annotations

import json
import uuid
from pathlib import Path

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto import agent as helix_agent
from helix_proto import helix_cli


def _test_root() -> Path:
    return Path.cwd() / "verification" / "cli-sessions" / "_test" / uuid.uuid4().hex


def test_provider_registry_includes_cloud_local_and_openai_compatible() -> None:
    assert "deepinfra" in helix_cli.PROVIDERS
    assert "ollama" in helix_cli.PROVIDERS
    assert "llamacpp" in helix_cli.PROVIDERS
    assert "local" in helix_cli.PROVIDERS
    assert helix_cli.PROVIDERS["deepinfra"].token_env == "DEEPINFRA_API_TOKEN"
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
    assert suites["policy-rag-legal-debate"]["script_exists"] is True


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
    assert route["profile"] == "research"
    assert route["model"] == "Qwen/Qwen3.5-122B-A10B"
    assert route["blueprint"] == "balanced"


def test_auto_router_selects_gemma_for_reasoning_in_balanced_blueprint() -> None:
    route = helix_cli.route_model_for_task(
        "analiza los tradeoffs y desglosa la hipotesis paso a paso",
        provider_name="deepinfra",
        policy="balanced",
    )
    assert route["intent"] == "reasoning"
    assert route["profile"] == "reasoning"
    assert route["model"] == "google/gemma-4-31B"


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


def test_model_alias_resolution() -> None:
    assert helix_cli.resolve_model_alias("mistral").startswith("mistralai/")
    assert helix_cli.resolve_model_alias("sonnet") == helix_cli.DEEPINFRA_MODEL_PROFILES["sonnet"].model_id
    assert helix_cli.resolve_model_alias("qwen") == helix_cli.DEEPINFRA_MODEL_PROFILES["qwen-122b"].model_id
    assert helix_cli.resolve_model_alias("gemma") == helix_cli.DEEPINFRA_MODEL_PROFILES["gemma"].model_id
    assert helix_cli.resolve_model_alias("llama-vision") == helix_cli.DEEPINFRA_MODEL_PROFILES["llama-vision"].model_id
    assert helix_cli.resolve_model_alias("auto") == "auto"


def test_router_blueprints_report_lists_current_and_hybrid_presets() -> None:
    blueprints = {item["name"]: item for item in helix_cli.router_blueprints_report()}
    assert "balanced" in blueprints
    assert "current" in blueprints
    assert "qwen-gemma-mistral" in blueprints
    assert blueprints["balanced"]["reasoning_alias"] == "reasoning"
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
    markup = helix_cli._prompt_toolbar_markup(session)
    assert "thread" in markup
    assert "provider" in markup
    assert "model" in markup
    assert "router" in markup
    assert "theme" in markup
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
    assert not helix_cli._is_identity_question("hola")


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
    assert memory["receipt"]["key_provenance"] == "ephemeral_preregistered"


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
    assert '"key_provenance": "ephemeral_preregistered"' in system
    assert '"tombstone_boundary"' in system


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
    assert "search_text" in names
    assert "evidence.refresh" in names
    assert "suite.list" in names
