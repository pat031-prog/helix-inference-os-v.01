from pathlib import Path

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto.api import HelixRuntime, _AssistantTokenCleaner, _sanitize_assistant_text
from helix_proto.workspace import model_workspace, save_model_info


def _save_model(root: Path, alias: str) -> None:
    model_dir = model_workspace(alias, root)
    save_model_info(
        model_dir,
        {
            "alias": alias,
            "alias_slug": alias.lower().replace(" ", "-"),
            "model_ref": alias,
            "model_dir": str(model_dir),
            "export_dir": str(model_dir / "export"),
        },
    )


def test_stream_assistant_text_builds_messages(monkeypatch, tmp_path: Path) -> None:
    _save_model(tmp_path, "tiny-gpt2")
    runtime = HelixRuntime(root=tmp_path)
    captured = {}

    def fake_stream_text(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        yield {"event": "token", "token_text": "hola"}
        yield {"event": "done"}

    monkeypatch.setattr(runtime, "stream_text", fake_stream_text)

    events = list(
        runtime.stream_assistant_text(
            assistant_id="general",
            message="Decime hola",
            max_new_tokens=32,
        )
    )

    assert captured["alias"] == "tiny-gpt2"
    assert captured["prompt"] is None
    assert captured["messages"][0]["role"] == "system"
    assert "/no_think" in captured["messages"][0]["content"]
    assert captured["messages"][1] == {"role": "user", "content": "Decime hola\n\n/no_think"}
    assert events[0]["assistant_id"] == "general"


def test_assistant_token_cleaner_drops_thinking_prefix() -> None:
    cleaner = _AssistantTokenCleaner()

    assert cleaner.push("Thinking") == ""
    assert cleaner.push(" Process:\n\n1. paso\n\n") == ""
    assert cleaner.push("Helix es un runtime local.") == "Helix es un runtime local."


def test_assistant_token_cleaner_drops_numbered_reasoning_scaffold() -> None:
    cleaner = _AssistantTokenCleaner()

    assert cleaner.push("1. **Analyze the Request:**\n\n") == ""
    assert cleaner.push("2. **Determine the Response:**\n\n") == ""
    assert cleaner.push("Hola, en que te puedo ayudar?") == "Hola, en que te puedo ayudar?"


def test_sanitize_assistant_text_removes_no_think_suffix() -> None:
    assert _sanitize_assistant_text("hola\n\n/no_think") == "hola"
    assert _sanitize_assistant_text("/no_think") == ""


def test_generate_assistant_text_uses_plain_completion_for_llama(monkeypatch, tmp_path: Path) -> None:
    _save_model(tmp_path, "tiny-gpt2")
    runtime = HelixRuntime(root=tmp_path)
    captured = {}

    monkeypatch.setattr(runtime, "_uses_llama_cpp", lambda alias: True)
    monkeypatch.setattr(runtime, "_engine", lambda alias, cache_mode: object())

    def fake_chat_completion(engine, **kwargs):  # noqa: ANN001, ANN003
        del engine
        captured.update(kwargs)
        return {"choices": [{"message": {"content": " hola"}}]}

    monkeypatch.setattr(runtime, "_assistant_llama_chat_completion", fake_chat_completion)

    result = runtime.generate_assistant_text(
        assistant_id="general",
        message="hola",
        max_new_tokens=32,
    )

    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "hola\n\n/no_think"
    assert captured["stream"] is False
    assert result["completion_text"] == "hola"


def test_prewarm_assistants_loads_unique_alias_once(monkeypatch, tmp_path: Path) -> None:
    _save_model(tmp_path, "tiny-gpt2")
    runtime = HelixRuntime(root=tmp_path)
    calls = []

    monkeypatch.setattr(runtime, "_uses_llama_cpp", lambda alias: True)
    monkeypatch.setattr(runtime, "_engine", lambda alias, cache_mode: object())

    def fake_chat_completion(engine, **kwargs):  # noqa: ANN001, ANN003
        del engine
        calls.append(kwargs["messages"])
        return {"choices": [{"message": {"content": "hola"}}]}

    monkeypatch.setattr(runtime, "_assistant_llama_chat_completion", fake_chat_completion)

    warmed = runtime.prewarm_assistants()

    assert len(warmed) == 1
    assert warmed[0]["alias"] == "tiny-gpt2"
    assert len(calls) == 1


def test_openai_chat_completion_wraps_generate_text(monkeypatch, tmp_path: Path) -> None:
    runtime = HelixRuntime(root=tmp_path)
    captured = {}

    def fake_generate_text(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "completion_text": "Observation: safe claim.",
            "prompt_ids": [1, 2, 3],
            "new_ids": [4, 5],
            "session_id": kwargs["session_id"],
            "session_dir": str(tmp_path / "sessions" / kwargs["session_id"]),
            "prefix_reuse_status": "hit",
        }

    monkeypatch.setattr(runtime, "generate_text", fake_generate_text)

    response = runtime.openai_chat_completion(
        {
            "model": "tiny-gpt2",
            "messages": [{"role": "user", "content": "hola"}],
            "max_tokens": 2,
            "extra_body": {
                "helix_session_id": "demo-session",
                "agent_id": "agent-a",
                "audit_policy": "deferred",
                "compression_mode": "turbo-int8-hadamard",
            },
        }
    )

    assert captured["alias"] == "tiny-gpt2"
    assert captured["session_id"] == "demo-session"
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "Observation: safe claim."
    assert response["usage"]["total_tokens"] == 5
    assert response["helix"]["agent_id"] == "agent-a"
    assert response["helix"]["prefix_reuse_status"] == "hit"


def test_openai_chat_completion_injects_memory_context_when_requested(monkeypatch, tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "session-os" / "memory.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="agent-a",
            memory_type="semantic",
            summary="Deferred audit claim wording",
            content="Pending means checkpoint is usable; verified means cryptographic audit is closed.",
            importance=9,
        )
    finally:
        catalog.close()
    runtime = HelixRuntime(root=tmp_path)
    captured = {}

    def fake_generate_text(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "completion_text": "Observation: safe claim.",
            "prompt_ids": [1, 2, 3],
            "new_ids": [4],
            "session_id": kwargs["session_id"],
            "session_dir": str(tmp_path / "sessions" / kwargs["session_id"]),
            "prefix_reuse_status": "not_checked",
        }

    monkeypatch.setattr(runtime, "generate_text", fake_generate_text)

    response = runtime.openai_chat_completion(
        {
            "model": "tiny-gpt2",
            "messages": [{"role": "user", "content": "How should we describe deferred audit?"}],
            "max_tokens": 1,
            "extra_body": {
                "helix_session_id": "demo-session",
                "agent_id": "agent-a",
                "helix_project": "helix",
                "helix_memory_mode": "search",
                "helix_recall_query": "pending verified audit",
                "helix_memory_budget_tokens": 120,
            },
        }
    )

    assert captured["messages"][0]["role"] == "system"
    assert "<helix-memory-context>" in captured["messages"][0]["content"]
    assert response["helix"]["memory_mode"] == "search"
    assert response["helix"]["memory_ids"] == [memory.memory_id]
    assert response["helix"]["memory_context_tokens"] > 0


def test_runtime_memory_endpoints_surface_catalog_search_context_and_graph(tmp_path: Path) -> None:
    runtime = HelixRuntime(root=tmp_path)

    observed = runtime.memory_observe(
        {
            "project": "helix",
            "agent_id": "agent-a",
            "session_id": "run-1",
            "event_type": "tool_call",
            "content": "Tool found that deferred audit means pending checkpoints verify later.",
            "summary": "Deferred audit observation",
            "tags": ["audit"],
            "importance": 8,
            "promote": True,
        }
    )
    search = runtime.memory_search({"project": "helix", "agent_id": "agent-a", "query": "pending verify", "top_k": 3})
    context = runtime.memory_context(
        {"project": "helix", "agent_id": "agent-a", "query": "deferred audit", "budget_tokens": 80}
    )
    graph = runtime.memory_graph({"project": "helix", "agent_id": "agent-a", "limit": 20})
    stats = runtime.memory_stats()

    assert observed["memory"]["memory_id"]
    assert search["results"][0]["kind"] == "memory"
    assert "<helix-memory-context>" in context["context"]
    assert graph["node_count"] >= 2
    assert graph["edge_count"] >= 1
    assert stats["memory_count"] == 1
