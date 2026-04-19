import json
from pathlib import Path

import pytest

from helix_kv.memory_catalog import MemoryCatalog
from helix_proto.agent import AgentRunner
from helix_proto.memory import add_knowledge_text, search_knowledge


@pytest.fixture
def permissive_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests exercise agent-runner memory injection with unsigned_legacy receipts;
    # opt out of the strict default so search returns the unsigned memories.
    monkeypatch.setenv("HELIX_RETRIEVAL_SIGNATURE_ENFORCEMENT", "permissive")


def test_knowledge_search_returns_relevant_chunk(tmp_path: Path) -> None:
    add_knowledge_text(
        "helix",
        "Josh described a fallback meta-agent pattern for compressed local models.",
        source="notes",
        root=tmp_path,
    )

    result = search_knowledge("helix", "fallback meta-agent", root=tmp_path, top_k=3)

    assert result["results"]
    assert "meta-agent" in result["results"][0]["text"]


def test_agent_runner_uses_rag_then_worker_model(tmp_path: Path) -> None:
    class FakeRuntime:
        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            prompt = kwargs["prompt"]
            if "Context:" in prompt:
                return {
                    "completion_text": "Use a local worker first, then fallback to remote planning when needed.",
                    "generated_text": "Use a local worker first, then fallback to remote planning when needed.",
                    "new_ids": [],
                    "generated_ids": [],
                }
            return {
                "completion_text": "generic completion",
                "generated_text": "generic completion",
                "new_ids": [],
                "generated_ids": [],
            }

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"completion_text": "resumed"}

    runner = AgentRunner(FakeRuntime(), root=tmp_path)
    runner.add_knowledge_text(
        "helix-agent",
        "The best policy is local worker first, remote planner fallback second.",
        source="design-notes",
    )

    result = runner.run(
        goal="How should the fallback policy work?",
        agent_name="helix-agent",
        default_model_alias="tiny-agent",
        max_steps=4,
    )

    assert result["final_answer"].startswith("Use a local worker first")
    assert result["steps"][0]["tool_name"] == "helix.search"
    assert result["steps"][1]["tool_name"] == "gpt.generate_text"
    trace_path = Path(result["trace_path"])
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["final_answer"] == result["final_answer"]
    assert trace["steps"][0]["hmem_memory_id"]
    catalog = MemoryCatalog.open(tmp_path / "session-os" / "memory.sqlite")
    try:
        stats = catalog.stats()
    finally:
        catalog.close()
    assert stats["observation_count"] >= 2
    assert stats["memory_count"] >= 2


def test_agent_runner_injects_hmem_context_into_planner(tmp_path: Path, permissive_retrieval: None) -> None:
    catalog = MemoryCatalog.open(tmp_path / "session-os" / "memory.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="memory-agent",
            memory_type="semantic",
            summary="Use deferred audit wording carefully",
            content="Pending checkpoints are usable; verified receipts close cryptographic integrity.",
            importance=9,
        )
    finally:
        catalog.close()

    class FakeRuntime:
        def __init__(self) -> None:
            self.planner_prompt = ""

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            self.planner_prompt = kwargs["prompt"]
            payload = {"kind": "final", "thought": "memory is enough", "final": "Use pending vs verified carefully."}
            text = json.dumps(payload)
            return {"completion_text": text, "generated_text": text}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)

    result = runner.run(
        goal="How should we phrase deferred audit?",
        agent_name="memory-agent",
        local_planner_alias="planner-local",
        max_steps=1,
    )

    assert result["memory_context"]["memory_ids"] == [memory.memory_id]
    assert "<helix-memory-context>" in runtime.planner_prompt
    assert result["final_answer"] == "Use pending vs verified carefully."


def test_agent_runner_injects_active_memory_into_worker_prompt(tmp_path: Path, permissive_retrieval: None) -> None:
    catalog = MemoryCatalog.open(tmp_path / "session-os" / "memory.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="active-agent",
            memory_type="semantic",
            summary="Deployment gate requires sigstore",
            content="The deployment policy requires sigstore verification before release.",
            importance=9,
        )
    finally:
        catalog.close()

    class FakeRuntime:
        def __init__(self) -> None:
            self.planner_calls = 0
            self.worker_prompt = ""

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            if kwargs["alias"] == "planner-local":
                self.planner_calls += 1
                if self.planner_calls == 1:
                    payload = {
                        "kind": "tool",
                        "thought": "Need worker draft.",
                        "tool_name": "gpt.generate_text",
                        "arguments": {
                            "alias": "tiny-agent",
                            "prompt": "Explain the deployment gate.",
                            "max_new_tokens": 16,
                        },
                    }
                else:
                    payload = {"kind": "final", "thought": "done", "final": "done"}
                text = json.dumps(payload)
                return {"completion_text": text, "generated_text": text}
            self.worker_prompt = kwargs["prompt"]
            return {"completion_text": "draft", "generated_text": "draft", "prompt_text": kwargs["prompt"]}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)

    result = runner.run(
        goal="What is the deployment gate?",
        agent_name="active-agent",
        default_model_alias="tiny-agent",
        local_planner_alias="planner-local",
        max_steps=2,
    )

    worker_event = next(item for item in result["active_memory_events"] if item["tool_name"] == "gpt.generate_text")
    assert "<helix-active-context>" in runtime.worker_prompt
    assert "sigstore verification" in runtime.worker_prompt
    assert worker_event["active_memory_used"] is True
    assert worker_event["memory_ids"] == [memory.memory_id]
    assert result["steps"][0]["active_memory"]["memory_ids"] == [memory.memory_id]
    assert "<helix-active-context>" not in json.dumps(result["observations"], ensure_ascii=True)


def test_agent_runner_memory_mode_off_does_not_inject_worker_prompt(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "session-os" / "memory.sqlite")
    try:
        catalog.remember(
            project="helix",
            agent_id="off-agent",
            memory_type="semantic",
            summary="Hidden policy",
            content="This should not be injected when memory mode is off.",
            importance=9,
        )
    finally:
        catalog.close()

    class FakeRuntime:
        def __init__(self) -> None:
            self.planner_calls = 0
            self.worker_prompt = ""

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            if kwargs["alias"] == "planner-local":
                self.planner_calls += 1
                if self.planner_calls == 1:
                    payload = {
                        "kind": "tool",
                        "thought": "Need worker draft.",
                        "tool_name": "gpt.generate_text",
                        "arguments": {"alias": "tiny-agent", "prompt": "Explain the policy."},
                    }
                else:
                    payload = {"kind": "final", "thought": "done", "final": "done"}
                text = json.dumps(payload)
                return {"completion_text": text, "generated_text": text}
            self.worker_prompt = kwargs["prompt"]
            return {"completion_text": "draft", "generated_text": "draft"}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)

    result = runner.run(
        goal="Explain the policy.",
        agent_name="off-agent",
        default_model_alias="tiny-agent",
        local_planner_alias="planner-local",
        max_steps=2,
        memory_mode="off",
    )

    worker_event = next(item for item in result["active_memory_events"] if item["tool_name"] == "gpt.generate_text")
    assert runtime.worker_prompt == "Explain the policy."
    assert worker_event["active_memory_used"] is False
    assert worker_event["mode"] == "off"


def test_agent_runner_refreshes_planner_context_after_tool_observation(tmp_path: Path, permissive_retrieval: None) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.planner_prompts = []

        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            self.planner_prompts.append(kwargs["prompt"])
            if len(self.planner_prompts) == 1:
                payload = {
                    "kind": "tool",
                    "thought": "Need model registry.",
                    "tool_name": "workspace.list_models",
                    "arguments": {},
                }
            else:
                payload = {"kind": "final", "thought": "memory refreshed", "final": "registry checked"}
            text = json.dumps(payload)
            return {"completion_text": text, "generated_text": text}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)

    result = runner.run(
        goal="Which models are available?",
        agent_name="refresh-agent",
        local_planner_alias="planner-local",
        max_steps=2,
    )

    assert result["final_answer"] == "registry checked"
    assert len(runtime.planner_prompts) == 2
    assert "workspace.list_models" in runtime.planner_prompts[1]
    assert any(item["tool_name"] == "__planner__" and item["active_memory_used"] for item in result["active_memory_events"])


def test_agent_runner_falls_back_when_local_planner_is_invalid(tmp_path: Path) -> None:
    class FakeRuntime:
        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            if kwargs["alias"] == "planner-local":
                return {"completion_text": "not valid json", "generated_text": "not valid json"}
            return {
                "completion_text": "Recovered through heuristic fallback.",
                "generated_text": "Recovered through heuristic fallback.",
                "new_ids": [],
                "generated_ids": [],
            }

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"completion_text": "resumed"}

    runner = AgentRunner(FakeRuntime(), root=tmp_path)
    runner.add_knowledge_text(
        "fallback-agent",
        "Fallback uses heuristic control when the planner emits invalid JSON.",
        source="design",
    )

    result = runner.run(
        goal="Explain the fallback behavior.",
        agent_name="fallback-agent",
        default_model_alias="tiny-agent",
        local_planner_alias="planner-local",
        max_steps=4,
    )

    assert result["final_answer"] == "Recovered through heuristic fallback."
    assert any("local:" in " ".join(item["errors"]) for item in result["planner_attempts"])


def test_agent_runner_records_kv_phase_trace_for_local_phases(tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.current_kv_mode = "fp32"
            self._kv_policy = None
            self.policy_calls = []
            self.switch_calls = []

        def set_kv_policy(self, policy, *, phase=None, allowed_modes=None):  # noqa: ANN001,ANN201
            self._kv_policy = policy
            self.policy_calls.append({"phase": phase, "allowed_modes": tuple(allowed_modes or ())})

        def switch_kv_mode(self, new_mode, *, reason=None):  # noqa: ANN001,ANN201
            self.current_kv_mode = str(new_mode)
            self.switch_calls.append({"new_mode": str(new_mode), "reason": reason})
            return self.current_kv_mode

    class FakeRuntime:
        def __init__(self) -> None:
            self.engines = {
                "planner-local": FakeEngine(),
                "tiny-agent": FakeEngine(),
            }

        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def _uses_llama_cpp(self, alias):  # noqa: ANN001,ANN202
            return False

        def _engine(self, alias, *, cache_mode):  # noqa: ANN001,ANN202
            assert cache_mode == "session"
            return self.engines[alias]

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            alias = kwargs["alias"]
            if alias == "planner-local":
                return {
                    "completion_text": '{"kind":"final","thought":"done","final":"respuesta final"}',
                    "generated_text": '{"kind":"final","thought":"done","final":"respuesta final"}',
                }
            return {
                "completion_text": "tool output",
                "generated_text": "tool output",
            }

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"completion_text": "resumed"}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)
    result = runner.run(
        goal="Decime algo breve.",
        agent_name="kv-phase-agent",
        default_model_alias="tiny-agent",
        local_planner_alias="planner-local",
        max_steps=1,
    )

    phases = [item["phase"] for item in result["kv_phase_trace"]]

    assert phases == ["plan", "final"]
    assert result["kv_phase_trace"][0]["current_mode_after"] == "turbo-int8-hadamard"
    assert result["kv_phase_trace"][1]["current_mode_after"] == "fp32"
    assert runtime.engines["planner-local"].policy_calls[0]["allowed_modes"] == ("turbo-int8-hadamard", "fp32")
    assert runtime.engines["tiny-agent"].switch_calls[-1]["new_mode"] == "fp32"


def test_agent_runner_uses_tool_phase_mode_band_for_local_text_tool(tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.current_kv_mode = "fp32"
            self._kv_policy = None
            self.policy_calls = []
            self.switch_calls = []

        def set_kv_policy(self, policy, *, phase=None, allowed_modes=None):  # noqa: ANN001,ANN201
            self._kv_policy = policy
            self.policy_calls.append({"phase": phase, "allowed_modes": tuple(allowed_modes or ())})

        def switch_kv_mode(self, new_mode, *, reason=None):  # noqa: ANN001,ANN201
            self.current_kv_mode = str(new_mode)
            self.switch_calls.append({"new_mode": str(new_mode), "reason": reason})
            return self.current_kv_mode

    class FakeRuntime:
        def __init__(self) -> None:
            self.engines = {
                "planner-local": FakeEngine(),
                "tiny-agent": FakeEngine(),
            }
            self._planner_calls = 0

        def list_models(self):  # noqa: ANN202
            return [{"alias": "tiny-agent"}]

        def model_info(self, alias):  # noqa: ANN001,ANN202
            return {"alias": alias, "model_type": "gpt2"}

        def _uses_llama_cpp(self, alias):  # noqa: ANN001,ANN202
            return False

        def _engine(self, alias, *, cache_mode):  # noqa: ANN001,ANN202
            assert cache_mode == "session"
            return self.engines[alias]

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            alias = kwargs["alias"]
            if alias == "planner-local":
                self._planner_calls += 1
                if self._planner_calls == 1:
                    payload = {
                        "kind": "tool",
                        "thought": "Need worker generation.",
                        "tool_name": "gpt.generate_text",
                        "arguments": {"prompt": "hola"},
                    }
                else:
                    payload = {"kind": "final", "thought": "done", "final": "listo"}
                text = json.dumps(payload)
                return {"completion_text": text, "generated_text": text}
            return {
                "completion_text": "hola desde tool",
                "generated_text": "hola desde tool",
            }

        def resume_text(self, **kwargs):  # noqa: ANN003,ANN202
            return {"completion_text": "resumed"}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)
    result = runner.run(
        goal="Saludame usando la herramienta de texto.",
        agent_name="kv-tool-agent",
        default_model_alias="tiny-agent",
        local_planner_alias="planner-local",
        max_steps=3,
    )

    tool_phase = next(item for item in result["kv_phase_trace"] if item["phase"] == "tool_call")

    assert result["final_answer"] == "listo"
    assert tool_phase["current_mode_after"] == "turbo-4bit"
    assert tool_phase["allowed_modes"] == ["turbo-4bit", "turbo-int8-hadamard"]
    assert runtime.engines["tiny-agent"].policy_calls[0]["phase"] == "tool_call"


def test_agent_runner_stream_emits_active_memory_context_before_tool_call(tmp_path: Path, permissive_retrieval: None) -> None:
    catalog = MemoryCatalog.open(tmp_path / "session-os" / "memory.sqlite")
    try:
        memory = catalog.remember(
            project="helix",
            agent_id="stream-agent",
            memory_type="semantic",
            summary="Stream answer uses release memory",
            content="The stream worker should cite the release memory.",
            importance=9,
        )
    finally:
        catalog.close()

    class FakeRuntime:
        def __init__(self) -> None:
            self.planner_calls = 0
            self.stream_prompt = ""

        def generate_text(self, **kwargs):  # noqa: ANN003,ANN202
            self.planner_calls += 1
            if self.planner_calls == 1:
                payload = {
                    "kind": "tool",
                    "thought": "Need streamed worker.",
                    "tool_name": "gpt.generate_text",
                    "arguments": {"alias": "tiny-agent", "prompt": "Draft stream answer."},
                }
            else:
                payload = {"kind": "final", "thought": "done", "final": "done"}
            text = json.dumps(payload)
            return {"completion_text": text, "generated_text": text}

        def stream_text(self, **kwargs):  # noqa: ANN003,ANN202
            self.stream_prompt = kwargs["prompt"]
            yield {"event": "token", "token_text": "draft"}
            yield {"event": "done", "completion_text": "draft"}

    runtime = FakeRuntime()
    runner = AgentRunner(runtime, root=tmp_path)

    events = list(
        runner.run_stream(
            goal="Draft a release update.",
            agent_name="stream-agent",
            default_model_alias="tiny-agent",
            local_planner_alias="planner-local",
            max_steps=2,
        )
    )

    active_index = next(index for index, event in enumerate(events) if event["event"] == "active_memory_context")
    tool_index = next(index for index, event in enumerate(events) if event["event"] == "tool_call")
    assert active_index < tool_index
    assert events[active_index]["memory_ids"] == [memory.memory_id]
    assert "<helix-active-context>" in runtime.stream_prompt
