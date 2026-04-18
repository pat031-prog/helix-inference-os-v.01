from __future__ import annotations

from tools import run_local_multimodel_hypervisor as hypervisor


def test_multimodel_parser_defaults_to_deferred_qwen_gpt2() -> None:
    args = hypervisor.build_parser().parse_args([])

    assert args.scenario == "coder-writer"
    assert args.code_model == "qwen-1.5b"
    assert args.writer_model == "gpt2-fast"
    assert args.codec == "rust-hlx-buffered-flat"
    assert args.audit_policy == "deferred"


def test_registry_declares_transformer_and_hybrid_models() -> None:
    assert hypervisor.MODEL_REGISTRY["qwen-1.5b"]["arch"] == "transformer"
    assert hypervisor.MODEL_REGISTRY["gpt2-fast"]["arch"] == "transformer"
    assert hypervisor.MODEL_REGISTRY["zamba2-1.2b"]["arch"] == "hybrid-mamba-transformer"


def test_coder_writer_tasks_restore_code_reviewer_on_third_task() -> None:
    tasks = hypervisor._build_tasks(scenario="coder-writer", code_model_id="qwen-1.5b", writer_model_id="gpt2-fast")

    assert [task["agent_id"] for task in tasks] == ["code_reviewer", "writer", "code_reviewer"]
    assert [task["model_id"] for task in tasks] == ["qwen-1.5b", "gpt2-fast", "qwen-1.5b"]
    assert tasks[2]["expects_restore"] is True


def test_dynamic_prompts_include_external_transcript_handoff() -> None:
    transcript = [{"role": "code-review", "handoff_summary": "The session store must key by model_id plus agent_id."}]
    writer = hypervisor._build_tasks(scenario="coder-writer", code_model_id="qwen-1.5b", writer_model_id="gpt2-fast")[1]
    followup = hypervisor._build_tasks(scenario="coder-writer", code_model_id="qwen-1.5b", writer_model_id="gpt2-fast")[2]

    writer_prompt = hypervisor._prompt_for_task(writer, transcript)
    followup_prompt = hypervisor._prompt_for_task(followup, transcript)

    assert "model_id plus agent_id" in writer_prompt
    assert "Shared transcript hash" in writer_prompt
    assert "never restored into a GPT-2 or Zamba model" in followup_prompt


def test_safe_key_removes_path_separators() -> None:
    assert "/" not in hypervisor._safe_key("qwen/agent")
    assert "\\" not in hypervisor._safe_key("qwen\\agent")
