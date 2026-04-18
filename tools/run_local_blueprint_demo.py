from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for item in (REPO_ROOT, SRC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from helix_kv import rust_session  # noqa: E402
from helix_kv.session_os import SessionCatalog, token_hash  # noqa: E402
from helix_proto import hmem  # noqa: E402
from helix_proto.blueprints import (  # noqa: E402
    FALLBACK_SLOTS,
    architecture_summary,
    load_blueprint,
    load_stack_catalog,
    make_private_state_arrays,
    normalize_slots,
    quality_check_html,
    render_meta_microsite,
    sanitize_model_text,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value)).strip("-") or "item"


def _token_ids(text: str) -> list[int]:
    words = str(text or "").split()
    if not words:
        return [0]
    return [abs(hash(word)) % 32000 for word in words[:128]]


def _hf_cache_root() -> Path:
    import os

    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_ref_cached(model_ref: str) -> bool:
    ref = str(model_ref or "").strip()
    if not ref:
        return False
    snapshots = _hf_cache_root() / f"models--{ref.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return False
    return any((item / "config.json").exists() for item in snapshots.iterdir() if item.is_dir())


def _prepared_alias_available(alias: str) -> bool:
    if not alias:
        return False
    try:
        from helix_proto.workspace import resolve_model_info

        resolve_model_info(alias, REPO_ROOT)
        return True
    except Exception:  # noqa: BLE001
        return False


def _model_available(model: dict[str, Any], *, mode: str) -> bool:
    if mode == "mock-only":
        return False
    alias = str(model.get("alias") or model.get("model_id") or "")
    # Real execution never downloads models. It accepts either a prepared HeliX alias
    # or a Hugging Face model already present in the local cache.
    if _prepared_alias_available(alias):
        return True
    return _hf_ref_cached(str(model.get("ref") or model.get("model_ref") or ""))


def _fallback_for_task(slot: str, *, context: dict[str, Any], task_id: str) -> str:
    base = FALLBACK_SLOTS.get(slot, "HeliX records the step, stores context in hmem and keeps private state isolated.")
    context_hint = ""
    if context.get("context"):
        context_hint = " Context was injected from hmem before this task ran."
    return f"{base}{context_hint}"


def _run_prepared_alias(alias: str, prompt: str, *, max_new_tokens: int) -> tuple[str | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        from helix_proto.api import HelixRuntime

        runtime = HelixRuntime(root=REPO_ROOT)
        result = runtime.generate_text(
            alias=alias,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            cache_mode="session",
        )
        text = sanitize_model_text(str(result.get("completion_text") or result.get("generated_text") or ""))
        return text or None, {
            "backend": "helix-prepared-alias",
            "generation_time_ms": (time.perf_counter() - started) * 1000.0,
            "prompt_token_count": len(result.get("prompt_ids") or []),
            "generated_token_count": len(result.get("new_ids") or result.get("generated_ids") or []),
        }
    except Exception:  # noqa: BLE001
        return None, {
            "backend": "helix-prepared-alias",
            "error": "prepared_alias_generation_failed",
            "generation_time_ms": (time.perf_counter() - started) * 1000.0,
        }


def _run_hf_cached_ref(model_ref: str, prompt: str, *, max_new_tokens: int) -> tuple[str | None, dict[str, Any]]:
    load_started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return None, {"backend": "hf-transformers-local-cache", "error": f"missing_dependency:{type(exc).__name__}"}

    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        load_time_ms = (time.perf_counter() - load_started) * 1000.0
        prompt_text = prompt
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        generate_started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_time_ms = (time.perf_counter() - generate_started) * 1000.0
        new_tokens = output[0][prompt_token_count:]
        text = sanitize_model_text(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return text or None, {
            "backend": "hf-transformers-local-cache",
            "model_ref": model_ref,
            "load_time_ms": load_time_ms,
            "generation_time_ms": generation_time_ms,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": int(new_tokens.shape[-1]),
        }
    except Exception as exc:  # noqa: BLE001
        return None, {
            "backend": "hf-transformers-local-cache",
            "model_ref": model_ref,
            "error": f"{type(exc).__name__}:{exc}",
            "generation_time_ms": (time.perf_counter() - load_started) * 1000.0,
        }
    finally:
        del model
        del tokenizer
        gc.collect()


def _run_real_model(model: dict[str, Any], prompt: str, *, max_new_tokens: int) -> tuple[str | None, dict[str, Any]]:
    alias = str(model.get("alias") or model.get("model_id") or "")
    if _prepared_alias_available(alias):
        return _run_prepared_alias(alias, prompt, max_new_tokens=max_new_tokens)
    model_ref = str(model.get("ref") or model.get("model_ref") or "")
    if _hf_ref_cached(model_ref):
        return _run_hf_cached_ref(model_ref, prompt, max_new_tokens=max_new_tokens)
    return None, {"backend": "none", "error": "model_not_cached"}


def _write_side_artifacts(output_dir: Path, *, stack_catalog: dict[str, Any]) -> None:
    _write_json(output_dir / "local-inference-os-architecture-summary.json", architecture_summary())
    _write_json(output_dir / "local-blueprint-stack-catalog.json", stack_catalog)
    frontend = next((item for item in stack_catalog["stacks"] if item.get("id") == "frontend-factory"), {})
    _write_json(
        output_dir / "local-blueprint-frontend-factory-smoke.json",
        {
            "schema_version": 1,
            "title": "HeliX Frontend Factory Blueprint Smoke",
            "benchmark_kind": "inference-os-blueprint-frontend-factory-smoke-v0",
            "status": "completed",
            "blueprint_id": "frontend-factory",
            "stack": frontend,
            "smoke_scope": "spec_only",
            "claim_boundary": "This confirms the workload spec exists; it is not a real frontend model-quality run.",
        },
    )
    _write_json(
        output_dir / "local-blueprint-framework-showcase.json",
        {
            "schema_version": 1,
            "title": "HeliX Blueprint Framework Showcase",
            "benchmark_kind": "inference-os-blueprint-framework-showcase-v0",
            "status": "completed",
            "client_surface": "/v1/chat/completions",
            "blueprints_can_target_openai_compatible_api": True,
            "crewai_optional": True,
            "claim_boundary": "This is a framework-facing contract summary; external LangChain/CrewAI runtime quality is tested separately.",
        },
    )


def run_blueprint_demo(args: argparse.Namespace) -> dict[str, Any]:
    blueprint = load_blueprint(args.blueprint)
    output_dir = Path(args.output_dir)
    site_output = Path(args.site_output)
    run_id = f"{blueprint.blueprint_id}-{int(time.time())}"
    workspace = output_dir / "_blueprint-workspaces" / run_id
    sessions_root = workspace / "sessions"
    catalog = SessionCatalog.open(workspace / "session-catalog.sqlite")
    stack_catalog = load_stack_catalog(Path(args.blueprint).resolve().parent)
    memory_policy = dict(blueprint.payload["memory_policy"])
    session_policy = dict(blueprint.payload["session_policy"])
    project = str(memory_policy.get("project") or hmem.DEFAULT_PROJECT)
    memory_mode = str(memory_policy.get("mode") or "search")
    memory_budget = int(memory_policy.get("budget_tokens") or 700)
    codec = str(args.codec or session_policy.get("codec") or "rust-hlx-buffered-flat")
    audit_policy = str(args.audit_policy or session_policy.get("audit_policy") or "deferred")

    model_states: dict[str, dict[str, Any]] = {}
    for key, model in blueprint.payload["models"].items():
        available = _model_available(model, mode=str(args.mode))
        prepared_alias_available = _prepared_alias_available(str(model.get("alias") or model.get("model_id") or ""))
        hf_cache_available = _hf_ref_cached(str(model.get("ref") or model.get("model_ref") or ""))
        model_states[key] = {
            **model,
            "key": key,
            "available_prepared_alias": prepared_alias_available,
            "available_hf_cache": hf_cache_available,
            "available_real_model": available,
            "generation_mode": (
                "real-prepared-alias"
                if available and prepared_alias_available
                else "real-hf-cache"
                if available and hf_cache_available
                else "fallback-deterministic"
            ),
        }
    if args.mode == "real-only" and not all(item["available_real_model"] for item in model_states.values()):
        payload = {
            "schema_version": 1,
            "title": "HeliX Meta Microsite Blueprint Demo",
            "benchmark_kind": "inference-os-blueprint-meta-microsite-v0",
            "status": "skipped_model_not_cached",
            "mode": str(args.mode),
            "models_used": list(model_states.values()),
            "skip_reason": "real-only requires a prepared HeliX alias or Hugging Face cache hit for every model in the blueprint",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / blueprint.payload["outputs"]["artifact"], payload)
        return payload

    slots_raw: dict[str, str] = {}
    task_timeline: list[dict[str, Any]] = []
    model_lifecycle_events: list[dict[str, Any]] = []
    private_state_events: list[dict[str, Any]] = []
    hmem_events: list[dict[str, Any]] = []
    scheduler_decisions: list[dict[str, Any]] = []
    real_generation_events: list[dict[str, Any]] = []
    latest_session: dict[tuple[str, str], dict[str, Any]] = {}
    active_model_key: str | None = None

    for index, task in enumerate(blueprint.payload["tasks"]):
        started = time.perf_counter()
        agent_id = str(task["agent"])
        agent = blueprint.payload["agents"][agent_id]
        model_key = str(agent["model"])
        model = model_states[model_key]
        model_id = str(model.get("model_id") or model_key)
        slot = str(task["slot"])
        lifecycle_event = {
            "event": "model_reuse" if active_model_key == model_key else "model_activate",
            "previous_model_key": active_model_key,
            "model_key": model_key,
            "model_id": model_id,
            "generation_mode": model["generation_mode"],
            "load_time_ms": 0.0 if active_model_key == model_key else float(model.get("load_time_estimate_ms") or 0.0),
        }
        active_model_key = model_key
        model_lifecycle_events.append(lifecycle_event)
        restored = latest_session.get((model_id, agent_id))
        context = hmem.build_context(
            root=workspace,
            project=project,
            agent_id=agent_id,
            query=str(task.get("prompt") or ""),
            budget_tokens=memory_budget,
            mode=memory_mode,
        )
        prompt = (
            f"Role: {agent.get('role')}\nTask: {task.get('prompt')}\n"
            f"Memory context:\n{context.get('context') or '(empty)'}"
        )
        generated = None
        generation_meta: dict[str, Any] = {"backend": "fallback-deterministic"}
        if model["available_real_model"] and args.mode != "mock-only":
            generated, generation_meta = _run_real_model(model, prompt, max_new_tokens=int(args.max_new_tokens))
        if not generated:
            generated = _fallback_for_task(slot, context=context, task_id=str(task["task_id"]))
            generation_meta = {**generation_meta, "fallback_used": True}
        else:
            generation_meta = {**generation_meta, "fallback_used": False}
        real_generation_events.append(
            {
                "task_id": task["task_id"],
                "agent_id": agent_id,
                "model_id": model_id,
                "slot": slot,
                **generation_meta,
            }
        )
        slots_raw[slot] = generated
        observe = hmem.observe_event(
            root=workspace,
            project=project,
            agent_id=agent_id,
            session_id=run_id,
            event_type="blueprint_task",
            content=generated,
            summary=f"{task['task_id']}: {generated}",
            tags=["blueprint", blueprint.blueprint_id, str(task["task_id"]), slot],
            importance=7,
            promote=True,
            memory_type="episodic",
        )
        hmem_events.append(
            {
                "task_id": task["task_id"],
                "agent_id": agent_id,
                "memory_id": (observe.get("memory") or {}).get("memory_id"),
                "observation_id": (observe.get("observation") or {}).get("observation_id"),
                "memory_context_tokens": context.get("tokens", 0),
            }
        )
        token_ids = _token_ids(generated)
        session_dir = sessions_root / _safe(model_id) / _safe(agent_id) / f"v{index + 1:04d}"
        if restored is not None:
            _, _, load_receipt = rust_session.load_session_bundle(restored["path"], verify_policy="receipt-only")
            private_state_events.append(
                {
                    "event": "session_restored",
                    "task_id": task["task_id"],
                    "model_id": model_id,
                    "agent_id": agent_id,
                    "path": str(restored["path"]),
                    "session_hash_loaded": load_receipt.get("session_hash_loaded") or load_receipt.get("session_hash"),
                }
            )
        receipt = rust_session.save_session_bundle(
            session_dir,
            meta={
                "blueprint_id": blueprint.blueprint_id,
                "run_id": run_id,
                "task_id": task["task_id"],
                "model_id": model_id,
                "agent_id": agent_id,
                "session_token_ids": token_ids,
                "private_state_kind": "blueprint-control-plane-v0",
            },
            arrays=make_private_state_arrays(task_id=str(task["task_id"]), slot_text=generated),
            session_codec=codec,
            audit_policy=audit_policy,
        )
        session_hash = receipt.get("session_hash") or receipt.get("fast_payload_checksum")
        catalog.record_session(
            session_id=f"{_safe(model_id)}__{_safe(agent_id)}__v{index + 1:04d}",
            model_id=model_id,
            agent_id=agent_id,
            model_ref=str(model.get("ref") or model_id),
            arch=str(model.get("arch") or "transformer"),
            path=session_dir,
            token_ids=token_ids,
            session_bytes=int(receipt.get("session_total_bytes") or 0),
            codec=codec,
            audit_status=str(receipt.get("audit_status") or ("pending" if audit_policy == "deferred" else "verified")),
            session_hash=None if session_hash is None else str(session_hash),
            parent_session_id=None if restored is None else str(restored.get("session_id")),
        )
        latest_session[(model_id, agent_id)] = {"path": session_dir, "session_hash": session_hash, "session_id": f"{_safe(model_id)}__{_safe(agent_id)}__v{index + 1:04d}"}
        private_state_events.append(
            {
                "event": "session_saved",
                "task_id": task["task_id"],
                "model_id": model_id,
                "agent_id": agent_id,
                "path": str(session_dir),
                "audit_status": receipt.get("audit_status"),
                "session_total_bytes": receipt.get("session_total_bytes"),
                "session_hash": session_hash,
            }
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        decision = {
            "task_id": task["task_id"],
            "agent_id": agent_id,
            "selected_model_id": model_id,
            "candidate_models": [model_id, model.get("fallback")],
            "estimated_cost_ms": float(model.get("load_time_estimate_ms") or 0.0) + int(args.max_new_tokens),
            "actual_cost_ms": elapsed,
            "model_swapped": lifecycle_event["event"] == "model_activate",
            "session_restored": restored is not None,
            "hmem_context_tokens": context.get("tokens", 0),
            "audit_status": receipt.get("audit_status"),
            "generation_backend": generation_meta.get("backend"),
            "generation_fallback_used": bool(generation_meta.get("fallback_used")),
        }
        scheduler_decisions.append(decision)
        handoff = sanitize_model_text(generated, limit=320)
        task_timeline.append(
            {
                "task_id": task["task_id"],
                "agent_id": agent_id,
                "model_id": model_id,
                "slot": slot,
                "handoff_summary": handoff,
                "restored_private_state": restored is not None,
                "hmem_memory_id": hmem_events[-1]["memory_id"],
                "generation_backend": generation_meta.get("backend"),
                "generation_fallback_used": bool(generation_meta.get("fallback_used")),
            }
        )

    final_audits: list[dict[str, Any]] = []
    for session in latest_session.values():
        if audit_policy == "deferred":
            final_audits.append(rust_session.verify_deferred_session(session["path"]))
    final_audit_status = "verified" if not final_audits or all(item.get("audit_status") == "verified" for item in final_audits) else "failed"
    memory_graph = hmem.graph(root=workspace, project=project, agent_id=None, limit=100)
    content_slots, slot_quality = normalize_slots(slots_raw)

    artifact = {
        "schema_version": 1,
        "title": "HeliX Meta Microsite Blueprint Demo",
        "benchmark_kind": "inference-os-blueprint-meta-microsite-v0",
        "status": "completed",
        "mode": str(args.mode),
        "blueprint_id": blueprint.blueprint_id,
        "layers_demonstrated": ["active_model", "private_hlx_state", "shared_hmem", "multimodel_scheduler"],
        "models_used": list(model_states.values()),
        "agents": [
            {"agent_id": key, **value}
            for key, value in blueprint.payload["agents"].items()
        ],
        "task_timeline": task_timeline,
        "model_lifecycle_events": model_lifecycle_events,
        "private_state_events": private_state_events,
        "hmem_events": hmem_events,
        "scheduler_decisions": scheduler_decisions,
        "real_generation_events": real_generation_events,
        "memory_graph": {"node_count": memory_graph.get("node_count"), "edge_count": memory_graph.get("edge_count")},
        "content_slots": content_slots,
        "final_audits": final_audits,
        "final_audit_status": final_audit_status,
        "fallback_content_used": bool(slot_quality["fallback_content_used"]) or any(item.get("fallback_used") for item in real_generation_events),
        "fallback_mode_used": str(args.mode) == "mock-only" or not any(item["available_real_model"] for item in model_states.values()),
        "public_claim_level": (
            "real-cached-model-orchestration"
            if any(not item.get("fallback_used") for item in real_generation_events)
            else "orchestration-and-renderer"
        ),
        "claim_boundary": "Fallback or mixed mode proves Blueprint orchestration, hmem, .hlx private state and renderer quality; it does not prove model generation quality.",
    }
    html_text = render_meta_microsite(artifact)
    quality = quality_check_html(html_text)
    artifact["quality_checks"] = {**slot_quality, **quality}
    artifact["html_output_path"] = str(site_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    site_output.parent.mkdir(parents=True, exist_ok=True)
    site_output.write_text(html_text, encoding="utf-8")
    if not bool(getattr(args, "skip_web_copy", False)):
        web_meta = Path(args.web_copy_path)
        if not web_meta.is_absolute():
            web_meta = REPO_ROOT / web_meta
        web_meta.parent.mkdir(parents=True, exist_ok=True)
        web_meta.write_text(html_text, encoding="utf-8")
    _write_json(output_dir / blueprint.payload["outputs"]["artifact"], artifact)
    _write_side_artifacts(output_dir, stack_catalog=stack_catalog)
    catalog.close()
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local HeliX Blueprint Meta Microsite demo.")
    parser.add_argument("--blueprint", default="blueprints/meta-microsite.json")
    parser.add_argument("--mode", default="budgeted-local", choices=["budgeted-local", "real-only", "mock-only"])
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--site-output", default="site-dist/meta-demo.html")
    parser.add_argument("--web-copy-path", default="web/meta-demo.html")
    parser.add_argument("--codec", default="rust-hlx-buffered-flat")
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-web-copy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_blueprint_demo(args)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "artifact": str(Path(args.output_dir) / load_blueprint(args.blueprint).payload["outputs"]["artifact"]),
                "html_output_path": payload.get("html_output_path"),
                "quality_status": (payload.get("quality_checks") or {}).get("status"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
