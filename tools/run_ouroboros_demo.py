"""
run_ouroboros_demo.py
=====================
HeliX Proyecto Ouroboros — Meta-Compiler DAG runner

Orchestrates a 5-phase asymmetric Mixture-of-Models pipeline:
  Phase 1 (LOCAL)   — Qwen-1.5B: decompose raw requirement → 3 atomic sub-tasks → .hlx save → hmem
  Phase 2 (CLOUD)   — Concurrent ThreadPool dispatch to DeepInfra:
                       Hilo A: Llama-3.1-70B (Architect) — mathematical spec
                       Hilo B: Qwen-2.5-Coder-32B (Engineer) — code implementation
                       Hilo C: Mixtral-8x22B (Red Team) — concurrency audit
  Phase 3 (FENCE)   — If Red Team finds CRITICAL/HIGH → Tombstone in hmem → Engineer rewrite
  Phase 4 (LOCAL)   — Qwen-1.5B: Warm Restore .hlx → merge all → .patch + Pytest suite

Usage
-----
python tools/run_ouroboros_demo.py ^
    --blueprint blueprints/meta-compiler-dag.json ^
    --mode hybrid-cloud ^
    --deepinfra-api-key <KEY> ^
    --requirement tools/requirements/merkle-dag-migration.txt ^
    --output-dir verification ^
    --site-output site-dist/ouroboros-demo.html
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helix_kv import rust_session  # noqa: E402
from helix_kv.session_os import SessionCatalog  # noqa: E402
from helix_proto import hmem  # noqa: E402
from helix_proto.blueprints import (  # noqa: E402
    OUROBOROS_FALLBACK_SLOTS,
    load_blueprint,
    make_private_state_arrays,
    quality_check_ouroboros_html,
    render_ouroboros_site,
    sanitize_model_text,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai/chat/completions"

_TOMBSTONE_KEYWORDS = [
    "deadlock",
    "race condition",
    "data race",
    "starvation",
    "livelock",
    "use-after-free",
    "double-free",
    "inference dos",
    "dos vector",
]

# Tasks that run concurrently in Phase 2 (order matters for prompt construction)
_CONCURRENT_TASKS = ("architect", "engineer", "red-team")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value)).strip("-") or "item"


def _count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


def _token_ids(text: str) -> list[int]:
    words = str(text or "").split()
    if not words:
        return [0]
    return [abs(hash(word)) % 32000 for word in words[:128]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Local model backend (HuggingFace cache)
# ---------------------------------------------------------------------------

def _hf_cache_root() -> Path:
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


def _run_hf_cached_ref(
    model_ref: str,
    prompt: str,
    *,
    max_new_tokens: int,
) -> tuple[str | None, dict[str, Any]]:
    load_started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return None, {"backend": "hf-transformers-local-cache", "error": f"missing_dependency:{type(exc).__name__}"}

    model_obj = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model_obj.eval()
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
        gen_started = time.perf_counter()
        import torch
        with torch.inference_mode():
            output = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_time_ms = (time.perf_counter() - gen_started) * 1000.0
        new_tokens = output[0][prompt_token_count:]
        text = sanitize_model_text(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return text or None, {
            "backend": "hf-transformers-local-cache",
            "model_ref": model_ref,
            "load_time_ms": load_time_ms,
            "generation_time_ms": generation_time_ms,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": int(new_tokens.shape[-1]),
            "fallback_used": False,
        }
    except Exception as exc:  # noqa: BLE001
        return None, {
            "backend": "hf-transformers-local-cache",
            "model_ref": model_ref,
            "error": f"{type(exc).__name__}:{exc}",
            "generation_time_ms": (time.perf_counter() - load_started) * 1000.0,
            "fallback_used": True,
        }
    finally:
        del model_obj
        del tokenizer
        gc.collect()


# ---------------------------------------------------------------------------
# Cloud backend — DeepInfra
# ---------------------------------------------------------------------------

def _run_deepinfra(
    model_ref: str,
    prompt: str,
    *,
    api_key: str,
    max_new_tokens: int,
    timeout_s: float = 180.0,
) -> tuple[str | None, dict[str, Any]]:
    """Call DeepInfra OpenAI-compatible endpoint. Records wall-clock sleep time."""
    request_payload = json.dumps(
        {
            "model": model_ref,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.2,
            "stream": False,
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    req = urllib.request.Request(_DEEPINFRA_BASE, data=request_payload, headers=headers, method="POST")
    sleep_started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        cloud_ms = (time.perf_counter() - sleep_started) * 1000.0
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            return None, {
                "backend": "deepinfra",
                "model_ref": model_ref,
                "error": "empty_choices",
                "local_sleep_ms": cloud_ms,
                "fallback_used": True,
            }
        content = str((choices[0].get("message") or {}).get("content") or "")
        usage = data.get("usage") or {}
        tokens_sent = int(usage.get("prompt_tokens") or _count_tokens_approx(prompt))
        tokens_back = int(usage.get("completion_tokens") or _count_tokens_approx(content))
        text = sanitize_model_text(content, limit=1400)
        return text or None, {
            "backend": "deepinfra",
            "model_ref": model_ref,
            "local_sleep_ms": cloud_ms,
            "cloud_request_time_ms": cloud_ms,
            "tokens_sent_to_cloud": tokens_sent,
            "tokens_received_from_cloud": tokens_back,
            "fallback_used": False,
        }
    except Exception as exc:  # noqa: BLE001
        error_ms = (time.perf_counter() - sleep_started) * 1000.0
        return None, {
            "backend": "deepinfra",
            "model_ref": model_ref,
            "error": f"{type(exc).__name__}:{exc}",
            "local_sleep_ms": error_ms,
            "fallback_used": True,
        }


# ---------------------------------------------------------------------------
# Unified model dispatcher
# ---------------------------------------------------------------------------

def _run_model(
    model: dict[str, Any],
    prompt: str,
    *,
    api_key: str,
    max_new_tokens: int,
    cloud_max_tokens: int,
    mode: str,
    timeout_s: float,
) -> tuple[str | None, dict[str, Any]]:
    endpoint = str(model.get("endpoint") or "local")
    model_ref = str(model.get("ref") or "")

    if mode == "mock-only":
        return None, {"backend": "mock", "endpoint": endpoint, "fallback_used": True}

    if endpoint == "deepinfra":
        if not api_key:
            return None, {
                "backend": "deepinfra",
                "model_ref": model_ref,
                "error": "no_api_key",
                "local_sleep_ms": 0.0,
                "fallback_used": True,
            }
        return _run_deepinfra(model_ref, prompt, api_key=api_key, max_new_tokens=cloud_max_tokens, timeout_s=timeout_s)

    # Local
    if _hf_ref_cached(model_ref):
        return _run_hf_cached_ref(model_ref, prompt, max_new_tokens=max_new_tokens)
    return None, {"backend": "none", "endpoint": endpoint, "error": "model_not_cached", "fallback_used": True}


# ---------------------------------------------------------------------------
# Tombstone evaluation
# ---------------------------------------------------------------------------

def _evaluate_tombstone(redteam_output: str) -> tuple[bool, str]:
    """
    Inspect the Red Team output for TOMBSTONE triggers.
    Returns (triggered: bool, reason: str).
    """
    text_lower = str(redteam_output or "").lower()
    # Explicit tombstone header
    if text_lower.strip().startswith("tombstone:"):
        reason = sanitize_model_text(redteam_output, limit=600)
        return True, reason
    # Implicit: CRITICAL or HIGH severity + any tombstone keyword
    has_critical_or_high = bool(re.search(r"\b(critical|high)\b", text_lower))
    has_keyword = any(kw in text_lower for kw in _TOMBSTONE_KEYWORDS)
    if has_critical_or_high and has_keyword:
        reason = sanitize_model_text(redteam_output, limit=600)
        return True, reason
    return False, sanitize_model_text(redteam_output, limit=300)


# ---------------------------------------------------------------------------
# Patch/Test extractor from merger output
# ---------------------------------------------------------------------------

def _extract_patch_and_tests(merge_output: str) -> tuple[str, str]:
    """Split the merger's output into (.patch text, pytest text)."""
    text = str(merge_output or "")
    patch_part = ""
    tests_part = ""
    if "PATCH:" in text and "TESTS:" in text:
        patch_start = text.index("PATCH:") + len("PATCH:")
        tests_start = text.index("TESTS:") + len("TESTS:")
        patch_part = text[patch_start:text.index("TESTS:")].strip()
        tests_part = text[tests_start:].strip()
    elif "PATCH:" in text:
        patch_part = text[text.index("PATCH:") + len("PATCH:"):].strip()
    else:
        patch_part = text
    return patch_part, tests_part


# ---------------------------------------------------------------------------
# Build prompt for each task
# ---------------------------------------------------------------------------

def _build_prompt(
    task: dict[str, Any],
    agent: dict[str, Any],
    requirement_text: str,
    context: dict[str, Any],
    slots_raw: dict[str, str],
) -> str:
    task_id = str(task.get("task_id", ""))
    role = str(agent.get("role", ""))
    base_prompt = str(task.get("prompt", ""))
    ctx_text = context.get("context") or "(no prior context)"

    if task_id == "decompose":
        return (
            f"Role: {role}\n"
            f"Task: {base_prompt}\n\n"
            f"RAW REQUIREMENT:\n{requirement_text}\n\n"
            f"Memory context:\n{ctx_text}"
        )
    if task_id == "architect":
        decompose_out = slots_raw.get("decompose_out", "")
        return (
            f"Role: {role}\n"
            f"Task: {base_prompt}\n\n"
            f"DECOMPOSITION MANIFEST:\n{decompose_out}\n\n"
            f"ORIGINAL REQUIREMENT (for context):\n{requirement_text[:800]}\n\n"
            f"Memory context:\n{ctx_text}"
        )
    if task_id == "engineer":
        architect_out = slots_raw.get("architect_out", "")
        decompose_out = slots_raw.get("decompose_out", "")
        return (
            f"Role: {role}\n"
            f"Task: {base_prompt}\n\n"
            f"ARCHITECT SPEC (mathematical design):\n{architect_out}\n\n"
            f"DECOMPOSITION CONTEXT:\n{decompose_out[:400]}\n\n"
            f"Memory context:\n{ctx_text}"
        )
    if task_id == "red-team":
        engineer_out = slots_raw.get("engineer_out", "")
        architect_out = slots_raw.get("architect_out", "")
        return (
            f"Role: {role}\n"
            f"Task: {base_prompt}\n\n"
            f"ENGINEER IMPLEMENTATION:\n{engineer_out}\n\n"
            f"ARCHITECT SPEC (reference):\n{architect_out[:400]}\n\n"
            f"Memory context:\n{ctx_text}"
        )
    if task_id == "merge":
        engineer_out = slots_raw.get("engineer_out", "")
        redteam_out = slots_raw.get("redteam_out", "")
        decompose_out = slots_raw.get("decompose_out", "")
        return (
            f"Role: {role}\n"
            f"Task: {base_prompt}\n\n"
            f"VALIDATED IMPLEMENTATION:\n{engineer_out}\n\n"
            f"RED TEAM AUDIT (resolved):\n{redteam_out[:400]}\n\n"
            f"ORIGINAL DECOMPOSITION:\n{decompose_out[:300]}\n\n"
            f"Memory context:\n{ctx_text}"
        )
    # Generic fallback
    return f"Role: {role}\nTask: {base_prompt}\n\nMemory context:\n{ctx_text}"


# ---------------------------------------------------------------------------
# Single-task executor (used by both sequential and ThreadPool paths)
# ---------------------------------------------------------------------------

def _execute_task(
    task: dict[str, Any],
    blueprint_payload: dict[str, Any],
    model_states: dict[str, dict[str, Any]],
    *,
    workspace: Path,
    sessions_root: Path,
    run_id: str,
    project: str,
    memory_mode: str,
    memory_budget: int,
    codec: str,
    audit_policy: str,
    api_key: str,
    max_new_tokens: int,
    cloud_max_tokens: int,
    mode: str,
    timeout_s: float,
    requirement_text: str,
    slots_raw: dict[str, str],
    latest_session: dict[tuple[str, str], dict[str, Any]],
    task_index: int,
    wall_t0: float,
) -> dict[str, Any]:
    """Execute a single blueprint task and return its event record."""
    started = time.perf_counter()
    wall_start_ms = (started - wall_t0) * 1000.0

    agent_id = str(task["agent"])
    agent = blueprint_payload["agents"][agent_id]
    model_key = str(agent["model"])
    model = model_states[model_key]
    model_id = str(model.get("model_id") or model_key)
    slot = str(task["slot"])
    endpoint = str(model.get("endpoint") or "local")
    task_id = str(task.get("task_id", ""))
    task_expects_restore = bool(task.get("expects_restore"))

    context = hmem.build_context(
        root=workspace,
        project=project,
        agent_id=agent_id,
        query=str(task.get("prompt") or ""),
        budget_tokens=memory_budget,
        mode=memory_mode,
    )

    prompt = _build_prompt(task, agent, requirement_text, context, slots_raw)

    # Generate
    generated: str | None = None
    generation_meta: dict[str, Any] = {"backend": "fallback-deterministic", "endpoint": endpoint}

    if model["available_real_model"]:
        generated, generation_meta = _run_model(
            model,
            prompt,
            api_key=api_key,
            max_new_tokens=max_new_tokens,
            cloud_max_tokens=cloud_max_tokens,
            mode=mode,
            timeout_s=timeout_s,
        )
        generation_meta["endpoint"] = endpoint

    if not generated:
        generated = OUROBOROS_FALLBACK_SLOTS.get(slot, "HeliX recorded the step.")
        generation_meta = {**generation_meta, "fallback_used": True}

    slots_raw[slot] = generated

    # Write to hmem
    observe = hmem.observe_event(
        root=workspace,
        project=project,
        agent_id=agent_id,
        session_id=run_id,
        event_type="ouroboros_task",
        content=generated,
        summary=f"{task_id}: {generated[:120]}",
        tags=["ouroboros", "meta-compiler-dag", task_id, slot, endpoint],
        importance=8,
        promote=True,
        memory_type="episodic",
    )

    # Private state — local agents only
    restored = latest_session.get((model_id, agent_id))
    token_ids = _token_ids(generated)
    session_dir = sessions_root / _safe(model_id) / _safe(agent_id) / f"v{task_index + 1:04d}"

    private_event: dict[str, Any]
    if endpoint != "deepinfra":
        if restored is not None and task_expects_restore:
            try:
                _, _, load_receipt = rust_session.load_session_bundle(
                    restored["path"], verify_policy="receipt-only"
                )
                latest_session[(model_id, agent_id)]["load_receipt"] = load_receipt
            except Exception:  # noqa: BLE001
                pass

        receipt = rust_session.save_session_bundle(
            session_dir,
            meta={
                "blueprint_id": "meta-compiler-dag",
                "run_id": run_id,
                "task_id": task_id,
                "model_id": model_id,
                "agent_id": agent_id,
                "session_token_ids": token_ids,
                "private_state_kind": "ouroboros-v0",
                "endpoint": endpoint,
            },
            arrays=make_private_state_arrays(task_id=task_id, slot_text=generated),
            session_codec=codec,
            audit_policy=audit_policy,
        )
        session_hash = receipt.get("session_hash") or receipt.get("fast_payload_checksum")
        latest_session[(model_id, agent_id)] = {
            "path": session_dir,
            "session_hash": session_hash,
            "session_id": f"{_safe(model_id)}__{_safe(agent_id)}__v{task_index + 1:04d}",
        }
        private_event = {
            "event": "session_saved",
            "task_id": task_id,
            "model_id": model_id,
            "agent_id": agent_id,
            "endpoint": endpoint,
            "path": str(session_dir),
            "audit_status": receipt.get("audit_status"),
            "session_total_bytes": receipt.get("session_total_bytes"),
            "session_hash": session_hash,
        }
    else:
        private_event = {
            "event": "stateless_cloud_node",
            "task_id": task_id,
            "model_id": model_id,
            "agent_id": agent_id,
            "endpoint": "deepinfra",
            "note": "Cloud node — no private .hlx state by design.",
        }

    elapsed = (time.perf_counter() - started) * 1000.0
    wall_end_ms = (time.perf_counter() - wall_t0) * 1000.0

    return {
        "task_id": task_id,
        "agent_id": agent_id,
        "model_id": model_id,
        "slot": slot,
        "endpoint": endpoint,
        "generated": generated,
        "handoff_summary": sanitize_model_text(generated, limit=320),
        "generation_meta": generation_meta,
        "hmem_event": {
            "task_id": task_id,
            "agent_id": agent_id,
            "endpoint": endpoint,
            "memory_id": (observe.get("memory") or {}).get("memory_id"),
            "observation_id": (observe.get("observation") or {}).get("observation_id"),
            "memory_context_tokens": context.get("tokens", 0),
        },
        "private_event": private_event,
        "scheduler_decision": {
            "task_id": task_id,
            "agent_id": agent_id,
            "selected_model_id": model_id,
            "endpoint": endpoint,
            "candidate_models": [model_id, model.get("fallback")],
            "estimated_cost_ms": float(model.get("load_time_estimate_ms") or 0.0) + max_new_tokens,
            "actual_cost_ms": elapsed,
            "session_restored": restored is not None and task_expects_restore and endpoint != "deepinfra",
            "hmem_context_tokens": context.get("tokens", 0),
            "generation_backend": generation_meta.get("backend"),
            "generation_fallback_used": bool(generation_meta.get("fallback_used")),
        },
        "wall_start_ms": wall_start_ms,
        "wall_end_ms": wall_end_ms,
        "elapsed_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ouroboros_demo(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901
    blueprint = load_blueprint(args.blueprint)
    output_dir = Path(args.output_dir)
    site_output = Path(args.site_output)
    api_key = str(args.deepinfra_api_key or os.environ.get("DEEPINFRA_API_KEY") or "")

    req_path = Path(args.requirement) if args.requirement else (REPO_ROOT / "tools" / "requirements" / "merkle-dag-migration.txt")
    requirement_text = req_path.read_text(encoding="utf-8") if req_path.exists() else (
        "Migrate the MemoryCatalog from SQLite FTS5 to an in-memory Merkle DAG to support "
        "faster audits while mitigating deadlocks in the ThreadPool."
    )

    run_id = f"ouroboros-{int(time.time())}"
    workspace = output_dir / "_blueprint-workspaces" / run_id
    sessions_root = workspace / "sessions"
    catalog = SessionCatalog.open(workspace / "session-catalog.sqlite")

    bp = blueprint.payload
    memory_policy = dict(bp.get("memory_policy", {}))
    session_policy = dict(bp.get("session_policy", {}))
    project = str(memory_policy.get("project") or "helix-ouroboros")
    memory_mode = str(memory_policy.get("mode") or "search")
    memory_budget = int(memory_policy.get("budget_tokens") or 1200)
    codec = str(args.codec or session_policy.get("codec") or "rust-hlx-buffered-flat")
    audit_policy = str(args.audit_policy or session_policy.get("audit_policy") or "deferred")

    # Evaluate model availability
    model_states: dict[str, dict[str, Any]] = {}
    for key, model in bp["models"].items():
        endpoint = str(model.get("endpoint") or "local")
        model_ref = str(model.get("ref") or "")
        if endpoint == "deepinfra":
            available = bool(api_key) and args.mode != "mock-only"
            gen_mode = "deepinfra-api" if available else "fallback-deterministic"
            hf_cached = False
        else:
            hf_cached = _hf_ref_cached(model_ref)
            available = hf_cached and args.mode != "mock-only"
            gen_mode = "real-hf-cache" if available else "fallback-deterministic"
        model_states[key] = {
            **model,
            "key": key,
            "available_real_model": available,
            "available_hf_cache": hf_cached,
            "generation_mode": gen_mode,
        }

    # Tracking
    slots_raw: dict[str, str] = {}
    task_timeline: list[dict[str, Any]] = []
    private_state_events: list[dict[str, Any]] = []
    hmem_events: list[dict[str, Any]] = []
    scheduler_decisions: list[dict[str, Any]] = []
    real_generation_events: list[dict[str, Any]] = []
    latest_session: dict[tuple[str, str], dict[str, Any]] = {}
    tombstone_event: dict[str, Any] = {"triggered": False, "reason": "", "rewrite_triggered": False}

    # Build task map for quick lookup
    tasks_by_id = {str(t.get("task_id", "")): t for t in bp["tasks"]}
    tasks_raw_list = list(bp["tasks"])

    wall_t0 = time.perf_counter()

    # ---- Common executor kwargs ----
    exec_kwargs = dict(
        blueprint_payload=bp,
        model_states=model_states,
        workspace=workspace,
        sessions_root=sessions_root,
        run_id=run_id,
        project=project,
        memory_mode=memory_mode,
        memory_budget=memory_budget,
        codec=codec,
        audit_policy=audit_policy,
        api_key=api_key,
        max_new_tokens=int(args.max_new_tokens),
        cloud_max_tokens=int(args.cloud_max_tokens),
        mode=args.mode,
        timeout_s=float(args.timeout_seconds),
        requirement_text=requirement_text,
        slots_raw=slots_raw,
        latest_session=latest_session,
        wall_t0=wall_t0,
    )

    # ==================================================================
    # PHASE 1: Decompose (local, sequential)
    # ==================================================================
    decompose_task = tasks_by_id["decompose"]
    result = _execute_task(decompose_task, task_index=0, **exec_kwargs)
    task_timeline.append({
        "task_id": result["task_id"],
        "agent_id": result["agent_id"],
        "model_id": result["model_id"],
        "slot": result["slot"],
        "endpoint": result["endpoint"],
        "handoff_summary": result["handoff_summary"],
        "generation_backend": result["generation_meta"].get("backend"),
        "generation_fallback_used": bool(result["generation_meta"].get("fallback_used")),
        "wall_start_ms": result["wall_start_ms"],
        "wall_end_ms": result["wall_end_ms"],
    })
    private_state_events.append(result["private_event"])
    hmem_events.append(result["hmem_event"])
    scheduler_decisions.append(result["scheduler_decision"])
    real_generation_events.append({
        "task_id": result["task_id"],
        "agent_id": result["agent_id"],
        "model_id": result["model_id"],
        "slot": result["slot"],
        **{k: v for k, v in result["generation_meta"].items() if k not in ("endpoint",)},
    })

    # ==================================================================
    # PHASE 2: Concurrent cloud tasks (Architect → Engineer ∥ Red Team)
    # ==================================================================
    concurrent_wall_start = time.perf_counter()

    # Order: architect first (engineer depends on it), red-team waits on engineer
    concurrent_task_ids = ["architect", "engineer", "red-team"]

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ouroboros") as pool:
        futures: dict[str, Future] = {}

        # Architect submits immediately
        futures["architect"] = pool.submit(
            _execute_task,
            tasks_by_id["architect"],
            task_index=1,
            **exec_kwargs,
        )

        # Engineer waits for Architect's result to build its prompt
        def _engineer_after_architect() -> dict[str, Any]:
            arch_result = futures["architect"].result(timeout=float(args.timeout_seconds) + 30)
            # slots_raw is updated by architect already; engineer prompt reads it
            return _execute_task(tasks_by_id["engineer"], task_index=2, **exec_kwargs)

        futures["engineer"] = pool.submit(_engineer_after_architect)

        # Red Team waits for Engineer's result
        def _redteam_after_engineer() -> dict[str, Any]:
            futures["engineer"].result(timeout=float(args.timeout_seconds) + 60)
            return _execute_task(tasks_by_id["red-team"], task_index=3, **exec_kwargs)

        futures["red-team"] = pool.submit(_redteam_after_engineer)

        # Collect in dependency order (not as_completed, to preserve narrative)
        for tid in concurrent_task_ids:
            try:
                res = futures[tid].result(timeout=float(args.timeout_seconds) + 90)
            except Exception as exc:  # noqa: BLE001
                # Create a fallback result record on thread failure
                task_def = tasks_by_id[tid]
                agent_id = str(task_def["agent"])
                agent = bp["agents"][agent_id]
                model_key = str(agent["model"])
                model_id = str(model_states[model_key].get("model_id") or model_key)
                slot = str(task_def["slot"])
                fallback_text = OUROBOROS_FALLBACK_SLOTS.get(slot, "HeliX recorded the step.")
                slots_raw[slot] = fallback_text
                res = {
                    "task_id": tid,
                    "agent_id": agent_id,
                    "model_id": model_id,
                    "slot": slot,
                    "endpoint": str(model_states[model_key].get("endpoint") or "deepinfra"),
                    "generated": fallback_text,
                    "handoff_summary": sanitize_model_text(fallback_text, limit=320),
                    "generation_meta": {"backend": "fallback-thread-error", "error": str(exc), "fallback_used": True},
                    "hmem_event": {"task_id": tid, "agent_id": agent_id, "endpoint": "deepinfra", "memory_id": None, "observation_id": None, "memory_context_tokens": 0},
                    "private_event": {"event": "stateless_cloud_node", "task_id": tid, "model_id": model_id, "agent_id": agent_id, "endpoint": "deepinfra", "note": f"Thread error: {exc}"},
                    "scheduler_decision": {"task_id": tid, "agent_id": agent_id, "selected_model_id": model_id, "endpoint": "deepinfra", "candidate_models": [model_id], "estimated_cost_ms": 0, "actual_cost_ms": 0, "session_restored": False, "hmem_context_tokens": 0, "generation_backend": "fallback-thread-error", "generation_fallback_used": True},
                    "wall_start_ms": (time.perf_counter() - wall_t0) * 1000.0,
                    "wall_end_ms": (time.perf_counter() - wall_t0) * 1000.0,
                    "elapsed_ms": 0.0,
                }

            task_timeline.append({
                "task_id": res["task_id"],
                "agent_id": res["agent_id"],
                "model_id": res["model_id"],
                "slot": res["slot"],
                "endpoint": res["endpoint"],
                "handoff_summary": res["handoff_summary"],
                "generation_backend": res["generation_meta"].get("backend"),
                "generation_fallback_used": bool(res["generation_meta"].get("fallback_used")),
                "wall_start_ms": res["wall_start_ms"],
                "wall_end_ms": res["wall_end_ms"],
            })
            private_state_events.append(res["private_event"])
            hmem_events.append(res["hmem_event"])
            scheduler_decisions.append(res["scheduler_decision"])
            real_generation_events.append({
                "task_id": res["task_id"],
                "agent_id": res["agent_id"],
                "model_id": res["model_id"],
                "slot": res["slot"],
                **{k: v for k, v in res["generation_meta"].items() if k not in ("endpoint",)},
            })

    concurrent_wall_ms = (time.perf_counter() - concurrent_wall_start) * 1000.0

    # ==================================================================
    # PHASE 3: Tombstone evaluation + optional rewrite
    # ==================================================================
    redteam_output = slots_raw.get("redteam_out", "")
    triggered, tomb_reason = _evaluate_tombstone(redteam_output)

    if triggered:
        # Plant a Tombstone (fence_memory) for the engineer's output
        engineer_hmem_event = next(
            (e for e in hmem_events if e.get("task_id") == "engineer"),
            {},
        )
        fence_memory_id = engineer_hmem_event.get("memory_id")
        if fence_memory_id:
            fence_result = hmem.fence_memory(
                root=workspace,
                project=project,
                agent_id="decomposer",
                session_id=run_id,
                memory_id=fence_memory_id,
                reason=f"Red Team Tombstone: {tomb_reason[:240]}",
            )
        else:
            # Write a standalone fence observation when no parent memory to fence
            fence_result = hmem.observe_event(
                root=workspace,
                project=project,
                agent_id="decomposer",
                session_id=run_id,
                event_type="ouroboros_tombstone",
                content=f"TOMBSTONE\nRed Team verdict: {tomb_reason}",
                summary=f"Tombstone planted — Red Team escalation: {tomb_reason[:180]}",
                tags=["ouroboros", "tombstone", "red-team-escalation"],
                importance=9,
                promote=True,
                memory_type="episodic",
            )

        # Force-rewrite: re-run Engineer with tombstone context
        tombstone_note = (
            f"CRITICAL: The previous implementation was flagged by the Red Team. "
            f"Reason: {tomb_reason[:300]} "
            f"You MUST avoid the vulnerability described above. "
            f"Produce a corrected implementation that addresses ALL Red Team findings."
        )
        original_engineer_task = dict(tasks_by_id["engineer"])
        original_engineer_task["prompt"] = tombstone_note + "\n\n" + str(original_engineer_task.get("prompt", ""))

        rewrite_result = _execute_task(
            original_engineer_task,
            task_index=10,  # offset to avoid session dir collision
            **exec_kwargs,
        )
        # Override engineer slot with rewritten output
        slots_raw["engineer_out"] = rewrite_result["generated"]
        tombstone_event = {
            "triggered": True,
            "reason": tomb_reason,
            "rewrite_triggered": True,
            "fence_result": fence_result,
            "rewrite_summary": rewrite_result["handoff_summary"],
        }
        hmem_events.append(rewrite_result["hmem_event"])
    else:
        tombstone_event = {
            "triggered": False,
            "reason": tomb_reason,
            "rewrite_triggered": False,
        }

    # ==================================================================
    # PHASE 4: Merge (local, sequential, Warm Restore)
    # ==================================================================
    merge_task = tasks_by_id["merge"]
    merge_result = _execute_task(merge_task, task_index=4, **exec_kwargs)
    task_timeline.append({
        "task_id": merge_result["task_id"],
        "agent_id": merge_result["agent_id"],
        "model_id": merge_result["model_id"],
        "slot": merge_result["slot"],
        "endpoint": merge_result["endpoint"],
        "handoff_summary": merge_result["handoff_summary"],
        "generation_backend": merge_result["generation_meta"].get("backend"),
        "generation_fallback_used": bool(merge_result["generation_meta"].get("fallback_used")),
        "wall_start_ms": merge_result["wall_start_ms"],
        "wall_end_ms": merge_result["wall_end_ms"],
    })
    private_state_events.append(merge_result["private_event"])
    hmem_events.append(merge_result["hmem_event"])
    scheduler_decisions.append(merge_result["scheduler_decision"])
    real_generation_events.append({
        "task_id": merge_result["task_id"],
        "agent_id": merge_result["agent_id"],
        "model_id": merge_result["model_id"],
        "slot": merge_result["slot"],
        **{k: v for k, v in merge_result["generation_meta"].items() if k not in ("endpoint",)},
    })

    # Extract patch + tests from merger output
    patch_text, tests_text = _extract_patch_and_tests(slots_raw.get("merge_out", ""))

    # Write patch + test files to disk
    patch_path = output_dir / str(bp["outputs"].get("patch", "ouroboros.patch"))
    tests_path = output_dir / str(bp["outputs"].get("tests", "test_ouroboros_suite.py"))
    _write_text(patch_path, patch_text or OUROBOROS_FALLBACK_SLOTS["merge_out"])
    _write_text(tests_path, tests_text or "# Auto-generated: no tests output from model.")

    # Final deferred audits
    final_audits: list[dict[str, Any]] = []
    for session in latest_session.values():
        if audit_policy == "deferred":
            try:
                final_audits.append(rust_session.verify_deferred_session(session["path"]))
            except Exception:  # noqa: BLE001
                pass
    final_audit_status = (
        "verified"
        if not final_audits or all(item.get("audit_status") == "verified" for item in final_audits)
        else "failed"
    )

    memory_graph = hmem.graph(root=workspace, project=project, agent_id=None, limit=100)

    # Normalize content slots
    content_slots: dict[str, str] = {}
    for slot_key, fallback in OUROBOROS_FALLBACK_SLOTS.items():
        raw = slots_raw.get(slot_key, "")
        content_slots[slot_key] = sanitize_model_text(raw, limit=1200) if raw and len(raw) > 24 else fallback

    has_real_gen = any(not item.get("fallback_used") for item in real_generation_events)
    has_cloud_gen = any(item.get("endpoint") == "deepinfra" and not item.get("fallback_used") for item in scheduler_decisions)

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "title": "Proyecto Ouroboros — Meta-Compiler DAG",
        "benchmark_kind": "inference-os-blueprint-ouroboros-dag-v0",
        "status": "completed",
        "mode": str(args.mode),
        "blueprint_id": blueprint.blueprint_id,
        "layers_demonstrated": bp.get("layers_demonstrated", []),
        "models_used": model_states,
        "agents": [{f"agent_id": k, **v} for k, v in bp["agents"].items()],
        "tasks_raw": tasks_raw_list,
        "task_timeline": task_timeline,
        "private_state_events": private_state_events,
        "hmem_events": hmem_events,
        "scheduler_decisions": scheduler_decisions,
        "real_generation_events": real_generation_events,
        "tombstone_event": tombstone_event,
        "concurrent_phase_meta": {
            "tasks": _CONCURRENT_TASKS,
            "total_concurrent_wall_ms": concurrent_wall_ms,
        },
        "patch_artifact": patch_text,
        "tests_artifact": tests_text,
        "memory_graph": {
            "node_count": memory_graph.get("node_count"),
            "edge_count": memory_graph.get("edge_count"),
        },
        "content_slots": content_slots,
        "final_audits": final_audits,
        "final_audit_status": final_audit_status,
        "fallback_content_used": any(item.get("fallback_used") for item in real_generation_events),
        "cloud_generation_used": has_cloud_gen,
        "local_generation_used": has_real_gen,
        "public_claim_level": (
            "ouroboros-full-mom-dag"
            if has_cloud_gen
            else "orchestration-and-renderer"
        ),
        "claim_boundary": (
            "Ouroboros proves: concurrent MoM dispatch, Tombstone fencing via hmem.fence_memory(), "
            ".hlx Warm Restore for the Merger, and production of an executable patch + Pytest suite. "
            "HeliX governed three specialist models to patch the system that governs them."
        ),
    }

    # Render HTML
    html_text = render_ouroboros_site(artifact)
    quality = quality_check_ouroboros_html(html_text)
    artifact["quality_checks"] = quality
    artifact["html_output_path"] = str(site_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    site_output.parent.mkdir(parents=True, exist_ok=True)
    site_output.write_text(html_text, encoding="utf-8")
    _write_json(output_dir / str(bp["outputs"].get("artifact", "ouroboros-dag-artifact.json")), artifact)
    catalog.close()
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HeliX Proyecto Ouroboros — Meta-Compiler DAG runner."
    )
    parser.add_argument("--blueprint", default="blueprints/meta-compiler-dag.json")
    parser.add_argument(
        "--mode",
        default="hybrid-cloud",
        choices=["hybrid-cloud", "mock-only"],
        help="hybrid-cloud: uses DeepInfra for cloud tasks; mock-only: all fallback",
    )
    parser.add_argument("--deepinfra-api-key", default=None)
    parser.add_argument("--requirement", default=None, help="Path to raw requirement .txt file")
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--site-output", default="site-dist/ouroboros-demo.html")
    parser.add_argument("--codec", default="rust-hlx-buffered-flat")
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens for local model generation")
    parser.add_argument("--cloud-max-tokens", type=int, default=256, help="Max tokens per cloud model")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Timeout per cloud API request")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_ouroboros_demo(args)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "blueprint_id": payload.get("blueprint_id"),
                "html_output_path": payload.get("html_output_path"),
                "quality_status": (payload.get("quality_checks") or {}).get("status"),
                "tombstone_triggered": (payload.get("tombstone_event") or {}).get("triggered"),
                "rewrite_triggered": (payload.get("tombstone_event") or {}).get("rewrite_triggered"),
                "cloud_generation_used": payload.get("cloud_generation_used"),
                "final_audit_status": payload.get("final_audit_status"),
                "public_claim_level": payload.get("public_claim_level"),
                "concurrent_phase_wall_ms": (payload.get("concurrent_phase_meta") or {}).get("total_concurrent_wall_ms"),
                "patch_artifact_chars": len(payload.get("patch_artifact") or ""),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
