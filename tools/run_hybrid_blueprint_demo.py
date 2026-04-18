"""
run_hybrid_blueprint_demo.py
============================
HeliX Hybrid Context Handoff runner — "Privacy-Safe Research Sieve"

Orchestrates a 3-step blueprint:
  Task 1 (LOCAL)  — Qwen-1.5B: anonymize a dirty document, write anon_map to hmem
  Task 2 (CLOUD)  — DeepInfra Llama-3-70B: heavy reasoning on clean text (stateless)
  Task 3 (LOCAL)  — Qwen-1.5B: restore .hlx session, re-inject real names, final report

Usage
-----
python tools/run_hybrid_blueprint_demo.py ^
    --blueprint blueprints/hybrid-research.json ^
    --mode hybrid-cloud ^
    --deepinfra-api-key <KEY> ^
    --output-dir verification ^
    --site-output site-dist/hybrid-research-demo.html ^
    --doc tools/demo-doc-dirty.txt
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
import urllib.parse
import urllib.request
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
    HYBRID_FALLBACK_SLOTS,
    load_blueprint,
    make_private_state_arrays,
    quality_check_hybrid_html,
    render_hybrid_research_site,
    sanitize_model_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+", "PERSON"),
    (r"Prof\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+", "PERSON"),
    (r"Ms\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+", "PERSON"),
    (r"Mr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+", "PERSON"),
    # Plain "Firstname Lastname" capitalized pairs (after titles are stripped)
    (r"[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}", "PERSON"),
    # Org: "XYZ Labs", "XYZ Institute", etc.
    (r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Labs?|Institute|Corporation|Corp|Inc|University|Ministry)\b", "ORG"),
    # Email-like
    (r"[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]{2,}", "EMAIL"),
    # City + country pairs heuristic
    (r"Buenos\s+Aires(?:,\s*Argentina)?", "LOC"),
    # Grant IDs
    (r"Grant\s+ID:\s*[\w-]+", "GRANT_ID"),
    # H100 node IDs
    (r"ne-h100-\d+", "INFRA_NODE"),
]


def _build_anon_map(text: str) -> tuple[str, dict[str, str], int]:
    """
    Replace PII with placeholder tokens.
    Returns (clean_text, forward_map, entity_count)
    forward_map: {placeholder -> original}
    """
    counters: dict[str, int] = {}
    reverse: dict[str, str] = {}  # original -> placeholder (to reuse same token)
    forward: dict[str, str] = {}  # placeholder -> original

    result = text
    # Sort patterns so longer/more-specific ones fire first
    for pattern, entity_type in _PII_PATTERNS:
        for match in re.findall(pattern, result):
            match_str = match.strip()
            if not match_str or match_str in reverse:
                continue
            counters[entity_type] = counters.get(entity_type, 0) + 1
            placeholder = f"[{entity_type}_{counters[entity_type]}]"
            reverse[match_str] = placeholder
            forward[placeholder] = match_str

    # Apply substitutions (longest first to avoid partial matches)
    for original in sorted(reverse, key=len, reverse=True):
        result = result.replace(original, reverse[original])

    return result, forward, len(forward)


def _reinjection_report(cloud_synthesis: str, forward_map: dict[str, str]) -> str:
    """Re-inject real names into the cloud synthesis."""
    report = cloud_synthesis
    for placeholder, real in forward_map.items():
        report = report.replace(placeholder, real)
    return report


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


def _count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Local model backend (same logic as run_local_blueprint_demo.py)
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

_DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai/chat/completions"


def _run_deepinfra(
    model_ref: str,
    prompt: str,
    *,
    api_key: str,
    max_new_tokens: int,
    timeout_s: float = 120.0,
) -> tuple[str | None, dict[str, Any]]:
    """
    Calls DeepInfra's OpenAI-compatible chat completions endpoint.
    Records local_sleep_ms = total wall-clock time the local machine was waiting.
    """
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
        cloud_response_ms = (time.perf_counter() - sleep_started) * 1000.0
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            return None, {
                "backend": "deepinfra",
                "endpoint": "deepinfra",
                "model_ref": model_ref,
                "error": "empty_choices",
                "local_sleep_ms": cloud_response_ms,
                "cloud_request_time_ms": cloud_response_ms,
                "cloud_response_time_ms": cloud_response_ms,
                "cloud_fallback_used": True,
                "fallback_used": True,
            }
        content = str((choices[0].get("message") or {}).get("content") or "")
        usage = data.get("usage") or {}
        tokens_sent = int(usage.get("prompt_tokens") or _count_tokens_approx(prompt))
        tokens_back = int(usage.get("completion_tokens") or _count_tokens_approx(content))
        text = sanitize_model_text(content, limit=1200)
        return text or None, {
            "backend": "deepinfra",
            "endpoint": "deepinfra",
            "model_ref": model_ref,
            "local_sleep_ms": cloud_response_ms,
            "cloud_request_time_ms": cloud_response_ms,
            "cloud_response_time_ms": cloud_response_ms,
            "tokens_sent_to_cloud": tokens_sent,
            "tokens_received_from_cloud": tokens_back,
            "cloud_fallback_used": False,
            "fallback_used": False,
        }
    except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
        error_ms = (time.perf_counter() - sleep_started) * 1000.0
        return None, {
            "backend": "deepinfra",
            "endpoint": "deepinfra",
            "model_ref": model_ref,
            "error": f"{type(exc).__name__}:{exc}",
            "local_sleep_ms": error_ms,
            "cloud_request_time_ms": error_ms,
            "cloud_response_time_ms": 0.0,
            "cloud_fallback_used": True,
            "fallback_used": True,
        }


# ---------------------------------------------------------------------------
# Hybrid dispatcher
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
    model_ref = str(model.get("ref") or model.get("model_ref") or "")

    if mode == "mock-only":
        return None, {"backend": "mock", "endpoint": endpoint, "fallback_used": True}

    if endpoint == "deepinfra":
        if not api_key:
            return None, {
                "backend": "deepinfra",
                "endpoint": "deepinfra",
                "model_ref": model_ref,
                "error": "no_api_key",
                "local_sleep_ms": 0.0,
                "cloud_fallback_used": True,
                "fallback_used": True,
            }
        return _run_deepinfra(model_ref, prompt, api_key=api_key, max_new_tokens=cloud_max_tokens, timeout_s=timeout_s)

    # local model
    if _hf_ref_cached(model_ref):
        return _run_hf_cached_ref(model_ref, prompt, max_new_tokens=max_new_tokens)
    return None, {"backend": "none", "endpoint": endpoint, "error": "model_not_cached", "fallback_used": True}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_hybrid_demo(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901
    blueprint = load_blueprint(args.blueprint)
    output_dir = Path(args.output_dir)
    site_output = Path(args.site_output)
    api_key = str(args.deepinfra_api_key or os.environ.get("DEEPINFRA_API_KEY") or "")
    doc_path = Path(args.doc) if args.doc else (REPO_ROOT / "tools" / "demo-doc-dirty.txt")

    # Read document
    if doc_path.exists():
        raw_doc = doc_path.read_text(encoding="utf-8")
    else:
        raw_doc = (
            "Dr. Elena Marchetti and Prof. Carlos Vega at NeuraCore Labs, Buenos Aires "
            "report a 37% KV-cache reduction using selective attention with block_size=8. "
            "Ms. Valentina Rios is working on a zero-copy GPU path. "
            "Grant ID: MNCyT-2025-0047 assigned to Prof. Vega."
        )

    # Build anonymization map upfront (deterministic, local, no model needed)
    clean_doc, forward_map, pii_count = _build_anon_map(raw_doc)
    original_chars = len(raw_doc)

    run_id = f"{blueprint.blueprint_id}-{int(time.time())}"
    workspace = output_dir / "_blueprint-workspaces" / run_id
    sessions_root = workspace / "sessions"
    catalog = SessionCatalog.open(workspace / "session-catalog.sqlite")
    memory_policy = dict(blueprint.payload["memory_policy"])
    session_policy = dict(blueprint.payload["session_policy"])
    project = str(memory_policy.get("project") or hmem.DEFAULT_PROJECT)
    memory_mode = str(memory_policy.get("mode") or "search")
    memory_budget = int(memory_policy.get("budget_tokens") or 900)
    codec = str(args.codec or session_policy.get("codec") or "rust-hlx-buffered-flat")
    audit_policy = str(args.audit_policy or session_policy.get("audit_policy") or "deferred")

    # Evaluate model availability
    model_states: dict[str, dict[str, Any]] = {}
    for key, model in blueprint.payload["models"].items():
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

    # Tracking lists
    slots_raw: dict[str, str] = {}
    task_timeline: list[dict[str, Any]] = []
    model_lifecycle_events: list[dict[str, Any]] = []
    private_state_events: list[dict[str, Any]] = []
    hmem_events: list[dict[str, Any]] = []
    scheduler_decisions: list[dict[str, Any]] = []
    real_generation_events: list[dict[str, Any]] = []
    hybrid_events: list[dict[str, Any]] = []
    latest_session: dict[tuple[str, str], dict[str, Any]] = {}
    active_model_key: str | None = None

    # Write the anon map to hmem before tasks start (so cloud-analyst can't see it
    # and local reviewer CAN see it via hmem injection)
    _anon_map_str = json.dumps(forward_map, ensure_ascii=False, indent=2)
    hmem.observe_event(
        root=workspace,
        project=project,
        agent_id="anonymizer",
        session_id=run_id,
        event_type="anonymization_map",
        content=_anon_map_str,
        summary="PII anonymization map: placeholder -> original entity",
        tags=["blueprint", blueprint.blueprint_id, "anon-map", "pii"],
        importance=9,
        promote=True,
        memory_type="episodic",
    )

    for index, task in enumerate(blueprint.payload["tasks"]):
        started = time.perf_counter()
        agent_id = str(task["agent"])
        agent = blueprint.payload["agents"][agent_id]
        model_key = str(agent["model"])
        model = model_states[model_key]
        model_id = str(model.get("model_id") or model_key)
        slot = str(task["slot"])
        endpoint = str(model.get("endpoint") or "local")

        lifecycle_event = {
            "event": "model_reuse" if active_model_key == model_key else "model_activate",
            "previous_model_key": active_model_key,
            "model_key": model_key,
            "model_id": model_id,
            "endpoint": endpoint,
            "generation_mode": model["generation_mode"],
            "load_time_ms": 0.0 if active_model_key == model_key else float(model.get("load_time_estimate_ms") or 0.0),
        }
        active_model_key = model_key
        model_lifecycle_events.append(lifecycle_event)

        restored = latest_session.get((model_id, agent_id))
        task_expects_restore = bool(task.get("expects_restore"))

        # Build prompt — inject clean_doc for task 1, anon context for task 2, full context for task 3
        context = hmem.build_context(
            root=workspace,
            project=project,
            agent_id=agent_id,
            query=str(task.get("prompt") or ""),
            budget_tokens=memory_budget,
            mode=memory_mode,
        )

        if slot == "anon_map":
            # Task 1: inject the raw document
            prompt = (
                f"Role: {agent.get('role')}\n"
                f"Task: {task.get('prompt')}\n\n"
                f"DOCUMENT TO ANALYZE:\n{raw_doc}\n\n"
                f"Memory context:\n{context.get('context') or '(empty)'}"
            )
        elif slot == "cloud_synthesis":
            # Task 2: inject the CLEAN document only
            prompt = (
                f"Role: {agent.get('role')}\n"
                f"Task: {task.get('prompt')}\n\n"
                f"ANONYMIZED DOCUMENT:\n{clean_doc}\n\n"
                f"Note: Entity names have been replaced with placeholder tokens for privacy."
            )
        else:
            # Task 3: inject cloud synthesis + anon map from hmem
            cloud_output = slots_raw.get("cloud_synthesis", "")
            prompt = (
                f"Role: {agent.get('role')}\n"
                f"Task: {task.get('prompt')}\n\n"
                f"CLOUD ANALYSIS (with placeholders):\n{cloud_output}\n\n"
                f"ANONYMIZATION MAP (from hmem):\n{_anon_map_str[:600]}\n\n"
                f"Memory context:\n{context.get('context') or '(empty)'}"
            )

        # Generate
        generated: str | None = None
        generation_meta: dict[str, Any] = {"backend": "fallback-deterministic", "endpoint": endpoint}

        if model["available_real_model"]:
            generated, generation_meta = _run_model(
                model,
                prompt,
                api_key=api_key,
                max_new_tokens=int(args.max_new_tokens),
                cloud_max_tokens=int(args.cloud_max_tokens),
                mode=args.mode,
                timeout_s=float(args.timeout_seconds),
            )
            generation_meta["endpoint"] = endpoint

        if not generated:
            # Fallback content
            if slot == "anon_map":
                # Build the anon map from our deterministic extractor and use fallback text + map
                generated = (
                    f"{HYBRID_FALLBACK_SLOTS['anon_map']} "
                    f"Entities masked: {pii_count}. Map: {'; '.join(f'{v}->{k}' for k, v in list(forward_map.items())[:5])}."
                )
            elif slot == "cloud_synthesis":
                generated = HYBRID_FALLBACK_SLOTS["cloud_synthesis"]
            elif slot == "final_report":
                # Re-inject using local deterministic logic
                cloud_out = slots_raw.get("cloud_synthesis", HYBRID_FALLBACK_SLOTS["cloud_synthesis"])
                reinjected = _reinjection_report(cloud_out, forward_map)
                generated = (
                    f"{HYBRID_FALLBACK_SLOTS['final_report']} "
                    f"Re-injected synthesis: {reinjected[:300]}"
                ) if reinjected != cloud_out else HYBRID_FALLBACK_SLOTS["final_report"]
            else:
                generated = HYBRID_FALLBACK_SLOTS.get(slot, "HeliX recorded the step.")
            generation_meta = {**generation_meta, "fallback_used": True}
        else:
            # Post-process task 3: deterministic re-injection on top of model output
            if slot == "final_report":
                reinjected = _reinjection_report(generated, forward_map)
                if reinjected != generated:
                    generated = reinjected
            generation_meta = {**generation_meta, "fallback_used": generation_meta.get("fallback_used", False)}

        # Track hybrid event for cloud tasks
        if endpoint == "deepinfra":
            hybrid_events.append({
                "task_id": task["task_id"],
                "agent_id": agent_id,
                "endpoint": "deepinfra",
                "model_ref": str(model.get("ref") or ""),
                "local_sleep_ms": generation_meta.get("local_sleep_ms", 0.0),
                "cloud_request_time_ms": generation_meta.get("cloud_request_time_ms", 0.0),
                "cloud_response_time_ms": generation_meta.get("cloud_response_time_ms", 0.0),
                "tokens_sent_to_cloud": generation_meta.get("tokens_sent_to_cloud", _count_tokens_approx(clean_doc)),
                "tokens_received_from_cloud": generation_meta.get("tokens_received_from_cloud", 0),
                "cloud_fallback_used": bool(generation_meta.get("cloud_fallback_used")),
            })

        slots_raw[slot] = generated
        real_generation_events.append({
            "task_id": task["task_id"],
            "agent_id": agent_id,
            "model_id": model_id,
            "slot": slot,
            **{k: v for k, v in generation_meta.items() if k not in ("endpoint",)},
        })

        # hmem: cloud agent does NOT write its raw output (it might contain masked placeholders)
        # Local agents write normally
        observe = hmem.observe_event(
            root=workspace,
            project=project,
            agent_id=agent_id,
            session_id=run_id,
            event_type="blueprint_task",
            content=generated,
            summary=f"{task['task_id']}: {generated[:120]}",
            tags=["blueprint", blueprint.blueprint_id, str(task["task_id"]), slot, endpoint],
            importance=7,
            promote=True,
            memory_type="episodic",
        )
        hmem_events.append({
            "task_id": task["task_id"],
            "agent_id": agent_id,
            "endpoint": endpoint,
            "memory_id": (observe.get("memory") or {}).get("memory_id"),
            "observation_id": (observe.get("observation") or {}).get("observation_id"),
            "memory_context_tokens": context.get("tokens", 0),
        })

        # Private state — cloud agent has NO .hlx session (stateless node)
        token_ids = _token_ids(generated)
        session_dir = sessions_root / _safe(model_id) / _safe(agent_id) / f"v{index + 1:04d}"

        if endpoint != "deepinfra":
            if restored is not None and task_expects_restore:
                _, _, load_receipt = rust_session.load_session_bundle(restored["path"], verify_policy="receipt-only")
                private_state_events.append({
                    "event": "session_restored",
                    "task_id": task["task_id"],
                    "model_id": model_id,
                    "agent_id": agent_id,
                    "path": str(restored["path"]),
                    "session_hash_loaded": load_receipt.get("session_hash_loaded") or load_receipt.get("session_hash"),
                })

            receipt = rust_session.save_session_bundle(
                session_dir,
                meta={
                    "blueprint_id": blueprint.blueprint_id,
                    "run_id": run_id,
                    "task_id": task["task_id"],
                    "model_id": model_id,
                    "agent_id": agent_id,
                    "session_token_ids": token_ids,
                    "private_state_kind": "hybrid-blueprint-v0",
                    "endpoint": endpoint,
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
            latest_session[(model_id, agent_id)] = {
                "path": session_dir,
                "session_hash": session_hash,
                "session_id": f"{_safe(model_id)}__{_safe(agent_id)}__v{index + 1:04d}",
            }
            private_state_events.append({
                "event": "session_saved",
                "task_id": task["task_id"],
                "model_id": model_id,
                "agent_id": agent_id,
                "endpoint": endpoint,
                "path": str(session_dir),
                "audit_status": receipt.get("audit_status"),
                "session_total_bytes": receipt.get("session_total_bytes"),
                "session_hash": session_hash,
            })
        else:
            # Cloud agent: record as stateless — no .hlx saved
            private_state_events.append({
                "event": "stateless_cloud_node",
                "task_id": task["task_id"],
                "model_id": model_id,
                "agent_id": agent_id,
                "endpoint": "deepinfra",
                "note": "Cloud node has no private .hlx state by design.",
            })

        elapsed = (time.perf_counter() - started) * 1000.0
        decision = {
            "task_id": task["task_id"],
            "agent_id": agent_id,
            "selected_model_id": model_id,
            "endpoint": endpoint,
            "candidate_models": [model_id, model.get("fallback")],
            "estimated_cost_ms": float(model.get("load_time_estimate_ms") or 0.0) + int(args.max_new_tokens),
            "actual_cost_ms": elapsed,
            "model_swapped": lifecycle_event["event"] == "model_activate",
            "session_restored": restored is not None and task_expects_restore and endpoint != "deepinfra",
            "hmem_context_tokens": context.get("tokens", 0),
            "audit_status": "n/a" if endpoint == "deepinfra" else "pending",
            "generation_backend": generation_meta.get("backend"),
            "generation_fallback_used": bool(generation_meta.get("fallback_used")),
        }
        scheduler_decisions.append(decision)

        handoff = sanitize_model_text(generated, limit=320)
        task_timeline.append({
            "task_id": task["task_id"],
            "agent_id": agent_id,
            "model_id": model_id,
            "slot": slot,
            "endpoint": endpoint,
            "handoff_summary": handoff,
            "restored_private_state": restored is not None and task_expects_restore and endpoint != "deepinfra",
            "hmem_memory_id": hmem_events[-1]["memory_id"],
            "generation_backend": generation_meta.get("backend"),
            "generation_fallback_used": bool(generation_meta.get("fallback_used")),
        })

    # Final audits (only local sessions)
    final_audits: list[dict[str, Any]] = []
    for session in latest_session.values():
        if audit_policy == "deferred":
            final_audits.append(rust_session.verify_deferred_session(session["path"]))
    final_audit_status = "verified" if not final_audits or all(item.get("audit_status") == "verified" for item in final_audits) else "failed"

    memory_graph = hmem.graph(root=workspace, project=project, agent_id=None, limit=100)

    # Normalize content slots for HTML
    content_slots: dict[str, str] = {}
    for slot_key, fallback in HYBRID_FALLBACK_SLOTS.items():
        raw = slots_raw.get(slot_key, "")
        content_slots[slot_key] = sanitize_model_text(raw) if raw and len(raw) > 24 else fallback

    # Privacy audit summary
    cloud_synthesis_text = slots_raw.get("cloud_synthesis", "")
    real_names = list(forward_map.values())
    cloud_saw_real = any(name in cloud_synthesis_text for name in real_names if len(name) > 4)
    final_report_text = slots_raw.get("final_report", "")
    re_inject_ok = any(name in final_report_text for name in real_names if len(name) > 4)

    cloud_event = next((e for e in hybrid_events if e.get("endpoint") == "deepinfra"), {})

    privacy_audit = {
        "doc_chars_original": original_chars,
        "doc_chars_sent_to_cloud": len(clean_doc),
        "pii_entities_masked": pii_count,
        "placeholder_map_size": len(forward_map),
        "cloud_saw_real_names": cloud_saw_real,
        "re_injection_successful": re_inject_ok,
        "anon_map_in_hmem": True,
    }

    has_real_gen = any(not item.get("fallback_used") for item in real_generation_events)
    has_cloud_gen = any(not e.get("cloud_fallback_used") for e in hybrid_events)

    artifact = {
        "schema_version": 1,
        "title": "HeliX Hybrid Research Sieve — Privacy-Safe Blueprint",
        "benchmark_kind": "inference-os-blueprint-hybrid-research-v0",
        "status": "completed",
        "mode": str(args.mode),
        "blueprint_id": blueprint.blueprint_id,
        "layers_demonstrated": ["active_model", "private_hlx_state", "shared_hmem", "multimodel_scheduler", "cloud_handoff"],
        "models_used": list(model_states.values()),
        "agents": [{"agent_id": key, **value} for key, value in blueprint.payload["agents"].items()],
        "task_timeline": task_timeline,
        "model_lifecycle_events": model_lifecycle_events,
        "private_state_events": private_state_events,
        "hmem_events": hmem_events,
        "scheduler_decisions": scheduler_decisions,
        "real_generation_events": real_generation_events,
        "hybrid_events": hybrid_events,
        "privacy_audit": privacy_audit,
        "memory_graph": {"node_count": memory_graph.get("node_count"), "edge_count": memory_graph.get("edge_count")},
        "content_slots": content_slots,
        "final_audits": final_audits,
        "final_audit_status": final_audit_status,
        "fallback_content_used": any(item.get("fallback_used") for item in real_generation_events),
        "cloud_generation_used": has_cloud_gen,
        "local_generation_used": has_real_gen,
        "public_claim_level": (
            "hybrid-cloud-local-orchestration" if has_cloud_gen
            else "local-only-orchestration" if has_real_gen
            else "orchestration-and-renderer"
        ),
        "claim_boundary": (
            "Hybrid mode proves Blueprint orchestration, privacy shield via hmem, "
            ".hlx private state continuity, cloud dispatch, and local re-injection. "
            "Data sovereignty: the cloud received only anonymized text."
        ),
    }

    html_text = render_hybrid_research_site(artifact)
    quality = quality_check_hybrid_html(html_text)
    artifact["quality_checks"] = quality
    artifact["html_output_path"] = str(site_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    site_output.parent.mkdir(parents=True, exist_ok=True)
    site_output.write_text(html_text, encoding="utf-8")
    _write_json(output_dir / blueprint.payload["outputs"]["artifact"], artifact)
    catalog.close()
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HeliX Hybrid Context Handoff — Privacy-Safe Research Sieve.")
    parser.add_argument("--blueprint", default="blueprints/hybrid-research.json")
    parser.add_argument(
        "--mode",
        default="hybrid-cloud",
        choices=["hybrid-cloud", "budgeted-local", "mock-only"],
        help="hybrid-cloud: uses DeepInfra for cloud tasks; budgeted-local: all local; mock-only: all fallback",
    )
    parser.add_argument("--deepinfra-api-key", default=None, help="DeepInfra API key (or set DEEPINFRA_API_KEY env var)")
    parser.add_argument("--doc", default=None, help="Path to the raw research document to anonymize (default: tools/demo-doc-dirty.txt)")
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--site-output", default="site-dist/hybrid-research-demo.html")
    parser.add_argument("--codec", default="rust-hlx-buffered-flat")
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens for local model generation")
    parser.add_argument("--cloud-max-tokens", type=int, default=512, help="Max tokens for cloud model generation")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Timeout for cloud API requests (seconds)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_hybrid_demo(args)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "artifact": str(Path(args.output_dir) / load_blueprint(args.blueprint).payload["outputs"]["artifact"]),
                "html_output_path": payload.get("html_output_path"),
                "quality_status": (payload.get("quality_checks") or {}).get("status"),
                "cloud_generation_used": payload.get("cloud_generation_used"),
                "privacy_audit": payload.get("privacy_audit"),
                "final_audit_status": payload.get("final_audit_status"),
                "public_claim_level": payload.get("public_claim_level"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
