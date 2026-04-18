from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(max(1, min(2, int(torch.get_num_threads()))))

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_kv import rust_session  # noqa: E402
from helix_kv.session_os import SessionCatalog, SessionScheduler  # noqa: E402
from helix_kv.transformers_cache import _load_benchmark_cache, _save_benchmark_cache  # noqa: E402
from tools.run_local_hybrid_stress import _encode_prompt_text, _json_ready, _run_generation_trace, _write_json  # noqa: E402
from tools.run_local_session_core import PROFILE, _hf_model_cached, _load_model_bundle  # noqa: E402


SHOWCASE_DIR = REPO_ROOT / "benchmarks" / "agent_showcase"
HETEROGENEOUS_FIXTURE = SHOWCASE_DIR / "heterogeneous_pr_handoff.md"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gpt2-fast": {
        "ref": "gpt2",
        "arch": "transformer",
        "profile_key": "gpt2",
        "capabilities": ["fallback", "drafting", "smoke"],
        "ram_bytes_estimate": 548_000_000,
        "load_time_estimate_ms": 1_500,
    },
    "qwen-1.5b": {
        "ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "arch": "transformer",
        "profile_key": "qwen",
        "capabilities": ["general", "reasoning", "drafting", "code-review"],
        "ram_bytes_estimate": 3_100_000_000,
        "load_time_estimate_ms": 5_000,
    },
    "qwen-coder-1.5b": {
        "ref": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "arch": "transformer",
        "profile_key": "qwen",
        "capabilities": ["code", "debugging", "review"],
        "ram_bytes_estimate": 3_100_000_000,
        "load_time_estimate_ms": 5_000,
    },
    "qwen-3b-hxq": {
        "ref": "EchoLabs33/qwen2.5-3b-instruct-helix",
        "arch": "transformer-hxq",
        "profile_key": "qwen",
        "capabilities": ["general", "reasoning", "drafting", "hxq"],
        "ram_bytes_estimate": 3_500_000_000,
        "load_time_estimate_ms": 8_000,
        "experimental": True,
    },
    "zamba2-1.2b": {
        "ref": "Zyphra/Zamba2-1.2B-Instruct-v2",
        "arch": "hybrid-mamba-transformer",
        "profile_key": "zamba",
        "capabilities": ["long-context", "continuity", "hybrid"],
        "ram_bytes_estimate": 2_800_000_000,
        "load_time_estimate_ms": 7_000,
    },
}


@dataclass
class ActiveBundle:
    model_id: str
    model_ref: str
    arch: str
    profile_key: str
    model: Any
    adapter: Any
    input_adapter: str
    variant: dict[str, Any]


@dataclass
class SessionEntry:
    model_id: str
    agent_id: str
    model_ref: str
    arch: str
    version: int
    path: Path
    session_total_bytes: int
    audit_status: str | None
    session_hash: str | None
    created_ms: float


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value)).strip("-") or "item"


def _read_session_receipt(session_dir: Path) -> dict[str, Any]:
    receipt_path = session_dir / "session-hlx-receipt.json"
    if not receipt_path.exists():
        return {}
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _receipt_session_marker(receipt: dict[str, Any]) -> str | None:
    if not receipt:
        return None
    inner = receipt.get("receipt") if isinstance(receipt.get("receipt"), dict) else receipt
    return (
        inner.get("session_hash")
        or receipt.get("session_hash")
        or inner.get("fast_payload_checksum")
        or receipt.get("fast_payload_checksum")
    )


def _transcript_hash(transcript: list[dict[str, Any]]) -> str:
    payload = json.dumps(_json_ready(transcript), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truncate(text: str, limit: int = 420) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _fixture_text() -> str:
    return HETEROGENEOUS_FIXTURE.read_text(encoding="utf-8")


def _compact_fixture_text() -> str:
    return (
        "Mini PR context:\n"
        "- save_session_bundle now forwards audit_policy into rust-hlx-buffered-flat.\n"
        "- SessionStore changed from store.load(agent_id) to store.load(model_id, agent_id).\n"
        "- The key safety risk is restoring a cache under the wrong model id.\n"
        "- Qwen, GPT-2, and Zamba sessions are not interchangeable because their cache shapes and semantics differ.\n"
        "- Deferred audit means pending checkpoints can be restored quickly, but public integrity claims require final audit_status=verified."
    )


def _registry_public(local_files_only: bool) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for model_id, item in MODEL_REGISTRY.items():
        ref = str(item["ref"])
        payload[model_id] = {
            "ref": ref,
            "arch": item["arch"],
            "capabilities": list(item.get("capabilities") or []),
            "ram_bytes_estimate": item.get("ram_bytes_estimate"),
            "load_time_estimate_ms": item.get("load_time_estimate_ms"),
            "cached": _hf_model_cached(ref),
            "experimental": bool(item.get("experimental", False)),
        }
        if local_files_only and not payload[model_id]["cached"]:
            payload[model_id]["availability"] = "skipped_not_cached"
        else:
            payload[model_id]["availability"] = "available"
    return payload


def _model_available(model_id: str, *, local_files_only: bool) -> bool:
    if model_id not in MODEL_REGISTRY:
        return False
    return (not local_files_only) or _hf_model_cached(str(MODEL_REGISTRY[model_id]["ref"]))


def _choose_model(requested: str, fallbacks: list[str], *, local_files_only: bool, strict: bool) -> str | None:
    candidates = [requested] + [item for item in fallbacks if item != requested]
    for model_id in candidates:
        if model_id in MODEL_REGISTRY and _model_available(model_id, local_files_only=local_files_only):
            return model_id
        if strict and model_id == requested:
            return None
    return None


class ModelLifecycle:
    def __init__(self, *, device: torch.device, local_files_only: bool) -> None:
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.active: ActiveBundle | None = None
        self.events: list[dict[str, Any]] = []

    @property
    def active_model_id(self) -> str | None:
        return None if self.active is None else self.active.model_id

    def activate(self, model_id: str) -> tuple[ActiveBundle, dict[str, Any]]:
        if self.active is not None and self.active.model_id == model_id:
            event = {
                "event": "model_reuse",
                "model_id": model_id,
                "model_ref": self.active.model_ref,
                "swap_time_ms": 0.0,
                "load_time_ms": 0.0,
                "unload_time_ms": 0.0,
            }
            self.events.append(event)
            return self.active, event
        previous_model_id = self.active_model_id
        unload_ms = 0.0
        if self.active is not None:
            unload_ms = self.deactivate()["unload_time_ms"]
        entry = MODEL_REGISTRY[model_id]
        start = time.perf_counter()
        model, adapter, input_adapter = _load_model_bundle(
            str(entry["ref"]),
            device=self.device,
            local_files_only=self.local_files_only,
        )
        load_ms = (time.perf_counter() - start) * 1000.0
        profile_key = str(entry["profile_key"])
        self.active = ActiveBundle(
            model_id=model_id,
            model_ref=str(entry["ref"]),
            arch=str(entry["arch"]),
            profile_key=profile_key,
            model=model,
            adapter=adapter,
            input_adapter=input_adapter,
            variant=dict(PROFILE[profile_key]["variant"]),
        )
        event = {
            "event": "model_activate",
            "previous_model_id": previous_model_id,
            "model_id": model_id,
            "model_ref": entry["ref"],
            "arch": entry["arch"],
            "load_time_ms": load_ms,
            "unload_time_ms": unload_ms,
            "swap_time_ms": load_ms + unload_ms,
        }
        self.events.append(event)
        return self.active, event

    def deactivate(self) -> dict[str, Any]:
        if self.active is None:
            event = {"event": "model_deactivate", "model_id": None, "unload_time_ms": 0.0}
            self.events.append(event)
            return event
        model_id = self.active.model_id
        model_ref = self.active.model_ref
        start = time.perf_counter()
        del self.active.model
        del self.active.adapter
        self.active = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        unload_ms = (time.perf_counter() - start) * 1000.0
        event = {"event": "model_deactivate", "model_id": model_id, "model_ref": model_ref, "unload_time_ms": unload_ms}
        self.events.append(event)
        return event


class SessionStore:
    def __init__(self, root: Path, *, catalog: SessionCatalog | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self._latest: dict[tuple[str, str], SessionEntry] = {}
        self._all: list[SessionEntry] = []

    def latest(self, model_id: str, agent_id: str) -> SessionEntry | None:
        return self._latest.get((model_id, agent_id))

    def save(
        self,
        *,
        model_id: str,
        agent_id: str,
        model_ref: str,
        arch: str,
        cache: Any,
        model_config: Any,
        codec: str,
        audit_policy: str,
        token_ids: list[int] | None = None,
        parent_session_id: str | None = None,
    ) -> tuple[SessionEntry, dict[str, Any]]:
        key = (str(model_id), str(agent_id))
        version = 1 if key not in self._latest else self._latest[key].version + 1
        session_dir = self.root / f"{_safe_key(model_id)}__{_safe_key(agent_id)}" / f"v{version:04d}"
        if session_dir.exists():
            shutil.rmtree(session_dir)
        start = time.perf_counter()
        _save_benchmark_cache(cache, model_config=model_config, path=session_dir, session_codec=codec, audit_policy=audit_policy)
        save_ms = (time.perf_counter() - start) * 1000.0
        receipt = _read_session_receipt(session_dir)
        total_bytes = int(sum(item.stat().st_size for item in session_dir.iterdir() if item.is_file()))
        index = {
            "model_id": model_id,
            "agent_id": agent_id,
            "model_ref": model_ref,
            "arch": arch,
            "version": version,
            "codec": codec,
            "audit_policy": audit_policy,
            "session_token_ids": [int(item) for item in (token_ids or [])],
            "parent_session_id": parent_session_id,
        }
        (session_dir / "session-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        session_json = session_dir / "session.json"
        if session_json.exists() and token_ids is not None:
            try:
                meta = json.loads(session_json.read_text(encoding="utf-8"))
                meta["session_token_ids"] = [int(item) for item in token_ids]
                meta["parent_session_id"] = parent_session_id
                session_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except json.JSONDecodeError:
                pass
        entry = SessionEntry(
            model_id=model_id,
            agent_id=agent_id,
            model_ref=model_ref,
            arch=arch,
            version=version,
            path=session_dir,
            session_total_bytes=total_bytes,
            audit_status=receipt.get("audit_status"),
            session_hash=_receipt_session_marker(receipt),
            created_ms=time.time() * 1000.0,
        )
        self._latest[key] = entry
        self._all.append(entry)
        if self.catalog is not None:
            self.catalog.record_session(
                session_id=f"{_safe_key(model_id)}__{_safe_key(agent_id)}__v{version:04d}",
                model_id=model_id,
                agent_id=agent_id,
                model_ref=model_ref,
                arch=arch,
                path=session_dir,
                token_ids=token_ids or [],
                session_bytes=total_bytes,
                codec=codec,
                audit_status=entry.audit_status,
                session_hash=entry.session_hash,
                merkle_root=receipt.get("merkle_root"),
                parent_session_id=parent_session_id,
            )
        return entry, {"save_time_ms": save_ms, "session_total_bytes": total_bytes, "receipt": receipt}

    def load(
        self,
        *,
        model_id: str,
        agent_id: str,
        model_config: Any,
        device: torch.device,
        verify_policy: str,
    ) -> tuple[Any | None, SessionEntry | None, dict[str, Any]]:
        entry = self.latest(model_id, agent_id)
        if entry is None:
            return None, None, {"load_time_ms": 0.0, "session_hash_loaded": None}
        index = json.loads((entry.path / "session-index.json").read_text(encoding="utf-8"))
        if index.get("model_id") != model_id or index.get("agent_id") != agent_id:
            raise ValueError("session index mismatch")
        start = time.perf_counter()
        cache = _load_benchmark_cache(entry.path, model_config=model_config, device=device, verify_policy=verify_policy)
        load_ms = (time.perf_counter() - start) * 1000.0
        return cache, entry, {"load_time_ms": load_ms, "session_hash_loaded": _receipt_session_marker(_read_session_receipt(entry.path))}

    def verify_all(self, *, codec: str, audit_policy: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for entry in self._all:
            start = time.perf_counter()
            try:
                if str(audit_policy) == "deferred":
                    receipt = rust_session.verify_deferred_session(entry.path)
                elif str(codec) in {"rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat"}:
                    receipt = rust_session.verify_hlx_session(entry.path)
                else:
                    receipt = _read_session_receipt(entry.path)
                audit_status = str(receipt.get("audit_status") or "verified")
                ok = bool(receipt.get("ok", True)) and audit_status != "failed"
            except Exception as exc:
                receipt = {"audit_status": "failed", "audit_error": str(exc)}
                audit_status = "failed"
                ok = False
            results.append(
                {
                    "model_id": entry.model_id,
                    "agent_id": entry.agent_id,
                    "model_ref": entry.model_ref,
                    "arch": entry.arch,
                    "version": entry.version,
                    "path": str(entry.path),
                    "audit_status": audit_status,
                    "ok": ok,
                    "audit_time_ms": (time.perf_counter() - start) * 1000.0,
                    "session_hash": receipt.get("session_hash") or _receipt_session_marker(receipt),
                    "merkle_root": receipt.get("merkle_root"),
                }
            )
        return results

    def session_sizes(self) -> dict[str, int]:
        return {f"{entry.model_id}/{entry.agent_id}/v{entry.version:04d}": entry.session_total_bytes for entry in self._all}


def _resolve_model_pair(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    if args.scenario == "qwen-zamba-handoff":
        code_fallbacks = ["qwen-1.5b", "gpt2-fast"]
        writer_fallbacks = ["zamba2-1.2b", "qwen-1.5b"]
        code = _choose_model(args.code_model, code_fallbacks, local_files_only=args.local_files_only, strict=args.strict_models)
        writer = _choose_model(args.hybrid_model, writer_fallbacks, local_files_only=args.local_files_only, strict=args.strict_models)
    else:
        code_fallbacks = ["qwen-coder-1.5b", "qwen-1.5b", "gpt2-fast"]
        writer_fallbacks = ["gpt2-fast", "qwen-1.5b"]
        code = _choose_model(args.code_model, code_fallbacks, local_files_only=args.local_files_only, strict=args.strict_models)
        writer = _choose_model(args.writer_model, writer_fallbacks, local_files_only=args.local_files_only, strict=args.strict_models)
    if code is None or writer is None:
        return code, writer, "required_model_not_available"
    if code == writer:
        alternate = _choose_model("gpt2-fast" if code != "gpt2-fast" else "qwen-1.5b", [], local_files_only=args.local_files_only, strict=False)
        if alternate is not None and alternate != code:
            writer = alternate
    if code == writer:
        return code, writer, "could_not_resolve_two_distinct_models"
    return code, writer, None


def _build_tasks(*, scenario: str, code_model_id: str, writer_model_id: str) -> list[dict[str, Any]]:
    return [
        {
            "task_id": "code-review-initial",
            "agent_id": "code_reviewer",
            "model_id": code_model_id,
            "capability": "code-review",
            "prompt": (
                "code-review-initial"
            ),
            "expects_restore": False,
        },
        {
            "task_id": "writer-summary",
            "agent_id": "writer",
            "model_id": writer_model_id,
            "capability": "drafting",
            "prompt": (
                "writer-summary"
            ),
            "expects_restore": False,
        },
        {
            "task_id": "code-review-followup",
            "agent_id": "code_reviewer",
            "model_id": code_model_id,
            "capability": "code-review",
            "prompt": (
                "code-review-followup"
            ),
            "expects_restore": True,
        },
    ]


def _prompt_for_task(task: dict[str, Any], transcript: list[dict[str, Any]]) -> str:
    fixture = _fixture_text()
    task_id = str(task["task_id"])
    if task_id == "code-review-initial":
        return (
            "You are the code_reviewer agent. Review this mini PR and name the highest-risk bug. "
            "Be specific about model_id + agent_id session isolation. Start with Observation:\n\n"
            f"{_compact_fixture_text()}\n\nFull fixture title: {fixture.splitlines()[0] if fixture else 'fixture'}"
        )
    if task_id == "writer-summary":
        return (
            "You are the writer agent. Convert the prior engineering handoff into one public-safe summary. "
            "Do not claim cryptographic verification until audit_status is verified. Start with Observation:\n\n"
            f"Shared transcript hash: {_transcript_hash(transcript)}\n"
            f"Prior handoffs: {json.dumps(_json_ready(transcript[-2:]), ensure_ascii=False)}"
        )
    if task_id == "code-review-followup":
        return (
            "Continue the same code_reviewer session after another model handled the writer task. "
            "Name one test that proves a Qwen session is never restored into a GPT-2 or Zamba model. "
            "Start with Observation:\n\n"
            f"Shared transcript hash: {_transcript_hash(transcript)}\n"
            f"Prior handoffs: {json.dumps(_json_ready(transcript[-3:]), ensure_ascii=False)}"
        )
    return str(task.get("prompt") or "")


def _run_task(
    *,
    task: dict[str, Any],
    lifecycle: ModelLifecycle,
    store: SessionStore,
    codec: str,
    audit_policy: str,
    max_new_tokens: int,
    prompt_tokens: int,
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    route_decision = None
    if store.catalog is not None:
        route_task = {
            "model_id": str(task["model_id"]),
            "agent_id": str(task["agent_id"]),
            "capability": str(task["capability"]),
            "max_new_tokens": int(max_new_tokens),
            "prefill_ms_per_token": 1.0,
            "decode_ms_per_token": 1.0,
        }
        route_decision = SessionScheduler(store.catalog).route(route_task, MODEL_REGISTRY, lifecycle)
    bundle, swap_event = lifecycle.activate(str(task["model_id"]))
    cache, restored_entry, load = store.load(
        model_id=bundle.model_id,
        agent_id=str(task["agent_id"]),
        model_config=bundle.model.config,
        device=lifecycle.device,
        verify_policy="receipt-only" if str(audit_policy) == "deferred" else "full",
    )
    session_hash_before = None if restored_entry is None else restored_entry.session_hash
    prompt_inputs, prompt_ids = _encode_prompt_text(
        model_ref=bundle.model_ref,
        adapter=bundle.adapter,
        input_adapter=bundle.input_adapter,
        prompt_text=_prompt_for_task(task, transcript),
        max_tokens=int(prompt_tokens),
    )
    start = time.perf_counter()
    result = _run_generation_trace(
        bundle.model,
        prompt_inputs=prompt_inputs,
        prompt_ids=prompt_ids,
        variant=bundle.variant,
        max_new_tokens=int(max_new_tokens),
        adapter=bundle.adapter,
        device=lifecycle.device,
        initial_cache=cache,
        resume_prompt_token_by_token=bundle.arch == "hybrid-mamba-transformer" and cache is not None,
        return_cache=True,
    )
    generation_time_ms = (time.perf_counter() - start) * 1000.0
    updated_cache = result.pop("_cache")
    entry, save = store.save(
        model_id=bundle.model_id,
        agent_id=str(task["agent_id"]),
        model_ref=bundle.model_ref,
        arch=bundle.arch,
        cache=updated_cache,
        model_config=bundle.model.config,
        codec=codec,
        audit_policy=audit_policy,
        token_ids=prompt_ids,
        parent_session_id=None if restored_entry is None else f"{_safe_key(restored_entry.model_id)}__{_safe_key(restored_entry.agent_id)}__v{restored_entry.version:04d}",
    )
    answer_text = str(result.get("answer_text") or "")
    answer_tokens = list(result.get("answer_token_ids") or [])
    handoff_summary = _truncate(answer_text if answer_text.strip() else f"generated_token_ids={answer_tokens[:12]}", 360)
    transcript.append(
        {
            "task_id": task["task_id"],
            "agent_id": task["agent_id"],
            "model_id": bundle.model_id,
            "role": task["capability"],
            "handoff_summary": handoff_summary,
            "session_version": entry.version,
        }
    )
    event = {
        "task_id": task["task_id"],
        "agent_id": task["agent_id"],
        "model_id": bundle.model_id,
        "model_ref": bundle.model_ref,
        "arch": bundle.arch,
        "capability": task["capability"],
        "restored_session": restored_entry is not None,
        "expects_restore": bool(task.get("expects_restore", False)),
        "session_hash_before": session_hash_before,
        "session_hash_loaded": load.get("session_hash_loaded"),
        "restore_hash_match": None if restored_entry is None else session_hash_before == load.get("session_hash_loaded"),
        "session_hash_after": entry.session_hash,
        "session_version_after": entry.version,
        "swap_event": swap_event,
        "load_time_ms": load["load_time_ms"],
        "generation_time_ms": generation_time_ms,
        "save_time_ms": save["save_time_ms"],
        "session_total_bytes": save["session_total_bytes"],
        "audit_status": save["receipt"].get("audit_status"),
        "restore_from_pending": str(audit_policy) == "deferred" and restored_entry is not None,
        "route_decision": None if route_decision is None else route_decision.to_dict(),
        "prompt_token_count": len(prompt_ids),
        "generated_token_ids": answer_tokens,
        "answer_text": answer_text,
        "answer_preview": _truncate(answer_text, 240),
        "handoff_out": handoff_summary,
        "shared_transcript_hash": _transcript_hash(transcript),
    }
    del updated_cache
    del cache
    gc.collect()
    return event


def run_multimodel_hypervisor(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_model_id, writer_model_id, skip_reason = _resolve_model_pair(args)
    registry = _registry_public(bool(args.local_files_only))
    artifact_name = str(args.artifact_name)
    if skip_reason is not None or code_model_id is None or writer_model_id is None:
        payload = {
            "title": "HeliX Local Multimodel Hypervisor v0",
            "demo": "heterogeneous-multi-model",
            "scenario": str(args.scenario),
            "status": "skipped",
            "skip_reason": skip_reason,
            "registry": registry,
            "requested_models": {"code_model": args.code_model, "writer_model": args.writer_model, "hybrid_model": args.hybrid_model},
        }
        _write_json(output_dir / artifact_name, payload)
        return payload
    device = torch.device(args.device)
    transcript: list[dict[str, Any]] = []
    task_events: list[dict[str, Any]] = []
    start_wall = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="helix-multimodel-hypervisor-") as temp:
        lifecycle = ModelLifecycle(device=device, local_files_only=bool(args.local_files_only))
        catalog = SessionCatalog.open(Path(temp) / "sessions" / "session-catalog.sqlite")
        store = SessionStore(Path(temp) / "sessions", catalog=catalog)
        try:
            for task in _build_tasks(
                scenario=str(args.scenario),
                code_model_id=code_model_id,
                writer_model_id=writer_model_id,
            ):
                task_events.append(
                    _run_task(
                        task=task,
                        lifecycle=lifecycle,
                        store=store,
                        codec=str(args.codec),
                        audit_policy=str(args.audit_policy),
                        max_new_tokens=int(args.max_new_tokens),
                        prompt_tokens=int(args.prompt_tokens),
                        transcript=transcript,
                    )
                )
            final_deactivate = lifecycle.deactivate()
            final_audits = store.verify_all(codec=str(args.codec), audit_policy=str(args.audit_policy))
            session_sizes = store.session_sizes()
            catalog_stats = catalog.stats()
            catalog_sessions = [session.to_dict() for session in catalog.list_sessions()]
            lifecycle_events = lifecycle.events + [final_deactivate] if lifecycle.events[-1:] != [final_deactivate] else lifecycle.events
        finally:
            if "lifecycle" in locals() and lifecycle.active is not None:
                lifecycle.deactivate()
            if "catalog" in locals():
                catalog.close()
    activation_events = [event for event in lifecycle_events if event.get("event") == "model_activate"]
    restored_events = [event for event in task_events if event.get("restored_session")]
    models_used = []
    for event in task_events:
        if event["model_id"] not in models_used:
            models_used.append(event["model_id"])
    payload = {
        "title": "HeliX Local Multimodel Hypervisor v0",
        "demo": "heterogeneous-multi-model",
        "scenario": str(args.scenario),
        "status": "completed",
        "profile": str(args.profile),
        "codec": str(args.codec),
        "audit_policy": str(args.audit_policy),
        "local_files_only": bool(args.local_files_only),
        "device": str(args.device),
        "models_used": models_used,
        "model_refs_used": {model_id: MODEL_REGISTRY[model_id]["ref"] for model_id in models_used},
        "model_arches_used": {model_id: MODEL_REGISTRY[model_id]["arch"] for model_id in models_used},
        "agents": sorted({str(event["agent_id"]) for event in task_events}),
        "model_swaps": len(activation_events),
        "swap_times_ms": [event.get("swap_time_ms") for event in activation_events],
        "lifecycle_events": lifecycle_events,
        "task_events": task_events,
        "session_sizes_bytes": session_sizes,
        "session_catalog": {"stats": catalog_stats, "sessions": catalog_sessions},
        "final_audits": final_audits,
        "shared_transcript": transcript,
        "shared_transcript_hash": _transcript_hash(transcript),
        "total_wall_time_s": time.perf_counter() - start_wall,
        "coordination_mode": "external-message-handoff",
        "state_sharing": "none; sessions are keyed by (model_id, agent_id) and cannot be restored across model ids",
        "claim_boundary": (
            "Pending receipts mean the checkpoint was written and accepted for fast restore. "
            "Cryptographic integrity is claimed only for final audit_status=verified receipts."
        ),
        "all_restore_hash_matches": all(event.get("restore_hash_match") is True for event in restored_events) if restored_events else None,
        "all_pending_receipts_loaded": all(event.get("restore_hash_match") is True for event in restored_events) if restored_events else None,
        "all_final_audits_verified": all(audit.get("ok") is True and audit.get("audit_status") == "verified" for audit in final_audits),
        "restored_session_count": len(restored_events),
        "registry": registry,
    }
    _write_json(output_dir / artifact_name, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local heterogeneous multi-model session hypervisor demo.")
    parser.add_argument("--scenario", default="coder-writer", choices=["coder-writer", "qwen-zamba-handoff"])
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--code-model", default="qwen-1.5b", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--writer-model", default="gpt2-fast", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--hybrid-model", default="zamba2-1.2b", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--strict-models", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--codec", default="rust-hlx-buffered-flat", choices=["rust-hlx", "rust-hlx-buffered", "rust-hlx-buffered-flat", "python-npz"])
    parser.add_argument("--audit-policy", default="deferred", choices=["blocking", "deferred"])
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--artifact-name", default="local-multimodel-hypervisor-demo.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_multimodel_hypervisor(args)
    print(json.dumps(_json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
