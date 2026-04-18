from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

torch.set_num_threads(max(1, min(2, int(torch.get_num_threads()))))

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helix_kv.transformers_cache import (  # noqa: E402
    _DEFAULT_MAMBA_STATE_BLOCK_SIZE,
    _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT,
    _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD,
    _DEFAULT_MAMBA_STATE_SCALE_FLOOR,
    _build_prompt_attention_mask,
    _device_for_benchmark,
    _is_zamba2_hybrid_model_config,
    _load_benchmark_cache,
    _load_causal_model,
    _load_text_adapter,
    _quantizer_max_iter_for_variant,
    _save_benchmark_cache,
    _serialize_transformers_cache_bytes,
    _session_snapshot_receipt,
    _supports_forward_arg,
    _sync_device,
    TransformersCompressedKVCache,
    TransformersHybridKVCache,
    build_transformers_hybrid_state_variants,
)


FIXTURE_DIR = REPO_ROOT / "benchmarks" / "hybrid_stress"
LONG_CONTEXT_FIXTURE = FIXTURE_DIR / "long_context_coder.py"
STATE_JUGGLER_FIXTURE = FIXTURE_DIR / "state_juggler_log.txt"
CONTEXT_SWITCHER_FIXTURE = FIXTURE_DIR / "context_switcher_mix.txt"

MISSION_ARTIFACT_NAMES = {
    "long-context-coder": "local-zamba2-stress-long-context.json",
    "state-juggler": "local-zamba2-stress-state-juggler.json",
    "context-switcher": "local-zamba2-stress-context-switcher.json",
    "restore-equivalence": "local-zamba2-stress-restore-equivalence.json",
}

PUBLIC_LOCAL_PROFILES: dict[str, dict[str, Any]] = {
    "laptop-12gb": {
        "model_ref": "Zyphra/Zamba2-1.2B-Instruct-v2",
        "kv_hot_window": 4,
        "kv_quant_seed": 7,
        "mamba_state_block_size": 64,
        "mamba_state_scale_floor": 1e-8,
        "mamba_state_clip_threshold_pct": 2.0,
        "mamba_state_rel_rmse_threshold": 0.2,
        "mamba_state_auto_promote": False,
        "mission1_prompt_tokens_staged": 640,
        "mission1_prompt_tokens_heavy": 3072,
        "mission1_max_new_tokens": 6,
        "mission2_prompt_tokens": 768,
        "mission2_max_new_tokens": 18,
        "mission3_prompt_tokens": 1024,
        "mission3_max_new_tokens": 24,
        "mission3_clip_threshold_pct": 0.0,
        "mission3_rel_rmse_threshold": 0.08,
        "mission3_auto_promote": True,
        "mission4_prompt_tokens": 512,
        "mission4_probe_tokens": 96,
        "mission4_max_new_tokens": 4,
    }
}

LONG_CONTEXT_TASKS = [
    {
        "task_id": "bug-localization",
        "title": "Long-Context Coder / Bug localization",
        "focus_symbol": "merge_runtime_rows",
        "expected_identifiers": [
            "PROJECT_SENTINEL",
            "BOOT_CHANNELS",
            "project_boot_digest",
            "resume_runtime_session",
        ],
        "question": (
            "Review the Python module below. Find the most likely subtle bug inside merge_runtime_rows, "
            "explain the fix briefly, and name two symbols defined near the top of the file that would also be affected."
        ),
    },
    {
        "task_id": "async-refactor",
        "title": "Long-Context Coder / Async refactor",
        "focus_symbol": "load_remote_manifest",
        "expected_identifiers": [
            "emit_receipt_checkpoint",
            "resume_runtime_session",
            "BOOT_CHANNELS",
        ],
        "question": (
            "Refactor the remote manifest and receipt upload path so it becomes asynchronous. "
            "Focus on load_remote_manifest and upload_receipts_batch, and mention one helper defined near the top "
            "that should still be reused."
        ),
    },
]

STATE_JUGGLER_PROMPTS = {
    "question_one": (
        "Question: Which subsystem first reported the fault, and which process was running when the lockup started?\n"
        "Answer:"
    ),
    "question_two": (
        "\nFollow-up question: After restore, tell me which CPU core handled the panic and which device was involved.\n"
        "Answer:"
    ),
}

STATE_JUGGLER_EXPECTED = {
    "question_one": ["nvme", "backup-agent"],
    "question_two": ["cpu#3", "nvme0n1"],
}

CONTEXT_SWITCHER_PROMPT = (
    "Extrae el valor closing_price_usd del ticker HELIX y explicalo con el mismo tono del ensayo en español. "
    "No inventes cifras. Responde en un solo parrafo.\n\nRespuesta:"
)
CONTEXT_SWITCHER_EXPECTED_VALUE = "142.75"

RESTORE_EQUIVALENCE_PROMPT = (
    "\nRestore-equivalence probe: continue deterministically by naming the failed subsystem, process, CPU core, "
    "and device from the prior log. Keep the answer short.\nAnswer:"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _truncate_text(text: str, limit: int = 420) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= int(limit):
        return clean
    return clean[: max(int(limit) - 3, 0)] + "..."


def _decode_ids(adapter: Any | None, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
    if adapter is None or not token_ids:
        return ""
    try:
        return str(adapter.decode(list(token_ids), skip_special_tokens=skip_special_tokens))
    except Exception:
        return ""


def _encode_prompt_text(
    *,
    model_ref: str,
    adapter: Any,
    input_adapter: str,
    prompt_text: str,
    max_tokens: int | None,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    if input_adapter == "processor-text":
        prompt_inputs = adapter.apply_chat_template(
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = prompt_inputs["input_ids"][0].tolist()
        if max_tokens is not None:
            input_ids = input_ids[: int(max_tokens)]
            prompt_inputs = {
                key: value[..., : int(max_tokens)] if isinstance(value, torch.Tensor) else value
                for key, value in prompt_inputs.items()
            }
        return {
            "input_ids": prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
        }, [int(token_id) for token_id in input_ids]

    token_ids = adapter.encode(prompt_text, add_special_tokens=False)
    if max_tokens is not None:
        token_ids = token_ids[: int(max_tokens)]
    if not token_ids:
        raise ValueError(f"empty encoded prompt for {model_ref}")
    return {
        "input_ids": torch.tensor([token_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
    }, [int(token_id) for token_id in token_ids]


def _build_cache_for_variant(model: Any, variant: dict[str, Any], *, device: torch.device, kv_backend: str = "torch") -> tuple[Any, str]:
    cache_precision = str(variant.get("kv_cache_precision", "native-dense"))
    common_kwargs = {
        "kv_cache_precision": cache_precision,
        "kv_key_precision": variant.get("kv_key_precision"),
        "kv_value_precision": variant.get("kv_value_precision"),
        "kv_key_scaling_strategy": variant.get("kv_key_scaling_strategy"),
        "kv_value_scaling_strategy": variant.get("kv_value_scaling_strategy"),
        "kv_rotation_mode": str(variant.get("kv_rotation_mode", "hadamard")),
        "kv_hot_window": int(variant.get("kv_hot_window", 0)),
        "kv_quant_seed": int(variant.get("kv_quant_seed", 7)),
        "kv_calibration_tokens": int(variant.get("kv_calibration_tokens", 128)),
        "kv_adaptive_high_kurtosis": float(variant.get("kv_adaptive_high_kurtosis", 10.0)),
        "kv_adaptive_medium_kurtosis": float(variant.get("kv_adaptive_medium_kurtosis", 3.0)),
        "protected_layer_indices": variant.get("protected_layer_indices"),
        "kv_key_fourbit_max_iter": _quantizer_max_iter_for_variant(variant, "key"),
        "kv_value_fourbit_max_iter": _quantizer_max_iter_for_variant(variant, "value"),
        "kv_backend": str(kv_backend),
        "kv_async_compression": False,
    }
    if _is_zamba2_hybrid_model_config(model.config):
        cache = TransformersHybridKVCache(
            model.config,
            batch_size=1,
            dtype=next(model.parameters()).dtype,
            device=device,
            mamba_state_precision=str(variant.get("mamba_state_precision", "native-dense")),
            mamba_state_block_size=int(variant.get("mamba_state_block_size", _DEFAULT_MAMBA_STATE_BLOCK_SIZE)),
            mamba_state_scale_floor=float(variant.get("mamba_state_scale_floor", _DEFAULT_MAMBA_STATE_SCALE_FLOOR)),
            mamba_state_clip_threshold_pct=float(
                variant.get("mamba_state_clip_threshold_pct", _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT)
            ),
            mamba_state_rel_rmse_threshold=float(
                variant.get("mamba_state_rel_rmse_threshold", _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD)
            ),
            mamba_state_auto_promote=bool(variant.get("mamba_state_auto_promote", True)),
            mamba_receipts_enabled=bool(variant.get("mamba_receipts_enabled", False)),
            mamba_receipts_path=variant.get("mamba_receipts_path"),
            mamba_receipt_run_id=variant.get("mamba_receipt_run_id"),
            **common_kwargs,
        )
        return cache, str(variant["name"])
    if cache_precision == "native-dense":
        return DynamicCache(config=model.config), "native-dense"
    cache = TransformersCompressedKVCache(model.config, **common_kwargs)
    return cache, str(variant["name"])


def _hybrid_cache_runtime_bytes(cache: Any) -> int | None:
    if isinstance(cache, TransformersHybridKVCache):
        return int(cache.hybrid_total_runtime_cache_bytes)
    return None


def _hybrid_cache_total_bytes(cache: Any) -> int | None:
    if isinstance(cache, TransformersHybridKVCache):
        return int(cache.hybrid_total_cache_bytes)
    return None


def _hybrid_fallback_counts(cache: Any) -> dict[str, int] | None:
    if isinstance(cache, TransformersHybridKVCache):
        return dict(cache.mamba_state_fallback_counts)
    return None


def _run_generation_trace(
    model: Any,
    *,
    prompt_inputs: dict[str, torch.Tensor],
    prompt_ids: list[int],
    variant: dict[str, Any],
    max_new_tokens: int,
    adapter: Any | None,
    device: torch.device,
    initial_cache: Any | None = None,
    save_session_path: Path | None = None,
    resume_prompt_token_by_token: bool = False,
    capture_selection_logits: bool = False,
    return_cache: bool = False,
) -> dict[str, Any]:
    supports_attention_mask = _supports_forward_arg(model, "attention_mask")
    supports_cache_position = _supports_forward_arg(model, "cache_position")
    run_input_ids = prompt_inputs["input_ids"].to(device=device)
    prompt_inputs = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in prompt_inputs.items()
    }

    if initial_cache is None:
        cache, current_mode = _build_cache_for_variant(model, variant, device=device)
        attention_mask = prompt_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = _build_prompt_attention_mask(device, run_input_ids.shape[1])
        model_inputs: dict[str, Any] = {"input_ids": run_input_ids, "past_key_values": cache, "use_cache": True, "return_dict": True}
        if supports_attention_mask:
            model_inputs["attention_mask"] = attention_mask
        if supports_cache_position:
            model_inputs["cache_position"] = torch.arange(run_input_ids.shape[1], device=device)
    else:
        cache = initial_cache
        current_mode = str(variant["name"])
        if isinstance(cache, TransformersHybridKVCache):
            cache.mamba_state_precision = str(variant.get("mamba_state_precision", cache.mamba_state_precision))
            cache.mamba_state_block_size = int(variant.get("mamba_state_block_size", cache.mamba_state_block_size))
            cache.mamba_state_scale_floor = float(variant.get("mamba_state_scale_floor", cache.mamba_state_scale_floor))
            cache.mamba_state_clip_threshold_pct = float(
                variant.get("mamba_state_clip_threshold_pct", cache.mamba_state_clip_threshold_pct)
            )
            cache.mamba_state_rel_rmse_threshold = float(
                variant.get("mamba_state_rel_rmse_threshold", cache.mamba_state_rel_rmse_threshold)
            )
            cache.mamba_state_auto_promote = bool(
                variant.get("mamba_state_auto_promote", cache.mamba_state_auto_promote)
            )
            cache.mamba_receipts_enabled = bool(variant.get("mamba_receipts_enabled", cache.mamba_receipts_enabled))
            receipts_path = variant.get("mamba_receipts_path", cache.mamba_receipts_path)
            cache.mamba_receipts_path = None if receipts_path is None else Path(receipts_path)
            cache.mamba_receipt_run_id = variant.get("mamba_receipt_run_id", cache.mamba_receipt_run_id)
            cache._mamba_receipt_prev_hash = "0" * 64
            cache._mamba_receipt_step_index = 0
            cache._mamba_receipt_count = 0
        base_seq_length = int(cache.get_seq_length())
        attention_mask = torch.ones((1, base_seq_length), dtype=torch.long, device=device)
        model_inputs = None
        if not resume_prompt_token_by_token:
            attention_mask = torch.ones((1, base_seq_length + run_input_ids.shape[1]), dtype=torch.long, device=device)
            model_inputs = {"input_ids": run_input_ids, "past_key_values": cache, "use_cache": True, "return_dict": True}
            if supports_attention_mask:
                model_inputs["attention_mask"] = attention_mask
            if supports_cache_position:
                model_inputs["cache_position"] = torch.arange(
                    base_seq_length, base_seq_length + run_input_ids.shape[1], device=device
                )

    if isinstance(cache, TransformersHybridKVCache):
        cache.materialize_mamba_state_runtime()

    with torch.inference_mode():
        _sync_device(device)
        total_start = time.perf_counter()
        if resume_prompt_token_by_token and initial_cache is not None:
            prompt_loss = torch.tensor(0.0, device=device)
            last_logits = None
            for token_index in range(run_input_ids.shape[1]):
                current_token = run_input_ids[:, token_index : token_index + 1]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                    dim=1,
                )
                if isinstance(cache, TransformersHybridKVCache):
                    cache.materialize_mamba_state_runtime()
                resume_inputs: dict[str, Any] = {
                    "input_ids": current_token,
                    "past_key_values": cache,
                    "use_cache": True,
                    "return_dict": True,
                }
                if supports_attention_mask:
                    resume_inputs["attention_mask"] = attention_mask
                if supports_cache_position:
                    resume_inputs["cache_position"] = torch.arange(
                        cache.get_seq_length(), cache.get_seq_length() + 1, device=device
                    )
                outputs = model(**resume_inputs)
                _sync_device(device)
                cache = getattr(outputs, "past_key_values", cache)
                if isinstance(cache, TransformersHybridKVCache):
                    cache.compress_mamba_state_runtime()
                last_logits = outputs.logits[:, -1, :]
            if last_logits is None:
                raise ValueError("resume_prompt_token_by_token produced no logits")
        else:
            outputs = model(**model_inputs)
            _sync_device(device)
            cache = getattr(outputs, "past_key_values", cache)
            if isinstance(cache, TransformersHybridKVCache):
                cache.compress_mamba_state_runtime()
            last_logits = outputs.logits[:, -1, :]
            prompt_logits = outputs.logits[:, :-1, :]
            prompt_labels = run_input_ids[:, 1:]
            prompt_loss = (
                F.cross_entropy(
                    prompt_logits.reshape(-1, prompt_logits.shape[-1]),
                    prompt_labels.reshape(-1),
                    reduction="mean",
                )
                if prompt_labels.numel() > 0
                else torch.tensor(0.0, device=device)
            )

        answer_token_ids: list[int] = []
        generated_ids = list(prompt_ids)
        step_times_ms: list[float] = []
        step_trace: list[dict[str, Any]] = []
        selection_logits: list[list[float]] = []
        for step_index in range(int(max_new_tokens)):
            if capture_selection_logits:
                selection_logits.append(last_logits.detach().to(dtype=torch.float32, device="cpu")[0].tolist())
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            answer_token_ids.append(int(next_token.item()))
            generated_ids.append(int(next_token.item()))
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            next_inputs: dict[str, Any] = {
                "input_ids": next_token,
                "past_key_values": cache,
                "use_cache": True,
                "return_dict": True,
            }
            if isinstance(cache, TransformersHybridKVCache):
                cache.materialize_mamba_state_runtime()
                next_inputs["past_key_values"] = cache
            if supports_attention_mask:
                next_inputs["attention_mask"] = attention_mask
            if supports_cache_position:
                next_inputs["cache_position"] = torch.arange(
                    cache.get_seq_length(), cache.get_seq_length() + 1, device=device
                )
            _sync_device(device)
            step_start = time.perf_counter()
            outputs = model(**next_inputs)
            _sync_device(device)
            cache = getattr(outputs, "past_key_values", cache)
            if isinstance(cache, TransformersHybridKVCache):
                cache.compress_mamba_state_runtime()
            last_logits = outputs.logits[:, -1, :]
            step_ms = (time.perf_counter() - step_start) * 1000.0
            step_times_ms.append(step_ms)
            step_trace.append(
                {
                    "step_index": int(step_index),
                    "token_id": int(next_token.item()),
                    "step_time_ms": float(step_ms),
                    "cache_seq_length": int(cache.get_seq_length()),
                    "hybrid_total_runtime_cache_bytes": _hybrid_cache_runtime_bytes(cache),
                    "mamba_state_fallback_counts": _hybrid_fallback_counts(cache),
                }
            )
        total_time_s = time.perf_counter() - total_start

    session_bytes = _serialize_transformers_cache_bytes(cache)
    session_receipt = None
    session_save_time_ms = None
    if save_session_path is not None:
        destination = Path(save_session_path)
        if destination.exists():
            shutil.rmtree(destination)
        save_start = time.perf_counter()
        _save_benchmark_cache(cache, model_config=model.config, path=destination)
        session_save_time_ms = (time.perf_counter() - save_start) * 1000.0
        session_receipt = _session_snapshot_receipt(destination)
        session_receipt["save_time_ms"] = float(session_save_time_ms)

    answer_text = _decode_ids(adapter, answer_token_ids, skip_special_tokens=True)
    result = {
        "variant_name": str(variant["name"]),
        "prompt_token_count": int(len(prompt_ids)),
        "generation_length": int(len(answer_token_ids)),
        "generated_ids": generated_ids,
        "answer_token_ids": answer_token_ids,
        "answer_text": answer_text,
        "answer_preview": _truncate_text(answer_text),
        "prompt_perplexity": float(np.exp(float(prompt_loss.item()))),
        "total_time_s": float(total_time_s),
        "avg_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
          "step_times_ms": [float(item) for item in step_times_ms],
          "step_trace": step_trace,
          "selection_logits": selection_logits if capture_selection_logits else None,
          "current_mode": current_mode,
        "session_total_bytes": int(session_bytes["session_total_bytes"]),
        "session_meta_bytes": int(session_bytes["session_meta_bytes"]),
        "session_npz_bytes": int(session_bytes["session_npz_bytes"]),
        "session_save_time_ms": None if session_save_time_ms is None else float(session_save_time_ms),
        "session_receipt": session_receipt,
        "hybrid_total_runtime_cache_bytes": _hybrid_cache_runtime_bytes(cache),
        "hybrid_total_cache_bytes": _hybrid_cache_total_bytes(cache),
        "mamba_state_runtime_ratio_vs_native": None
        if not isinstance(cache, TransformersHybridKVCache)
        else float(cache.mamba_state_runtime_ratio_vs_native),
        "mamba_state_fallback_counts": _hybrid_fallback_counts(cache),
        "mamba_receipt_count": None if not isinstance(cache, TransformersHybridKVCache) else int(cache.mamba_receipt_count),
        "logits_finite": bool(torch.isfinite(last_logits).all().item()),
    }
    if return_cache:
        result["_cache"] = cache
    return result


def _load_receipts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _receipt_summary(path: Path) -> dict[str, Any]:
    receipts = _load_receipts(path)
    if not receipts:
        return {
            "path": str(path),
            "receipt_count": 0,
            "run_ids": [],
            "hash_chain_ok": True,
            "step_windows": [],
        }
    run_ids = sorted({str(item["run_id"]) for item in receipts})
    hash_chain_ok = True
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in receipts:
        by_run[str(item["run_id"])].append(item)
    for items in by_run.values():
        prev_hash = "0" * 64
        for entry in items:
            if str(entry.get("prev_hash")) != prev_hash:
                hash_chain_ok = False
                break
            prev_hash = str(entry.get("receipt_hash"))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in receipts:
        grouped[int(entry.get("step_index", 0))].append(entry)
    step_windows: list[dict[str, Any]] = []
    for step_index in sorted(grouped):
        entries = grouped[step_index]
        total_dense = sum(int(item.get("dense_bytes", 0)) for item in entries)
        total_compressed = sum(int(item.get("compressed_bytes", 0)) for item in entries)
        fallback_counts = Counter(str(item.get("fallback_precision", "unknown")) for item in entries)
        step_windows.append(
            {
                "step_index": int(step_index),
                "receipt_count": len(entries),
                "ratio": float(total_dense / total_compressed) if total_compressed > 0 else 1.0,
                "max_clip_pct": float(max(float(item.get("clip_pct", 0.0)) for item in entries)),
                "max_rel_rmse": float(max(float(item.get("rel_rmse", 0.0)) for item in entries)),
                "promoted_block_count": int(sum(int(item.get("promoted_block_count", 0)) for item in entries)),
                "block_count": int(sum(int(item.get("block_count", 0)) for item in entries)),
                "int4_block_count": int(sum(int(item.get("int4_block_count", 0)) for item in entries)),
                "int8_block_count": int(sum(int(item.get("int8_block_count", 0)) for item in entries)),
                "dense_block_count": int(sum(int(item.get("dense_block_count", 0)) for item in entries)),
                "max_abs_value": float(max(float(item.get("max_abs_value", 0.0)) for item in entries)),
                "max_state_norm": float(max(float(item.get("state_norm", 0.0)) for item in entries)),
                "fallback_counts": dict(fallback_counts),
            }
        )
    return {
        "path": str(path),
        "receipt_count": len(receipts),
        "run_ids": run_ids,
        "hash_chain_ok": hash_chain_ok,
        "step_windows": step_windows,
        "max_clip_pct": float(max(float(item.get("clip_pct", 0.0)) for item in receipts)),
        "max_rel_rmse": float(max(float(item.get("rel_rmse", 0.0)) for item in receipts)),
        "promoted_block_total": int(sum(int(item.get("promoted_block_count", 0)) for item in receipts)),
    }


def _variant_result_compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_name": result["variant_name"],
        "prompt_token_count": result["prompt_token_count"],
        "generation_length": result["generation_length"],
        "total_time_s": result["total_time_s"],
        "avg_step_ms": result["avg_step_ms"],
        "prompt_perplexity": result["prompt_perplexity"],
        "session_total_bytes": result["session_total_bytes"],
        "session_save_time_ms": result.get("session_save_time_ms"),
        "hybrid_total_runtime_cache_bytes": result.get("hybrid_total_runtime_cache_bytes"),
        "hybrid_total_cache_bytes": result.get("hybrid_total_cache_bytes"),
        "mamba_state_runtime_ratio_vs_native": result.get("mamba_state_runtime_ratio_vs_native"),
        "mamba_state_fallback_counts": result.get("mamba_state_fallback_counts"),
        "mamba_receipt_count": result.get("mamba_receipt_count"),
        "answer_text": result.get("answer_text"),
        "answer_preview": result.get("answer_preview"),
        "step_trace": result.get("step_trace"),
        "step_times_ms": result.get("step_times_ms"),
        "logits_finite": result.get("logits_finite"),
    }


def _logit_arrays(result: dict[str, Any]) -> list[np.ndarray]:
    return [np.asarray(item, dtype=np.float32) for item in (result.get("selection_logits") or [])]


def _compare_restore_equivalence(pre_result: dict[str, Any], post_result: dict[str, Any]) -> dict[str, Any]:
    pre_ids = [int(item) for item in pre_result.get("answer_token_ids") or []]
    post_ids = [int(item) for item in post_result.get("answer_token_ids") or []]
    pre_logits = _logit_arrays(pre_result)
    post_logits = _logit_arrays(post_result)
    comparable = min(len(pre_logits), len(post_logits))
    deltas: list[np.ndarray] = []
    pre_top1: list[int] = []
    post_top1: list[int] = []
    finite_pre = bool(pre_result.get("logits_finite", False))
    finite_post = bool(post_result.get("logits_finite", False))
    for index in range(comparable):
        if pre_logits[index].shape != post_logits[index].shape:
            continue
        finite_pre = finite_pre and bool(np.isfinite(pre_logits[index]).all())
        finite_post = finite_post and bool(np.isfinite(post_logits[index]).all())
        pre_top1.append(int(np.argmax(pre_logits[index])))
        post_top1.append(int(np.argmax(post_logits[index])))
        deltas.append(np.abs(pre_logits[index] - post_logits[index]).astype(np.float32))
    if deltas:
        flattened_delta = np.concatenate([item.reshape(-1) for item in deltas])
        pre_flat = np.concatenate([item.reshape(-1) for item in pre_logits[:comparable]]).astype(np.float32)
        post_flat = np.concatenate([item.reshape(-1) for item in post_logits[:comparable]]).astype(np.float32)
        denom = float(np.linalg.norm(pre_flat) * np.linalg.norm(post_flat))
        cosine = float(np.dot(pre_flat, post_flat) / denom) if denom > 0.0 else None
        max_abs = float(np.max(flattened_delta))
        mean_abs = float(np.mean(flattened_delta))
    else:
        cosine = None
        max_abs = None
        mean_abs = None
    return {
        "generated_ids_match": bool(pre_ids == post_ids),
        "top1_match_all": bool(pre_top1 == post_top1 and len(pre_top1) == comparable),
        "pre_answer_token_ids": pre_ids,
        "post_answer_token_ids": post_ids,
        "pre_top1_ids": pre_top1,
        "post_top1_ids": post_top1,
        "logit_step_count": int(comparable),
        "max_abs_logit_delta": max_abs,
        "mean_abs_logit_delta": mean_abs,
        "cosine_similarity": cosine,
        "finite_before": finite_pre,
        "finite_after": finite_post,
    }


def _speedup(native_result: dict[str, Any], combined_result: dict[str, Any]) -> float | None:
    baseline = float(native_result["total_time_s"])
    target = float(combined_result["total_time_s"])
    return float(baseline / target) if target > 0 else None


def _runtime_ratio(native_result: dict[str, Any], combined_result: dict[str, Any]) -> float | None:
    baseline = native_result.get("hybrid_total_runtime_cache_bytes") or native_result.get("hybrid_total_cache_bytes")
    target = combined_result.get("hybrid_total_runtime_cache_bytes") or combined_result.get("hybrid_total_cache_bytes")
    if not baseline or not target:
        return None
    return float(baseline) / float(target)


def _identifier_hits(answer_text: str, identifiers: list[str]) -> list[str]:
    lowered = answer_text.lower()
    return [identifier for identifier in identifiers if identifier.lower() in lowered]


def _fixture_excerpt(path: Path, *, focus_symbol: str, tier: str) -> str:
    text = _read_text(path)
    if tier == "heavy":
        return text
    lines = text.splitlines()
    selected: set[int] = set(range(min(len(lines), 72)))
    focus_symbols = [focus_symbol]
    if focus_symbol == "load_remote_manifest":
        focus_symbols.append("upload_receipts_batch")
    for symbol in focus_symbols:
        for index, line in enumerate(lines):
            if symbol in line:
                start = max(index - 26, 0)
                end = min(index + 42, len(lines))
                selected.update(range(start, end))
                break
    selected.update(range(max(len(lines) - 18, 0), len(lines)))
    chunks: list[str] = []
    previous = -2
    for index in sorted(selected):
        if index != previous + 1:
            chunks.append("...")
        chunks.append(f"{index + 1:04d}: {lines[index]}")
        previous = index
    return "\n".join(chunks)


def _build_prompt_with_fixture(path: Path, question: str, *, focus_symbol: str = "", tier: str = "staged") -> str:
    fixture_text = _fixture_excerpt(path, focus_symbol=focus_symbol, tier=tier)
    return (
        f"### TASK\n{question}\n\n"
        f"### FIXTURE EXCERPT: {path.name}\n"
        f"{fixture_text}\n\n"
        "### ANSWER\n"
    )


def _load_model_stack(model_ref: str, *, local_files_only: bool, trust_remote_code: bool, device: torch.device) -> tuple[Any, Any | None, str]:
    model = _load_causal_model(model_ref, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    adapter, input_adapter, _, _ = _load_text_adapter(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    return model, adapter, input_adapter


def _pick_variant(
    profile: dict[str, Any],
    *,
    name: str,
    receipts_path: Path | None = None,
    run_id: str | None = None,
    clip_threshold_pct: float | None = None,
    rel_rmse_threshold: float | None = None,
    auto_promote: bool | None = None,
) -> dict[str, Any]:
    variants = build_transformers_hybrid_state_variants(
        kv_quant_seed=int(profile["kv_quant_seed"]),
        kv_hot_window=int(profile["kv_hot_window"]),
        mamba_state_block_size=int(profile["mamba_state_block_size"]),
        mamba_state_scale_floor=float(profile["mamba_state_scale_floor"]),
        mamba_state_clip_threshold_pct=float(
            profile["mamba_state_clip_threshold_pct"] if clip_threshold_pct is None else clip_threshold_pct
        ),
        mamba_state_rel_rmse_threshold=float(
            profile["mamba_state_rel_rmse_threshold"] if rel_rmse_threshold is None else rel_rmse_threshold
        ),
        mamba_state_auto_promote=bool(
            profile["mamba_state_auto_promote"] if auto_promote is None else auto_promote
        ),
        mamba_receipts_enabled=receipts_path is not None,
        mamba_receipts_path=receipts_path,
        mamba_receipt_run_id=run_id,
    )
    for variant in variants:
        if str(variant["name"]) == str(name):
            return dict(variant)
    raise KeyError(name)


def _mission_result_header(*, mission_id: str, title: str, profile_name: str, model_ref: str, fixture_path: Path) -> dict[str, Any]:
    return {
        "mission_id": mission_id,
        "title": title,
        "profile": profile_name,
        "model_ref": model_ref,
        "fixture_path": str(fixture_path),
        "fixture_line_count": _line_count(fixture_path),
        "status": "completed",
    }


def run_long_context_coder(
    args: argparse.Namespace,
    *,
    profile_name: str,
    profile: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    tier = str(args.mission1_tier)
    model_ref = str(profile["model_ref"])
    device = _device_for_benchmark(args.device)
    prompt_budget = int(profile["mission1_prompt_tokens_staged"] if tier == "staged" else profile["mission1_prompt_tokens_heavy"])
    max_new_tokens = int(profile["mission1_max_new_tokens"])
    model, adapter, input_adapter = _load_model_stack(
        model_ref,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        device=device,
    )

    result = _mission_result_header(
        mission_id="long-context-coder",
        title="Long-Context Coder",
        profile_name=profile_name,
        model_ref=model_ref,
        fixture_path=LONG_CONTEXT_FIXTURE,
    )
    result["tier"] = tier
    tasks_payload: list[dict[str, Any]] = []
    claim_parts: list[str] = []
    for task in LONG_CONTEXT_TASKS:
        prompt_text = _build_prompt_with_fixture(
            LONG_CONTEXT_FIXTURE,
            task["question"],
            focus_symbol=str(task["focus_symbol"]),
            tier=tier,
        )
        prompt_inputs, prompt_ids = _encode_prompt_text(
            model_ref=model_ref,
            adapter=adapter,
            input_adapter=input_adapter,
            prompt_text=prompt_text,
            max_tokens=prompt_budget,
        )
        native_variant = _pick_variant(profile, name="native-dense")
        receipts_path = output_dir / f"local-zamba2-stress-long-context-{task['task_id']}.jsonl.gz"
        if receipts_path.exists():
            receipts_path.unlink()
        combined_variant = _pick_variant(
            profile,
            name="turbo-int8-hadamard+q-mamba-dsq-int4",
            receipts_path=receipts_path,
            run_id=f"stress-long-context-{task['task_id']}",
        )
        native_result = _run_generation_trace(
            model,
            prompt_inputs=prompt_inputs,
            prompt_ids=prompt_ids,
            variant=native_variant,
            max_new_tokens=max_new_tokens,
            adapter=adapter,
            device=device,
        )
        combined_result = _run_generation_trace(
            model,
            prompt_inputs=prompt_inputs,
            prompt_ids=prompt_ids,
            variant=combined_variant,
            max_new_tokens=max_new_tokens,
            adapter=adapter,
            device=device,
        )
        hits = _identifier_hits(str(combined_result["answer_text"]), list(task["expected_identifiers"]))
        receipt_summary = _receipt_summary(receipts_path)
        speedup = _speedup(native_result, combined_result)
        runtime_ratio = _runtime_ratio(native_result, combined_result)
        tasks_payload.append(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "focus_symbol": task["focus_symbol"],
                "prompt_token_count": int(len(prompt_ids)),
                "expected_identifiers": list(task["expected_identifiers"]),
                "identifier_hits": hits,
                "identifier_recall_ok": bool(hits),
                "speedup_vs_native": speedup,
                "hybrid_total_runtime_cache_ratio_vs_native": runtime_ratio,
                "native_dense": _variant_result_compact(native_result),
                "combined": _variant_result_compact(combined_result),
                "receipt_summary": receipt_summary,
            }
        )
        claim_parts.append(
            f"{task['task_id']} {runtime_ratio:.2f}x runtime-cache"
            if runtime_ratio is not None
            else f"{task['task_id']} completed"
        )

    result["tasks"] = tasks_payload
    identifier_passes = sum(1 for item in tasks_payload if item["identifier_recall_ok"])
    result["headline_metrics"] = {
        "identifier_recall_passes": identifier_passes,
        "task_count": len(tasks_payload),
        "best_runtime_ratio": max(
            (
                float(item["hybrid_total_runtime_cache_ratio_vs_native"])
                for item in tasks_payload
                if item["hybrid_total_runtime_cache_ratio_vs_native"] is not None
            ),
            default=1.0,
        ),
        "best_speedup_vs_native": max(
            (float(item["speedup_vs_native"]) for item in tasks_payload if item["speedup_vs_native"] is not None),
            default=0.0,
        ),
    }
    result["strongest_claim"] = (
        f"Staged long-context coder reached {result['headline_metrics']['best_runtime_ratio']:.2f}x runtime-cache reduction "
        f"while recovering early identifiers in {identifier_passes}/{len(tasks_payload)} tasks."
    )
    result["claims"] = claim_parts
    mission_path = _write_json(output_dir / "local-zamba2-stress-long-context.json", result)
    return result, mission_path


def _write_internal_phase(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _run_state_juggler_phase_a(args: argparse.Namespace, *, profile: dict[str, Any]) -> dict[str, Any]:
    model_ref = str(profile["model_ref"])
    device = _device_for_benchmark(args.device)
    model, adapter, input_adapter = _load_model_stack(
        model_ref,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        device=device,
    )
    prompt_text = _read_text(STATE_JUGGLER_FIXTURE) + "\n\n" + STATE_JUGGLER_PROMPTS["question_one"]
    prompt_inputs, prompt_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=prompt_text,
        max_tokens=int(profile["mission2_prompt_tokens"]),
    )
    receipts_path = Path(args.internal_receipts_path)
    if receipts_path.exists():
        receipts_path.unlink()
    variant = _pick_variant(
        profile,
        name="turbo-int8-hadamard+q-mamba-dsq-int4",
        receipts_path=receipts_path,
        run_id="stress-state-juggler-phase-a",
    )
    result = _run_generation_trace(
        model,
        prompt_inputs=prompt_inputs,
        prompt_ids=prompt_ids,
        variant=variant,
        max_new_tokens=int(profile["mission2_max_new_tokens"]),
        adapter=adapter,
        device=device,
        save_session_path=Path(args.internal_session_dir),
    )
    payload = {
        "phase": "phase-a",
        "model_ref": model_ref,
        "answer_text": result["answer_text"],
        "answer_preview": result["answer_preview"],
        "prompt_token_count": result["prompt_token_count"],
        "receipts_path": str(receipts_path),
        "receipt_summary": _receipt_summary(receipts_path),
        "session_receipt": {
            "mission_id": "state-juggler",
            "phase": "phase-a",
            **(result["session_receipt"] or {}),
            "load_time_ms": None,
            "hash_match": None,
        },
        "context_keywords_hit": [
            word for word in STATE_JUGGLER_EXPECTED["question_one"] if word.lower() in str(result["answer_text"]).lower()
        ],
    }
    _write_internal_phase(Path(args.internal_output_path), payload)
    return payload


def _run_state_juggler_phase_b(args: argparse.Namespace, *, profile: dict[str, Any]) -> dict[str, Any]:
    phase_a = json.loads(Path(args.internal_input_path).read_text(encoding="utf-8"))
    model_ref = str(profile["model_ref"])
    device = _device_for_benchmark(args.device)
    model, adapter, input_adapter = _load_model_stack(
        model_ref,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        device=device,
    )
    session_dir = Path(args.internal_session_dir)
    load_start = time.perf_counter()
    restored_cache = _load_benchmark_cache(session_dir, model_config=model.config, device=device)
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    restore_snapshot_dir = session_dir.parent / f"{session_dir.name}-restored"
    if restore_snapshot_dir.exists():
        shutil.rmtree(restore_snapshot_dir)
    _save_benchmark_cache(restored_cache, model_config=model.config, path=restore_snapshot_dir)
    restored_receipt = _session_snapshot_receipt(restore_snapshot_dir)
    restored_receipt["load_time_ms"] = float(load_time_ms)
    restored_receipt["hash_match"] = bool(restored_receipt["session_hash"] == phase_a["session_receipt"]["session_hash"])

    prompt_inputs, prompt_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=STATE_JUGGLER_PROMPTS["question_two"],
        max_tokens=None,
    )
    receipts_path = Path(args.internal_receipts_path)
    if receipts_path.exists():
        receipts_path.unlink()
    variant = _pick_variant(
        profile,
        name="turbo-int8-hadamard+q-mamba-dsq-int4",
        receipts_path=receipts_path,
        run_id="stress-state-juggler-phase-b",
    )
    result = _run_generation_trace(
        model,
        prompt_inputs=prompt_inputs,
        prompt_ids=prompt_ids,
        variant=variant,
        max_new_tokens=int(profile["mission2_max_new_tokens"]),
        adapter=adapter,
        device=device,
        initial_cache=restored_cache,
        resume_prompt_token_by_token=True,
    )
    payload = {
        "phase": "phase-b",
        "model_ref": model_ref,
        "answer_text": result["answer_text"],
        "answer_preview": result["answer_preview"],
        "prompt_token_count": result["prompt_token_count"],
        "receipts_path": str(receipts_path),
        "receipt_summary": _receipt_summary(receipts_path),
        "session_receipt": {
            "mission_id": "state-juggler",
            "phase": "phase-b",
            **restored_receipt,
            "save_time_ms": phase_a["session_receipt"].get("save_time_ms"),
        },
        "context_keywords_hit": [
            word for word in STATE_JUGGLER_EXPECTED["question_two"] if word.lower() in str(result["answer_text"]).lower()
        ],
    }
    _write_internal_phase(Path(args.internal_output_path), payload)
    if restore_snapshot_dir.exists():
        shutil.rmtree(restore_snapshot_dir)
    return payload


def _run_internal_state_juggler(args: argparse.Namespace, *, profile: dict[str, Any]) -> int:
    if args.internal_state_juggler_phase == "phase-a":
        _run_state_juggler_phase_a(args, profile=profile)
    else:
        _run_state_juggler_phase_b(args, profile=profile)
    return 0


def run_state_juggler(
    args: argparse.Namespace,
    *,
    profile_name: str,
    profile: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    phase_root = output_dir / "_stress_state_juggler"
    if phase_root.exists():
        shutil.rmtree(phase_root)
    phase_root.mkdir(parents=True, exist_ok=True)
    phase_a_json = phase_root / "phase-a.json"
    phase_b_json = phase_root / "phase-b.json"
    session_dir = phase_root / "saved-session"
    phase_a_receipts = output_dir / "local-zamba2-stress-state-juggler-phase-a.jsonl.gz"
    phase_b_receipts = output_dir / "local-zamba2-stress-state-juggler-phase-b.jsonl.gz"
    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mission",
        "state-juggler",
        "--profile",
        profile_name,
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--internal-session-dir",
        str(session_dir),
    ]
    if bool(args.local_files_only):
        base_cmd.append("--local-files-only")
    if bool(args.trust_remote_code):
        base_cmd.append("--trust-remote-code")

    subprocess.run(
        base_cmd
        + [
            "--internal-state-juggler-phase",
            "phase-a",
            "--internal-output-path",
            str(phase_a_json),
            "--internal-receipts-path",
            str(phase_a_receipts),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        base_cmd
        + [
            "--internal-state-juggler-phase",
            "phase-b",
            "--internal-output-path",
            str(phase_b_json),
            "--internal-input-path",
            str(phase_a_json),
            "--internal-receipts-path",
            str(phase_b_receipts),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    phase_a = json.loads(phase_a_json.read_text(encoding="utf-8"))
    phase_b = json.loads(phase_b_json.read_text(encoding="utf-8"))
    hash_match = bool(phase_b["session_receipt"]["hash_match"])
    result = _mission_result_header(
        mission_id="state-juggler",
        title="State Juggler",
        profile_name=profile_name,
        model_ref=str(profile["model_ref"]),
        fixture_path=STATE_JUGGLER_FIXTURE,
    )
    result["phase_a"] = phase_a
    result["phase_b"] = phase_b
    result["headline_metrics"] = {
        "hash_match": hash_match,
        "session_total_bytes": int(phase_a["session_receipt"]["session_total_bytes"]),
        "save_time_ms": phase_a["session_receipt"]["save_time_ms"],
        "load_time_ms": phase_b["session_receipt"]["load_time_ms"],
    }
    result["strongest_claim"] = (
        "Hybrid session hashes matched across a real process restart."
        if hash_match
        else "Hybrid session restore changed the serialized snapshot."
    )
    result["claims"] = [
        f"question_one keywords={len(phase_a['context_keywords_hit'])}",
        f"question_two keywords={len(phase_b['context_keywords_hit'])}",
    ]
    session_receipt_path = _write_json(
        output_dir / "local-zamba2-stress-state-juggler-session.json",
        {
            "mission_id": "state-juggler",
            "profile": profile_name,
            "phase_a": phase_a["session_receipt"],
            "phase_b": phase_b["session_receipt"],
            "hash_match": hash_match,
        },
    )
    result["session_receipt_path"] = str(session_receipt_path)
    mission_path = _write_json(output_dir / "local-zamba2-stress-state-juggler.json", result)
    if phase_root.exists():
        shutil.rmtree(phase_root)
    return result, mission_path


def run_context_switcher(
    args: argparse.Namespace,
    *,
    profile_name: str,
    profile: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    model_ref = str(profile["model_ref"])
    device = _device_for_benchmark(args.device)
    model, adapter, input_adapter = _load_model_stack(
        model_ref,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        device=device,
    )
    prompt_text = _read_text(CONTEXT_SWITCHER_FIXTURE) + "\n\n" + CONTEXT_SWITCHER_PROMPT
    prompt_inputs, prompt_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=prompt_text,
        max_tokens=int(profile["mission3_prompt_tokens"]),
    )
    native_variant = _pick_variant(profile, name="native-dense")
    receipts_path = output_dir / "local-zamba2-stress-context-switcher.jsonl.gz"
    if receipts_path.exists():
        receipts_path.unlink()
    combined_variant = _pick_variant(
        profile,
        name="turbo-int8-hadamard+q-mamba-dsq-int4",
        receipts_path=receipts_path,
        run_id="stress-context-switcher",
        clip_threshold_pct=float(profile["mission3_clip_threshold_pct"]),
        rel_rmse_threshold=float(profile["mission3_rel_rmse_threshold"]),
        auto_promote=bool(profile["mission3_auto_promote"]),
    )
    native_result = _run_generation_trace(
        model,
        prompt_inputs=prompt_inputs,
        prompt_ids=prompt_ids,
        variant=native_variant,
        max_new_tokens=int(profile["mission3_max_new_tokens"]),
        adapter=adapter,
        device=device,
    )
    combined_result = _run_generation_trace(
        model,
        prompt_inputs=prompt_inputs,
        prompt_ids=prompt_ids,
        variant=combined_variant,
        max_new_tokens=int(profile["mission3_max_new_tokens"]),
        adapter=adapter,
        device=device,
    )
    receipt_summary = _receipt_summary(receipts_path)
    native_steps = native_result["step_times_ms"]
    combined_steps = combined_result["step_times_ms"]
    step_windows: list[dict[str, Any]] = []
    receipt_steps = {int(item["step_index"]): item for item in receipt_summary["step_windows"]}
    for step_index in range(min(len(native_steps), len(combined_steps))):
        native_ms = float(native_steps[step_index])
        combined_ms = float(combined_steps[step_index])
        receipt_step = dict(receipt_steps.get(step_index, {}))
        step_windows.append(
            {
                "step_index": int(step_index),
                "native_step_time_ms": native_ms,
                "combined_step_time_ms": combined_ms,
                "step_time_ratio_vs_native": float(native_ms / combined_ms) if combined_ms > 0 else None,
                "receipt_count": int(receipt_step.get("receipt_count", 0)),
                "max_clip_pct": float(receipt_step.get("max_clip_pct", 0.0)),
                "max_rel_rmse": float(receipt_step.get("max_rel_rmse", 0.0)),
                "promoted_block_count": int(receipt_step.get("promoted_block_count", 0)),
                "int4_block_count": int(receipt_step.get("int4_block_count", 0)),
                "int8_block_count": int(receipt_step.get("int8_block_count", 0)),
                "dense_block_count": int(receipt_step.get("dense_block_count", 0)),
                "fallback_counts": receipt_step.get("fallback_counts", {}),
            }
        )

    answer_contains_expected_value = CONTEXT_SWITCHER_EXPECTED_VALUE in str(combined_result["answer_text"])
    result = _mission_result_header(
        mission_id="context-switcher",
        title="Context Switcher",
        profile_name=profile_name,
        model_ref=model_ref,
        fixture_path=CONTEXT_SWITCHER_FIXTURE,
    )
    result["native_dense"] = _variant_result_compact(native_result)
    result["combined"] = _variant_result_compact(combined_result)
    result["answer_contains_expected_value"] = bool(answer_contains_expected_value)
    result["expected_value"] = CONTEXT_SWITCHER_EXPECTED_VALUE
    result["receipt_summary"] = receipt_summary
    result["step_windows"] = step_windows
    result["headline_metrics"] = {
        "speedup_vs_native": _speedup(native_result, combined_result),
        "hybrid_total_runtime_cache_ratio_vs_native": _runtime_ratio(native_result, combined_result),
        "promoted_block_total": int(receipt_summary.get("promoted_block_total", 0)),
        "logits_finite": bool(combined_result["logits_finite"]),
    }
    result["strongest_claim"] = (
        f"Context switcher kept logits finite and surfaced {receipt_summary.get('promoted_block_total', 0)} promoted blocks."
    )
    result["claims"] = [
        f"expected_value_present={answer_contains_expected_value}",
        f"max_clip_pct={receipt_summary.get('max_clip_pct', 0.0):.2f}",
    ]
    mission_path = _write_json(output_dir / "local-zamba2-stress-context-switcher.json", result)
    return result, mission_path


def _restore_equivalence_phase_a(args: argparse.Namespace, *, profile: dict[str, Any]) -> dict[str, Any]:
    model_ref = str(profile["model_ref"])
    device = _device_for_benchmark(args.device)
    model, adapter, input_adapter = _load_model_stack(
        model_ref,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        device=device,
    )
    context_text = _read_text(STATE_JUGGLER_FIXTURE) + "\n\nBuild a resumable hybrid session from this diagnostic log.\n"
    context_inputs, context_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=context_text,
        max_tokens=int(profile["mission4_prompt_tokens"]),
    )
    probe_inputs, probe_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=RESTORE_EQUIVALENCE_PROMPT,
        max_tokens=int(profile["mission4_probe_tokens"]),
    )
    receipts_path = Path(args.internal_receipts_path)
    if receipts_path.exists():
        receipts_path.unlink()
    variant = _pick_variant(
        profile,
        name="turbo-int8-hadamard+q-mamba-dsq-int4",
        receipts_path=receipts_path,
        run_id="stress-restore-equivalence-pre",
    )
    seed_result = _run_generation_trace(
        model,
        prompt_inputs=context_inputs,
        prompt_ids=context_ids,
        variant=variant,
        max_new_tokens=0,
        adapter=adapter,
        device=device,
        save_session_path=Path(args.internal_session_dir),
        return_cache=True,
    )
    cache = seed_result.pop("_cache")
    pre_result = _run_generation_trace(
        model,
        prompt_inputs=probe_inputs,
        prompt_ids=probe_ids,
        variant=variant,
        max_new_tokens=int(profile["mission4_max_new_tokens"]),
        adapter=adapter,
        device=device,
        initial_cache=cache,
        resume_prompt_token_by_token=True,
        capture_selection_logits=True,
    )
    payload = {
        "phase": "phase-a",
        "model_ref": model_ref,
        "context_prompt_token_count": int(len(context_ids)),
        "probe_prompt_token_count": int(len(probe_ids)),
        "session_receipt": seed_result["session_receipt"],
        "seed_result": _variant_result_compact(seed_result),
        "pre_restore": {
            **_variant_result_compact(pre_result),
            "answer_token_ids": pre_result.get("answer_token_ids"),
            "selection_logits": pre_result.get("selection_logits"),
        },
        "receipts_path": str(receipts_path),
        "receipt_summary": _receipt_summary(receipts_path),
    }
    _write_internal_phase(Path(args.internal_output_path), payload)
    return payload


def _restore_equivalence_phase_b(args: argparse.Namespace, *, profile: dict[str, Any]) -> dict[str, Any]:
    phase_a = json.loads(Path(args.internal_input_path).read_text(encoding="utf-8"))
    model_ref = str(profile["model_ref"])
    device = _device_for_benchmark(args.device)
    model, adapter, input_adapter = _load_model_stack(
        model_ref,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        device=device,
    )
    session_dir = Path(args.internal_session_dir)
    load_start = time.perf_counter()
    restored_cache = _load_benchmark_cache(session_dir, model_config=model.config, device=device)
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    restore_snapshot_dir = session_dir.parent / "restored-snapshot"
    if restore_snapshot_dir.exists():
        shutil.rmtree(restore_snapshot_dir)
    _save_benchmark_cache(restored_cache, model_config=model.config, path=restore_snapshot_dir)
    restored_receipt = _session_snapshot_receipt(restore_snapshot_dir)
    restored_receipt["load_time_ms"] = float(load_time_ms)
    restored_receipt["hash_match"] = bool(
        restored_receipt["session_hash"] == phase_a["session_receipt"]["session_hash"]
    )
    probe_inputs, probe_ids = _encode_prompt_text(
        model_ref=model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_text=RESTORE_EQUIVALENCE_PROMPT,
        max_tokens=int(profile["mission4_probe_tokens"]),
    )
    receipts_path = Path(args.internal_receipts_path)
    if receipts_path.exists():
        receipts_path.unlink()
    variant = _pick_variant(
        profile,
        name="turbo-int8-hadamard+q-mamba-dsq-int4",
        receipts_path=receipts_path,
        run_id="stress-restore-equivalence-post",
    )
    post_result = _run_generation_trace(
        model,
        prompt_inputs=probe_inputs,
        prompt_ids=probe_ids,
        variant=variant,
        max_new_tokens=int(profile["mission4_max_new_tokens"]),
        adapter=adapter,
        device=device,
        initial_cache=restored_cache,
        resume_prompt_token_by_token=True,
        capture_selection_logits=True,
    )
    payload = {
        "phase": "phase-b",
        "model_ref": model_ref,
        "probe_prompt_token_count": int(len(probe_ids)),
        "session_receipt": restored_receipt,
        "post_restore": {
            **_variant_result_compact(post_result),
            "answer_token_ids": post_result.get("answer_token_ids"),
            "selection_logits": post_result.get("selection_logits"),
        },
        "receipts_path": str(receipts_path),
        "receipt_summary": _receipt_summary(receipts_path),
    }
    _write_internal_phase(Path(args.internal_output_path), payload)
    if restore_snapshot_dir.exists():
        shutil.rmtree(restore_snapshot_dir)
    return payload


def _run_internal_restore_equivalence(args: argparse.Namespace, *, profile: dict[str, Any]) -> int:
    if args.internal_restore_equivalence_phase == "phase-a":
        _restore_equivalence_phase_a(args, profile=profile)
    else:
        _restore_equivalence_phase_b(args, profile=profile)
    return 0


def run_restore_equivalence(
    args: argparse.Namespace,
    *,
    profile_name: str,
    profile: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    phase_root = output_dir / "_stress_restore_equivalence"
    if phase_root.exists():
        shutil.rmtree(phase_root)
    phase_root.mkdir(parents=True, exist_ok=True)
    phase_a_json = phase_root / "phase-a.json"
    phase_b_json = phase_root / "phase-b.json"
    session_dir = phase_root / "saved-session"
    pre_receipts = output_dir / "local-zamba2-stress-restore-equivalence-pre.jsonl.gz"
    post_receipts = output_dir / "local-zamba2-stress-restore-equivalence-post.jsonl.gz"
    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mission",
        "restore-equivalence",
        "--profile",
        profile_name,
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--internal-session-dir",
        str(session_dir),
    ]
    if bool(args.local_files_only):
        base_cmd.append("--local-files-only")
    if bool(args.trust_remote_code):
        base_cmd.append("--trust-remote-code")
    subprocess.run(
        base_cmd
        + [
            "--internal-restore-equivalence-phase",
            "phase-a",
            "--internal-output-path",
            str(phase_a_json),
            "--internal-receipts-path",
            str(pre_receipts),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        base_cmd
        + [
            "--internal-restore-equivalence-phase",
            "phase-b",
            "--internal-output-path",
            str(phase_b_json),
            "--internal-input-path",
            str(phase_a_json),
            "--internal-receipts-path",
            str(post_receipts),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    phase_a = json.loads(phase_a_json.read_text(encoding="utf-8"))
    phase_b = json.loads(phase_b_json.read_text(encoding="utf-8"))
    comparison = _compare_restore_equivalence(phase_a["pre_restore"], phase_b["post_restore"])
    hash_match = bool(phase_b["session_receipt"]["hash_match"])
    result = _mission_result_header(
        mission_id="restore-equivalence",
        title="Restore Equivalence",
        profile_name=profile_name,
        model_ref=str(profile["model_ref"]),
        fixture_path=STATE_JUGGLER_FIXTURE,
    )
    result["phase_a"] = {
        key: value
        for key, value in phase_a.items()
        if key not in {"pre_restore"}
    }
    result["phase_b"] = {
        key: value
        for key, value in phase_b.items()
        if key not in {"post_restore"}
    }
    result["pre_restore"] = {
        key: value for key, value in phase_a["pre_restore"].items() if key != "selection_logits"
    }
    result["post_restore"] = {
        key: value for key, value in phase_b["post_restore"].items() if key != "selection_logits"
    }
    result["comparison"] = comparison
    result["headline_metrics"] = {
        "hash_match": hash_match,
        "generated_ids_match": bool(comparison["generated_ids_match"]),
        "top1_match_all": bool(comparison["top1_match_all"]),
        "max_abs_logit_delta": comparison["max_abs_logit_delta"],
        "mean_abs_logit_delta": comparison["mean_abs_logit_delta"],
        "finite_before": bool(comparison["finite_before"]),
        "finite_after": bool(comparison["finite_after"]),
        "session_total_bytes": int(phase_a["session_receipt"]["session_total_bytes"]),
        "save_time_ms": phase_a["session_receipt"].get("save_time_ms"),
        "load_time_ms": phase_b["session_receipt"].get("load_time_ms"),
    }
    result["strongest_claim"] = (
        "Restored hybrid session produced the same deterministic continuation as the pre-restore cache."
        if hash_match and comparison["generated_ids_match"] and comparison["top1_match_all"]
        else "Restored hybrid session was integrity-checked, but continuation equivalence needs caveats."
    )
    result["claims"] = [
        f"hash_match={hash_match}",
        f"generated_ids_match={comparison['generated_ids_match']}",
        f"top1_match_all={comparison['top1_match_all']}",
        f"max_abs_logit_delta={comparison['max_abs_logit_delta']}",
    ]
    session_receipt_path = _write_json(
        output_dir / "local-zamba2-stress-restore-equivalence-session.json",
        {
            "mission_id": "restore-equivalence",
            "profile": profile_name,
            "phase_a": phase_a["session_receipt"],
            "phase_b": phase_b["session_receipt"],
            "hash_match": hash_match,
        },
    )
    result["session_receipt_path"] = str(session_receipt_path)
    mission_path = _write_json(output_dir / "local-zamba2-stress-restore-equivalence.json", result)
    if phase_root.exists():
        shutil.rmtree(phase_root)
    return result, mission_path


def _dashboard_payload(profile_name: str, profile: dict[str, Any], missions: list[tuple[dict[str, Any], Path]]) -> dict[str, Any]:
    return {
        "title": "HeliX Stress Missions v1",
        "profile": profile_name,
        "model_ref": str(profile["model_ref"]),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "missions": [
            {
                **payload,
                "artifact_path": str(path),
            }
            for payload, path in missions
        ],
        "strongest_claims": [payload["strongest_claim"] for payload, _ in missions],
    }


def run_stress_suite(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_name = str(args.profile or "laptop-12gb")
    profile = dict(PUBLIC_LOCAL_PROFILES.get(profile_name, PUBLIC_LOCAL_PROFILES["laptop-12gb"]))
    selected = str(args.mission)

    missions: list[tuple[dict[str, Any], Path]] = []
    if selected in {"long-context-coder", "all"}:
        missions.append(run_long_context_coder(args, profile_name=profile_name, profile=profile, output_dir=output_dir))
    if selected in {"state-juggler", "all"}:
        missions.append(run_state_juggler(args, profile_name=profile_name, profile=profile, output_dir=output_dir))
    if selected in {"context-switcher", "all"}:
        missions.append(run_context_switcher(args, profile_name=profile_name, profile=profile, output_dir=output_dir))
    if selected in {"restore-equivalence", "all"}:
        missions.append(run_restore_equivalence(args, profile_name=profile_name, profile=profile, output_dir=output_dir))

    mission_by_id: dict[str, tuple[dict[str, Any], Path]] = {
        payload["mission_id"]: (payload, path) for payload, path in missions
    }
    for mission_id, artifact_name in MISSION_ARTIFACT_NAMES.items():
        if mission_id in mission_by_id:
            continue
        artifact_path = output_dir / artifact_name
        if not artifact_path.exists():
            continue
        try:
            mission_by_id[mission_id] = (
                json.loads(artifact_path.read_text(encoding="utf-8")),
                artifact_path,
            )
        except Exception:
            continue
    ordered_missions = [
        mission_by_id[mission_id]
        for mission_id in MISSION_ARTIFACT_NAMES
        if mission_id in mission_by_id
    ]

    dashboard_path = _write_json(
        output_dir / "local-zamba2-stress-dashboard.json",
        _dashboard_payload(profile_name, profile, ordered_missions),
    )
    outputs = {payload["mission_id"]: path for payload, path in missions}
    outputs["dashboard"] = dashboard_path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local hybrid stress missions for Zamba2 on a laptop-safe profile.")
    parser.add_argument(
        "--mission",
        choices=["long-context-coder", "state-juggler", "context-switcher", "restore-equivalence", "all"],
        default="all",
    )
    parser.add_argument("--profile", default="laptop-12gb")
    parser.add_argument("--output-dir", type=Path, default=Path("verification"))
    parser.add_argument("--mission1-tier", choices=["staged", "heavy"], default="staged")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--internal-state-juggler-phase", choices=["phase-a", "phase-b"])
    parser.add_argument("--internal-restore-equivalence-phase", choices=["phase-a", "phase-b"])
    parser.add_argument("--internal-output-path", type=Path)
    parser.add_argument("--internal-input-path", type=Path)
    parser.add_argument("--internal-session-dir", type=Path)
    parser.add_argument("--internal-receipts-path", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    profile_name = str(args.profile or "laptop-12gb")
    profile = dict(PUBLIC_LOCAL_PROFILES.get(profile_name, PUBLIC_LOCAL_PROFILES["laptop-12gb"]))
    if args.internal_state_juggler_phase:
        if args.internal_output_path is None or args.internal_session_dir is None:
            raise SystemExit("internal state-juggler phases require --internal-output-path and --internal-session-dir")
        if args.internal_receipts_path is None:
            raise SystemExit("internal state-juggler phases require --internal-receipts-path")
        if args.internal_state_juggler_phase == "phase-b" and args.internal_input_path is None:
            raise SystemExit("phase-b requires --internal-input-path")
        return _run_internal_state_juggler(args, profile=profile)
    if args.internal_restore_equivalence_phase:
        if args.internal_output_path is None or args.internal_session_dir is None:
            raise SystemExit("internal restore-equivalence phases require --internal-output-path and --internal-session-dir")
        if args.internal_receipts_path is None:
            raise SystemExit("internal restore-equivalence phases require --internal-receipts-path")
        if args.internal_restore_equivalence_phase == "phase-b" and args.internal_input_path is None:
            raise SystemExit("restore-equivalence phase-b requires --internal-input-path")
        return _run_internal_restore_equivalence(args, profile=profile)
    outputs = run_stress_suite(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
