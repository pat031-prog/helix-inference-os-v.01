from __future__ import annotations

import contextlib
import gzip
import hashlib
import inspect
import io
import importlib
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, DynamicCache
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicLayer

from helix_proto.hf import (
    _HotWindowKVArray,
    _Turbo4BitKVArray,
    _TurboInt8KVArray,
    _TurboQJLKVArray,
    _build_kv_rotation,
    _compute_lloyd_max_codebook,
    _gaussian_qjl_matrix,
)
from helix_kv.torch_quant import (
    Torch4BitKVArray,
    Torch4BitQuantizer,
    TorchHotWindowKVArray,
    TorchInt8KVArray,
    TorchRotation,
    _pearson_kurtosis,
)

_DEFAULT_TRANSFORMERS_BENCHMARK_PROMPT = (
    "Helix benchmarks how KV-cache compression changes prompt perplexity, token selection, "
    "GPU memory, and serialized session size during real inference on instruction-tuned models. "
    "The passage intentionally mixes technical nouns, verbs, numbers, and varied sentence lengths "
    "so the prompt distribution is not a trivial repetition loop. Local-first agents, compressed "
    "memory systems, retrieval traces, JSON tool outputs, and long-context summaries all appear in "
    "the text because the benchmark should resemble realistic product traffic rather than synthetic noise."
)
_DEFAULT_WARMUP_PROMPT_LENGTH = 32
_DEFAULT_WARMUP_MAX_NEW_TOKENS = 4
_DEFAULT_PROTECTED_LAYER_COUNT = 2
_DEFAULT_ONLINE_FOURBIT_MAX_ITER = 5
_DEFAULT_FIXED_FOURBIT_MAX_ITER = 0
_DEFAULT_MAMBA_STATE_BLOCK_SIZE = 64
_DEFAULT_MAMBA_STATE_SCALE_FLOOR = 1e-8
_DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT = 2.0
_DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD = 0.2
_MODE_RANK = {"turbo-4bit": 0, "turbo-int8": 1, "native-dense": 2}
_GEMMA_REF_PREFIXES = ("google/gemma-3", "google/gemma-4")
_HXQ_REF_PREFIX = "echolabs33/"
_HXQ_MODEL_ALIASES = {
    "echolabs33/zamba2-1.2b-helix": "EchoLabs33/zamba2-1.2b-hxq",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_src_path() -> Path:
    return _repo_root() / "src"


def _safe_resolve_path(path_value: str) -> Path | None:
    try:
        return Path(path_value).resolve()
    except OSError:
        return None


def _normalize_model_ref(model_ref: str | Path) -> str:
    return str(model_ref).replace("\\", "/")


def _canonical_model_ref(model_ref: str | Path) -> str:
    normalized = _normalize_model_ref(model_ref)
    return _HXQ_MODEL_ALIASES.get(normalized.lower(), normalized)


def _is_hxq_model_ref(model_ref: str | Path) -> bool:
    return _normalize_model_ref(model_ref).lower().startswith(_HXQ_REF_PREFIX)


def _is_gemma_model_ref(model_ref: str | Path) -> bool:
    normalized = _normalize_model_ref(model_ref).lower()
    return any(normalized.startswith(prefix) for prefix in _GEMMA_REF_PREFIXES)


def _is_gemma3_model_ref(model_ref: str | Path) -> bool:
    return _normalize_model_ref(model_ref).lower().startswith("google/gemma-3")


def _is_gemma4_model_ref(model_ref: str | Path) -> bool:
    return _normalize_model_ref(model_ref).lower().startswith("google/gemma-4")


def _is_gated_model_ref(model_ref: str | Path) -> bool:
    return _is_gemma_model_ref(model_ref)


def _decoder_model_config(model_config: Any) -> Any:
    return model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config


def _layers_block_type(model_config: Any) -> list[str]:
    decoder_config = _decoder_model_config(model_config)
    values = getattr(decoder_config, "layers_block_type", None) or []
    return [str(value) for value in values]


def _is_zamba2_hybrid_model_config(model_config: Any) -> bool:
    decoder_config = _decoder_model_config(model_config)
    model_type = str(getattr(decoder_config, "model_type", "") or "").lower()
    block_types = _layers_block_type(decoder_config)
    return model_type == "zamba2" or ("hybrid" in block_types and "mamba" in block_types)


def _promote_mode(current_mode: str, minimum_mode: str) -> str:
    if _MODE_RANK[str(current_mode)] < _MODE_RANK[str(minimum_mode)]:
        return str(minimum_mode)
    return str(current_mode)


def _default_protected_layer_indices(num_layers: int) -> list[int]:
    protected: set[int] = set()
    if num_layers <= 0:
        return []
    for offset in range(min(_DEFAULT_PROTECTED_LAYER_COUNT, num_layers)):
        protected.add(offset)
        protected.add(max(num_layers - 1 - offset, 0))
    return sorted(protected)


def _quantizer_max_iter_for_variant(variant: dict[str, Any], cache_name: str) -> int:
    cache_specific = variant.get(f"kv_{cache_name}_fourbit_max_iter")
    if cache_specific is not None:
        return int(cache_specific)
    return int(variant.get("kv_fourbit_max_iter", _DEFAULT_ONLINE_FOURBIT_MAX_ITER))


def _directory_size_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _huggingface_cache_size_bytes(model_ref: str | Path) -> int | None:
    local_path = Path(str(model_ref))
    if local_path.exists():
        return _directory_size_bytes(local_path) if local_path.is_dir() else int(local_path.stat().st_size)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{str(model_ref).replace('/', '--')}"
    if not cache_root.exists():
        return None
    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            latest = max(snapshots, key=lambda path: path.stat().st_mtime)
            return _directory_size_bytes(latest)
    return _directory_size_bytes(cache_root)


def _model_vram_bytes(model: Any) -> int | None:
    try:
        footprint = model.get_memory_footprint()
    except Exception:
        footprint = None
    if footprint is not None:
        return int(footprint)
    total = 0
    found = False
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += int(tensor.numel() * tensor.element_size())
        found = True
    return total if found else None


def _imported_module_origin(module_name: str) -> str | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    return getattr(module, "__file__", None)


def _weight_runtime_source_for_model(model_ref: str | Path) -> str | None:
    if not _is_hxq_model_ref(model_ref):
        return None
    origin = _imported_module_origin("helix_substrate")
    if origin is None:
        return "not-imported"
    resolved_origin = _safe_resolve_path(origin)
    repo_src = _repo_src_path().resolve()
    if resolved_origin is not None and repo_src in resolved_origin.parents:
        return "repo-fallback"
    return "pypi"


def _ensure_hxq_hf_integration_registered() -> None:
    importlib.import_module("helix_substrate")
    last_error: Exception | None = None
    for module_name in ("helix_substrate.hf_integration", "helix_substrate.hf_quantizer"):
        try:
            importlib.import_module(module_name)
            return
        except ModuleNotFoundError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


@contextlib.contextmanager
def _prefer_installed_substrate() -> Any:
    repo_src = _repo_src_path().resolve()
    original_sys_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path if _safe_resolve_path(entry) != repo_src]
    for module_name in list(sys.modules):
        if module_name == "helix_substrate" or module_name.startswith("helix_substrate."):
            sys.modules.pop(module_name, None)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


@contextlib.contextmanager
def _prefer_repo_substrate() -> Any:
    repo_src = _repo_src_path().resolve()
    original_sys_path = list(sys.path)
    sys.path[:] = [str(repo_src)] + [entry for entry in sys.path if _safe_resolve_path(entry) != repo_src]
    for module_name in list(sys.modules):
        if module_name == "helix_substrate" or module_name.startswith("helix_substrate."):
            sys.modules.pop(module_name, None)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


def _logit_comparison(current: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float32).reshape(-1)
    diff = np.abs(current - baseline)
    denom = float(np.linalg.norm(current) * np.linalg.norm(baseline))
    cosine = float(np.dot(current, baseline) / denom) if denom > 0.0 else 0.0
    return {
        "cosine_similarity": cosine,
        "max_abs_err": float(np.max(diff)),
        "mean_abs_err": float(np.mean(diff)),
    }


def _storage_mode_name(mode: str | None) -> str | None:
    if mode is None:
        return None
    lowered = str(mode).strip().lower()
    aliases = {
        "fp32": "native-dense",
        "native": "native-dense",
        "native-dense": "native-dense",
        "turbo-int8": "turbo-int8",
        "turbo-int8-hadamard": "turbo-int8",
        "turbo-int8-qr": "turbo-int8",
        "turbo-4bit": "turbo-4bit",
        "turbo-qjl": "turbo-qjl",
        "adaptive": "adaptive",
        "adaptive-asymmetric": "adaptive-asymmetric",
    }
    try:
        return aliases[lowered]
    except KeyError as exc:
        raise ValueError(f"unsupported Transformers KV mode: {mode}") from exc


def _public_mode_name(mode: str, *, rotation_mode: str) -> str:
    if mode == "native-dense":
        return "native-dense"
    if mode == "turbo-int8":
        return "turbo-int8-hadamard" if str(rotation_mode) == "hadamard" else "turbo-int8-qr"
    return str(mode)


def _normalize_mamba_state_precision(value: str | None) -> str:
    normalized = "native-dense" if value is None else str(value).strip().lower()
    if normalized not in _VALID_MAMBA_STATE_PRECISIONS:
        raise ValueError(f"unsupported Mamba state precision: {value}")
    return normalized


def _scaling_strategy_name(strategy: str | None, *, default: str | None = None) -> str | None:
    if strategy is None:
        return None if default is None else str(default)
    lowered = str(strategy).strip().lower()
    if lowered in {"channel", "per-channel", "per_channel"}:
        return "per-channel"
    if lowered in {"token", "per-token", "per_token"}:
        return "per-token"
    raise ValueError(f"unsupported scaling strategy: {strategy}")


def _cache_bytes(cache: Any) -> int:
    if cache is None:
        return 0
    if isinstance(cache, torch.Tensor):
        return int(cache.numel() * cache.element_size())
    if isinstance(cache, TorchHotWindowKVArray):
        return int(cache.nbytes)
    if isinstance(cache, (TorchInt8KVArray, Torch4BitKVArray)):
        return int(cache.nbytes)
    if isinstance(cache, _HotWindowKVArray):
        return int(cache.nbytes)
    if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
        return int(cache.nbytes)
    return int(np.asarray(cache, dtype=np.float32).nbytes)


def _materialize_cache(cache: Any) -> np.ndarray | None:
    if cache is None:
        return None
    if isinstance(cache, (_HotWindowKVArray, _TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
        return cache.to_float32().astype(np.float32)
    return np.asarray(cache, dtype=np.float32)


def _materialize_cache_torch(cache: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if cache is None:
        return None
    if isinstance(cache, torch.Tensor):
        return cache.to(device=device, dtype=dtype)
    if isinstance(cache, TorchHotWindowKVArray):
        return cache.to_float(dtype=dtype).to(device=device, dtype=dtype)
    if isinstance(cache, (TorchInt8KVArray, Torch4BitKVArray)):
        return cache.to_float(dtype=dtype).to(device=device, dtype=dtype)
    if isinstance(cache, (_HotWindowKVArray, _TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
        array = cache.to_float32().astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)
    array = np.asarray(cache, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)


def _append_cache(
    cache: Any,
    values: np.ndarray,
    *,
    store_dense_fn,
    store_fn,
    hot_window: int,
) -> Any:
    values = np.asarray(values, dtype=np.float32)
    if cache is None:
        return store_fn(values)
    if values.shape[1] == 1:
        if isinstance(cache, _HotWindowKVArray):
            return cache.append_token(values, max(int(hot_window), 0), store_dense_fn)
        if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
            return cache.append_compressed(values)
    current = _materialize_cache(cache)
    combined = np.concatenate([current, values], axis=1) if current is not None else values
    return store_fn(combined)


def _torch_cache_bytes(cache: Cache) -> int:
    total = 0
    for layer in getattr(cache, "layers", []):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is not None and isinstance(keys, torch.Tensor):
            total += int(keys.numel() * keys.element_size())
        if values is not None and isinstance(values, torch.Tensor):
            total += int(values.numel() * values.element_size())
    return total


def _dynamic_cache_stats(cache: Cache) -> dict[str, Any]:
    total = 0
    dtype_name: str | None = None
    element_size: int | None = None
    for layer in getattr(cache, "layers", []):
        for attr_name in ("keys", "values"):
            tensor = getattr(layer, attr_name, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            total += int(tensor.numel() * tensor.element_size())
            if dtype_name is None:
                dtype_name = str(tensor.dtype).replace("torch.", "")
                element_size = int(tensor.element_size())
    return {
        "bytes": int(total),
        "dtype": dtype_name,
        "element_size_bytes": element_size,
    }


def _int4_packed_bytes(numel: int) -> int:
    return int((int(numel) + 1) // 2)


_VALID_MAMBA_STATE_PRECISIONS = {"native-dense", "q-mamba-dsq-int4"}


def _pack_int4_signed(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(dtype=torch.int16)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)], dim=0)
    encoded = (flat + 8).clamp_(0, 15).to(dtype=torch.uint8)
    low = encoded[0::2]
    high = encoded[1::2]
    return low | (high << 4)


def _unpack_int4_signed(packed: torch.Tensor, *, numel: int) -> torch.Tensor:
    encoded = packed.reshape(-1).to(dtype=torch.uint8)
    unpacked = torch.empty(encoded.numel() * 2, dtype=torch.int16, device=encoded.device)
    unpacked[0::2] = (encoded & 0x0F).to(dtype=torch.int16)
    unpacked[1::2] = ((encoded >> 4) & 0x0F).to(dtype=torch.int16)
    return unpacked[: int(numel)].to(dtype=torch.int8) - 8


class PackedDSQInt4Tensor:
    def __init__(
        self,
        *,
        packed: torch.Tensor,
        channel_scale: torch.Tensor,
        state_scale: torch.Tensor,
        shape: tuple[int, ...],
        dtype_name: str,
        original_bytes: int,
        mse: float,
        mae: float,
        max_abs_err: float,
    ) -> None:
        self.packed = packed.contiguous()
        self.channel_scale = channel_scale.contiguous()
        self.state_scale = state_scale.contiguous()
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype_name = str(dtype_name)
        self.original_bytes = int(original_bytes)
        self.mse = float(mse)
        self.mae = float(mae)
        self.max_abs_err = float(max_abs_err)

    @property
    def numel(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= int(dim)
        return int(numel)

    @property
    def compressed_bytes(self) -> int:
        return int(
            self.packed.numel() * self.packed.element_size()
            + self.channel_scale.numel() * self.channel_scale.element_size()
            + self.state_scale.numel() * self.state_scale.element_size()
        )

    @property
    def compression_ratio(self) -> float:
        compressed = self.compressed_bytes
        return float(self.original_bytes / compressed) if compressed > 0 else 1.0

    def decompress(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.packed.device if device is None else torch.device(device)
        target_dtype = _torch_dtype_from_name(self.dtype_name) if dtype is None else dtype
        quantized = _unpack_int4_signed(self.packed.to(device=target_device), numel=self.numel).reshape(self.shape)
        combined_scale = torch.sqrt(
            self.channel_scale.to(device=target_device, dtype=torch.float32)
            * self.state_scale.to(device=target_device, dtype=torch.float32)
        ).clamp(min=1e-8)
        restored = quantized.to(dtype=torch.float32) * combined_scale
        return restored.to(dtype=target_dtype)


class DenseRuntimeStateTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor.detach().contiguous()
        self.shape = tuple(int(dim) for dim in tensor.shape)
        self.dtype_name = _torch_dtype_name(tensor.dtype)
        self.original_bytes = int(tensor.numel() * tensor.element_size())
        self.numel = int(tensor.numel())
        self.mse = 0.0
        self.mae = 0.0
        self.max_abs_err = 0.0
        self.rel_rmse = 0.0
        self.clip_pct = 0.0
        self.precision = "dense"

    @property
    def compressed_bytes(self) -> int:
        return int(self.original_bytes)

    @property
    def compression_ratio(self) -> float:
        return 1.0

    def decompress(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        target_dtype = _torch_dtype_from_name(self.dtype_name) if dtype is None else dtype
        target_device = self.tensor.device if device is None else torch.device(device)
        return self.tensor.to(device=target_device, dtype=target_dtype)


class PackedBlockwiseQuantizedTensor:
    def __init__(
        self,
        *,
        bits: int,
        storage: torch.Tensor,
        scales: torch.Tensor,
        shape: tuple[int, ...],
        dtype_name: str,
        block_size: int,
        original_bytes: int,
        mse: float,
        mae: float,
        max_abs_err: float,
        rel_rmse: float,
        clip_pct: float,
        numel: int,
    ) -> None:
        self.bits = int(bits)
        self.storage = storage.contiguous()
        self.scales = scales.contiguous()
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype_name = str(dtype_name)
        self.block_size = int(block_size)
        self.original_bytes = int(original_bytes)
        self.mse = float(mse)
        self.mae = float(mae)
        self.max_abs_err = float(max_abs_err)
        self.rel_rmse = float(rel_rmse)
        self.clip_pct = float(clip_pct)
        self.numel = int(numel)
        self.precision = "int4" if self.bits == 4 else "int8"

    @property
    def compressed_bytes(self) -> int:
        return int(
            self.storage.numel() * self.storage.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    @property
    def compression_ratio(self) -> float:
        compressed = self.compressed_bytes
        return float(self.original_bytes / compressed) if compressed > 0 else 1.0

    def decompress(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        target_device = self.storage.device if device is None else torch.device(device)
        target_dtype = _torch_dtype_from_name(self.dtype_name) if dtype is None else dtype
        if self.bits == 4:
            quantized = _unpack_int4_signed(self.storage.to(device=target_device), numel=self.scales.shape[0] * self.block_size)
        else:
            quantized = self.storage.to(device=target_device, dtype=torch.int8).reshape(-1)
        blocks = quantized.to(dtype=torch.float32).reshape(-1, self.block_size)
        restored = (blocks * self.scales.to(device=target_device, dtype=torch.float32)).reshape(-1)[: self.numel]
        return restored.reshape(self.shape).to(dtype=target_dtype)


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _session_snapshot_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    session_json = source / "session.json"
    kv_npz = source / "kv_cache.npz"
    meta_hash = _sha256_file(session_json)
    npz_hash = _sha256_file(kv_npz)
    session_total_bytes = int(session_json.stat().st_size + kv_npz.stat().st_size)
    session_hash = _sha256_json(
        {
            "session_meta_hash": meta_hash,
            "session_npz_hash": npz_hash,
            "session_total_bytes": session_total_bytes,
        }
    )
    return {
        "session_hash": session_hash,
        "session_meta_hash": meta_hash,
        "session_npz_hash": npz_hash,
        "session_total_bytes": session_total_bytes,
    }


def _append_receipt_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "at", encoding="utf-8") as handle:
            handle.write(line)
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _blockwise_quantize_tensor(
    tensor: torch.Tensor,
    *,
    bits: int,
    block_size: int,
    scale_floor: float,
) -> PackedBlockwiseQuantizedTensor | DenseRuntimeStateTensor:
    source = tensor.detach().to(dtype=torch.float32)
    if source.numel() == 0:
        return DenseRuntimeStateTensor(tensor)
    if not bool(torch.isfinite(source).all().item()):
        return DenseRuntimeStateTensor(tensor)
    qmax = 7 if int(bits) == 4 else 127
    flat = source.reshape(-1)
    padded_numel = int(math.ceil(float(flat.numel()) / float(block_size)) * block_size)
    if padded_numel > flat.numel():
        flat = F.pad(flat, (0, padded_numel - flat.numel()))
    blocks = flat.reshape(-1, int(block_size))
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=float(scale_floor))
    normalized = blocks / scales
    clip_mask = normalized.abs() > float(qmax)
    quantized = torch.round(normalized).clamp(min=-float(qmax), max=float(qmax)).to(dtype=torch.int8)
    restored = (quantized.to(dtype=torch.float32) * scales).reshape(-1)[: source.numel()].reshape(source.shape)
    diff = restored - source
    rms = torch.sqrt(source.square().mean()).item()
    rel_rmse = float(torch.sqrt(diff.square().mean()).item() / max(float(rms), float(scale_floor)))
    clip_pct = float(clip_mask.sum().item()) * 100.0 / float(clip_mask.numel())
    storage: torch.Tensor
    if int(bits) == 4:
        storage = _pack_int4_signed(quantized.reshape(-1))
    else:
        storage = quantized.reshape(-1).to(dtype=torch.int8)
    return PackedBlockwiseQuantizedTensor(
        bits=int(bits),
        storage=storage,
        scales=scales.to(dtype=torch.float16),
        shape=tuple(int(dim) for dim in source.shape),
        dtype_name=_torch_dtype_name(tensor.dtype),
        block_size=int(block_size),
        original_bytes=int(tensor.numel() * tensor.element_size()),
        mse=float(diff.square().mean().item()),
        mae=float(diff.abs().mean().item()),
        max_abs_err=float(diff.abs().max().item()),
        rel_rmse=rel_rmse,
        clip_pct=clip_pct,
        numel=int(source.numel()),
    )


def _runtime_precision_name(value: Any) -> str:
    precision = getattr(value, "precision", None)
    return "dense" if precision is None else str(precision)


def _dsq_quantize_tensor_int4(
    tensor: torch.Tensor,
    *,
    channel_reduce_dims: tuple[int, ...],
    state_reduce_dims: tuple[int, ...],
) -> PackedDSQInt4Tensor:
    source = tensor.detach().to(dtype=torch.float32)
    if source.numel() == 0:
        return PackedDSQInt4Tensor(
            packed=torch.zeros(0, dtype=torch.uint8, device=source.device),
            channel_scale=torch.zeros(0, dtype=torch.float16, device=source.device),
            state_scale=torch.zeros(0, dtype=torch.float16, device=source.device),
            shape=tuple(int(dim) for dim in source.shape),
            dtype_name=_torch_dtype_name(tensor.dtype),
            original_bytes=0,
            mse=0.0,
            mae=0.0,
            max_abs_err=0.0,
        )
    eps = 1e-8
    channel_scale = source.abs().amax(dim=channel_reduce_dims, keepdim=True).clamp(min=eps)
    state_scale = source.abs().amax(dim=state_reduce_dims, keepdim=True).clamp(min=eps)
    combined_scale = torch.sqrt(channel_scale * state_scale).clamp(min=eps)
    quantized = torch.round(source / combined_scale).clamp(min=-7, max=7).to(dtype=torch.int8)
    reconstructed = quantized.to(dtype=torch.float32) * combined_scale
    diff = reconstructed - source
    return PackedDSQInt4Tensor(
        packed=_pack_int4_signed(quantized),
        channel_scale=channel_scale.to(dtype=torch.float16),
        state_scale=state_scale.to(dtype=torch.float16),
        shape=tuple(int(dim) for dim in source.shape),
        dtype_name=_torch_dtype_name(tensor.dtype),
        original_bytes=int(tensor.numel() * tensor.element_size()),
        mse=float(diff.square().mean().item()),
        mae=float(diff.abs().mean().item()),
        max_abs_err=float(diff.abs().max().item()),
    )


def _quantize_tensor_dsq_int4(
    tensor: torch.Tensor,
    *,
    channel_reduce_dims: tuple[int, ...],
    state_reduce_dims: tuple[int, ...],
) -> dict[str, Any]:
    packed = _dsq_quantize_tensor_int4(
        tensor,
        channel_reduce_dims=channel_reduce_dims,
        state_reduce_dims=state_reduce_dims,
    )
    if packed.numel == 0:
        return {
            "original_bytes": 0,
            "compressed_bytes": 0,
            "compression_ratio": 1.0,
            "mse": 0.0,
            "mae": 0.0,
            "max_abs_err": 0.0,
            "numel": 0,
        }
    return {
        "original_bytes": int(packed.original_bytes),
        "compressed_bytes": int(packed.compressed_bytes),
        "compression_ratio": float(packed.compression_ratio),
        "mse": float(packed.mse),
        "mae": float(packed.mae),
        "max_abs_err": float(packed.max_abs_err),
        "numel": int(packed.numel),
    }


def _compress_mamba_state_tensor(
    tensor: torch.Tensor,
    *,
    block_size: int,
    scale_floor: float,
    clip_threshold_pct: float,
    rel_rmse_threshold: float,
    auto_promote: bool,
) -> tuple[Any, dict[str, Any]]:
    finite_before = bool(torch.isfinite(tensor).all().item())
    block_count = int(math.ceil(float(tensor.numel()) / float(max(int(block_size), 1)))) if tensor.numel() > 0 else 0
    max_abs_value = float(torch.nan_to_num(tensor.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()) if tensor.numel() > 0 else 0.0
    state_norm = float(torch.nan_to_num(tensor.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).norm().item()) if tensor.numel() > 0 else 0.0
    if not finite_before:
        dense = DenseRuntimeStateTensor(tensor)
        return dense, {
            "precision": "dense",
            "fallback_precision": "dense",
            "fallback_reason": "non_finite_source",
            "dense_bytes": int(dense.original_bytes),
            "compressed_bytes": int(dense.compressed_bytes),
            "ratio": float(dense.compression_ratio),
            "rel_rmse": 0.0,
            "clip_pct": 0.0,
            "finite_before": False,
            "finite_after": False,
            "block_count": block_count,
            "int4_block_count": 0,
            "int8_block_count": 0,
            "dense_block_count": block_count,
            "promoted_block_count": block_count,
            "max_abs_value": max_abs_value,
            "state_norm": state_norm,
        }

    candidate = _blockwise_quantize_tensor(
        tensor,
        bits=4,
        block_size=int(block_size),
        scale_floor=float(scale_floor),
    )
    fallback_reason: str | None = None
    if auto_promote and (
        getattr(candidate, "clip_pct", 0.0) > float(clip_threshold_pct)
        or getattr(candidate, "rel_rmse", 0.0) > float(rel_rmse_threshold)
    ):
        fallback_reason = "int4_threshold_exceeded"
        candidate = _blockwise_quantize_tensor(
            tensor,
            bits=8,
            block_size=int(block_size),
            scale_floor=float(scale_floor),
        )
        if (
            getattr(candidate, "clip_pct", 0.0) > float(clip_threshold_pct)
            or getattr(candidate, "rel_rmse", 0.0) > float(rel_rmse_threshold)
            or not bool(torch.isfinite(candidate.decompress(dtype=tensor.dtype, device=tensor.device)).all().item())
        ):
            fallback_reason = "int8_threshold_exceeded"
            candidate = DenseRuntimeStateTensor(tensor)
    finite_after = bool(torch.isfinite(candidate.decompress(dtype=tensor.dtype, device=tensor.device)).all().item())
    precision = _runtime_precision_name(candidate)
    int4_block_count = block_count if precision == "int4" else 0
    int8_block_count = block_count if precision == "int8" else 0
    dense_block_count = block_count if precision == "dense" else 0
    return candidate, {
        "precision": precision,
        "fallback_precision": precision,
        "fallback_reason": fallback_reason,
        "dense_bytes": int(getattr(candidate, "original_bytes", tensor.numel() * tensor.element_size())),
        "compressed_bytes": int(getattr(candidate, "compressed_bytes", tensor.numel() * tensor.element_size())),
        "ratio": float(getattr(candidate, "compression_ratio", 1.0)),
        "rel_rmse": float(getattr(candidate, "rel_rmse", 0.0)),
        "clip_pct": float(getattr(candidate, "clip_pct", 0.0)),
        "finite_before": finite_before,
        "finite_after": finite_after,
        "block_count": block_count,
        "int4_block_count": int4_block_count,
        "int8_block_count": int8_block_count,
        "dense_block_count": dense_block_count,
        "promoted_block_count": int(block_count if precision != "int4" else 0),
        "max_abs_value": max_abs_value,
        "state_norm": state_norm,
    }


def _probe_mamba_state_compression(cache: Any) -> dict[str, Any] | None:
    conv_states = getattr(cache, "conv_states", None)
    ssm_states = getattr(cache, "ssm_states", None)
    if not isinstance(conv_states, dict) or not isinstance(ssm_states, dict):
        return None

    conv_entries: list[dict[str, Any]] = []
    ssm_entries: list[dict[str, Any]] = []
    for layer_idx in sorted(conv_states):
        tensor = conv_states[layer_idx]
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            stats = _quantize_tensor_dsq_int4(
                tensor,
                channel_reduce_dims=(2,),
                state_reduce_dims=(1,),
            )
            stats["layer_index"] = int(layer_idx)
            conv_entries.append(stats)
    for layer_idx in sorted(ssm_states):
        tensor = ssm_states[layer_idx]
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            stats = _quantize_tensor_dsq_int4(
                tensor,
                channel_reduce_dims=(3,),
                state_reduce_dims=(1, 2),
            )
            stats["layer_index"] = int(layer_idx)
            ssm_entries.append(stats)

    if not conv_entries and not ssm_entries:
        return None

    total_original = sum(int(item["original_bytes"]) for item in conv_entries + ssm_entries)
    total_compressed = sum(int(item["compressed_bytes"]) for item in conv_entries + ssm_entries)
    total_numel = sum(int(item["numel"]) for item in conv_entries + ssm_entries)

    def _weighted_mean(key: str, entries: list[dict[str, Any]]) -> float:
        denom = sum(int(item["numel"]) for item in entries)
        if denom <= 0:
            return 0.0
        return float(sum(float(item[key]) * int(item["numel"]) for item in entries) / denom)

    return {
        "method": "q-mamba-dsq-int4-probe",
        "original_bytes": int(total_original),
        "compressed_bytes": int(total_compressed),
        "compression_ratio": float(total_original / total_compressed) if total_compressed > 0 else 1.0,
        "mse": _weighted_mean("mse", conv_entries + ssm_entries),
        "mae": _weighted_mean("mae", conv_entries + ssm_entries),
        "max_abs_err": max(float(item["max_abs_err"]) for item in conv_entries + ssm_entries),
        "conv_layers_profile": conv_entries,
        "ssm_layers_profile": ssm_entries,
    }


def _native_fp32_equivalent_bytes(native_bytes: int, native_element_size_bytes: int | None) -> int:
    if not native_element_size_bytes or native_element_size_bytes <= 0:
        return int(native_bytes)
    return int(round(float(native_bytes) * (4.0 / float(native_element_size_bytes))))


def _tensor_to_numpy_preserve_bytes(tensor: torch.Tensor) -> np.ndarray:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    if cpu_tensor.dtype == torch.bfloat16:
        return cpu_tensor.view(torch.uint16).numpy()
    return cpu_tensor.numpy()


def _device_for_benchmark(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_causal_model(
    model_ref: str | Path,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    load_kwargs = {
        "local_files_only": local_files_only,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    except Exception as primary_error:
        if not _is_gemma3_model_ref(model_ref):
            if _is_gemma4_model_ref(model_ref):
                try:
                    auto_image_text = getattr(importlib.import_module("transformers"), "AutoModelForImageTextToText")
                except Exception:
                    raise
                return auto_image_text.from_pretrained(model_ref, **load_kwargs)
            raise
        try:
            gemma3_cls = getattr(importlib.import_module("transformers"), "Gemma3ForConditionalGeneration")
        except Exception:
            raise primary_error
        return gemma3_cls.from_pretrained(model_ref, **load_kwargs)


def _load_text_adapter(
    model_ref: str | Path,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[Any | None, str, bool, bool]:
    if _is_gemma_model_ref(model_ref):
        processor = AutoProcessor.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        return processor, "processor-text", True, True
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer, "tokenizer-causal", False, False


def _build_text_chat_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


def _slice_prompt_inputs(prompt_inputs: dict[str, torch.Tensor], prompt_length: int) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    for key, value in prompt_inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim >= 2 and value.shape[0] == 1 and value.shape[-1] >= int(prompt_length):
            sliced[key] = value[..., : int(prompt_length)]
        else:
            sliced[key] = value
    return sliced


def _resolve_prompt_inputs(
    model_ref: str | Path,
    *,
    adapter: Any | None,
    input_adapter: str,
    prompt_ids: list[int] | None,
    prompt_text: str | None,
    prompt_length: int,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    if prompt_ids is not None:
        resolved = [int(token_id) for token_id in prompt_ids]
        return {
            "input_ids": torch.tensor([resolved], dtype=torch.long),
            "attention_mask": torch.ones((1, len(resolved)), dtype=torch.long),
        }, resolved

    source_text = prompt_text or _DEFAULT_TRANSFORMERS_BENCHMARK_PROMPT
    if input_adapter == "processor-text":
        if adapter is None:
            raise ValueError("processor adapter selected without processor")
        prompt_inputs = adapter.apply_chat_template(
            _build_text_chat_messages(source_text),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = prompt_inputs["input_ids"][0].tolist()
        return _slice_prompt_inputs(prompt_inputs, prompt_length), [int(token_id) for token_id in input_ids[: int(prompt_length)]]

    if adapter is None:
        adapter = AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    base_ids = adapter.encode(source_text, add_special_tokens=False)
    if not base_ids:
        raise ValueError("tokenizer produced an empty prompt; pass explicit prompt_ids")
    resolved: list[int] = []
    while len(resolved) < int(prompt_length):
        resolved.extend(base_ids)
    resolved = resolved[: int(prompt_length)]
    return {
        "input_ids": torch.tensor([resolved], dtype=torch.long),
        "attention_mask": torch.ones((1, len(resolved)), dtype=torch.long),
    }, resolved


def _supports_forward_arg(model: Any, argument_name: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return argument_name in signature.parameters


def _sparse_v_skip_ratio(attentions: Any, *, threshold: float | None) -> float | None:
    if threshold is None or attentions is None:
        return None
    total = 0
    skipped = 0
    for attention in attentions:
        if not isinstance(attention, torch.Tensor):
            continue
        total += int(attention.numel())
        skipped += int((attention <= float(threshold)).sum().item())
    if total <= 0:
        return None
    return float(skipped / total)


class TransformersCompressedKVLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(
        self,
        *,
        layer_idx: int,
        num_layers: int,
        kv_cache_precision: str,
        kv_key_precision: str | None,
        kv_value_precision: str | None,
        kv_key_scaling_strategy: str | None,
        kv_value_scaling_strategy: str | None,
        kv_rotation_mode: str,
        kv_hot_window: int,
        kv_quant_seed: int,
        kv_calibration_tokens: int,
        kv_adaptive_high_kurtosis: float,
        kv_adaptive_medium_kurtosis: float,
        protected_layer_indices: list[int] | tuple[int, ...] | None = None,
        kv_key_fourbit_max_iter: int = _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        kv_value_fourbit_max_iter: int = _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        kv_backend: str = "torch",
        kv_async_compression: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.kv_cache_precision = _storage_mode_name(kv_cache_precision) or "native-dense"
        self.kv_key_precision = _storage_mode_name(kv_key_precision)
        self.kv_value_precision = _storage_mode_name(kv_value_precision)
        self.kv_key_scaling_strategy = _scaling_strategy_name(kv_key_scaling_strategy)
        self.kv_value_scaling_strategy = _scaling_strategy_name(kv_value_scaling_strategy)
        self.kv_rotation_mode = str(kv_rotation_mode)
        self.kv_hot_window = max(int(kv_hot_window), 0)
        self.kv_quant_seed = int(kv_quant_seed)
        self.kv_calibration_tokens = max(int(kv_calibration_tokens), 0)
        self.kv_adaptive_high_kurtosis = float(kv_adaptive_high_kurtosis)
        self.kv_adaptive_medium_kurtosis = float(kv_adaptive_medium_kurtosis)
        self.protected_layer_indices = tuple(
            int(index) for index in (protected_layer_indices or _default_protected_layer_indices(self.num_layers))
        )
        self._protected_layer_index_set = set(self.protected_layer_indices)
        self.kv_key_fourbit_max_iter = int(kv_key_fourbit_max_iter)
        self.kv_value_fourbit_max_iter = int(kv_value_fourbit_max_iter)
        self.kv_backend = str(kv_backend)
        self.kv_async_compression = bool(kv_async_compression)
        self.key_cache: Any | None = None
        self.value_cache: Any | None = None
        self.seq_length = 0
        self.batch_size = 0
        self.num_heads = 0
        self.head_dim = 0
        self._rotation = None
        self._initial_codebook = None
        self._qjl_matrix = None
        self._adaptive_enabled = self.kv_cache_precision in {"adaptive", "adaptive-asymmetric"}
        self._adaptive_asymmetric_enabled = self.kv_cache_precision == "adaptive-asymmetric"
        self._selected_mode: str | None = None
        self._selected_key_mode: str | None = None
        self._selected_value_mode: str | None = None
        self._kurtosis_profile: dict[str, Any] | None = None
        self._fourbit_quantizers: dict[str, Torch4BitQuantizer | None] = {"k": None, "v": None}
        self._compress_stream: torch.cuda.Stream | None = None
        self._pending_event: torch.cuda.Event | None = None
        self._pending_key_cache: Any | None = None
        self._pending_value_cache: Any | None = None
        self._pending_seq_increment = 0

    def _initialize_runtime_state(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        if self.batch_size != 1:
            raise ValueError("TransformersCompressedKVCache currently supports batch_size=1 only")
        modes = {self.kv_cache_precision}
        if self.kv_key_precision is not None:
            modes.add(self.kv_key_precision)
        if self.kv_value_precision is not None:
            modes.add(self.kv_value_precision)
        if self._adaptive_enabled:
            modes.update({"turbo-int8", "turbo-4bit"})
        if modes != {"native-dense"}:
            legacy_rotation = _build_kv_rotation(self.head_dim, self.kv_quant_seed, self.kv_rotation_mode)
            self._rotation = (
                TorchRotation.from_legacy(legacy_rotation, device=self.device)
                if self.kv_backend == "torch"
                else legacy_rotation
            )
        if modes.intersection({"turbo-4bit", "turbo-qjl", "adaptive"}) and self._rotation is not None:
            self._initial_codebook = _compute_lloyd_max_codebook(
                int(getattr(self._rotation, "rotated_dim", self.head_dim)),
                4,
            )
        if "turbo-qjl" in modes:
            self._qjl_matrix = _gaussian_qjl_matrix(self.head_dim, self.kv_quant_seed)
        if self.kv_backend == "torch" and self.device.type == "cuda" and self.kv_async_compression:
            self._compress_stream = torch.cuda.Stream(device=self.device)
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self._initialize_runtime_state(
            dtype=key_states.dtype,
            device=key_states.device,
            batch_size=int(key_states.shape[0]),
            num_heads=int(key_states.shape[1]),
            head_dim=int(key_states.shape[-1]),
        )

    def _terminal_band_protected(self) -> bool:
        return self.layer_idx in self._protected_layer_index_set

    def _adaptive_mode_from_kurtosis(self, kurtosis: float) -> str:
        if float(kurtosis) >= self.kv_adaptive_high_kurtosis:
            return "native-dense"
        if float(kurtosis) >= self.kv_adaptive_medium_kurtosis:
            return "turbo-int8"
        return "turbo-4bit"

    def _layer_mode_for(self, cache_name: str) -> str:
        if self._adaptive_asymmetric_enabled:
            if cache_name == "k":
                return str(self._selected_key_mode or "native-dense")
            if cache_name == "v":
                return str(self._selected_value_mode or "native-dense")
        if self._adaptive_enabled:
            return str(self._selected_mode or "native-dense")
        if cache_name == "k" and self.kv_key_precision is not None:
            return str(self.kv_key_precision)
        if cache_name == "v" and self.kv_value_precision is not None:
            return str(self.kv_value_precision)
        return str(self.kv_cache_precision)

    def _scaling_strategy_for(self, cache_name: str, *, mode: str) -> str:
        explicit = self.kv_key_scaling_strategy if cache_name == "k" else self.kv_value_scaling_strategy
        if explicit is not None:
            return str(explicit)
        if mode == "turbo-4bit":
            return "per-channel"
        return "per-token"

    def _fourbit_max_iter_for(self, cache_name: str) -> int:
        if cache_name == "k":
            return int(self.kv_key_fourbit_max_iter)
        return int(self.kv_value_fourbit_max_iter)

    def _adaptive_mode_from_stats(self, *, kurtosis: float, kv_norm_ratio: float | None, cache_name: str) -> str:
        selected = self._adaptive_mode_from_kurtosis(kurtosis)
        if kv_norm_ratio is None:
            return selected
        ratio = float(kv_norm_ratio)
        if ratio < 10.0:
            return selected
        if cache_name == "k":
            return _promote_mode(selected, "turbo-int8")
        if ratio >= 10.0:
            return selected
        return selected

    def _finalize_adaptive_mode_torch(self, key_values: torch.Tensor, value_values: torch.Tensor) -> None:
        if not self._adaptive_enabled or (self._selected_mode is not None and not self._adaptive_asymmetric_enabled):
            return
        calibration_tokens = int(self.kv_calibration_tokens) if self.kv_calibration_tokens > 0 else int(key_values.shape[1])
        calibration_tokens = max(1, min(calibration_tokens, int(key_values.shape[1])))
        calibration_keys = key_values[:, :calibration_tokens, :]
        calibration_values = value_values[:, :calibration_tokens, :]
        k_kurtosis = _pearson_kurtosis(calibration_keys)
        v_kurtosis = _pearson_kurtosis(calibration_values)
        k_norm_mean = float(calibration_keys.norm(dim=-1).mean().item())
        v_norm_mean = float(calibration_values.norm(dim=-1).mean().item())
        kv_norm_ratio = float(k_norm_mean / max(v_norm_mean, 1e-8))
        protected = self._terminal_band_protected()
        if self._adaptive_asymmetric_enabled:
            if kv_norm_ratio < 10.0:
                dominant = max(k_kurtosis, v_kurtosis)
                selected_k_mode = self._adaptive_mode_from_stats(
                    kurtosis=dominant,
                    kv_norm_ratio=kv_norm_ratio,
                    cache_name="k",
                )
                selected_v_mode = selected_k_mode
            else:
                selected_k_mode = self._adaptive_mode_from_stats(
                    kurtosis=k_kurtosis,
                    kv_norm_ratio=kv_norm_ratio,
                    cache_name="k",
                )
                selected_v_mode = self._adaptive_mode_from_stats(
                    kurtosis=v_kurtosis,
                    kv_norm_ratio=kv_norm_ratio,
                    cache_name="v",
                )
            if protected and selected_k_mode == "turbo-4bit":
                selected_k_mode = "turbo-int8"
            if protected and selected_v_mode == "turbo-4bit":
                selected_v_mode = "turbo-int8"
            self._selected_key_mode = selected_k_mode
            self._selected_value_mode = selected_v_mode
            self._selected_mode = None
            self._kurtosis_profile = {
                "layer_index": self.layer_idx,
                "k_kurtosis": float(k_kurtosis),
                "v_kurtosis": float(v_kurtosis),
                "k_norm_mean": k_norm_mean,
                "v_norm_mean": v_norm_mean,
                "kv_norm_ratio": kv_norm_ratio,
                "selected_k_mode": _public_mode_name(selected_k_mode, rotation_mode=self.kv_rotation_mode),
                "selected_v_mode": _public_mode_name(selected_v_mode, rotation_mode=self.kv_rotation_mode),
                "protected_layer": protected,
                "protected_terminal_band": protected,
                "protected_layer_indices": list(self.protected_layer_indices),
            }
            return
        dominant = max(k_kurtosis, v_kurtosis)
        selected_mode = self._adaptive_mode_from_stats(
            kurtosis=dominant,
            kv_norm_ratio=kv_norm_ratio,
            cache_name="k",
        )
        if protected and selected_mode == "turbo-4bit":
            selected_mode = "turbo-int8"
        self._selected_mode = selected_mode
        self._selected_key_mode = selected_mode
        self._selected_value_mode = selected_mode
        self._kurtosis_profile = {
            "layer_index": self.layer_idx,
            "k_kurtosis": float(k_kurtosis),
            "v_kurtosis": float(v_kurtosis),
            "k_norm_mean": k_norm_mean,
            "v_norm_mean": v_norm_mean,
            "kv_norm_ratio": kv_norm_ratio,
            "selected_mode": _public_mode_name(selected_mode, rotation_mode=self.kv_rotation_mode),
            "selected_k_mode": _public_mode_name(selected_mode, rotation_mode=self.kv_rotation_mode),
            "selected_v_mode": _public_mode_name(selected_mode, rotation_mode=self.kv_rotation_mode),
            "protected_layer": protected,
            "protected_terminal_band": protected,
            "protected_layer_indices": list(self.protected_layer_indices),
        }

    def _sync_pending_async(self) -> None:
        if self._pending_event is None:
            return
        self._pending_event.synchronize()
        self.key_cache = self._pending_key_cache
        self.value_cache = self._pending_value_cache
        self._pending_event = None
        self._pending_key_cache = None
        self._pending_value_cache = None
        self._pending_seq_increment = 0

    def _fit_fourbit_quantizer(self, cache_name: str, values: torch.Tensor, *, scaling_strategy: str) -> Torch4BitQuantizer:
        existing = self._fourbit_quantizers.get(cache_name)
        if existing is not None:
            return existing
        if self.kv_backend != "torch":
            raise ValueError("torch 4-bit quantizer requested on non-torch backend")
        if self._rotation is None or not isinstance(self._rotation, TorchRotation):
            raise ValueError("torch 4-bit quantizer requires torch rotation")
        calibration_tokens = int(self.kv_calibration_tokens) if self.kv_calibration_tokens > 0 else int(values.shape[1])
        calibration_tokens = max(1, min(calibration_tokens, int(values.shape[1])))
        calibration_values = values[:, :calibration_tokens, :].to(dtype=torch.float32)
        initial_centroids = torch.tensor(
            np.asarray(self._initial_codebook.centroids, dtype=np.float32),
            dtype=torch.float32,
            device=values.device,
        )
        quantizer = Torch4BitQuantizer.from_calibration(
            calibration_values,
            rotation=self._rotation,
            initial_centroids=initial_centroids,
            scaling_strategy=scaling_strategy,
            max_iter=self._fourbit_max_iter_for(cache_name),
        )
        self._fourbit_quantizers[cache_name] = quantizer
        return quantizer

    def _store_compact_cache_torch(self, values: torch.Tensor, *, mode: str, cache_name: str) -> Any:
        values = values.to(device=self.device)
        if mode == "native-dense":
            return values.to(dtype=self.dtype)
        if mode == "turbo-int8":
            if not isinstance(self._rotation, TorchRotation):
                raise ValueError("torch int8 cache requires torch rotation")
            scaling_strategy = self._scaling_strategy_for(cache_name, mode=mode)
            calibration_tokens = int(self.kv_calibration_tokens) if self.kv_calibration_tokens > 0 else int(values.shape[1])
            calibration_tokens = max(1, min(calibration_tokens, int(values.shape[1])))
            calibration_values = values[:, :calibration_tokens, :].to(dtype=torch.float32)
            return TorchInt8KVArray.from_values(
                values.to(dtype=torch.float32),
                rotation=self._rotation,
                scaling_strategy=scaling_strategy,
                calibration_values=calibration_values,
            )
        if mode == "turbo-4bit":
            scaling_strategy = self._scaling_strategy_for(cache_name, mode=mode)
            quantizer = self._fit_fourbit_quantizer(cache_name, values.to(dtype=torch.float32), scaling_strategy=scaling_strategy)
            return Torch4BitKVArray.from_values(values.to(dtype=torch.float32), quantizer=quantizer)
        raise ValueError(f"unsupported torch cache mode: {mode}")

    def _store_compact_cache_numpy(self, values: np.ndarray, *, mode: str) -> Any:
        values = np.asarray(values, dtype=np.float32)
        if mode == "native-dense":
            return values.astype(np.float32)
        if mode == "turbo-int8":
            return _TurboInt8KVArray(values, rotation=self._rotation)
        if mode == "turbo-4bit":
            return _Turbo4BitKVArray(values, rotation=self._rotation, codebook=self._initial_codebook)
        return _TurboQJLKVArray(
            values,
            rotation=self._rotation,
            codebook=self._initial_codebook,
            qjl_matrix=self._qjl_matrix,
        )

    def _store_cache(self, values: Any | None, *, mode: str, cache_name: str) -> Any:
        if values is None:
            return None
        if self.kv_backend == "torch":
            tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, device=self.device, dtype=self.dtype)
            if mode == "native-dense" or self.kv_hot_window <= 0:
                return self._store_compact_cache_torch(tensor, mode=mode, cache_name=cache_name)
            if tensor.shape[1] <= self.kv_hot_window:
                return TorchHotWindowKVArray(cold=None, hot=tensor.to(dtype=self.dtype))
            cold_length = tensor.shape[1] - self.kv_hot_window
            return TorchHotWindowKVArray(
                cold=self._store_compact_cache_torch(
                    tensor[:, :cold_length, :],
                    mode=mode,
                    cache_name=cache_name,
                ),
                hot=tensor[:, cold_length:, :].to(dtype=self.dtype),
            )
        array = np.asarray(values, dtype=np.float32)
        if mode == "native-dense" or self.kv_hot_window <= 0:
            return self._store_compact_cache_numpy(array, mode=mode)
        if array.shape[1] <= self.kv_hot_window:
            return _HotWindowKVArray(cold=None, hot=array)
        cold_length = array.shape[1] - self.kv_hot_window
        return _HotWindowKVArray(
            cold=self._store_compact_cache_numpy(array[:, :cold_length, :], mode=mode),
            hot=array[:, cold_length:, :],
        )

    def _append_layer_cache(self, cache: Any, values: Any, *, mode: str, cache_name: str) -> Any:
        if self.kv_backend == "torch":
            tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, device=self.device, dtype=self.dtype)
            if cache is None:
                return self._store_cache(tensor, mode=mode, cache_name=cache_name)
            if tensor.shape[1] == 1:
                if isinstance(cache, TorchHotWindowKVArray):
                    return cache.append_token(
                        tensor,
                        max_hot=self.kv_hot_window,
                        store_fn=lambda dense: self._store_compact_cache_torch(dense, mode=mode, cache_name=cache_name),
                    )
                if isinstance(cache, (TorchInt8KVArray, Torch4BitKVArray)):
                    return cache.append_compressed(tensor.to(dtype=torch.float32))
                if isinstance(cache, torch.Tensor):
                    return torch.cat([cache, tensor.to(dtype=cache.dtype)], dim=1)
            current = _materialize_cache_torch(cache, dtype=self.dtype, device=self.device)
            combined = torch.cat([current, tensor.to(dtype=current.dtype)], dim=1) if current is not None else tensor
            return self._store_cache(combined, mode=mode, cache_name=cache_name)
        return _append_cache(
            cache,
            values,
            store_dense_fn=lambda dense: self._store_compact_cache_numpy(dense, mode=mode),
            store_fn=lambda dense: self._store_cache(dense, mode=mode, cache_name=cache_name),
            hot_window=self.kv_hot_window,
        )

    def _dequantized_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv_backend == "torch":
            self._sync_pending_async()
            key_tensor = _materialize_cache_torch(self.key_cache, dtype=self.dtype, device=self.device)
            value_tensor = _materialize_cache_torch(self.value_cache, dtype=self.dtype, device=self.device)
            if key_tensor is None or value_tensor is None:
                raise ValueError("cache layer has not been initialized")
            return key_tensor.unsqueeze(0), value_tensor.unsqueeze(0)
        key_array = _materialize_cache(self.key_cache)
        value_array = _materialize_cache(self.value_cache)
        if key_array is None or value_array is None:
            raise ValueError("cache layer has not been initialized")
        key_tensor = torch.from_numpy(np.ascontiguousarray(np.expand_dims(key_array, axis=0))).to(device=self.device, dtype=self.dtype)
        value_tensor = torch.from_numpy(np.ascontiguousarray(np.expand_dims(value_array, axis=0))).to(device=self.device, dtype=self.dtype)
        return key_tensor, value_tensor

    @property
    def kv_cache_bytes(self) -> int:
        self._sync_pending_async()
        return _cache_bytes(self.key_cache) + _cache_bytes(self.value_cache)

    @property
    def kurtosis_profile(self) -> dict[str, Any] | None:
        return None if self._kurtosis_profile is None else dict(self._kurtosis_profile)

    def _supports_async_update(self, *, key_mode: str, value_mode: str, step_length: int) -> bool:
        return (
            self.kv_backend == "torch"
            and self._compress_stream is not None
            and self.device.type == "cuda"
            and step_length > 0
            and key_mode == "turbo-int8"
            and value_mode == "turbo-int8"
            and not self._adaptive_enabled
            and self.kv_rotation_mode == "hadamard"
        )

    def _returned_dense_for_async(self, cache: Any, new_values: torch.Tensor) -> torch.Tensor:
        current = _materialize_cache_torch(cache, dtype=self.dtype, device=self.device)
        if current is None:
            return new_values.to(dtype=self.dtype)
        return torch.cat([current, new_values.to(dtype=current.dtype)], dim=1)

    def _schedule_async_update(self, key_values: torch.Tensor, value_values: torch.Tensor, *, key_mode: str, value_mode: str) -> None:
        if self._compress_stream is None:
            raise ValueError("async compression requested without CUDA stream")
        with torch.cuda.stream(self._compress_stream):
            pending_key_cache = self._append_layer_cache(self.key_cache, key_values, mode=key_mode, cache_name="k")
            pending_value_cache = self._append_layer_cache(self.value_cache, value_values, mode=value_mode, cache_name="v")
        event = torch.cuda.Event()
        event.record(self._compress_stream)
        self._pending_key_cache = pending_key_cache
        self._pending_value_cache = pending_value_cache
        self._pending_event = event
        self._pending_seq_increment = int(key_values.shape[1])

    def finalize_pending(self) -> None:
        self._sync_pending_async()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.kv_backend == "torch":
            self._sync_pending_async()
            key_values = key_states[0].to(device=self.device, dtype=self.dtype)
            value_values = value_states[0].to(device=self.device, dtype=self.dtype)
            if self._adaptive_enabled and self._selected_mode is None:
                self._finalize_adaptive_mode_torch(key_values.to(dtype=torch.float32), value_values.to(dtype=torch.float32))
            key_mode = self._layer_mode_for("k")
            value_mode = self._layer_mode_for("v")
            if self._supports_async_update(key_mode=key_mode, value_mode=value_mode, step_length=int(key_values.shape[1])):
                returned_keys = self._returned_dense_for_async(self.key_cache, key_values).unsqueeze(0)
                returned_values = self._returned_dense_for_async(self.value_cache, value_values).unsqueeze(0)
                self._schedule_async_update(key_values, value_values, key_mode=key_mode, value_mode=value_mode)
                self.seq_length += int(key_values.shape[1])
                return returned_keys, returned_values
            self.key_cache = self._append_layer_cache(self.key_cache, key_values, mode=key_mode, cache_name="k")
            self.value_cache = self._append_layer_cache(self.value_cache, value_values, mode=value_mode, cache_name="v")
            self.seq_length += int(key_values.shape[1])
            return self._dequantized_tensors()
        key_np = key_states.detach().to(dtype=torch.float32, device="cpu").numpy()[0]
        value_np = value_states.detach().to(dtype=torch.float32, device="cpu").numpy()[0]
        if self._adaptive_enabled and self._selected_mode is None:
            self._finalize_adaptive_mode_torch(
                torch.from_numpy(np.ascontiguousarray(key_np)),
                torch.from_numpy(np.ascontiguousarray(value_np)),
            )
        self.key_cache = self._append_layer_cache(self.key_cache, key_np, mode=self._layer_mode_for("k"), cache_name="k")
        self.value_cache = self._append_layer_cache(self.value_cache, value_np, mode=self._layer_mode_for("v"), cache_name="v")
        self.seq_length += int(key_np.shape[1])
        return self._dequantized_tensors()

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = _query_length_from_cache_position(cache_position)
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return int(self.seq_length)

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self.key_cache = None
        self.value_cache = None
        self.seq_length = 0
        self._selected_mode = None if self._adaptive_enabled else self._selected_mode
        self._selected_key_mode = None if self._adaptive_enabled else self._selected_key_mode
        self._selected_value_mode = None if self._adaptive_enabled else self._selected_value_mode
        if self._adaptive_enabled:
            self._kurtosis_profile = None
        self._fourbit_quantizers = {"k": None, "v": None}
        self._pending_event = None
        self._pending_key_cache = None
        self._pending_value_cache = None
        self._pending_seq_increment = 0
        self.is_initialized = False


class TransformersCompressedKVCache(Cache):
    def __init__(
        self,
        model_config: Any,
        *,
        kv_cache_precision: str = "native-dense",
        kv_key_precision: str | None = None,
        kv_value_precision: str | None = None,
        kv_key_scaling_strategy: str | None = None,
        kv_value_scaling_strategy: str | None = None,
        kv_rotation_mode: str = "hadamard",
        kv_hot_window: int = 0,
        kv_quant_seed: int = 7,
        kv_calibration_tokens: int = 128,
        kv_adaptive_high_kurtosis: float = 10.0,
        kv_adaptive_medium_kurtosis: float = 3.0,
        protected_layer_indices: list[int] | tuple[int, ...] | None = None,
        kv_key_fourbit_max_iter: int = _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        kv_value_fourbit_max_iter: int = _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        kv_backend: str = "torch",
        kv_async_compression: bool = True,
    ) -> None:
        super().__init__(layers=[])
        decoder_config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        self.model_config = model_config
        self.num_layers = int(
            getattr(decoder_config, "num_hidden_layers", getattr(decoder_config, "n_layer", 0))
        )
        if self.num_layers <= 0:
            raise ValueError("could not infer number of transformer layers from model config")
        self.decoder_config = decoder_config
        self.kv_cache_precision = _storage_mode_name(kv_cache_precision) or "native-dense"
        self.kv_key_precision = _storage_mode_name(kv_key_precision)
        self.kv_value_precision = _storage_mode_name(kv_value_precision)
        self.kv_key_scaling_strategy = _scaling_strategy_name(kv_key_scaling_strategy)
        self.kv_value_scaling_strategy = _scaling_strategy_name(kv_value_scaling_strategy)
        self.kv_rotation_mode = str(kv_rotation_mode)
        self.kv_hot_window = max(int(kv_hot_window), 0)
        self.kv_quant_seed = int(kv_quant_seed)
        self.kv_calibration_tokens = int(kv_calibration_tokens)
        self.kv_adaptive_high_kurtosis = float(kv_adaptive_high_kurtosis)
        self.kv_adaptive_medium_kurtosis = float(kv_adaptive_medium_kurtosis)
        self.protected_layer_indices = list(
            int(index) for index in (protected_layer_indices or _default_protected_layer_indices(self.num_layers))
        )
        self.kv_key_fourbit_max_iter = int(kv_key_fourbit_max_iter)
        self.kv_value_fourbit_max_iter = int(kv_value_fourbit_max_iter)
        self.kv_backend = str(kv_backend)
        self.kv_async_compression = bool(kv_async_compression)

    def _build_layer(self, layer_idx: int) -> TransformersCompressedKVLayer:
        return TransformersCompressedKVLayer(
            layer_idx=layer_idx,
            num_layers=self.num_layers,
            kv_cache_precision=self.kv_cache_precision,
            kv_key_precision=self.kv_key_precision,
            kv_value_precision=self.kv_value_precision,
            kv_key_scaling_strategy=self.kv_key_scaling_strategy,
            kv_value_scaling_strategy=self.kv_value_scaling_strategy,
            kv_rotation_mode=self.kv_rotation_mode,
            kv_hot_window=self.kv_hot_window,
            kv_quant_seed=self.kv_quant_seed,
            kv_calibration_tokens=self.kv_calibration_tokens,
            kv_adaptive_high_kurtosis=self.kv_adaptive_high_kurtosis,
            kv_adaptive_medium_kurtosis=self.kv_adaptive_medium_kurtosis,
            protected_layer_indices=self.protected_layer_indices,
            kv_key_fourbit_max_iter=self.kv_key_fourbit_max_iter,
            kv_value_fourbit_max_iter=self.kv_value_fourbit_max_iter,
            kv_backend=self.kv_backend,
            kv_async_compression=self.kv_async_compression,
        )

    @property
    def current_kv_mode(self) -> str:
        if self.kv_cache_precision == "adaptive-asymmetric":
            return "adaptive-asymmetric"
        return _public_mode_name(self.kv_cache_precision, rotation_mode=self.kv_rotation_mode)

    @property
    def kv_cache_bytes(self) -> int:
        self.finalize_pending()
        return int(sum(layer.kv_cache_bytes for layer in self.layers if isinstance(layer, TransformersCompressedKVLayer)))

    @property
    def kv_kurtosis_profile(self) -> list[dict[str, Any]] | None:
        profile = [layer.kurtosis_profile for layer in self.layers if isinstance(layer, TransformersCompressedKVLayer)]
        compact = [item for item in profile if item is not None]
        return compact or None

    @property
    def kv_norm_ratio_per_layer(self) -> list[dict[str, Any]] | None:
        profile = self.kv_kurtosis_profile
        if profile is None:
            return None
        return [
            {
                "layer_index": int(item["layer_index"]),
                "k_norm_mean": float(item["k_norm_mean"]),
                "v_norm_mean": float(item["v_norm_mean"]),
                "kv_norm_ratio": float(item["kv_norm_ratio"]),
                "protected_layer": bool(item.get("protected_layer", False)),
            }
            for item in profile
            if "kv_norm_ratio" in item
        ]

    @property
    def layer_mode_counts(self) -> dict[str, int] | None:
        if self.kv_cache_precision != "adaptive":
            return None
        counts = {"native-dense": 0, "turbo-int8": 0, "turbo-4bit": 0}
        for layer in self.layers:
            if not isinstance(layer, TransformersCompressedKVLayer):
                continue
            mode = str(layer._selected_mode or "native-dense")
            if mode in counts:
                counts[mode] += 1
        return counts

    @property
    def layer_kv_mode_counts(self) -> dict[str, int] | None:
        if self.kv_cache_precision != "adaptive-asymmetric":
            return None
        counts: dict[str, int] = {}
        for layer in self.layers:
            if not isinstance(layer, TransformersCompressedKVLayer):
                continue
            key_mode = _public_mode_name(str(layer._selected_key_mode or "native-dense"), rotation_mode=self.kv_rotation_mode)
            value_mode = _public_mode_name(str(layer._selected_value_mode or "native-dense"), rotation_mode=self.kv_rotation_mode)
            pair = f"{key_mode}/{value_mode}"
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @property
    def cache_device(self) -> str | None:
        self.finalize_pending()
        for layer in self.layers:
            if isinstance(layer, TransformersCompressedKVLayer) and hasattr(layer, "device"):
                return str(layer.device)
        return None

    def finalize_pending(self) -> None:
        for layer in self.layers:
            if isinstance(layer, TransformersCompressedKVLayer):
                layer.finalize_pending()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(self._build_layer(len(self.layers)))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def save(self, path: str | Path, *, session_codec: str = "python-npz", audit_policy: str = "blocking") -> Path:
        self.finalize_pending()
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        meta, arrays = _serialize_transformers_cache_payload(self)
        _write_session_payload(destination, meta=meta, arrays=arrays, session_codec=session_codec, audit_policy=audit_policy)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_config: Any | None = None,
        device: str | torch.device | None = None,
        verify_policy: str = "full",
    ) -> "TransformersCompressedKVCache":
        source = Path(path)
        meta, arrays, _ = _read_session_payload(source, verify_policy=verify_policy)
        if str(meta.get("cache_kind")) != "transformers-kv":
            raise ValueError(f"unsupported cache kind in session: {meta.get('cache_kind')!r}")
        if str(meta.get("format")) != "transformers-compressed-kv-v3":
            raise ValueError(f"unsupported compressed HF cache format: {meta.get('format')!r}")
        resolved_device = _device_for_benchmark(None if device is None else str(device))
        resolved_model_config = model_config or _SessionModelConfig(dict(meta.get("model_config") or {}))
        cache = cls(
            resolved_model_config,
            kv_cache_precision=str(meta.get("kv_cache_precision", "native-dense")),
            kv_key_precision=meta.get("kv_key_precision"),
            kv_value_precision=meta.get("kv_value_precision"),
            kv_key_scaling_strategy=meta.get("kv_key_scaling_strategy"),
            kv_value_scaling_strategy=meta.get("kv_value_scaling_strategy"),
            kv_rotation_mode=str(meta.get("kv_rotation_mode", "hadamard")),
            kv_hot_window=int(meta.get("kv_hot_window", 0)),
            kv_quant_seed=int(meta.get("kv_quant_seed", 7)),
            kv_calibration_tokens=int(meta.get("kv_calibration_tokens", 128)),
            kv_adaptive_high_kurtosis=float(meta.get("kv_adaptive_high_kurtosis", 10.0)),
            kv_adaptive_medium_kurtosis=float(meta.get("kv_adaptive_medium_kurtosis", 3.0)),
            protected_layer_indices=meta.get("protected_layer_indices"),
            kv_key_fourbit_max_iter=int(meta.get("kv_key_fourbit_max_iter", _DEFAULT_ONLINE_FOURBIT_MAX_ITER)),
            kv_value_fourbit_max_iter=int(meta.get("kv_value_fourbit_max_iter", _DEFAULT_ONLINE_FOURBIT_MAX_ITER)),
            kv_backend=str(meta.get("kv_backend", "torch")),
            kv_async_compression=False,
        )
        cache.layers = []
        for layer_meta in meta.get("layers", []):
            layer_idx = int(layer_meta.get("layer_index", len(cache.layers)))
            while len(cache.layers) <= layer_idx:
                cache.layers.append(cache._build_layer(len(cache.layers)))
            layer = cache.layers[layer_idx]
            assert isinstance(layer, TransformersCompressedKVLayer)
            dtype_name = str(layer_meta.get("dtype") or "float32")
            num_heads = int(layer_meta.get("num_heads", 0))
            head_dim = int(layer_meta.get("head_dim", 0))
            layer._initialize_runtime_state(
                dtype=_torch_dtype_from_name(dtype_name),
                device=resolved_device,
                batch_size=int(layer_meta.get("batch_size", 1)),
                num_heads=num_heads,
                head_dim=head_dim,
            )
            layer.seq_length = int(layer_meta.get("seq_length", 0))
            layer._selected_mode = layer_meta.get("selected_mode")
            layer._selected_key_mode = layer_meta.get("selected_k_mode")
            layer._selected_value_mode = layer_meta.get("selected_v_mode")
            layer._kurtosis_profile = layer_meta.get("kurtosis_profile")
            key_meta = layer_meta.get("key_cache")
            value_meta = layer_meta.get("value_cache")
            if key_meta is not None:
                layer.key_cache = _deserialize_cache_value(key_meta, arrays, layer=layer, cache_name="k", device=resolved_device)
            if value_meta is not None:
                layer.value_cache = _deserialize_cache_value(value_meta, arrays, layer=layer, cache_name="v", device=resolved_device)
        return cache


class TransformersHybridKVCache(Cache):
    def __init__(
        self,
        model_config: Any,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
        kv_cache_precision: str = "native-dense",
        kv_key_precision: str | None = None,
        kv_value_precision: str | None = None,
        kv_key_scaling_strategy: str | None = None,
        kv_value_scaling_strategy: str | None = None,
        kv_rotation_mode: str = "hadamard",
        kv_hot_window: int = 0,
        kv_quant_seed: int = 7,
        kv_calibration_tokens: int = 128,
        kv_adaptive_high_kurtosis: float = 10.0,
        kv_adaptive_medium_kurtosis: float = 3.0,
        protected_layer_indices: list[int] | tuple[int, ...] | None = None,
        kv_key_fourbit_max_iter: int = _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        kv_value_fourbit_max_iter: int = _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        kv_backend: str = "torch",
        kv_async_compression: bool = True,
        mamba_state_precision: str = "native-dense",
        mamba_state_block_size: int = _DEFAULT_MAMBA_STATE_BLOCK_SIZE,
        mamba_state_scale_floor: float = _DEFAULT_MAMBA_STATE_SCALE_FLOOR,
        mamba_state_clip_threshold_pct: float = _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT,
        mamba_state_rel_rmse_threshold: float = _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD,
        mamba_state_auto_promote: bool = True,
        mamba_receipts_enabled: bool = False,
        mamba_receipts_path: str | Path | None = None,
        mamba_receipt_run_id: str | None = None,
    ) -> None:
        decoder_config = _decoder_model_config(model_config)
        self.model_config = model_config
        self.decoder_config = decoder_config
        self.num_layers = int(getattr(decoder_config, "num_hidden_layers", 0))
        if self.num_layers <= 0:
            raise ValueError("could not infer number of hybrid layers from model config")
        self.layers_block_type = _layers_block_type(decoder_config)
        if len(self.layers_block_type) != self.num_layers:
            raise ValueError("hybrid cache requires explicit layers_block_type metadata")
        self.transformer_layers = [idx for idx, block_type in enumerate(self.layers_block_type) if block_type == "hybrid"]
        if not self.transformer_layers:
            raise ValueError("hybrid cache requires at least one transformer-backed layer")

        self.kv_cache_precision = _storage_mode_name(kv_cache_precision) or "native-dense"
        self.kv_key_precision = _storage_mode_name(kv_key_precision)
        self.kv_value_precision = _storage_mode_name(kv_value_precision)
        self.kv_key_scaling_strategy = _scaling_strategy_name(kv_key_scaling_strategy)
        self.kv_value_scaling_strategy = _scaling_strategy_name(kv_value_scaling_strategy)
        self.kv_rotation_mode = str(kv_rotation_mode)
        self.kv_hot_window = max(int(kv_hot_window), 0)
        self.kv_quant_seed = int(kv_quant_seed)
        self.kv_calibration_tokens = int(kv_calibration_tokens)
        self.kv_adaptive_high_kurtosis = float(kv_adaptive_high_kurtosis)
        self.kv_adaptive_medium_kurtosis = float(kv_adaptive_medium_kurtosis)
        self.protected_layer_indices = list(
            int(index) for index in (protected_layer_indices or _default_protected_layer_indices(self.num_layers))
        )
        self.kv_key_fourbit_max_iter = int(kv_key_fourbit_max_iter)
        self.kv_value_fourbit_max_iter = int(kv_value_fourbit_max_iter)
        self.kv_backend = str(kv_backend)
        self.kv_async_compression = bool(kv_async_compression)
        self.mamba_state_precision = _normalize_mamba_state_precision(mamba_state_precision)
        self.mamba_state_block_size = max(int(mamba_state_block_size), 1)
        self.mamba_state_scale_floor = float(mamba_state_scale_floor)
        self.mamba_state_clip_threshold_pct = float(mamba_state_clip_threshold_pct)
        self.mamba_state_rel_rmse_threshold = float(mamba_state_rel_rmse_threshold)
        self.mamba_state_auto_promote = bool(mamba_state_auto_promote)
        self.mamba_receipts_enabled = bool(mamba_receipts_enabled)
        self.mamba_receipts_path = None if mamba_receipts_path is None else Path(mamba_receipts_path)
        self.mamba_receipt_run_id = str(mamba_receipt_run_id or uuid.uuid4().hex)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = torch.device(device)
        self.has_previous_state = False
        self._compressed_conv_states: dict[int, Any] = {}
        self._compressed_ssm_states: dict[int, Any] = {}
        self._mamba_state_compress_time_ms = 0.0
        self._mamba_state_materialize_time_ms = 0.0
        self._mamba_receipt_prev_hash = "0" * 64
        self._mamba_receipt_step_index = 0
        self._mamba_receipt_count = 0

        layers: list[CacheLayerMixin] = []
        for layer_idx, block_type in enumerate(self.layers_block_type):
            if block_type == "hybrid" and self.kv_cache_precision != "native-dense":
                layers.append(
                    TransformersCompressedKVLayer(
                        layer_idx=layer_idx,
                        num_layers=self.num_layers,
                        kv_cache_precision=self.kv_cache_precision,
                        kv_key_precision=self.kv_key_precision,
                        kv_value_precision=self.kv_value_precision,
                        kv_key_scaling_strategy=self.kv_key_scaling_strategy,
                        kv_value_scaling_strategy=self.kv_value_scaling_strategy,
                        kv_rotation_mode=self.kv_rotation_mode,
                        kv_hot_window=self.kv_hot_window,
                        kv_quant_seed=self.kv_quant_seed,
                        kv_calibration_tokens=self.kv_calibration_tokens,
                        kv_adaptive_high_kurtosis=self.kv_adaptive_high_kurtosis,
                        kv_adaptive_medium_kurtosis=self.kv_adaptive_medium_kurtosis,
                        protected_layer_indices=self.protected_layer_indices,
                        kv_key_fourbit_max_iter=self.kv_key_fourbit_max_iter,
                        kv_value_fourbit_max_iter=self.kv_value_fourbit_max_iter,
                        kv_backend=self.kv_backend,
                        kv_async_compression=self.kv_async_compression,
                    )
                )
            else:
                layers.append(DynamicLayer())
        super().__init__(layers=layers)

        self.intermediate_size = int(getattr(decoder_config, "mamba_expand", 2) * getattr(decoder_config, "hidden_size", 0))
        self.ssm_state_size = int(getattr(decoder_config, "mamba_d_state", 0))
        self.conv_kernel_size = int(getattr(decoder_config, "mamba_d_conv", 0))
        self.n_mamba_heads = int(getattr(decoder_config, "n_mamba_heads", 0))
        self.mamba_head_dim = int(getattr(decoder_config, "mamba_headdim", 0))
        self.mamba_ngroups = int(getattr(decoder_config, "mamba_ngroups", 1))

        conv_width = self.intermediate_size + 2 * self.mamba_ngroups * self.ssm_state_size
        self.conv_states = {
            idx: torch.zeros(
                self.batch_size,
                conv_width,
                self.conv_kernel_size,
                device=self.device,
                dtype=self.dtype,
            )
            for idx in range(self.num_layers)
        }
        self.ssm_states = {
            idx: torch.zeros(
                self.batch_size,
                self.n_mamba_heads,
                self.mamba_head_dim,
                self.ssm_state_size,
                device=self.device,
                dtype=self.dtype,
            )
            for idx in range(self.num_layers)
        }

    @property
    def current_kv_mode(self) -> str:
        if self.kv_cache_precision == "adaptive-asymmetric":
            return "adaptive-asymmetric"
        return _public_mode_name(self.kv_cache_precision, rotation_mode=self.kv_rotation_mode)

    @property
    def kv_cache_bytes(self) -> int:
        self.finalize_pending()
        if self.kv_cache_precision == "native-dense":
            return int(
                sum(
                    int(layer.keys.numel() * layer.keys.element_size()) + int(layer.values.numel() * layer.values.element_size())
                    for idx, layer in enumerate(self.layers)
                    if idx in self.transformer_layers and isinstance(getattr(layer, "keys", None), torch.Tensor)
                    and isinstance(getattr(layer, "values", None), torch.Tensor)
                )
            )
        return int(
            sum(
                layer.kv_cache_bytes
                for idx, layer in enumerate(self.layers)
                if idx in self.transformer_layers and isinstance(layer, TransformersCompressedKVLayer)
            )
        )

    @property
    def mamba_state_bytes(self) -> int:
        total = 0
        seen_layers: set[tuple[str, int]] = set()
        for layer_idx, tensor in self.conv_states.items():
            if isinstance(tensor, torch.Tensor):
                total += int(tensor.numel() * tensor.element_size())
                seen_layers.add(("conv", int(layer_idx)))
        for layer_idx, packed in self._compressed_conv_states.items():
            if ("conv", int(layer_idx)) not in seen_layers:
                total += int(packed.original_bytes)
        for layer_idx, tensor in self.ssm_states.items():
            if isinstance(tensor, torch.Tensor):
                total += int(tensor.numel() * tensor.element_size())
                seen_layers.add(("ssm", int(layer_idx)))
        for layer_idx, packed in self._compressed_ssm_states.items():
            if ("ssm", int(layer_idx)) not in seen_layers:
                total += int(packed.original_bytes)
        return int(total)

    @property
    def mamba_state_runtime_enabled(self) -> bool:
        return self.mamba_state_precision != "native-dense"

    @property
    def mamba_state_runtime_bytes(self) -> int:
        total = 0
        for tensor in list(self.conv_states.values()) + list(self.ssm_states.values()):
            if isinstance(tensor, torch.Tensor):
                total += int(tensor.numel() * tensor.element_size())
        for packed in list(self._compressed_conv_states.values()) + list(self._compressed_ssm_states.values()):
            total += int(packed.compressed_bytes)
        return int(total)

    @property
    def hybrid_total_cache_bytes(self) -> int:
        return int(self.kv_cache_bytes + self.mamba_state_bytes)

    @property
    def hybrid_total_runtime_cache_bytes(self) -> int:
        return int(self.kv_cache_bytes + self.mamba_state_runtime_bytes)

    @property
    def mamba_state_runtime_ratio_vs_native(self) -> float:
        native = self.mamba_state_bytes
        runtime = self.mamba_state_runtime_bytes
        return float(native / runtime) if runtime > 0 else 1.0

    @property
    def mamba_state_fallback_counts(self) -> dict[str, int]:
        counts = {"int4": 0, "int8": 0, "dense": 0}
        for value in list(self._compressed_conv_states.values()) + list(self._compressed_ssm_states.values()):
            precision = _runtime_precision_name(value)
            counts[precision] = counts.get(precision, 0) + 1
        if not any(counts.values()):
            dense_layers = sum(1 for tensor in list(self.conv_states.values()) + list(self.ssm_states.values()) if isinstance(tensor, torch.Tensor))
            counts["dense"] = int(dense_layers)
        return counts

    @property
    def mamba_receipt_count(self) -> int:
        return int(self._mamba_receipt_count)

    @property
    def mamba_state_compress_time_ms(self) -> float:
        return float(self._mamba_state_compress_time_ms)

    @property
    def mamba_state_materialize_time_ms(self) -> float:
        return float(self._mamba_state_materialize_time_ms)

    @property
    def kv_kurtosis_profile(self) -> list[dict[str, Any]] | None:
        if self.kv_cache_precision == "native-dense":
            return None
        profile = [
            layer.kurtosis_profile
            for idx, layer in enumerate(self.layers)
            if idx in self.transformer_layers and isinstance(layer, TransformersCompressedKVLayer)
        ]
        compact = [item for item in profile if item is not None]
        return compact or None

    @property
    def kv_norm_ratio_per_layer(self) -> list[dict[str, Any]] | None:
        profile = self.kv_kurtosis_profile
        if profile is None:
            return None
        return [
            {
                "layer_index": int(item["layer_index"]),
                "k_norm_mean": float(item["k_norm_mean"]),
                "v_norm_mean": float(item["v_norm_mean"]),
                "kv_norm_ratio": float(item["kv_norm_ratio"]),
                "protected_layer": bool(item.get("protected_layer", False)),
            }
            for item in profile
            if "kv_norm_ratio" in item
        ]

    @property
    def layer_mode_counts(self) -> dict[str, int] | None:
        if self.kv_cache_precision != "adaptive":
            return None
        counts = {"native-dense": 0, "turbo-int8": 0, "turbo-4bit": 0}
        for idx, layer in enumerate(self.layers):
            if idx not in self.transformer_layers or not isinstance(layer, TransformersCompressedKVLayer):
                continue
            mode = str(layer._selected_mode or "native-dense")
            if mode in counts:
                counts[mode] += 1
        return counts

    @property
    def layer_kv_mode_counts(self) -> dict[str, int] | None:
        if self.kv_cache_precision != "adaptive-asymmetric":
            return None
        counts: dict[str, int] = {}
        for idx, layer in enumerate(self.layers):
            if idx not in self.transformer_layers or not isinstance(layer, TransformersCompressedKVLayer):
                continue
            key_mode = _public_mode_name(str(layer._selected_key_mode or "native-dense"), rotation_mode=self.kv_rotation_mode)
            value_mode = _public_mode_name(str(layer._selected_value_mode or "native-dense"), rotation_mode=self.kv_rotation_mode)
            pair = f"{key_mode}/{value_mode}"
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @property
    def cache_device(self) -> str | None:
        for layer in self.layers:
            if isinstance(layer, TransformersCompressedKVLayer) and hasattr(layer, "device"):
                return str(layer.device)
            keys = getattr(layer, "keys", None)
            if isinstance(keys, torch.Tensor):
                return str(keys.device)
        return str(self.device)

    def finalize_pending(self) -> None:
        for idx, layer in enumerate(self.layers):
            if idx in self.transformer_layers and isinstance(layer, TransformersCompressedKVLayer):
                layer.finalize_pending()

    def _emit_mamba_receipt(
        self,
        *,
        layer_index: int,
        state_kind: str,
        receipt: dict[str, Any],
    ) -> None:
        if not self.mamba_receipts_enabled:
            return
        payload = {
            "run_id": self.mamba_receipt_run_id,
            "step_index": int(self._mamba_receipt_step_index),
            "token_index": int(self._mamba_receipt_step_index),
            "layer_index": int(layer_index),
            "state_kind": str(state_kind),
            "dense_bytes": int(receipt["dense_bytes"]),
            "compressed_bytes": int(receipt["compressed_bytes"]),
            "ratio": float(receipt["ratio"]),
            "rel_rmse": float(receipt["rel_rmse"]),
            "clip_pct": float(receipt["clip_pct"]),
            "finite_before": bool(receipt["finite_before"]),
            "finite_after": bool(receipt["finite_after"]),
            "fallback_precision": str(receipt["fallback_precision"]),
            "fallback_reason": receipt.get("fallback_reason"),
            "block_count": int(receipt.get("block_count", 0)),
            "int4_block_count": int(receipt.get("int4_block_count", 0)),
            "int8_block_count": int(receipt.get("int8_block_count", 0)),
            "dense_block_count": int(receipt.get("dense_block_count", 0)),
            "promoted_block_count": int(receipt.get("promoted_block_count", 0)),
            "max_abs_value": float(receipt.get("max_abs_value", 0.0)),
            "state_norm": float(receipt.get("state_norm", 0.0)),
            "prev_hash": self._mamba_receipt_prev_hash,
        }
        payload["receipt_hash"] = _sha256_json(payload)
        self._mamba_receipt_prev_hash = str(payload["receipt_hash"])
        self._mamba_receipt_count += 1
        if self.mamba_receipts_path is not None:
            _append_receipt_jsonl(self.mamba_receipts_path, payload)

    def compress_mamba_state_runtime(self) -> None:
        if not self.mamba_state_runtime_enabled or self.mamba_state_precision != "q-mamba-dsq-int4":
            return
        if self._compressed_conv_states or self._compressed_ssm_states:
            return
        start = time.perf_counter()
        compressed_conv: dict[int, Any] = {}
        compressed_ssm: dict[int, Any] = {}
        for layer_idx, tensor in self.conv_states.items():
            if isinstance(tensor, torch.Tensor):
                compressed_value, receipt = _compress_mamba_state_tensor(
                    tensor,
                    block_size=int(self.mamba_state_block_size),
                    scale_floor=float(self.mamba_state_scale_floor),
                    clip_threshold_pct=float(self.mamba_state_clip_threshold_pct),
                    rel_rmse_threshold=float(self.mamba_state_rel_rmse_threshold),
                    auto_promote=bool(self.mamba_state_auto_promote),
                )
                compressed_conv[int(layer_idx)] = compressed_value
                self._emit_mamba_receipt(layer_index=int(layer_idx), state_kind="conv", receipt=receipt)
        for layer_idx, tensor in self.ssm_states.items():
            if isinstance(tensor, torch.Tensor):
                compressed_value, receipt = _compress_mamba_state_tensor(
                    tensor,
                    block_size=int(self.mamba_state_block_size),
                    scale_floor=float(self.mamba_state_scale_floor),
                    clip_threshold_pct=float(self.mamba_state_clip_threshold_pct),
                    rel_rmse_threshold=float(self.mamba_state_rel_rmse_threshold),
                    auto_promote=bool(self.mamba_state_auto_promote),
                )
                compressed_ssm[int(layer_idx)] = compressed_value
                self._emit_mamba_receipt(layer_index=int(layer_idx), state_kind="ssm", receipt=receipt)
        self._compressed_conv_states = compressed_conv
        self._compressed_ssm_states = compressed_ssm
        self.conv_states = {idx: None for idx in range(self.num_layers)}
        self.ssm_states = {idx: None for idx in range(self.num_layers)}
        self._mamba_receipt_step_index += 1
        self._mamba_state_compress_time_ms += (time.perf_counter() - start) * 1000.0

    def materialize_mamba_state_runtime(self) -> None:
        if not self._compressed_conv_states and not self._compressed_ssm_states:
            return
        start = time.perf_counter()
        conv_states: dict[int, torch.Tensor] = {}
        ssm_states: dict[int, torch.Tensor] = {}
        for layer_idx in range(self.num_layers):
            packed_conv = self._compressed_conv_states.get(layer_idx)
            conv_states[layer_idx] = (
                packed_conv.decompress(dtype=self.dtype, device=self.device)
                if packed_conv is not None
                else torch.zeros(
                    self.batch_size,
                    self.intermediate_size + 2 * self.mamba_ngroups * self.ssm_state_size,
                    self.conv_kernel_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            packed_ssm = self._compressed_ssm_states.get(layer_idx)
            ssm_states[layer_idx] = (
                packed_ssm.decompress(dtype=self.dtype, device=self.device)
                if packed_ssm is not None
                else torch.zeros(
                    self.batch_size,
                    self.n_mamba_heads,
                    self.mamba_head_dim,
                    self.ssm_state_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        self.conv_states = conv_states
        self.ssm_states = ssm_states
        self._compressed_conv_states = {}
        self._compressed_ssm_states = {}
        self._mamba_state_materialize_time_ms += (time.perf_counter() - start) * 1000.0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(layer_idx) not in self.transformer_layers:
            return key_states, value_states
        layer = self.layers[int(layer_idx)]
        if isinstance(layer, TransformersCompressedKVLayer):
            return layer.update(key_states, value_states, cache_kwargs)
        return layer.update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        target_idx = self.transformer_layers[0] if layer_idx is None or layer_idx not in self.transformer_layers else int(layer_idx)
        layer = self.layers[target_idx]
        if isinstance(layer, TransformersCompressedKVLayer):
            return int(layer.get_seq_length())
        return int(layer.get_seq_length())

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        kv_offset = 0
        query_length = _query_length_from_cache_position(cache_position)
        kv_length = self.get_seq_length(layer_idx=layer_idx) + query_length
        return int(kv_length), int(kv_offset)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.materialize_mamba_state_runtime()
        for layer in self.layers:
            if isinstance(layer, TransformersCompressedKVLayer):
                layer.finalize_pending()
                key_cache = layer.key_cache
                value_cache = layer.value_cache
                if isinstance(key_cache, torch.Tensor):
                    layer.key_cache = key_cache.index_select(0, beam_idx.to(key_cache.device))
                if isinstance(value_cache, torch.Tensor):
                    layer.value_cache = value_cache.index_select(0, beam_idx.to(value_cache.device))
            else:
                layer.batch_select_indices(beam_idx)
        for layer_idx in range(self.num_layers):
            conv_state = self.conv_states[layer_idx]
            ssm_state = self.ssm_states[layer_idx]
            self.conv_states[layer_idx] = conv_state.index_select(0, beam_idx.to(conv_state.device))
            self.ssm_states[layer_idx] = ssm_state.index_select(0, beam_idx.to(ssm_state.device))

    def reset(self) -> None:
        for idx, layer in enumerate(self.layers):
            if idx in self.transformer_layers and isinstance(layer, TransformersCompressedKVLayer):
                layer.reset()
            else:
                layer.keys = torch.tensor([], dtype=self.dtype, device=self.device)
                layer.values = torch.tensor([], dtype=self.dtype, device=self.device)
                layer.is_initialized = False
        self._compressed_conv_states = {}
        self._compressed_ssm_states = {}
        self._mamba_receipt_prev_hash = "0" * 64
        self._mamba_receipt_step_index = 0
        self._mamba_receipt_count = 0
        for tensor in self.conv_states.values():
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
        for tensor in self.ssm_states.values():
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
        self.has_previous_state = False

    def save(self, path: str | Path, *, session_codec: str = "python-npz", audit_policy: str = "blocking") -> Path:
        self.finalize_pending()
        self.materialize_mamba_state_runtime()
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        meta, arrays = _serialize_transformers_cache_payload(self)
        _write_session_payload(destination, meta=meta, arrays=arrays, session_codec=session_codec, audit_policy=audit_policy)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_config: Any | None = None,
        device: str | torch.device | None = None,
        verify_policy: str = "full",
    ) -> "TransformersHybridKVCache":
        source = Path(path)
        meta, arrays, _ = _read_session_payload(source, verify_policy=verify_policy)
        if str(meta.get("format")) != "transformers-hybrid-cache-v1":
            raise ValueError(f"unsupported hybrid cache format: {meta.get('format')!r}")
        resolved_device = _device_for_benchmark(None if device is None else str(device))
        resolved_model_config = model_config or _SessionModelConfig(dict(meta.get("model_config") or {}))
        batch_size = int(meta.get("batch_size", 1))
        cache = cls(
            resolved_model_config,
            batch_size=batch_size,
            dtype=_torch_dtype_from_name(str(meta.get("dtype") or "float32")),
            device=resolved_device,
            kv_cache_precision=str(meta.get("kv_cache_precision", "native-dense")),
            kv_key_precision=meta.get("kv_key_precision"),
            kv_value_precision=meta.get("kv_value_precision"),
            kv_key_scaling_strategy=meta.get("kv_key_scaling_strategy"),
            kv_value_scaling_strategy=meta.get("kv_value_scaling_strategy"),
            kv_rotation_mode=str(meta.get("kv_rotation_mode", "hadamard")),
            kv_hot_window=int(meta.get("kv_hot_window", 0)),
            kv_quant_seed=int(meta.get("kv_quant_seed", 7)),
            kv_calibration_tokens=int(meta.get("kv_calibration_tokens", 128)),
            kv_adaptive_high_kurtosis=float(meta.get("kv_adaptive_high_kurtosis", 10.0)),
            kv_adaptive_medium_kurtosis=float(meta.get("kv_adaptive_medium_kurtosis", 3.0)),
            protected_layer_indices=meta.get("protected_layer_indices"),
            kv_key_fourbit_max_iter=int(meta.get("kv_key_fourbit_max_iter", _DEFAULT_ONLINE_FOURBIT_MAX_ITER)),
            kv_value_fourbit_max_iter=int(meta.get("kv_value_fourbit_max_iter", _DEFAULT_ONLINE_FOURBIT_MAX_ITER)),
            kv_backend=str(meta.get("kv_backend", "torch")),
            kv_async_compression=False,
            mamba_state_precision=str(meta.get("mamba_state_precision", "native-dense")),
            mamba_state_block_size=int(meta.get("mamba_state_block_size", _DEFAULT_MAMBA_STATE_BLOCK_SIZE)),
            mamba_state_scale_floor=float(meta.get("mamba_state_scale_floor", _DEFAULT_MAMBA_STATE_SCALE_FLOOR)),
            mamba_state_clip_threshold_pct=float(
                meta.get("mamba_state_clip_threshold_pct", _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT)
            ),
            mamba_state_rel_rmse_threshold=float(
                meta.get("mamba_state_rel_rmse_threshold", _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD)
            ),
            mamba_state_auto_promote=bool(meta.get("mamba_state_auto_promote", True)),
        )
        cache.has_previous_state = bool(meta.get("has_previous_state", False))
        for layer_meta in meta.get("layers", []):
            layer_idx = int(layer_meta["layer_index"])
            layer = cache.layers[layer_idx]
            if isinstance(layer, TransformersCompressedKVLayer):
                dtype_name = str(layer_meta.get("dtype") or "float32")
                num_heads = int(layer_meta.get("num_heads", 0))
                head_dim = int(layer_meta.get("head_dim", 0))
                layer._initialize_runtime_state(
                    dtype=_torch_dtype_from_name(dtype_name),
                    device=resolved_device,
                    batch_size=int(layer_meta.get("batch_size", batch_size)),
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
                layer.seq_length = int(layer_meta.get("seq_length", 0))
                layer._selected_mode = layer_meta.get("selected_mode")
                layer._selected_key_mode = layer_meta.get("selected_k_mode")
                layer._selected_value_mode = layer_meta.get("selected_v_mode")
                layer._kurtosis_profile = layer_meta.get("kurtosis_profile")
                if layer_meta.get("key_cache") is not None:
                    layer.key_cache = _deserialize_cache_value(
                        layer_meta["key_cache"],
                        arrays,
                        layer=layer,
                        cache_name="k",
                        device=resolved_device,
                    )
                if layer_meta.get("value_cache") is not None:
                    layer.value_cache = _deserialize_cache_value(
                        layer_meta["value_cache"],
                        arrays,
                        layer=layer,
                        cache_name="v",
                        device=resolved_device,
                    )
            else:
                if layer_meta.get("key_cache") is not None:
                    tensor_info = layer_meta["key_cache"]["tensor"]
                    layer.keys = _tensor_from_numpy(
                        arrays[str(tensor_info["array"])],
                        dtype_name=str(tensor_info["dtype"]),
                        device=resolved_device,
                    )
                    layer.is_initialized = True
                    layer.dtype = layer.keys.dtype
                    layer.device = layer.keys.device
                if layer_meta.get("value_cache") is not None:
                    tensor_info = layer_meta["value_cache"]["tensor"]
                    layer.values = _tensor_from_numpy(
                        arrays[str(tensor_info["array"])],
                        dtype_name=str(tensor_info["dtype"]),
                        device=resolved_device,
                    )
                    layer.is_initialized = True
                    layer.dtype = layer.values.dtype
                    layer.device = layer.values.device
            if layer_meta.get("conv_state") is not None:
                tensor_info = layer_meta["conv_state"]["tensor"]
                cache.conv_states[layer_idx] = _tensor_from_numpy(
                    arrays[str(tensor_info["array"])],
                    dtype_name=str(tensor_info["dtype"]),
                    device=resolved_device,
                )
            if layer_meta.get("ssm_state") is not None:
                tensor_info = layer_meta["ssm_state"]["tensor"]
                cache.ssm_states[layer_idx] = _tensor_from_numpy(
                    arrays[str(tensor_info["array"])],
                    dtype_name=str(tensor_info["dtype"]),
                    device=resolved_device,
                )
        return cache


def _write_session_payload(
    destination: Path,
    *,
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    session_codec: str = "python-npz",
    audit_policy: str = "blocking",
) -> dict[str, Any]:
    from helix_kv import rust_session

    return rust_session.save_session_bundle(destination, meta=meta, arrays=arrays, session_codec=session_codec, audit_policy=audit_policy)


def _read_session_payload(source: Path, *, verify_policy: str = "full") -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from helix_kv import rust_session

    return rust_session.load_session_bundle(source, verify_policy=verify_policy)


def _save_dynamic_cache(
    cache: DynamicCache,
    *,
    model_config: Any,
    path: str | Path,
    session_codec: str = "python-npz",
    audit_policy: str = "blocking",
) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    meta, arrays = _serialize_transformers_cache_payload(cache)
    meta["model_config"] = _session_model_config_dict(model_config)
    _write_session_payload(destination, meta=meta, arrays=arrays, session_codec=session_codec, audit_policy=audit_policy)
    return destination


def _load_dynamic_cache(path: str | Path, *, device: str | torch.device | None = None, verify_policy: str = "full") -> DynamicCache:
    source = Path(path)
    meta, arrays, _ = _read_session_payload(source, verify_policy=verify_policy)
    if str(meta.get("format")) != "transformers-dynamic-cache-v3":
        raise ValueError(f"unsupported dynamic cache format: {meta.get('format')!r}")
    resolved_device = _device_for_benchmark(None if device is None else str(device))
    cache = DynamicCache(config=_SessionModelConfig(dict(meta.get("model_config") or {})))
    for layer_meta in meta.get("layers", []):
        layer = cache.layers[int(layer_meta["layer_index"])]
        if layer_meta.get("key_cache") is not None:
            tensor_info = layer_meta["key_cache"]["tensor"]
            layer.keys = _tensor_from_numpy(
                arrays[str(tensor_info["array"])],
                dtype_name=str(tensor_info["dtype"]),
                device=resolved_device,
            )
        if layer_meta.get("value_cache") is not None:
            tensor_info = layer_meta["value_cache"]["tensor"]
            layer.values = _tensor_from_numpy(
                arrays[str(tensor_info["array"])],
                dtype_name=str(tensor_info["dtype"]),
                device=resolved_device,
            )
    return cache


def _save_benchmark_cache(
    cache: Any,
    *,
    model_config: Any,
    path: str | Path,
    session_codec: str = "python-npz",
    audit_policy: str = "blocking",
) -> Path:
    if isinstance(cache, TransformersHybridKVCache):
        return cache.save(path, session_codec=session_codec, audit_policy=audit_policy)
    if isinstance(cache, TransformersCompressedKVCache):
        return cache.save(path, session_codec=session_codec, audit_policy=audit_policy)
    if isinstance(cache, DynamicCache):
        return _save_dynamic_cache(cache, model_config=model_config, path=path, session_codec=session_codec, audit_policy=audit_policy)
    raise TypeError(f"unsupported cache type for benchmark session save: {type(cache)!r}")


def _load_benchmark_cache(
    path: str | Path,
    *,
    model_config: Any | None,
    device: str | torch.device | None = None,
    verify_policy: str = "full",
) -> Any:
    source = Path(path)
    meta = json.loads((source / "session.json").read_text(encoding="utf-8"))
    if str(meta.get("format")) == "transformers-hybrid-cache-v1":
        return TransformersHybridKVCache.load(source, model_config=model_config, device=device, verify_policy=verify_policy)
    if str(meta.get("format")) == "transformers-compressed-kv-v3":
        return TransformersCompressedKVCache.load(source, model_config=model_config, device=device, verify_policy=verify_policy)
    if str(meta.get("format")) == "transformers-dynamic-cache-v3":
        return _load_dynamic_cache(source, device=device, verify_policy=verify_policy)
    raise ValueError(f"unsupported benchmark session format: {meta.get('format')!r}")


def _variant_display_mode(variant: dict[str, Any]) -> str:
    return _public_mode_name(
        str(variant.get("kv_cache_precision", "native-dense")),
        rotation_mode=str(variant.get("kv_rotation_mode", "hadamard")),
    )


def build_gpu_transformers_variants(
    *,
    kv_quant_seed: int = 7,
    kv_hot_window: int = 4,
    kv_calibration_tokens: int = 128,
    kv_adaptive_medium_kurtosis: float = 9.0,
    kv_adaptive_high_kurtosis: float = 20.0,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "native-dense",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
        },
        {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
        },
        {
            "name": "turbo-int8k-4bitv",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_value_fourbit_max_iter": _DEFAULT_FIXED_FOURBIT_MAX_ITER,
        },
        {
            "name": "adaptive-m9-h20",
            "kv_cache_precision": "adaptive",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_calibration_tokens": int(kv_calibration_tokens),
            "kv_adaptive_medium_kurtosis": float(kv_adaptive_medium_kurtosis),
            "kv_adaptive_high_kurtosis": float(kv_adaptive_high_kurtosis),
        },
    ]


def build_transformers_asymmetry_sweep_variants(
    *,
    kv_quant_seed: int = 7,
    kv_hot_window: int = 4,
    kv_calibration_tokens: int = 128,
    kv_adaptive_medium_kurtosis: float = 9.0,
    kv_adaptive_high_kurtosis: float = 20.0,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "turbo-int8k-4bitv",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_value_fourbit_max_iter": _DEFAULT_FIXED_FOURBIT_MAX_ITER,
        },
        {
            "name": "turbo-4bitk-int8v",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-4bit",
            "kv_value_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
        },
        {
            "name": "turbo-int8k-4bitv-perchannel",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_key_scaling_strategy": "per-channel",
            "kv_value_scaling_strategy": "per-token",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_calibration_tokens": int(kv_calibration_tokens),
            "kv_value_fourbit_max_iter": _DEFAULT_FIXED_FOURBIT_MAX_ITER,
        },
        {
            "name": "turbo-4bit-perchannel",
            "kv_cache_precision": "turbo-4bit",
            "kv_key_scaling_strategy": "per-channel",
            "kv_value_scaling_strategy": "per-token",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_calibration_tokens": int(kv_calibration_tokens),
        },
        {
            "name": "adaptive-asymmetric-m9-h20",
            "kv_cache_precision": "adaptive-asymmetric",
            "kv_key_scaling_strategy": "per-channel",
            "kv_value_scaling_strategy": "per-token",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_calibration_tokens": int(kv_calibration_tokens),
            "kv_adaptive_medium_kurtosis": float(kv_adaptive_medium_kurtosis),
            "kv_adaptive_high_kurtosis": float(kv_adaptive_high_kurtosis),
        },
    ]


def build_transformers_community_variants(
    *,
    kv_quant_seed: int = 7,
    kv_hot_window: int = 4,
    kv_calibration_tokens: int = 128,
    kv_adaptive_medium_kurtosis: float = 9.0,
    kv_adaptive_high_kurtosis: float = 20.0,
    sparse_v_threshold: float = 1e-4,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "native-dense",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
        },
        {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
        },
        {
            "name": "turbo-int8k-4bitv",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_value_fourbit_max_iter": _DEFAULT_FIXED_FOURBIT_MAX_ITER,
        },
        {
            "name": "turbo-int8k-4bitv-online",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_value_fourbit_max_iter": _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
        },
        {
            "name": "helix-optimal",
            "kv_cache_precision": "adaptive-asymmetric",
            "kv_key_scaling_strategy": "per-channel",
            "kv_value_scaling_strategy": "per-token",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "kv_calibration_tokens": int(kv_calibration_tokens),
            "kv_adaptive_medium_kurtosis": float(kv_adaptive_medium_kurtosis),
            "kv_adaptive_high_kurtosis": float(kv_adaptive_high_kurtosis),
            "kv_key_fourbit_max_iter": _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
            "kv_value_fourbit_max_iter": _DEFAULT_ONLINE_FOURBIT_MAX_ITER,
            "kv_sparse_v_threshold": float(sparse_v_threshold),
        },
    ]


def build_transformers_hybrid_state_variants(
    *,
    kv_quant_seed: int = 7,
    kv_hot_window: int = 4,
    mamba_state_block_size: int = _DEFAULT_MAMBA_STATE_BLOCK_SIZE,
    mamba_state_scale_floor: float = _DEFAULT_MAMBA_STATE_SCALE_FLOOR,
    mamba_state_clip_threshold_pct: float = _DEFAULT_MAMBA_STATE_CLIP_THRESHOLD_PCT,
    mamba_state_rel_rmse_threshold: float = _DEFAULT_MAMBA_STATE_REL_RMSE_THRESHOLD,
    mamba_state_auto_promote: bool = True,
    mamba_receipts_enabled: bool = False,
    mamba_receipts_path: str | Path | None = None,
    mamba_receipt_run_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "native-dense",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
            "mamba_state_precision": "native-dense",
        },
        {
            "name": "q-mamba-dsq-int4",
            "kv_cache_precision": "native-dense",
            "kv_rotation_mode": "qr",
            "kv_hot_window": int(kv_hot_window),
            "mamba_state_precision": "q-mamba-dsq-int4",
            "mamba_state_block_size": int(mamba_state_block_size),
            "mamba_state_scale_floor": float(mamba_state_scale_floor),
            "mamba_state_clip_threshold_pct": float(mamba_state_clip_threshold_pct),
            "mamba_state_rel_rmse_threshold": float(mamba_state_rel_rmse_threshold),
            "mamba_state_auto_promote": bool(mamba_state_auto_promote),
            "mamba_receipts_enabled": bool(mamba_receipts_enabled),
            "mamba_receipts_path": None if mamba_receipts_path is None else str(mamba_receipts_path),
            "mamba_receipt_run_id": None if mamba_receipt_run_id is None else f"{mamba_receipt_run_id}:q-mamba-dsq-int4",
        },
        {
            "name": "turbo-int8-hadamard+q-mamba-dsq-int4",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": int(kv_hot_window),
            "kv_quant_seed": int(kv_quant_seed),
            "mamba_state_precision": "q-mamba-dsq-int4",
            "mamba_state_block_size": int(mamba_state_block_size),
            "mamba_state_scale_floor": float(mamba_state_scale_floor),
            "mamba_state_clip_threshold_pct": float(mamba_state_clip_threshold_pct),
            "mamba_state_rel_rmse_threshold": float(mamba_state_rel_rmse_threshold),
            "mamba_state_auto_promote": bool(mamba_state_auto_promote),
            "mamba_receipts_enabled": bool(mamba_receipts_enabled),
            "mamba_receipts_path": None if mamba_receipts_path is None else str(mamba_receipts_path),
            "mamba_receipt_run_id": None
            if mamba_receipt_run_id is None
            else f"{mamba_receipt_run_id}:turbo-int8-hadamard+q-mamba-dsq-int4",
        },
    ]


def _build_prompt_attention_mask(device: torch.device, length: int) -> torch.Tensor:
    return torch.ones((1, int(length)), dtype=torch.long, device=device)


def _query_length_from_cache_position(cache_position: Any) -> int:
    if isinstance(cache_position, torch.Tensor):
        if cache_position.ndim == 0:
            return 1
        return int(cache_position.shape[0])
    if isinstance(cache_position, (int, np.integer)):
        return 1
    if cache_position is None:
        return 1
    try:
        return int(len(cache_position))
    except TypeError:
        return 1


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_runtime_device(cache: Any) -> str | None:
    if isinstance(cache, TransformersHybridKVCache):
        return cache.cache_device
    if isinstance(cache, DynamicCache):
        for layer in getattr(cache, "layers", []):
            keys = getattr(layer, "keys", None)
            if isinstance(keys, torch.Tensor):
                return str(keys.device)
            values = getattr(layer, "values", None)
            if isinstance(values, torch.Tensor):
                return str(values.device)
        return None
    if isinstance(cache, TransformersCompressedKVCache):
        return cache.cache_device
    return None


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, str(dtype_name))
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype name: {dtype_name}") from exc


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": _torch_dtype_name(tensor.dtype),
        "shape": list(tensor.shape),
    }


def _tensor_from_numpy(array: np.ndarray, *, dtype_name: str, device: torch.device) -> torch.Tensor:
    dtype = _torch_dtype_from_name(dtype_name)
    contiguous = np.ascontiguousarray(array)
    tensor = torch.from_numpy(contiguous)
    if dtype == torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def _serialize_cache_value(
    prefix: str,
    value: (
        np.ndarray
        | torch.Tensor
        | TorchInt8KVArray
        | Torch4BitKVArray
        | TorchHotWindowKVArray
        | _TurboInt8KVArray
        | _Turbo4BitKVArray
        | _TurboQJLKVArray
        | _HotWindowKVArray
    ),
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    if isinstance(value, TorchHotWindowKVArray):
        arrays[f"{prefix}_hot"] = _tensor_to_numpy_preserve_bytes(value.hot)
        return {
            "kind": "torch-hot-window",
            "hot": {"array": f"{prefix}_hot", **_tensor_meta(value.hot)},
            "cold": None if value.cold is None else _serialize_cache_value(f"{prefix}_cold", value.cold, arrays),
        }
    if isinstance(value, TorchInt8KVArray):
        arrays[f"{prefix}_q"] = _tensor_to_numpy_preserve_bytes(value.q)
        arrays[f"{prefix}_scales"] = _tensor_to_numpy_preserve_bytes(value.scales)
        return {
            "kind": "torch-int8",
            "scaling_strategy": value.scaling_strategy,
            "q": {"array": f"{prefix}_q", **_tensor_meta(value.q)},
            "scales": {"array": f"{prefix}_scales", **_tensor_meta(value.scales)},
        }
    if isinstance(value, Torch4BitKVArray):
        arrays[f"{prefix}_packed"] = _tensor_to_numpy_preserve_bytes(value.packed)
        arrays[f"{prefix}_scales"] = _tensor_to_numpy_preserve_bytes(value.scales)
        arrays[f"{prefix}_centroids"] = _tensor_to_numpy_preserve_bytes(value.quantizer.centroids)
        return {
            "kind": "torch-4bit",
            "scaling_strategy": value.quantizer.scaling_strategy,
            "packed": {"array": f"{prefix}_packed", **_tensor_meta(value.packed)},
            "scales": {"array": f"{prefix}_scales", **_tensor_meta(value.scales)},
            "centroids": {"array": f"{prefix}_centroids", **_tensor_meta(value.quantizer.centroids)},
        }
    if isinstance(value, torch.Tensor):
        arrays[prefix] = _tensor_to_numpy_preserve_bytes(value)
        return {"kind": "torch-tensor", "tensor": {"array": prefix, **_tensor_meta(value)}}
    if isinstance(value, _HotWindowKVArray):
        arrays[f"{prefix}_hot"] = value.hot.astype(np.float32)
        return {
            "kind": "numpy-hot-window",
            "hot": {"array": f"{prefix}_hot", "dtype": "float32", "shape": list(value.hot.shape)},
            "cold": None if value.cold is None else _serialize_cache_value(f"{prefix}_cold", value.cold, arrays),
        }
    if isinstance(value, _TurboInt8KVArray):
        arrays[f"{prefix}_q"] = value.q
        arrays[f"{prefix}_scales"] = value.scales
        return {
            "kind": "numpy-int8",
            "q": {"array": f"{prefix}_q", "dtype": str(value.q.dtype), "shape": list(value.q.shape)},
            "scales": {"array": f"{prefix}_scales", "dtype": str(value.scales.dtype), "shape": list(value.scales.shape)},
        }
    if isinstance(value, _Turbo4BitKVArray):
        arrays[f"{prefix}_packed"] = value.packed
        arrays[f"{prefix}_norms"] = value.norms
        return {
            "kind": "numpy-4bit",
            "packed": {"array": f"{prefix}_packed", "dtype": str(value.packed.dtype), "shape": list(value.packed.shape)},
            "norms": {"array": f"{prefix}_norms", "dtype": str(value.norms.dtype), "shape": list(value.norms.shape)},
        }
    if isinstance(value, _TurboQJLKVArray):
        arrays[f"{prefix}_packed"] = value.packed
        arrays[f"{prefix}_norms"] = value.norms
        arrays[f"{prefix}_qjl_bits"] = value.qjl_bits
        arrays[f"{prefix}_qjl_norms"] = value.residual_norms
        return {
            "kind": "numpy-qjl",
            "packed": {"array": f"{prefix}_packed", "dtype": str(value.packed.dtype), "shape": list(value.packed.shape)},
            "norms": {"array": f"{prefix}_norms", "dtype": str(value.norms.dtype), "shape": list(value.norms.shape)},
            "qjl_bits": {"array": f"{prefix}_qjl_bits", "dtype": str(value.qjl_bits.dtype), "shape": list(value.qjl_bits.shape)},
            "residual_norms": {
                "array": f"{prefix}_qjl_norms",
                "dtype": str(value.residual_norms.dtype),
                "shape": list(value.residual_norms.shape),
            },
        }
    array = np.asarray(value, dtype=np.float32)
    arrays[prefix] = array
    return {"kind": "numpy-array", "tensor": {"array": prefix, "dtype": str(array.dtype), "shape": list(array.shape)}}


def _session_model_config_dict(model_config: Any) -> dict[str, Any]:
    if hasattr(model_config, "to_dict"):
        return dict(model_config.to_dict())
    return {
        "num_hidden_layers": int(getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", 0))),
        "num_attention_heads": int(getattr(model_config, "num_attention_heads", 0) or 0),
        "num_key_value_heads": int(getattr(model_config, "num_key_value_heads", 0) or 0),
    }


def _serialize_transformers_cache_payload(cache: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any]

    if isinstance(cache, TransformersHybridKVCache):
        cache.finalize_pending()
        cache.materialize_mamba_state_runtime()
        model_config_dict = _session_model_config_dict(cache.model_config)
        layers_meta: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(cache.layers):
            layer_meta: dict[str, Any] = {
                "layer_index": int(layer_index),
                "block_type": str(cache.layers_block_type[layer_index]),
                "key_cache": None,
                "value_cache": None,
                "conv_state": _serialize_cache_value(f"layer_{layer_index}_conv", cache.conv_states[layer_index], arrays),
                "ssm_state": _serialize_cache_value(f"layer_{layer_index}_ssm", cache.ssm_states[layer_index], arrays),
            }
            if isinstance(layer, TransformersCompressedKVLayer):
                layer_meta.update(
                    {
                        "layer_kind": "compressed-kv",
                        "seq_length": int(layer.seq_length),
                        "batch_size": int(getattr(layer, "batch_size", cache.batch_size) or cache.batch_size),
                        "num_heads": int(getattr(layer, "num_heads", 0) or 0),
                        "head_dim": int(getattr(layer, "head_dim", 0) or 0),
                        "dtype": None if not hasattr(layer, "dtype") else _torch_dtype_name(layer.dtype),
                        "selected_mode": layer._selected_mode,
                        "selected_k_mode": layer._selected_key_mode,
                        "selected_v_mode": layer._selected_value_mode,
                        "kurtosis_profile": layer.kurtosis_profile,
                    }
                )
                if layer.key_cache is not None:
                    layer_meta["key_cache"] = _serialize_cache_value(f"layer_{layer_index}_k", layer.key_cache, arrays)
                if layer.value_cache is not None:
                    layer_meta["value_cache"] = _serialize_cache_value(f"layer_{layer_index}_v", layer.value_cache, arrays)
            else:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                layer_meta["layer_kind"] = "dynamic-kv"
                if isinstance(keys, torch.Tensor):
                    layer_meta["key_cache"] = _serialize_cache_value(f"layer_{layer_index}_k", keys, arrays)
                if isinstance(values, torch.Tensor):
                    layer_meta["value_cache"] = _serialize_cache_value(f"layer_{layer_index}_v", values, arrays)
            layers_meta.append(layer_meta)
        meta = {
            "cache_kind": "transformers-kv",
            "format_version": 1,
            "format": "transformers-hybrid-cache-v1",
            "hybrid_model_type": "zamba2",
            "kv_cache_precision": cache.kv_cache_precision,
            "kv_key_precision": cache.kv_key_precision,
            "kv_value_precision": cache.kv_value_precision,
            "kv_key_scaling_strategy": cache.kv_key_scaling_strategy,
            "kv_value_scaling_strategy": cache.kv_value_scaling_strategy,
            "kv_rotation_mode": cache.kv_rotation_mode,
            "kv_hot_window": int(cache.kv_hot_window),
            "kv_quant_seed": int(cache.kv_quant_seed),
            "kv_calibration_tokens": int(cache.kv_calibration_tokens),
            "kv_adaptive_high_kurtosis": float(cache.kv_adaptive_high_kurtosis),
            "kv_adaptive_medium_kurtosis": float(cache.kv_adaptive_medium_kurtosis),
            "protected_layer_indices": list(cache.protected_layer_indices),
            "kv_key_fourbit_max_iter": int(cache.kv_key_fourbit_max_iter),
            "kv_value_fourbit_max_iter": int(cache.kv_value_fourbit_max_iter),
            "kv_backend": cache.kv_backend,
            "mamba_state_precision": cache.mamba_state_precision,
            "mamba_state_block_size": int(cache.mamba_state_block_size),
            "mamba_state_scale_floor": float(cache.mamba_state_scale_floor),
            "mamba_state_clip_threshold_pct": float(cache.mamba_state_clip_threshold_pct),
            "mamba_state_rel_rmse_threshold": float(cache.mamba_state_rel_rmse_threshold),
            "mamba_state_auto_promote": bool(cache.mamba_state_auto_promote),
            "mamba_state_runtime_enabled": bool(cache.mamba_state_runtime_enabled),
            "model_config": model_config_dict,
            "num_layers": int(cache.num_layers),
            "batch_size": int(cache.batch_size),
            "dtype": _torch_dtype_name(cache.dtype),
            "transformer_layers": list(cache.transformer_layers),
            "layers_block_type": list(cache.layers_block_type),
            "has_previous_state": bool(cache.has_previous_state),
            "mamba_state_bytes": int(cache.mamba_state_bytes),
            "mamba_state_runtime_bytes": int(cache.mamba_state_runtime_bytes),
            "layers": layers_meta,
        }
        return meta, arrays

    if isinstance(cache, DynamicCache):
        stats = _dynamic_cache_stats(cache)
        layers_meta: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(getattr(cache, "layers", [])):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            layer_meta: dict[str, Any] = {"layer_index": int(layer_index), "key_cache": None, "value_cache": None}
            if isinstance(keys, torch.Tensor):
                layer_meta["key_cache"] = _serialize_cache_value(f"layer_{layer_index}_k", keys, arrays)
            if isinstance(values, torch.Tensor):
                layer_meta["value_cache"] = _serialize_cache_value(f"layer_{layer_index}_v", values, arrays)
            layers_meta.append(layer_meta)
        meta = {
            "cache_kind": "transformers-kv",
            "format_version": 3,
            "format": "transformers-dynamic-cache-v3",
            "type": "native-dense",
            "native_kv_dtype": stats.get("dtype"),
            "native_element_size_bytes": stats.get("element_size_bytes"),
            "layers": layers_meta,
        }
        return meta, arrays

    if not isinstance(cache, TransformersCompressedKVCache):
        raise TypeError(f"unsupported cache type for serialization: {type(cache)!r}")

    cache.finalize_pending()
    model_config_dict = _session_model_config_dict(cache.model_config)
    meta = {
        "cache_kind": "transformers-kv",
        "format_version": 3,
        "format": "transformers-compressed-kv-v3",
        "kv_cache_precision": cache.kv_cache_precision,
        "kv_key_precision": cache.kv_key_precision,
        "kv_value_precision": cache.kv_value_precision,
        "kv_key_scaling_strategy": cache.kv_key_scaling_strategy,
        "kv_value_scaling_strategy": cache.kv_value_scaling_strategy,
        "kv_rotation_mode": cache.kv_rotation_mode,
        "kv_hot_window": int(cache.kv_hot_window),
        "kv_quant_seed": int(cache.kv_quant_seed),
        "kv_calibration_tokens": int(cache.kv_calibration_tokens),
        "kv_adaptive_high_kurtosis": float(cache.kv_adaptive_high_kurtosis),
        "kv_adaptive_medium_kurtosis": float(cache.kv_adaptive_medium_kurtosis),
        "protected_layer_indices": list(cache.protected_layer_indices),
        "kv_key_fourbit_max_iter": int(cache.kv_key_fourbit_max_iter),
        "kv_value_fourbit_max_iter": int(cache.kv_value_fourbit_max_iter),
        "kv_backend": cache.kv_backend,
        "layer_mode_counts": cache.layer_mode_counts,
        "layer_kv_mode_counts": cache.layer_kv_mode_counts,
        "kv_kurtosis_profile": cache.kv_kurtosis_profile,
        "kv_norm_ratio_per_layer": cache.kv_norm_ratio_per_layer,
        "stores_hot_window_exact_tail": bool(cache.kv_hot_window > 0),
        "model_config": model_config_dict,
        "num_layers": int(cache.num_layers),
        "layers": [],
    }
    for layer_index, layer in enumerate(cache.layers):
        if not isinstance(layer, TransformersCompressedKVLayer):
            continue
        layer_meta = {
            "layer_index": int(layer_index),
            "seq_length": int(layer.seq_length),
            "batch_size": int(getattr(layer, "batch_size", 1) or 1),
            "num_heads": int(getattr(layer, "num_heads", 0) or 0),
            "head_dim": int(getattr(layer, "head_dim", 0) or 0),
            "dtype": None if not hasattr(layer, "dtype") else _torch_dtype_name(layer.dtype),
            "selected_mode": layer._selected_mode,
            "selected_k_mode": layer._selected_key_mode,
            "selected_v_mode": layer._selected_value_mode,
            "kurtosis_profile": layer.kurtosis_profile,
            "key_cache": None,
            "value_cache": None,
        }
        if layer.key_cache is not None:
            layer_meta["key_cache"] = _serialize_cache_value(f"layer_{layer_index}_k", layer.key_cache, arrays)
        if layer.value_cache is not None:
            layer_meta["value_cache"] = _serialize_cache_value(f"layer_{layer_index}_v", layer.value_cache, arrays)
        meta["layers"].append(layer_meta)
    return meta, arrays


def _serialize_transformers_cache_bytes(cache: Any) -> dict[str, int]:
    meta, arrays = _serialize_transformers_cache_payload(cache)
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    npz_bytes = buffer.getvalue()
    return {
        "session_meta_bytes": int(len(meta_bytes)),
        "session_npz_bytes": int(len(npz_bytes)),
        "session_total_bytes": int(len(meta_bytes) + len(npz_bytes)),
    }


class _SessionTextConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)
        for key, value in values.items():
            setattr(self, key, value)


class _SessionModelConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        normalized = dict(values)
        if "num_hidden_layers" not in normalized and "n_layer" in normalized:
            normalized["num_hidden_layers"] = int(normalized["n_layer"])
        if "num_attention_heads" not in normalized and "n_head" in normalized:
            normalized["num_attention_heads"] = int(normalized["n_head"])
        if "num_key_value_heads" not in normalized:
            normalized["num_key_value_heads"] = int(
                normalized.get("num_attention_heads", normalized.get("n_head", 0)) or 0
            )
        self._values = normalized
        self._text_config = _SessionTextConfig(normalized)
        for key, value in normalized.items():
            setattr(self, key, value)

    def get_text_config(self, decoder: bool = True) -> _SessionTextConfig:  # noqa: ARG002
        return self._text_config

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)


def _deserialize_cache_value(
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    layer: TransformersCompressedKVLayer,
    cache_name: str,
    device: torch.device,
) -> Any:
    kind = str(meta["kind"])
    if kind == "torch-hot-window":
        hot_info = meta["hot"]
        hot = _tensor_from_numpy(arrays[str(hot_info["array"])], dtype_name=str(hot_info["dtype"]), device=device)
        cold_meta = meta.get("cold")
        cold = None if cold_meta is None else _deserialize_cache_value(cold_meta, arrays, layer=layer, cache_name=cache_name, device=device)
        return TorchHotWindowKVArray(cold=cold, hot=hot)
    if kind == "torch-int8":
        q_info = meta["q"]
        scales_info = meta["scales"]
        q = _tensor_from_numpy(arrays[str(q_info["array"])], dtype_name=str(q_info["dtype"]), device=device)
        scales = _tensor_from_numpy(arrays[str(scales_info["array"])], dtype_name=str(scales_info["dtype"]), device=device)
        return TorchInt8KVArray(
            q=q,
            scales=scales,
            rotation=layer._rotation,
            scaling_strategy=str(meta.get("scaling_strategy", "per-token")),
        )
    if kind == "torch-4bit":
        packed_info = meta["packed"]
        scales_info = meta["scales"]
        centroids_info = meta["centroids"]
        packed = _tensor_from_numpy(arrays[str(packed_info["array"])], dtype_name=str(packed_info["dtype"]), device=device)
        scales = _tensor_from_numpy(arrays[str(scales_info["array"])], dtype_name=str(scales_info["dtype"]), device=device)
        centroids = _tensor_from_numpy(arrays[str(centroids_info["array"])], dtype_name=str(centroids_info["dtype"]), device=device)
        scaling_strategy = str(meta.get("scaling_strategy", "per-channel"))
        quantizer = Torch4BitQuantizer(
            centroids=centroids.to(dtype=torch.float32),
            rotation=layer._rotation,
            scaling_strategy=scaling_strategy,
            reference_scales=scales if scaling_strategy == "per-channel" else None,
            max_iter=layer._fourbit_max_iter_for(cache_name),
        )
        layer._fourbit_quantizers[cache_name] = quantizer
        return Torch4BitKVArray(packed=packed, scales=scales, quantizer=quantizer)
    if kind == "torch-tensor":
        tensor_info = meta["tensor"]
        return _tensor_from_numpy(arrays[str(tensor_info["array"])], dtype_name=str(tensor_info["dtype"]), device=device)
    if kind == "numpy-hot-window":
        hot_info = meta["hot"]
        hot = np.asarray(arrays[str(hot_info["array"])], dtype=np.float32)
        cold_meta = meta.get("cold")
        cold = None if cold_meta is None else _deserialize_cache_value(cold_meta, arrays, layer=layer, cache_name=cache_name, device=device)
        return _HotWindowKVArray(cold=cold, hot=hot)
    if kind == "numpy-int8":
        q = np.asarray(arrays[str(meta["q"]["array"])], dtype=np.int8)
        scales = np.asarray(arrays[str(meta["scales"]["array"])], dtype=np.float16)
        return _TurboInt8KVArray.from_quantized(q, scales, rotation=layer._rotation)
    if kind == "numpy-4bit":
        packed = np.asarray(arrays[str(meta["packed"]["array"])], dtype=np.uint8)
        norms = np.asarray(arrays[str(meta["norms"]["array"])], dtype=np.float16)
        return _Turbo4BitKVArray.from_quantized(packed, norms, rotation=layer._rotation, codebook=layer._initial_codebook)
    if kind == "numpy-qjl":
        packed = np.asarray(arrays[str(meta["packed"]["array"])], dtype=np.uint8)
        norms = np.asarray(arrays[str(meta["norms"]["array"])], dtype=np.float16)
        qjl_bits = np.asarray(arrays[str(meta["qjl_bits"]["array"])], dtype=np.uint8)
        residual_norms = np.asarray(arrays[str(meta["residual_norms"]["array"])], dtype=np.float16)
        return _TurboQJLKVArray.from_quantized(
            packed,
            norms,
            qjl_bits,
            residual_norms,
            rotation=layer._rotation,
            codebook=layer._initial_codebook,
            qjl_matrix=layer._qjl_matrix,
        )
    if kind == "numpy-array":
        return np.asarray(arrays[str(meta["tensor"]["array"])], dtype=np.float32)
    raise ValueError(f"unsupported serialized cache kind: {kind}")


def _run_transformers_variant(
    model: Any,
    *,
    prompt_inputs: dict[str, torch.Tensor],
    prompt_ids: list[int],
    variant: dict[str, Any],
    max_new_tokens: int,
    warmup_prompt_inputs: dict[str, torch.Tensor] | None = None,
    warmup_max_new_tokens: int = _DEFAULT_WARMUP_MAX_NEW_TOKENS,
    kv_backend: str = "torch",
) -> dict[str, Any]:
    input_ids = prompt_inputs["input_ids"]
    device = input_ids.device
    supports_attention_mask = _supports_forward_arg(model, "attention_mask")
    supports_cache_position = _supports_forward_arg(model, "cache_position")
    supports_output_attentions = _supports_forward_arg(model, "output_attentions")
    model_device = next(model.parameters()).device
    sparse_v_threshold = variant.get("kv_sparse_v_threshold")
    sparse_v_probe_enabled = sparse_v_threshold is not None and supports_output_attentions
    is_hybrid_model = _is_zamba2_hybrid_model_config(model.config)

    def _is_compressed_cache(cache: Any) -> bool:
        return isinstance(cache, (TransformersCompressedKVCache, TransformersHybridKVCache)) and getattr(
            cache, "kv_cache_precision", "native-dense"
        ) != "native-dense"

    def _build_cache(batch_size: int) -> tuple[Any, str]:
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
            "kv_async_compression": bool(variant.get("kv_async_compression", True)),
        }
        if is_hybrid_model:
            built = TransformersHybridKVCache(
                model.config,
                batch_size=int(batch_size),
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
            return built, built.current_kv_mode
        if cache_precision == "native-dense":
            return DynamicCache(config=model.config), "native-dense"
        built = TransformersCompressedKVCache(model.config, **common_kwargs)
        return built, built.current_kv_mode

    def _execute_once(
        run_prompt_inputs: dict[str, torch.Tensor],
        *,
        run_max_new_tokens: int,
    ) -> tuple[dict[str, Any], Any]:
        run_input_ids = run_prompt_inputs["input_ids"]
        attention_mask = run_prompt_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = _build_prompt_attention_mask(device, run_input_ids.shape[1])
        cache, current_mode = _build_cache(int(run_input_ids.shape[0]))
        model_inputs: dict[str, Any] = {}
        for key, value in run_prompt_inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "attention_mask" and not supports_attention_mask:
                continue
            if key != "input_ids" and not _supports_forward_arg(model, key):
                continue
            model_inputs[key] = value
        model_inputs["past_key_values"] = cache
        model_inputs["use_cache"] = True
        model_inputs["return_dict"] = True
        if supports_attention_mask:
            model_inputs["attention_mask"] = attention_mask
        if supports_cache_position:
            model_inputs["cache_position"] = torch.arange(run_input_ids.shape[1], device=device)
        if sparse_v_probe_enabled:
            model_inputs["output_attentions"] = True

        with torch.inference_mode():
            _sync_device(device)
            total_start = time.perf_counter()
            if isinstance(cache, TransformersHybridKVCache):
                cache.materialize_mamba_state_runtime()
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
            generated_ids = list(prompt_ids)
            step_times_ms: list[float] = []
            sparse_v_ratios: list[float] = []
            initial_sparse_ratio = _sparse_v_skip_ratio(getattr(outputs, "attentions", None), threshold=sparse_v_threshold)
            if initial_sparse_ratio is not None:
                sparse_v_ratios.append(initial_sparse_ratio)
            for _ in range(int(run_max_new_tokens)):
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                generated_ids.append(int(next_token.item()))
                _sync_device(device)
                step_start = time.perf_counter()
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
                if sparse_v_probe_enabled:
                    next_inputs["output_attentions"] = True
                outputs = model(**next_inputs)
                _sync_device(device)
                cache = getattr(outputs, "past_key_values", cache)
                if isinstance(cache, TransformersHybridKVCache):
                    cache.compress_mamba_state_runtime()
                last_logits = outputs.logits[:, -1, :]
                sparse_ratio = _sparse_v_skip_ratio(getattr(outputs, "attentions", None), threshold=sparse_v_threshold)
                if sparse_ratio is not None:
                    sparse_v_ratios.append(sparse_ratio)
                step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            total_time_s = time.perf_counter() - total_start
        return (
            {
                "generated_ids": list(generated_ids),
                "prompt_perplexity": float(math.exp(float(prompt_loss.item()))),
                "total_time_s": float(total_time_s),
                "avg_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
                "tokens_per_second": float(run_max_new_tokens) / float(total_time_s) if total_time_s > 0 else 0.0,
                "last_logits": last_logits[0].detach().to(dtype=torch.float32, device="cpu").numpy(),
                "current_kv_mode": current_mode,
                "kv_kurtosis_profile": None
                if not _is_compressed_cache(cache)
                else cache.kv_kurtosis_profile,
                "layer_mode_counts": None
                if not _is_compressed_cache(cache)
                else cache.layer_mode_counts,
                "layer_kv_mode_counts": None
                if not _is_compressed_cache(cache)
                else cache.layer_kv_mode_counts,
                "kv_norm_ratio_per_layer": None
                if not _is_compressed_cache(cache)
                else cache.kv_norm_ratio_per_layer,
                "protected_layer_indices": None
                if not _is_compressed_cache(cache)
                else list(cache.protected_layer_indices),
                "sparse_v_skip_ratio": None if not sparse_v_ratios else float(np.mean(sparse_v_ratios)),
                "mamba_state_precision": None
                if not isinstance(cache, TransformersHybridKVCache)
                else str(cache.mamba_state_precision),
                "mamba_state_runtime_enabled": None
                if not isinstance(cache, TransformersHybridKVCache)
                else bool(cache.mamba_state_runtime_enabled),
                "mamba_state_bytes": None if not isinstance(cache, TransformersHybridKVCache) else int(cache.mamba_state_bytes),
                "mamba_state_runtime_bytes": None
                if not isinstance(cache, TransformersHybridKVCache)
                else int(cache.mamba_state_runtime_bytes),
                "mamba_state_runtime_ratio_vs_native": None
                if not isinstance(cache, TransformersHybridKVCache)
                else float(cache.mamba_state_runtime_ratio_vs_native),
                "mamba_state_compress_time_ms": None
                if not isinstance(cache, TransformersHybridKVCache)
                else float(cache.mamba_state_compress_time_ms),
                "mamba_state_materialize_time_ms": None
                if not isinstance(cache, TransformersHybridKVCache)
                else float(cache.mamba_state_materialize_time_ms),
                "mamba_state_fallback_counts": None
                if not isinstance(cache, TransformersHybridKVCache)
                else dict(cache.mamba_state_fallback_counts),
                "mamba_receipt_count": None
                if not isinstance(cache, TransformersHybridKVCache)
                else int(cache.mamba_receipt_count),
                "hybrid_total_cache_bytes": None
                if not isinstance(cache, TransformersHybridKVCache)
                else int(cache.hybrid_total_cache_bytes),
                "hybrid_total_runtime_cache_bytes": None
                if not isinstance(cache, TransformersHybridKVCache)
                else int(cache.hybrid_total_runtime_cache_bytes),
                "mamba_state_compression_probe": None,
                "transformer_layer_indices": None
                if not isinstance(cache, TransformersHybridKVCache)
                else list(cache.transformer_layers),
                "layers_block_type": None
                if not isinstance(cache, TransformersHybridKVCache)
                else list(cache.layers_block_type),
            },
            cache,
        )

    if warmup_prompt_inputs is not None and int(warmup_max_new_tokens) > 0:
        _execute_once(warmup_prompt_inputs, run_max_new_tokens=int(warmup_max_new_tokens))
        _sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run, cache = _execute_once(prompt_inputs, run_max_new_tokens=int(max_new_tokens))
    _sync_device(device)
    if isinstance(cache, TransformersHybridKVCache):
        cache.materialize_mamba_state_runtime()
        run["mamba_state_compression_probe"] = _probe_mamba_state_compression(cache)

    kv_cache_bytes = (
        int(cache.kv_cache_bytes)
        if isinstance(cache, TransformersHybridKVCache)
        else _torch_cache_bytes(cache)
        if isinstance(cache, DynamicCache)
        else cache.kv_cache_bytes
    )
    session_bytes = _serialize_transformers_cache_bytes(cache)
    with tempfile.TemporaryDirectory(prefix="helix-transformers-session-") as temp_dir:
        session_path = Path(temp_dir) / "session"
        save_start = time.perf_counter()
        _save_benchmark_cache(cache, model_config=model.config, path=session_path)
        session_save_time_ms = (time.perf_counter() - save_start) * 1000.0
        load_start = time.perf_counter()
        _ = _load_benchmark_cache(session_path, model_config=model.config, device=device)
        session_load_time_ms = (time.perf_counter() - load_start) * 1000.0
    gpu_peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else None
    )
    native_stats = (
        {}
        if isinstance(cache, TransformersHybridKVCache)
        else _dynamic_cache_stats(cache)
        if isinstance(cache, DynamicCache)
        else {}
    )
    return {
        "generated_ids": list(run["generated_ids"]),
        "prompt_perplexity": float(run["prompt_perplexity"]),
        "total_time_s": float(run["total_time_s"]),
        "avg_step_ms": float(run["avg_step_ms"]),
        "tokens_per_second": float(run["tokens_per_second"]),
        "kv_cache_bytes": int(kv_cache_bytes),
        "session_total_bytes": int(session_bytes["session_total_bytes"]),
        "session_meta_bytes": int(session_bytes["session_meta_bytes"]),
        "session_npz_bytes": int(session_bytes["session_npz_bytes"]),
        "session_save_time_ms": float(session_save_time_ms),
        "session_load_time_ms": float(session_load_time_ms),
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "last_logits": run["last_logits"],
        "current_kv_mode": run["current_kv_mode"],
        "kv_kurtosis_profile": run.get("kv_kurtosis_profile"),
        "layer_mode_counts": run.get("layer_mode_counts"),
        "layer_kv_mode_counts": run.get("layer_kv_mode_counts"),
        "kv_norm_ratio_per_layer": run.get("kv_norm_ratio_per_layer"),
        "protected_layer_indices": run.get("protected_layer_indices"),
        "sparse_v_skip_ratio": run.get("sparse_v_skip_ratio"),
        "mamba_state_precision": run.get("mamba_state_precision"),
        "mamba_state_runtime_enabled": run.get("mamba_state_runtime_enabled"),
        "mamba_state_bytes": run.get("mamba_state_bytes"),
        "mamba_state_runtime_bytes": run.get("mamba_state_runtime_bytes"),
        "mamba_state_runtime_ratio_vs_native": run.get("mamba_state_runtime_ratio_vs_native"),
        "mamba_state_compress_time_ms": run.get("mamba_state_compress_time_ms"),
        "mamba_state_materialize_time_ms": run.get("mamba_state_materialize_time_ms"),
        "mamba_state_fallback_counts": run.get("mamba_state_fallback_counts"),
        "mamba_receipt_count": run.get("mamba_receipt_count"),
        "hybrid_total_cache_bytes": run.get("hybrid_total_cache_bytes"),
        "hybrid_total_runtime_cache_bytes": run.get("hybrid_total_runtime_cache_bytes"),
        "mamba_state_compression_probe": run.get("mamba_state_compression_probe"),
        "transformer_layer_indices": run.get("transformer_layer_indices"),
        "layers_block_type": run.get("layers_block_type"),
        "model_device": str(model_device),
        "cache_device": _cache_runtime_device(cache),
        "native_kv_dtype": native_stats.get("dtype"),
        "native_element_size_bytes": native_stats.get("element_size_bytes"),
    }


def run_transformers_model_diagnostics(
    model_ref: str | Path,
    *,
    prompt_ids: list[int] | None = None,
    prompt_text: str | None = None,
    prompt_length: int = 16,
    local_files_only: bool = False,
    device: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    resolved_device = _device_for_benchmark(device)
    requested_model_ref = str(model_ref)
    effective_model_ref = _canonical_model_ref(model_ref)
    is_hxq_compressed = _is_hxq_model_ref(effective_model_ref)
    input_adapter = "tokenizer-causal"
    processor_used = False
    chat_template_used = False
    hxq_primary_error: str | None = None
    adapter: Any | None = None
    logs_buffer = io.StringIO()

    def _load_model_and_adapter() -> tuple[Any, Any | None, str, bool, bool]:
        if is_hxq_compressed:
            _ensure_hxq_hf_integration_registered()
        loaded_model = _load_causal_model(
            effective_model_ref,
            local_files_only=local_files_only,
            trust_remote_code=bool(trust_remote_code),
        )
        loaded_adapter, adapter_name, adapter_uses_processor, adapter_uses_chat_template = _load_text_adapter(
            effective_model_ref,
            local_files_only=local_files_only,
            trust_remote_code=bool(trust_remote_code),
        )
        return loaded_model, loaded_adapter, adapter_name, adapter_uses_processor, adapter_uses_chat_template

    model: Any | None = None
    load_error: str | None = None
    forward_error: str | None = None
    with contextlib.redirect_stdout(logs_buffer), contextlib.redirect_stderr(logs_buffer):
        try:
            if is_hxq_compressed:
                try:
                    with _prefer_installed_substrate():
                        model, adapter, input_adapter, processor_used, chat_template_used = _load_model_and_adapter()
                except Exception as primary_error:
                    hxq_primary_error = repr(primary_error)
                    with _prefer_repo_substrate():
                        model, adapter, input_adapter, processor_used, chat_template_used = _load_model_and_adapter()
            else:
                model, adapter, input_adapter, processor_used, chat_template_used = _load_model_and_adapter()
            model = model.to(resolved_device)
            model.eval()
        except Exception as exc:
            load_error = repr(exc)

    prompt_ids_resolved: list[int] = []
    logits_finite: bool | None = None
    nan_count: int | None = None
    inf_count: int | None = None
    logits_shape: list[int] | None = None
    max_abs_logit: float | None = None
    model_dtype: str | None = None
    model_device: str | None = None
    quantization_config: Any = None
    if model is not None and load_error is None:
        try:
            prompt_inputs_cpu, prompt_ids_resolved = _resolve_prompt_inputs(
                effective_model_ref,
                adapter=adapter,
                input_adapter=input_adapter,
                prompt_ids=prompt_ids,
                prompt_text=prompt_text,
                prompt_length=int(prompt_length),
                local_files_only=local_files_only,
                trust_remote_code=bool(trust_remote_code),
            )
            prompt_inputs = {
                key: value.to(device=resolved_device) if isinstance(value, torch.Tensor) else value
                for key, value in prompt_inputs_cpu.items()
            }
            model_inputs: dict[str, Any] = {
                "input_ids": prompt_inputs["input_ids"],
                "use_cache": True,
                "return_dict": True,
            }
            if "attention_mask" in prompt_inputs and _supports_forward_arg(model, "attention_mask"):
                model_inputs["attention_mask"] = prompt_inputs["attention_mask"]
            if _supports_forward_arg(model, "cache_position"):
                model_inputs["cache_position"] = torch.arange(prompt_inputs["input_ids"].shape[1], device=resolved_device)
            with contextlib.redirect_stdout(logs_buffer), contextlib.redirect_stderr(logs_buffer), torch.inference_mode():
                outputs = model(**model_inputs)
            logits = outputs.logits.detach()
            logits_shape = [int(dim) for dim in logits.shape]
            logits_finite = bool(torch.isfinite(logits).all().item())
            nan_count = int(torch.isnan(logits).sum().item())
            inf_count = int(torch.isinf(logits).sum().item())
            max_abs_logit = float(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
            model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
            model_device = str(next(model.parameters()).device)
            quantization_config = getattr(model, "quantization_config", None)
            if quantization_config is None:
                quantization_config = getattr(model.config, "quantization_config", None)
        except Exception as exc:
            forward_error = repr(exc)
            try:
                model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
                model_device = str(next(model.parameters()).device)
            except Exception:
                pass

    quantization_payload = quantization_config
    if hasattr(quantization_payload, "to_dict"):
        quantization_payload = quantization_payload.to_dict()
    elif quantization_payload is not None and not isinstance(quantization_payload, (dict, list, str, int, float, bool)):
        quantization_payload = repr(quantization_payload)

    return {
        "requested_model_ref": requested_model_ref,
        "effective_model_ref": effective_model_ref,
        "is_hxq_compressed": bool(is_hxq_compressed),
        "trust_remote_code_used": bool(trust_remote_code),
        "weight_runtime_source": _weight_runtime_source_for_model(effective_model_ref),
        "hxq_primary_error": hxq_primary_error,
        "input_adapter": input_adapter,
        "processor_used": bool(processor_used),
        "chat_template_used": bool(chat_template_used),
        "prompt_ids": list(prompt_ids_resolved),
        "prompt_length": int(len(prompt_ids_resolved)),
        "load_error": load_error,
        "forward_error": forward_error,
        "logits_finite": logits_finite,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "logits_shape": logits_shape,
        "max_abs_logit": max_abs_logit,
        "model_dtype": model_dtype,
        "model_device": model_device,
        "quantization_config": quantization_payload,
        "captured_logs": logs_buffer.getvalue(),
    }


def run_transformers_kv_benchmark(
    model_ref: str | Path,
    *,
    prompt_ids: list[int] | None = None,
    prompt_text: str | None = None,
    prompt_length: int = 128,
    max_new_tokens: int = 16,
    kv_variants: list[dict[str, Any]] | None = None,
    kv_quant_seed: int = 7,
    kv_hot_window: int = 4,
    kv_calibration_tokens: int = 128,
    kv_adaptive_high_kurtosis: float = 20.0,
    kv_adaptive_medium_kurtosis: float = 9.0,
    local_files_only: bool = False,
    device: str | None = None,
    kv_backend: str = "torch",
    trust_remote_code: bool = False,
    warmup_max_new_tokens: int | None = None,
) -> dict[str, Any]:
    resolved_device = _device_for_benchmark(device)
    normalized_model_ref = _normalize_model_ref(model_ref)
    is_hxq_compressed = _is_hxq_model_ref(model_ref)
    gated_model = _is_gated_model_ref(model_ref)
    hf_auth_required = gated_model
    input_adapter = "tokenizer-causal"
    processor_used = False
    chat_template_used = False
    adapter: Any | None = None
    hxq_primary_error: str | None = None

    def _load_model_and_adapter() -> tuple[Any, Any | None, str, bool, bool]:
        if is_hxq_compressed:
            _ensure_hxq_hf_integration_registered()
        loaded_model = _load_causal_model(
            model_ref,
            local_files_only=local_files_only,
            trust_remote_code=bool(trust_remote_code),
        )
        loaded_adapter, adapter_name, adapter_uses_processor, adapter_uses_chat_template = _load_text_adapter(
            model_ref,
            local_files_only=local_files_only,
            trust_remote_code=bool(trust_remote_code),
        )
        return loaded_model, loaded_adapter, adapter_name, adapter_uses_processor, adapter_uses_chat_template

    if is_hxq_compressed:
        try:
            with _prefer_installed_substrate():
                model, adapter, input_adapter, processor_used, chat_template_used = _load_model_and_adapter()
        except Exception as primary_error:
            hxq_primary_error = repr(primary_error)
            with _prefer_repo_substrate():
                model, adapter, input_adapter, processor_used, chat_template_used = _load_model_and_adapter()
    else:
        model, adapter, input_adapter, processor_used, chat_template_used = _load_model_and_adapter()

    model = model.to(resolved_device)
    model.eval()

    prompt_inputs_cpu, resolved_prompt_ids = _resolve_prompt_inputs(
        model_ref,
        adapter=adapter,
        input_adapter=input_adapter,
        prompt_ids=prompt_ids,
        prompt_text=prompt_text,
        prompt_length=int(prompt_length),
        local_files_only=local_files_only,
        trust_remote_code=bool(trust_remote_code),
    )
    prompt_inputs = {
        key: value.to(device=resolved_device) if isinstance(value, torch.Tensor) else value
        for key, value in prompt_inputs_cpu.items()
    }
    prompt_input_ids = prompt_inputs["input_ids"]
    warmup_prompt_length = int(min(int(_DEFAULT_WARMUP_PROMPT_LENGTH), prompt_input_ids.shape[1]))
    warmup_prompt_inputs = _slice_prompt_inputs(prompt_inputs, warmup_prompt_length)
    resolved_warmup_max_new_tokens = (
        int(_DEFAULT_WARMUP_MAX_NEW_TOKENS)
        if warmup_max_new_tokens is None
        else int(warmup_max_new_tokens)
    )
    variants = kv_variants or build_gpu_transformers_variants(
        kv_quant_seed=int(kv_quant_seed),
        kv_hot_window=int(kv_hot_window),
        kv_calibration_tokens=int(kv_calibration_tokens),
        kv_adaptive_medium_kurtosis=float(kv_adaptive_medium_kurtosis),
        kv_adaptive_high_kurtosis=float(kv_adaptive_high_kurtosis),
    )

    results: dict[str, Any] = {}
    baseline_logits: np.ndarray | None = None
    baseline_ids: list[int] | None = None
    baseline_time: float | None = None
    baseline_bytes_native: int | None = None
    baseline_bytes_fp32_equivalent: int | None = None
    baseline_session_bytes_native: int | None = None
    baseline_session_bytes_fp32_equivalent: int | None = None
    baseline_prompt_perplexity: float | None = None
    baseline_peak_memory_mb: float | None = None
    baseline_native_dtype: str | None = None
    baseline_native_element_size_bytes: int | None = None
    model_size_bytes = _huggingface_cache_size_bytes(model_ref)
    model_vram_bytes = _model_vram_bytes(model)
    weight_runtime_source = _weight_runtime_source_for_model(model_ref)

    for variant in variants:
        run = _run_transformers_variant(
            model,
            prompt_inputs=prompt_inputs,
            prompt_ids=resolved_prompt_ids,
            variant=variant,
            max_new_tokens=int(max_new_tokens),
            warmup_prompt_inputs=warmup_prompt_inputs,
            warmup_max_new_tokens=int(resolved_warmup_max_new_tokens),
            kv_backend=str(kv_backend),
        )
        entry = {
            "name": str(variant["name"]),
            "kv_cache_precision": str(variant.get("kv_cache_precision", "native-dense")),
            "kv_key_precision": variant.get("kv_key_precision"),
            "kv_value_precision": variant.get("kv_value_precision"),
            "kv_key_scaling_strategy": variant.get("kv_key_scaling_strategy"),
            "kv_value_scaling_strategy": variant.get("kv_value_scaling_strategy"),
            "kv_rotation_mode": str(variant.get("kv_rotation_mode", "hadamard")),
            "prompt_perplexity": float(run["prompt_perplexity"]),
            "total_time_s": float(run["total_time_s"]),
            "avg_step_ms": float(run["avg_step_ms"]),
            "tokens_per_second": float(run["tokens_per_second"]),
            "kv_cache_bytes": int(run["kv_cache_bytes"]),
            "session_total_bytes": int(run["session_total_bytes"]),
            "session_meta_bytes": int(run["session_meta_bytes"]),
            "session_npz_bytes": int(run["session_npz_bytes"]),
            "session_save_time_ms": float(run["session_save_time_ms"]),
            "session_load_time_ms": float(run["session_load_time_ms"]),
            "gpu_peak_memory_mb": run.get("gpu_peak_memory_mb"),
            "current_kv_mode": str(run["current_kv_mode"]),
            "kv_kurtosis_profile": run.get("kv_kurtosis_profile"),
            "layer_mode_counts": run.get("layer_mode_counts"),
            "layer_kv_mode_counts": run.get("layer_kv_mode_counts"),
            "kv_norm_ratio_per_layer": run.get("kv_norm_ratio_per_layer"),
            "protected_layer_indices": run.get("protected_layer_indices"),
            "sparse_v_skip_ratio": run.get("sparse_v_skip_ratio"),
            "mamba_state_precision": run.get("mamba_state_precision"),
            "mamba_state_runtime_enabled": run.get("mamba_state_runtime_enabled"),
            "mamba_state_bytes": run.get("mamba_state_bytes"),
            "mamba_state_runtime_bytes": run.get("mamba_state_runtime_bytes"),
            "mamba_state_runtime_ratio_vs_native": run.get("mamba_state_runtime_ratio_vs_native"),
            "mamba_state_compress_time_ms": run.get("mamba_state_compress_time_ms"),
            "mamba_state_materialize_time_ms": run.get("mamba_state_materialize_time_ms"),
            "mamba_state_fallback_counts": run.get("mamba_state_fallback_counts"),
            "mamba_receipt_count": run.get("mamba_receipt_count"),
            "hybrid_total_cache_bytes": run.get("hybrid_total_cache_bytes"),
            "hybrid_total_runtime_cache_bytes": run.get("hybrid_total_runtime_cache_bytes"),
            "mamba_state_compression_probe": run.get("mamba_state_compression_probe"),
            "transformer_layer_indices": run.get("transformer_layer_indices"),
            "layers_block_type": run.get("layers_block_type"),
            "model_device": run.get("model_device"),
            "cache_device": run.get("cache_device"),
            "generated_ids": list(run["generated_ids"]),
            "native_kv_dtype": run.get("native_kv_dtype"),
            "native_element_size_bytes": run.get("native_element_size_bytes"),
            "total_inference_footprint_bytes": None
            if model_vram_bytes is None
            else int(model_vram_bytes) + int(run.get("hybrid_total_runtime_cache_bytes") or run.get("hybrid_total_cache_bytes") or run["kv_cache_bytes"]),
        }
        if baseline_logits is None:
            baseline_logits = np.asarray(run["last_logits"], dtype=np.float32)
            baseline_ids = list(run["generated_ids"])
            baseline_time = float(run["total_time_s"])
            baseline_bytes_native = int(run["kv_cache_bytes"])
            baseline_native_dtype = run.get("native_kv_dtype")
            baseline_native_element_size_bytes = run.get("native_element_size_bytes")
            baseline_bytes_fp32_equivalent = _native_fp32_equivalent_bytes(
                baseline_bytes_native,
                baseline_native_element_size_bytes,
            )
            baseline_session_bytes_native = int(run["session_total_bytes"])
            baseline_session_bytes_fp32_equivalent = _native_fp32_equivalent_bytes(
                baseline_session_bytes_native,
                baseline_native_element_size_bytes,
            )
            baseline_prompt_perplexity = float(run["prompt_perplexity"])
            baseline_peak_memory_mb = None if run.get("gpu_peak_memory_mb") is None else float(run["gpu_peak_memory_mb"])
            entry["generated_match_vs_baseline"] = True
            entry["logit_comparison_vs_baseline"] = None
            entry["speedup_vs_native"] = 1.0
            entry["speedup_vs_fp32"] = 1.0
            entry["kv_cache_ratio_vs_native"] = 1.0
            entry["kv_cache_ratio_vs_fp32_equivalent"] = 1.0 if baseline_bytes_fp32_equivalent else 1.0
            entry["kv_cache_ratio_vs_fp32"] = 1.0 if baseline_bytes_fp32_equivalent else 1.0
            entry["session_size_ratio_vs_native"] = 1.0
            entry["session_size_ratio_vs_fp32_equivalent"] = 1.0 if baseline_session_bytes_fp32_equivalent else 1.0
            entry["session_size_ratio_vs_fp32"] = 1.0 if baseline_session_bytes_fp32_equivalent else 1.0
            entry["prompt_perplexity_delta_pct_vs_native"] = 0.0
            entry["prompt_perplexity_delta_pct_vs_fp32"] = 0.0
            entry["gpu_peak_memory_delta_vs_native_mb"] = 0.0 if baseline_peak_memory_mb is not None else None
            entry["gpu_peak_memory_delta_vs_fp32_mb"] = 0.0 if baseline_peak_memory_mb is not None else None
        else:
            entry["generated_match_vs_baseline"] = list(run["generated_ids"]) == baseline_ids
            entry["logit_comparison_vs_baseline"] = _logit_comparison(run["last_logits"], baseline_logits)
            entry["speedup_vs_native"] = float(baseline_time) / float(run["total_time_s"]) if float(run["total_time_s"]) else 0.0
            entry["speedup_vs_fp32"] = entry["speedup_vs_native"]
            entry["kv_cache_ratio_vs_native"] = (
                float(baseline_bytes_native) / float(run["kv_cache_bytes"])
                if baseline_bytes_native and int(run["kv_cache_bytes"])
                else 0.0
            )
            entry["kv_cache_ratio_vs_fp32_equivalent"] = (
                float(baseline_bytes_fp32_equivalent) / float(run["kv_cache_bytes"])
                if baseline_bytes_fp32_equivalent and int(run["kv_cache_bytes"])
                else 0.0
            )
            entry["kv_cache_ratio_vs_fp32"] = entry["kv_cache_ratio_vs_fp32_equivalent"]
            entry["session_size_ratio_vs_native"] = (
                float(baseline_session_bytes_native) / float(run["session_total_bytes"])
                if int(run["session_total_bytes"])
                else 0.0
            )
            entry["session_size_ratio_vs_fp32_equivalent"] = (
                float(baseline_session_bytes_fp32_equivalent) / float(run["session_total_bytes"])
                if baseline_session_bytes_fp32_equivalent and int(run["session_total_bytes"])
                else 0.0
            )
            entry["session_size_ratio_vs_fp32"] = entry["session_size_ratio_vs_fp32_equivalent"]
            entry["prompt_perplexity_delta_pct_vs_native"] = (
                ((float(run["prompt_perplexity"]) - float(baseline_prompt_perplexity)) / float(baseline_prompt_perplexity)) * 100.0
                if baseline_prompt_perplexity
                else 0.0
            )
            entry["prompt_perplexity_delta_pct_vs_fp32"] = entry["prompt_perplexity_delta_pct_vs_native"]
            entry["gpu_peak_memory_delta_vs_native_mb"] = (
                None
                if baseline_peak_memory_mb is None or run.get("gpu_peak_memory_mb") is None
                else float(run["gpu_peak_memory_mb"]) - float(baseline_peak_memory_mb)
            )
            entry["gpu_peak_memory_delta_vs_fp32_mb"] = entry["gpu_peak_memory_delta_vs_native_mb"]
        results[str(variant["name"])] = entry

    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
    attention_heads = int(getattr(model.config, "num_attention_heads", model_config.get("num_attention_heads", 0)) or 0)
    key_value_heads = int(getattr(model.config, "num_key_value_heads", model_config.get("num_key_value_heads", 0)) or 0)
    if key_value_heads <= 0:
        key_value_heads = attention_heads or 0
    gqa_group_size = (float(attention_heads) / float(key_value_heads)) if attention_heads > 0 and key_value_heads > 0 else 1.0
    model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    rows: list[dict[str, Any]] = []
    for variant_name in [str(variant["name"]) for variant in variants]:
        entry = results[variant_name]
        comparison = entry.get("logit_comparison_vs_baseline") or {}
        rows.append(
            {
                "name": variant_name,
                "kv_cache_precision": entry.get("kv_cache_precision"),
                "kv_key_precision": entry.get("kv_key_precision"),
                "kv_value_precision": entry.get("kv_value_precision"),
                "kv_key_scaling_strategy": entry.get("kv_key_scaling_strategy"),
                "kv_value_scaling_strategy": entry.get("kv_value_scaling_strategy"),
                "kv_rotation_mode": entry.get("kv_rotation_mode"),
                "prompt_perplexity": float(entry["prompt_perplexity"]),
                "total_time_s": float(entry["total_time_s"]),
                "avg_step_ms": float(entry["avg_step_ms"]),
                "tokens_per_second": float(entry["tokens_per_second"]),
                "speedup_vs_native": float(entry["speedup_vs_native"]),
                "speedup_vs_fp32": float(entry["speedup_vs_fp32"]),
                "kv_cache_bytes": int(entry["kv_cache_bytes"]),
                "kv_cache_ratio_vs_native": float(entry["kv_cache_ratio_vs_native"]),
                "kv_cache_ratio_vs_fp32_equivalent": float(entry["kv_cache_ratio_vs_fp32_equivalent"]),
                "kv_cache_ratio_vs_fp32": float(entry["kv_cache_ratio_vs_fp32"]),
                "session_total_bytes": int(entry["session_total_bytes"]),
                "session_size_ratio_vs_native": float(entry["session_size_ratio_vs_native"]),
                "session_size_ratio_vs_fp32_equivalent": float(entry["session_size_ratio_vs_fp32_equivalent"]),
                "session_size_ratio_vs_fp32": float(entry["session_size_ratio_vs_fp32"]),
                "session_meta_bytes": int(entry["session_meta_bytes"]),
                "session_npz_bytes": int(entry["session_npz_bytes"]),
                "session_save_time_ms": float(entry["session_save_time_ms"]),
                "session_load_time_ms": float(entry["session_load_time_ms"]),
                "prompt_perplexity_delta_pct_vs_native": float(entry["prompt_perplexity_delta_pct_vs_native"]),
                "prompt_perplexity_delta_pct_vs_fp32": float(entry["prompt_perplexity_delta_pct_vs_fp32"]),
                "generated_match_vs_baseline": bool(entry["generated_match_vs_baseline"]),
                "cosine_similarity": None if not comparison else float(comparison.get("cosine_similarity", 0.0)),
                "cosine_similarity_vs_baseline": None if not comparison else float(comparison.get("cosine_similarity", 0.0)),
                "max_abs_err": None if not comparison else float(comparison.get("max_abs_err", 0.0)),
                "max_abs_err_vs_baseline": None if not comparison else float(comparison.get("max_abs_err", 0.0)),
                "mean_abs_err": None if not comparison else float(comparison.get("mean_abs_err", 0.0)),
                "current_kv_mode": entry.get("current_kv_mode"),
                "kv_kurtosis_profile": entry.get("kv_kurtosis_profile"),
                "layer_mode_counts": entry.get("layer_mode_counts"),
                "layer_kv_mode_counts": entry.get("layer_kv_mode_counts"),
                "kv_norm_ratio_per_layer": entry.get("kv_norm_ratio_per_layer"),
                "protected_layer_indices": entry.get("protected_layer_indices"),
                "sparse_v_skip_ratio": entry.get("sparse_v_skip_ratio"),
                "mamba_state_precision": entry.get("mamba_state_precision"),
                "mamba_state_runtime_enabled": entry.get("mamba_state_runtime_enabled"),
                "mamba_state_bytes": entry.get("mamba_state_bytes"),
                "mamba_state_runtime_bytes": entry.get("mamba_state_runtime_bytes"),
                "mamba_state_runtime_ratio_vs_native": entry.get("mamba_state_runtime_ratio_vs_native"),
                "mamba_state_compress_time_ms": entry.get("mamba_state_compress_time_ms"),
                "mamba_state_materialize_time_ms": entry.get("mamba_state_materialize_time_ms"),
                "mamba_state_fallback_counts": entry.get("mamba_state_fallback_counts"),
                "mamba_receipt_count": entry.get("mamba_receipt_count"),
                "hybrid_total_cache_bytes": entry.get("hybrid_total_cache_bytes"),
                "hybrid_total_runtime_cache_bytes": entry.get("hybrid_total_runtime_cache_bytes"),
                "mamba_state_compression_probe": entry.get("mamba_state_compression_probe"),
                "transformer_layer_indices": entry.get("transformer_layer_indices"),
                "layers_block_type": entry.get("layers_block_type"),
                "gpu_peak_memory_mb": entry.get("gpu_peak_memory_mb"),
                "gpu_peak_memory_delta_vs_native_mb": entry.get("gpu_peak_memory_delta_vs_native_mb"),
                "gpu_peak_memory_delta_vs_fp32_mb": entry.get("gpu_peak_memory_delta_vs_fp32_mb"),
                "model_device": entry.get("model_device"),
                "cache_device": entry.get("cache_device"),
                "native_kv_dtype": entry.get("native_kv_dtype"),
                "native_element_size_bytes": entry.get("native_element_size_bytes"),
                "total_inference_footprint_bytes": entry.get("total_inference_footprint_bytes"),
            }
        )
    return {
        "model_ref": str(model_ref),
        "is_hxq_compressed": bool(is_hxq_compressed),
        "hxq_model_ref": str(model_ref) if is_hxq_compressed else None,
        "weight_compression_method": "hxq" if is_hxq_compressed else "none",
        "trust_remote_code_used": bool(trust_remote_code),
        "weight_runtime_source": weight_runtime_source,
        "hxq_primary_error": hxq_primary_error,
        "model_size_bytes": model_size_bytes,
        "model_vram_bytes": model_vram_bytes,
        "prompt_ids": list(resolved_prompt_ids),
        "prompt_length": len(resolved_prompt_ids),
        "max_new_tokens": int(max_new_tokens),
        "warmup_prompt_length": warmup_prompt_length,
        "warmup_max_new_tokens": int(resolved_warmup_max_new_tokens),
        "device": str(resolved_device),
        "variant_order": [str(variant["name"]) for variant in variants],
        "rows": rows,
        "supports_selective_attention_acceleration": False,
        "unsupported_features": ["kv_topk", "block_scoring", "layer_share"],
        "model_config": model_config,
        "model_dtype": model_dtype,
        "native_kv_dtype": baseline_native_dtype,
        "native_element_size_bytes": baseline_native_element_size_bytes,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": key_value_heads,
        "gqa_group_size": gqa_group_size,
        "input_adapter": input_adapter,
        "processor_used": bool(processor_used),
        "chat_template_used": bool(chat_template_used),
        "gated_model": bool(gated_model),
        "hf_auth_required": bool(hf_auth_required),
        "tokenizer_name_or_path": None
        if processor_used
        else None if adapter is None else str(getattr(adapter, "name_or_path", model_ref)),
        "processor_name_or_path": None
        if not processor_used
        else None if adapter is None else str(getattr(adapter, "name_or_path", model_ref)),
        "benchmark_prompt_text": prompt_text or _DEFAULT_TRANSFORMERS_BENCHMARK_PROMPT,
        "variants": results,
        "kv_backend": str(kv_backend),
        "weight_runtime_ref": normalized_model_ref,
    }
