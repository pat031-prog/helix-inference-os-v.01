from __future__ import annotations

import ctypes
import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from helix_kv import rust_session
from helix_kv.policy import AdaptiveKVPolicy
from helix_proto.format import (
    create_store,
    load_full_tensor,
    load_meta,
    load_tensor_rows,
    store_stats,
    streaming_matvec,
)


MANIFEST_FILE = "manifest.json"
_SESSION_CACHES: dict[str, _TensorRuntimeCache] = {}
_SESSION_FORMAT_VERSION = 2
_SWITCHABLE_SYMMETRIC_MODES = {"fp32", "turbo-int8", "turbo-4bit"}


def _canonical_switch_mode(mode: str) -> str:
    lowered = str(mode).strip().lower()
    aliases = {
        "fp32": "fp32",
        "turbo-int8": "turbo-int8",
        "turbo-int8-hadamard": "turbo-int8",
        "turbo-4bit": "turbo-4bit",
    }
    try:
        return aliases[lowered]
    except KeyError as exc:
        raise ValueError(f"unsupported switchable kv mode: {mode}") from exc


def _public_kv_mode_name(mode: str) -> str:
    lowered = str(mode).strip().lower()
    aliases = {
        "fp32": "fp32",
        "turbo-int8": "turbo-int8-hadamard",
        "turbo-int8-hadamard": "turbo-int8-hadamard",
        "turbo-4bit": "turbo-4bit",
        "turbo-qjl": "turbo-qjl",
        "adaptive": "adaptive",
    }
    try:
        return aliases[lowered]
    except KeyError:
        return lowered


class _TensorRuntimeCache:
    def __init__(self, *, max_tensor_bytes: int = 256 * 1024) -> None:
        self.max_tensor_bytes = max_tensor_bytes
        self._tensor_cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def tensor(self, store: Path) -> np.ndarray:
        key = str(store)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        tensor = load_full_tensor(store)
        if tensor.nbytes <= self.max_tensor_bytes:
            self._tensor_cache[key] = tensor
        return tensor

    def rows(self, store: Path, indices: list[int]) -> np.ndarray:
        meta = load_meta(store)
        estimated_bytes = int(np.prod(meta.shape)) * np.dtype(meta.dtype).itemsize
        if estimated_bytes <= self.max_tensor_bytes:
            tensor = self.tensor(store)
            return tensor[indices]
        return load_tensor_rows(store, indices)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._tensor_cache),
            "hits": self.hits,
            "misses": self.misses,
        }


class _KurtosisAccumulator:
    def __init__(
        self,
        *,
        count: int = 0,
        sum1: float = 0.0,
        sum2: float = 0.0,
        sum3: float = 0.0,
        sum4: float = 0.0,
    ) -> None:
        self.count = int(count)
        self.sum1 = float(sum1)
        self.sum2 = float(sum2)
        self.sum3 = float(sum3)
        self.sum4 = float(sum4)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        self.count += int(array.size)
        self.sum1 += float(np.sum(array, dtype=np.float64))
        self.sum2 += float(np.sum(array**2, dtype=np.float64))
        self.sum3 += float(np.sum(array**3, dtype=np.float64))
        self.sum4 += float(np.sum(array**4, dtype=np.float64))

    def pearson_kurtosis(self) -> float:
        if self.count <= 1:
            return 0.0
        count = float(self.count)
        mean = self.sum1 / count
        ex2 = self.sum2 / count
        ex3 = self.sum3 / count
        ex4 = self.sum4 / count
        variance = max(ex2 - mean * mean, 0.0)
        if variance <= 1e-12:
            return 0.0
        central4 = ex4 - (4.0 * mean * ex3) + (6.0 * (mean**2) * ex2) - (3.0 * (mean**4))
        return float(central4 / (variance * variance))

    def to_json(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "sum1": float(self.sum1),
            "sum2": float(self.sum2),
            "sum3": float(self.sum3),
            "sum4": float(self.sum4),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "_KurtosisAccumulator":
        return cls(
            count=int(payload.get("count", 0)),
            sum1=float(payload.get("sum1", 0.0)),
            sum2=float(payload.get("sum2", 0.0)),
            sum3=float(payload.get("sum3", 0.0)),
            sum4=float(payload.get("sum4", 0.0)),
        )


def _orthogonal_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng((int(seed) * 1_000_003) + int(dim))
    gaussian = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(gaussian.astype(np.float64))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).astype(np.float32)


def _next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    power = 1
    while power < n:
        power *= 2
    return power


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if callable(trapezoid):
        return float(trapezoid(values, grid))
    return float(np.trapz(values, grid))


def _fwht_last_axis(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32).copy()
    dim = int(transformed.shape[-1])
    if dim & (dim - 1):
        raise ValueError(f"fwht requires a power-of-two dimension, got {dim}")
    h = 1
    while h < dim:
        view = transformed.reshape(*transformed.shape[:-1], -1, 2 * h)
        a = view[..., :h].copy()
        b = view[..., h:].copy()
        view[..., :h] = a + b
        view[..., h:] = a - b
        h *= 2
    return transformed.astype(np.float32)


class _DenseOrthogonalRotation:
    def __init__(self, dim: int, seed: int) -> None:
        self.original_dim = int(dim)
        self.rotated_dim = int(dim)
        self.matrix = _orthogonal_rotation_matrix(dim, seed)

    def forward(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.einsum("...d,df->...f", array, self.matrix, optimize=True).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.einsum("...d,df->...f", array, self.matrix.T, optimize=True).astype(np.float32)


class _HadamardRotation:
    def __init__(self, dim: int, seed: int) -> None:
        self.original_dim = int(dim)
        self.rotated_dim = _next_power_of_two(dim)
        rng = np.random.default_rng((int(seed) * 1_700_011) + int(dim))
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.rotated_dim).astype(np.float32)
        self.scale = float(np.sqrt(self.rotated_dim))
        self.matrix: np.ndarray | None = None
        if self.rotated_dim <= 256:
            basis = np.eye(self.rotated_dim, dtype=np.float32)
            self.matrix = (_fwht_last_axis(basis * self.signs[None, :]) / self.scale).astype(np.float32)

    def forward(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if self.rotated_dim != self.original_dim:
            pad_width = [(0, 0)] * array.ndim
            pad_width[-1] = (0, self.rotated_dim - self.original_dim)
            array = np.pad(array, pad_width, mode="constant")
        if self.matrix is not None:
            return np.einsum("...d,df->...f", array, self.matrix, optimize=True).astype(np.float32)
        array = array * self.signs
        return (_fwht_last_axis(array) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.rotated_dim:
            raise ValueError(
                f"hadamard inverse expected last dim {self.rotated_dim}, got {array.shape[-1]}"
            )
        if self.matrix is not None:
            restored = np.einsum("...d,df->...f", array, self.matrix.T, optimize=True).astype(np.float32)
        else:
            restored = (_fwht_last_axis(array) / self.scale) * self.signs
        if self.rotated_dim != self.original_dim:
            restored = restored[..., : self.original_dim]
        return restored.astype(np.float32)


def _build_kv_rotation(dim: int, seed: int, mode: str) -> _DenseOrthogonalRotation | _HadamardRotation:
    if mode == "qr":
        return _DenseOrthogonalRotation(dim, seed)
    if mode == "hadamard":
        return _HadamardRotation(dim, seed)
    raise ValueError(f"unsupported kv rotation mode: {mode}")


def _apply_rotation(
    values: np.ndarray,
    rotation: np.ndarray | _DenseOrthogonalRotation | _HadamardRotation | None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if rotation is None:
        return array
    if isinstance(rotation, np.ndarray):
        return np.einsum("...d,df->...f", array, rotation, optimize=True).astype(np.float32)
    return rotation.forward(array)


def _apply_inverse_rotation(
    values: np.ndarray,
    rotation: np.ndarray | _DenseOrthogonalRotation | _HadamardRotation | None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if rotation is None:
        return array
    if isinstance(rotation, np.ndarray):
        return np.einsum("...d,df->...f", array, rotation.T, optimize=True).astype(np.float32)
    return rotation.inverse(array)


class _ScalarCodebook:
    def __init__(self, centroids: np.ndarray, boundaries: np.ndarray, *, dim: int, bits: int) -> None:
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.boundaries = np.asarray(boundaries, dtype=np.float32)
        self.dim = int(dim)
        self.bits = int(bits)
        self.levels = int(self.centroids.size)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.boundaries, np.asarray(values, dtype=np.float32), side="left") - 1
        return np.clip(idx, 0, self.levels - 1).astype(np.uint8)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        return self.centroids[np.asarray(indices, dtype=np.intp)].astype(np.float32)


def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    pdf = np.zeros_like(grid, dtype=np.float64)
    valid = (grid > -1.0) & (grid < 1.0)
    if not np.any(valid):
        return pdf
    coeff = math.exp(
        math.lgamma(dim / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2.0)
    )
    pdf[valid] = coeff * np.power(np.maximum(1.0 - grid[valid] ** 2, 1e-30), (dim - 3) / 2.0)
    return pdf


def _solve_lloyd_max(
    pdf: np.ndarray,
    grid: np.ndarray,
    levels: int,
    *,
    max_iter: int = 400,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    lo = float(grid[0])
    hi = float(grid[-1])
    centroids = np.linspace(lo, hi, num=levels + 2, dtype=np.float64)[1:-1]
    boundaries = np.empty(levels + 1, dtype=np.float64)
    boundaries[0] = lo
    boundaries[-1] = hi

    for _ in range(max_iter):
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        previous = centroids.copy()
        for index in range(levels):
            mask = (grid >= boundaries[index]) & (grid <= boundaries[index + 1])
            grid_slice = grid[mask]
            pdf_slice = pdf[mask]
            if grid_slice.size < 2:
                centroids[index] = 0.5 * (boundaries[index] + boundaries[index + 1])
                continue
            interval_mass = _trapz(pdf_slice, grid_slice)
            if interval_mass <= 1e-12:
                centroids[index] = 0.5 * (boundaries[index] + boundaries[index + 1])
            else:
                centroids[index] = _trapz(pdf_slice * grid_slice, grid_slice) / interval_mass
        if np.max(np.abs(centroids - previous)) < tol:
            break

    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.astype(np.float32), boundaries.astype(np.float32)


@lru_cache(maxsize=32)
def _compute_lloyd_max_codebook(dim: int, bits: int) -> _ScalarCodebook:
    levels = 2**bits
    sigma = 1.0 / math.sqrt(dim)
    lo = max(-1.0, -6.0 * sigma)
    hi = min(1.0, 6.0 * sigma)
    grid = np.linspace(lo, hi, num=8193, dtype=np.float64)
    if dim >= 64:
        pdf = np.exp(-0.5 * dim * grid**2) * math.sqrt(dim / (2.0 * math.pi))
    else:
        pdf = _beta_pdf(grid, dim)
    pdf = np.clip(pdf, 0.0, None)
    mass = _trapz(pdf, grid)
    if mass <= 0:
        raise ValueError(f"degenerate Lloyd-Max density for dim={dim}")
    pdf = pdf / mass
    centroids, boundaries = _solve_lloyd_max(pdf, grid, levels)
    return _ScalarCodebook(centroids, boundaries, dim=dim, bits=bits)


def _pack_nibbles(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint8)
    if values.shape[-1] % 2:
        pad_width = [(0, 0)] * values.ndim
        pad_width[-1] = (0, 1)
        values = np.pad(values, pad_width, mode="constant")
    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).astype(np.uint8)


def _unpack_nibbles(packed: np.ndarray, original_length: int) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    unpacked = np.empty(values.shape[:-1] + (values.shape[-1] * 2,), dtype=np.uint8)
    unpacked[..., 0::2] = values & 0x0F
    unpacked[..., 1::2] = (values >> 4) & 0x0F
    return unpacked[..., :original_length]


def _pack_sign_bits(signs: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(signs, dtype=np.uint8), axis=-1, bitorder="little")


def _unpack_sign_bits(packed: np.ndarray, original_length: int) -> np.ndarray:
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        axis=-1,
        count=original_length,
        bitorder="little",
    ).astype(np.uint8)


def _gaussian_qjl_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng((int(seed) * 2_300_027) + int(dim))
    return rng.standard_normal(size=(dim, dim), dtype=np.float32).astype(np.float32)


def _block_mean_summary(values: np.ndarray, block_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    block_size = max(int(block_size), 1)
    if values.shape[1] == 0:
        return np.empty((values.shape[0], 0, values.shape[2]), dtype=np.float32)
    summaries = []
    for start in range(0, values.shape[1], block_size):
        stop = min(start + block_size, values.shape[1])
        summaries.append(np.mean(values[:, start:stop, :], axis=1, dtype=np.float32))
    return np.stack(summaries, axis=1).astype(np.float32)


def _block_extrema_summary(values: np.ndarray, block_size: int) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    block_size = max(int(block_size), 1)
    if values.shape[1] == 0:
        empty = np.empty((values.shape[0], 0, values.shape[2]), dtype=np.float32)
        return {"mean": empty, "min": empty.copy(), "max": empty.copy()}
    means = []
    mins = []
    maxs = []
    for start in range(0, values.shape[1], block_size):
        stop = min(start + block_size, values.shape[1])
        block = values[:, start:stop, :]
        means.append(np.mean(block, axis=1, dtype=np.float32))
        mins.append(np.min(block, axis=1).astype(np.float32))
        maxs.append(np.max(block, axis=1).astype(np.float32))
    return {
        "mean": np.stack(means, axis=1).astype(np.float32),
        "min": np.stack(mins, axis=1).astype(np.float32),
        "max": np.stack(maxs, axis=1).astype(np.float32),
    }


def _append_block_mean_summary(
    existing: np.ndarray,
    *,
    prev_length: int,
    appended_values: np.ndarray,
    block_size: int,
) -> np.ndarray:
    existing = np.asarray(existing, dtype=np.float32)
    appended_values = np.asarray(appended_values, dtype=np.float32)
    block_size = max(int(block_size), 1)
    if appended_values.shape[1] == 0:
        return existing.copy()
    if int(prev_length) <= 0 or existing.shape[1] == 0:
        return _block_mean_summary(appended_values, block_size)

    result = existing.copy()
    cursor = 0
    tail_fill = int(prev_length) % block_size
    if tail_fill:
        fill = min(block_size - tail_fill, appended_values.shape[1])
        prev_sum = result[:, -1, :] * float(tail_fill)
        add_sum = np.sum(appended_values[:, :fill, :], axis=1, dtype=np.float32)
        result[:, -1, :] = (prev_sum + add_sum) / float(tail_fill + fill)
        cursor = fill
    if cursor < appended_values.shape[1]:
        tail_summaries = _block_mean_summary(appended_values[:, cursor:, :], block_size)
        result = np.concatenate([result, tail_summaries], axis=1)
    return result.astype(np.float32)


def _append_block_extrema_summary(
    existing: dict[str, np.ndarray] | np.ndarray,
    *,
    prev_length: int,
    appended_values: np.ndarray,
    block_size: int,
) -> dict[str, np.ndarray]:
    appended_values = np.asarray(appended_values, dtype=np.float32)
    block_size = max(int(block_size), 1)
    if isinstance(existing, dict):
        result = {key: np.asarray(value, dtype=np.float32).copy() for key, value in existing.items()}
    else:
        mean = np.asarray(existing, dtype=np.float32)
        result = {"mean": mean.copy(), "min": mean.copy(), "max": mean.copy()}
    if appended_values.shape[1] == 0:
        return result
    if int(prev_length) <= 0 or result["mean"].shape[1] == 0:
        return _block_extrema_summary(appended_values, block_size)

    cursor = 0
    tail_fill = int(prev_length) % block_size
    if tail_fill:
        fill = min(block_size - tail_fill, appended_values.shape[1])
        tail_values = appended_values[:, :fill, :]
        prev_sum = result["mean"][:, -1, :] * float(tail_fill)
        add_sum = np.sum(tail_values, axis=1, dtype=np.float32)
        result["mean"][:, -1, :] = (prev_sum + add_sum) / float(tail_fill + fill)
        result["min"][:, -1, :] = np.minimum(result["min"][:, -1, :], np.min(tail_values, axis=1))
        result["max"][:, -1, :] = np.maximum(result["max"][:, -1, :], np.max(tail_values, axis=1))
        cursor = fill
    if cursor < appended_values.shape[1]:
        tail_stats = _block_extrema_summary(appended_values[:, cursor:, :], block_size)
        for key in ("mean", "min", "max"):
            result[key] = np.concatenate([result[key], tail_stats[key]], axis=1).astype(np.float32)
    return result


def _append_block_summary_cache(
    cache: dict[int, np.ndarray | dict[str, np.ndarray]],
    *,
    prev_length: int,
    appended_values: np.ndarray,
) -> dict[int, np.ndarray | dict[str, np.ndarray]]:
    if not cache:
        return {}
    updated: dict[int, np.ndarray | dict[str, np.ndarray]] = {}
    for block_size, summary in cache.items():
        updated[int(block_size)] = _append_block_extrema_summary(
            summary,
            prev_length=prev_length,
            appended_values=appended_values,
            block_size=int(block_size),
        )
    return updated


def _merge_candidate_index_sets(
    *arrays: np.ndarray | None,
    max_candidates: int | None = None,
) -> np.ndarray:
    valid = [np.asarray(array, dtype=np.int64) for array in arrays if isinstance(array, np.ndarray) and array.size > 0]
    if not valid:
        return np.empty((0, 0), dtype=np.int64)
    num_heads = int(valid[0].shape[0])
    cap = None if max_candidates is None else max(int(max_candidates), 1)
    rows: list[np.ndarray] = []
    max_width = 0
    for head_index in range(num_heads):
        seen: set[int] = set()
        merged: list[int] = []
        for array in valid:
            for value in array[head_index]:
                token_index = int(value)
                if token_index in seen:
                    continue
                seen.add(token_index)
                merged.append(token_index)
                if cap is not None and len(merged) >= cap:
                    break
            if cap is not None and len(merged) >= cap:
                break
        row = np.asarray(merged, dtype=np.int64)
        rows.append(row)
        max_width = max(max_width, row.size)
    if max_width == 0:
        return np.empty((num_heads, 0), dtype=np.int64)
    merged = np.empty((num_heads, max_width), dtype=np.int64)
    for head_index, row in enumerate(rows):
        if row.size < max_width:
            pad_value = row[-1] if row.size else 0
            row = np.pad(row, (0, max_width - row.size), mode="constant", constant_values=pad_value)
        merged[head_index] = row
    return merged


def _expand_block_indices(block_indices: np.ndarray, *, cold_length: int, block_size: int) -> np.ndarray:
    block_indices = np.asarray(block_indices, dtype=np.int64)
    block_size = max(int(block_size), 1)
    rows: list[np.ndarray] = []
    max_width = 0
    for head_index in range(block_indices.shape[0]):
        tokens: list[int] = []
        seen: set[int] = set()
        for block_index in block_indices[head_index]:
            start = int(block_index) * block_size
            stop = min(start + block_size, int(cold_length))
            for token_index in range(start, stop):
                if token_index not in seen:
                    seen.add(token_index)
                    tokens.append(token_index)
        row = np.asarray(tokens, dtype=np.int64)
        rows.append(row)
        max_width = max(max_width, row.size)
    if max_width == 0:
        return np.empty((block_indices.shape[0], 0), dtype=np.int64)
    expanded = np.empty((block_indices.shape[0], max_width), dtype=np.int64)
    for head_index, row in enumerate(rows):
        if row.size < max_width:
            pad = np.full((max_width - row.size,), row[-1] if row.size else 0, dtype=np.int64)
            row = np.concatenate([row, pad], axis=0)
        expanded[head_index] = row
    return expanded


class _TurboInt8KVArray:
    kind = "turbo-int8"

    def __init__(
        self,
        values: np.ndarray,
        *,
        rotation: np.ndarray | _DenseOrthogonalRotation | _HadamardRotation | None = None,
    ) -> None:
        transformed = np.asarray(values, dtype=np.float32)
        self.rotation = rotation
        self._block_summary_cache: dict[int, np.ndarray | dict[str, np.ndarray]] = {}
        transformed = _apply_rotation(transformed, rotation)
        max_abs = np.max(np.abs(transformed), axis=-1, keepdims=True)
        safe_scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float16)
        self.q = np.clip(
            np.rint(transformed / safe_scale.astype(np.float32)),
            -127,
            127,
        ).astype(np.int8)
        self.scales = safe_scale

    @classmethod
    def from_quantized(
        cls,
        q: np.ndarray,
        scales: np.ndarray,
        *,
        rotation: np.ndarray | _DenseOrthogonalRotation | _HadamardRotation | None = None,
    ) -> "_TurboInt8KVArray":
        instance = cls.__new__(cls)
        instance.q = np.asarray(q, dtype=np.int8)
        instance.scales = np.asarray(scales, dtype=np.float16)
        instance.rotation = rotation
        instance._block_summary_cache = {}
        return instance

    @property
    def length(self) -> int:
        return int(self.q.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.q.nbytes + self.scales.nbytes)

    def to_float32(self) -> np.ndarray:
        restored = self.q.astype(np.float32) * self.scales.astype(np.float32)
        return _apply_inverse_rotation(restored, self.rotation).astype(np.float32)

    def approximate_scores(self, query: np.ndarray, *, head_dim: int) -> np.ndarray:
        """Compute approximate Q·K^T scores without full float32 materialization.

        For turbo-int8 the stored representation is: K_stored = R @ K_original * scale
        Since R is orthogonal, Q·K = Q·R^T·(R·K) = (R·Q)·(stored/scale * scale) = (R·Q)·stored_f32.
        We compute Q_rotated = R·Q, then dot against q*scales directly, avoiding the inverse
        rotation step entirely.

        Args:
            query: shape (num_heads, head_dim)
            head_dim: dimension per head for scaling

        Returns:
            shape (num_heads, seq_len) approximate attention scores (pre-softmax)
        """
        query = np.asarray(query, dtype=np.float32)
        q_rotated = _apply_rotation(query, self.rotation)  # (heads, rotated_dim)
        # self.q is (heads, seq_len, rotated_dim), self.scales is (heads, seq_len, 1)
        # Dequantize in rotated domain: restored_rotated = q_int8 * scales
        restored_rotated = self.q.astype(np.float32) * self.scales.astype(np.float32)
        # Scores = Q_rotated · K_rotated^T / sqrt(head_dim)
        scores = np.einsum("hd,hnd->hn", q_rotated, restored_rotated, optimize=True).astype(np.float32)
        scores /= np.sqrt(float(head_dim))
        return scores

    def approximate_block_scores(self, query: np.ndarray, *, head_dim: int, block_size: int) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        q_rotated = _apply_rotation(query, self.rotation)
        block_size = max(int(block_size), 1)
        if block_size not in self._block_summary_cache:
            restored_rotated = self.q.astype(np.float32) * self.scales.astype(np.float32)
            self._block_summary_cache[block_size] = _block_extrema_summary(restored_rotated, block_size)
        summaries = self._block_summary_cache[block_size]
        if isinstance(summaries, dict):
            positive = np.maximum(q_rotated[:, None, :], 0.0)
            negative = np.minimum(q_rotated[:, None, :], 0.0)
            scores = np.sum((positive * summaries["max"]) + (negative * summaries["min"]), axis=-1, dtype=np.float32)
        else:
            scores = np.einsum("hd,hbd->hb", q_rotated, summaries, optimize=True).astype(np.float32)
        scores /= np.sqrt(float(head_dim))
        return scores

    def materialize_indices(self, indices: np.ndarray) -> np.ndarray:
        """Materialize only selected token positions to float32.

        Args:
            indices: shape (num_heads, topk) integer indices into seq_len dimension

        Returns:
            shape (num_heads, topk, head_dim) float32 values
        """
        num_heads = self.q.shape[0]
        topk = indices.shape[1]
        head_dim_rotated = self.q.shape[2]
        # Gather selected positions
        selected_q = np.empty((num_heads, topk, head_dim_rotated), dtype=np.int8)
        selected_scales = np.empty((num_heads, topk, 1), dtype=np.float16)
        for h in range(num_heads):
            selected_q[h] = self.q[h, indices[h]]
            selected_scales[h] = self.scales[h, indices[h]]
        restored = selected_q.astype(np.float32) * selected_scales.astype(np.float32)
        return _apply_inverse_rotation(restored, self.rotation).astype(np.float32)

    def append_compressed(self, values: np.ndarray) -> "_TurboInt8KVArray":
        """Append new token(s) to the compressed array without full materialization.

        Args:
            values: shape (num_heads, new_tokens, head_dim) float32 values to compress and append

        Returns:
            New _TurboInt8KVArray with appended tokens
        """
        values = np.asarray(values, dtype=np.float32)
        prev_length = self.length
        transformed = _apply_rotation(values, self.rotation)
        max_abs = np.max(np.abs(transformed), axis=-1, keepdims=True)
        safe_scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float16)
        new_q = np.clip(
            np.rint(transformed / safe_scale.astype(np.float32)),
            -127, 127,
        ).astype(np.int8)
        result = _TurboInt8KVArray.__new__(_TurboInt8KVArray)
        result.rotation = self.rotation
        appended_rotated = new_q.astype(np.float32) * safe_scale.astype(np.float32)
        result._block_summary_cache = _append_block_summary_cache(
            self._block_summary_cache,
            prev_length=prev_length,
            appended_values=appended_rotated,
        )
        result.q = np.concatenate([self.q, new_q], axis=1)
        result.scales = np.concatenate([self.scales, safe_scale], axis=1)
        return result


class _Turbo4BitKVArray:
    kind = "turbo-4bit"

    def __init__(
        self,
        values: np.ndarray,
        *,
        rotation: _DenseOrthogonalRotation | _HadamardRotation,
        codebook: _ScalarCodebook,
    ) -> None:
        original = np.asarray(values, dtype=np.float32)
        self.rotation = rotation
        self.codebook = codebook
        self._block_summary_cache: dict[int, np.ndarray | dict[str, np.ndarray]] = {}
        norms = np.linalg.norm(original, axis=-1, keepdims=True).astype(np.float32)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        unit = original / safe_norms
        rotated = rotation.forward(unit)
        self.packed = _pack_nibbles(codebook.quantize(rotated))
        self.norms = norms.astype(np.float16)

    @classmethod
    def from_quantized(
        cls,
        packed: np.ndarray,
        norms: np.ndarray,
        *,
        rotation: _DenseOrthogonalRotation | _HadamardRotation,
        codebook: _ScalarCodebook,
    ) -> "_Turbo4BitKVArray":
        instance = cls.__new__(cls)
        instance.rotation = rotation
        instance.codebook = codebook
        instance._block_summary_cache = {}
        instance.packed = np.asarray(packed, dtype=np.uint8)
        instance.norms = np.asarray(norms, dtype=np.float16)
        return instance

    @property
    def length(self) -> int:
        return int(self.norms.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.packed.nbytes + self.norms.nbytes)

    def _base_float32(self) -> np.ndarray:
        rotated = self.codebook.dequantize(_unpack_nibbles(self.packed, self.rotation.rotated_dim))
        unit = self.rotation.inverse(rotated)
        return (unit * self.norms.astype(np.float32)).astype(np.float32)

    def to_float32(self) -> np.ndarray:
        return self._base_float32().astype(np.float32)

    def _rotated_dequantized(self) -> np.ndarray:
        """Dequantize in the rotated domain (before inverse rotation)."""
        return self.codebook.dequantize(_unpack_nibbles(self.packed, self.rotation.rotated_dim))

    def _packed_dot_scores(self, q_rotated: np.ndarray) -> np.ndarray:
        """Score packed 4-bit rows directly from bytes without full nibble unpack.

        Each packed byte contains two 4-bit codebook indices. For a given rotated query,
        we build a small per-pair lookup table over the 256 possible byte values and gather
        contributions directly from the packed tensor. This keeps the computation O(seq_len)
        while avoiding a full cold-prefix unpack on every step.
        """
        q_rotated = np.asarray(q_rotated, dtype=np.float32)
        if q_rotated.shape[-1] % 2:
            q_rotated = np.pad(q_rotated, ((0, 0), (0, 1)), mode="constant")
        q_pairs = q_rotated.reshape(q_rotated.shape[0], -1, 2)
        centroids = self.codebook.centroids.astype(np.float32)
        byte_values = np.arange(256, dtype=np.uint8)
        low_values = centroids[(byte_values & 0x0F).astype(np.intp)]
        high_values = centroids[(byte_values >> 4).astype(np.intp)]
        packed = self.packed.astype(np.intp, copy=False)
        norms = self.norms.astype(np.float32).squeeze(-1)
        scores = np.empty((packed.shape[0], packed.shape[1]), dtype=np.float32)

        for head_index in range(packed.shape[0]):
            lookup = (
                q_pairs[head_index, :, 0:1] * low_values[None, :]
                + q_pairs[head_index, :, 1:2] * high_values[None, :]
            ).astype(np.float32)
            selected = np.take_along_axis(lookup, packed[head_index].T, axis=1)
            scores[head_index] = np.sum(selected, axis=0, dtype=np.float32) * norms[head_index]
        return scores.astype(np.float32)

    def approximate_scores(self, query: np.ndarray, *, head_dim: int) -> np.ndarray:
        """Compute approximate Q·K^T scores without full float32 materialization.

        For turbo-4bit the stored representation is K = R_inv(codebook_dequant(packed)) * norms.
        Since R is orthogonal: Q·K = (R·Q) · (codebook_dequant(packed) * norms).
        We compute Q_rotated = R(Q), then dot against dequantized_rotated * norms.

        Args:
            query: shape (num_heads, head_dim)
            head_dim: dimension per head for scaling

        Returns:
            shape (num_heads, seq_len) approximate attention scores (pre-softmax)
        """
        query = np.asarray(query, dtype=np.float32)
        q_rotated = self.rotation.forward(query)  # (heads, rotated_dim)
        scores = self._packed_dot_scores(q_rotated)
        scores /= np.sqrt(float(head_dim))
        return scores

    def approximate_block_scores(self, query: np.ndarray, *, head_dim: int, block_size: int) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        q_rotated = self.rotation.forward(query)
        block_size = max(int(block_size), 1)
        if block_size not in self._block_summary_cache:
            rotated = self._rotated_dequantized() * self.norms.astype(np.float32)
            self._block_summary_cache[block_size] = _block_extrema_summary(rotated, block_size)
        summaries = self._block_summary_cache[block_size]
        if isinstance(summaries, dict):
            positive = np.maximum(q_rotated[:, None, :], 0.0)
            negative = np.minimum(q_rotated[:, None, :], 0.0)
            scores = np.sum((positive * summaries["max"]) + (negative * summaries["min"]), axis=-1, dtype=np.float32)
        else:
            scores = np.einsum("hd,hbd->hb", q_rotated, summaries, optimize=True).astype(np.float32)
        scores /= np.sqrt(float(head_dim))
        return scores

    def materialize_indices(self, indices: np.ndarray) -> np.ndarray:
        """Materialize only selected token positions to float32.

        Args:
            indices: shape (num_heads, topk) integer indices into seq_len dimension

        Returns:
            shape (num_heads, topk, head_dim) float32 values
        """
        # Gather packed rows first so we only unpack the selected top-K tokens rather than
        # scanning the entire cold prefix on every step.
        num_heads = indices.shape[0]
        topk = indices.shape[1]
        packed_width = self.packed.shape[2]
        selected_packed = np.empty((num_heads, topk, packed_width), dtype=np.uint8)
        selected_norms = np.empty((num_heads, topk, 1), dtype=np.float32)

        for h in range(num_heads):
            selected_packed[h] = self.packed[h, indices[h]]
            selected_norms[h] = self.norms[h, indices[h]]

        selected_rotated = self.codebook.dequantize(_unpack_nibbles(selected_packed, self.rotation.rotated_dim))
        unit = self.rotation.inverse(selected_rotated)
        return (unit * selected_norms).astype(np.float32)

    def append_compressed(self, values: np.ndarray) -> "_Turbo4BitKVArray":
        """Append new token(s) to the compressed array without full materialization.

        Args:
            values: shape (num_heads, new_tokens, head_dim) float32 values

        Returns:
            New _Turbo4BitKVArray with appended tokens
        """
        values = np.asarray(values, dtype=np.float32)
        prev_length = self.length
        norms = np.linalg.norm(values, axis=-1, keepdims=True).astype(np.float32)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        unit = values / safe_norms
        rotated = self.rotation.forward(unit)
        quantized = self.codebook.quantize(rotated)
        new_packed = _pack_nibbles(quantized)
        new_norms = norms.astype(np.float16)
        result = _Turbo4BitKVArray.__new__(_Turbo4BitKVArray)
        result.rotation = self.rotation
        result.codebook = self.codebook
        appended_rotated = self.codebook.dequantize(quantized) * norms.astype(np.float32)
        result._block_summary_cache = _append_block_summary_cache(
            self._block_summary_cache,
            prev_length=prev_length,
            appended_values=appended_rotated,
        )
        result.packed = np.concatenate([self.packed, new_packed], axis=1)
        result.norms = np.concatenate([self.norms, new_norms], axis=1)
        return result


class _TurboQJLKVArray(_Turbo4BitKVArray):
    kind = "turbo-qjl"

    def __init__(
        self,
        values: np.ndarray,
        *,
        rotation: _DenseOrthogonalRotation | _HadamardRotation,
        codebook: _ScalarCodebook,
        qjl_matrix: np.ndarray,
    ) -> None:
        super().__init__(values, rotation=rotation, codebook=codebook)
        self.qjl_matrix = np.asarray(qjl_matrix, dtype=np.float32)
        original = np.asarray(values, dtype=np.float32)
        residual = original - self._base_float32()
        residual_norms = np.linalg.norm(residual, axis=-1, keepdims=True).astype(np.float32)
        safe_residual_norms = np.where(residual_norms > 1e-8, residual_norms, 1.0)
        residual_unit = residual / safe_residual_norms
        projected = np.einsum("...d,df->...f", residual_unit, self.qjl_matrix, optimize=True).astype(np.float32)
        self.qjl_bits = _pack_sign_bits(projected >= 0)
        self.residual_norms = residual_norms.astype(np.float16)

    @classmethod
    def from_quantized(
        cls,
        packed: np.ndarray,
        norms: np.ndarray,
        qjl_bits: np.ndarray,
        residual_norms: np.ndarray,
        *,
        rotation: _DenseOrthogonalRotation | _HadamardRotation,
        codebook: _ScalarCodebook,
        qjl_matrix: np.ndarray,
    ) -> "_TurboQJLKVArray":
        instance = cls.__new__(cls)
        instance.rotation = rotation
        instance.codebook = codebook
        instance.packed = np.asarray(packed, dtype=np.uint8)
        instance.norms = np.asarray(norms, dtype=np.float16)
        instance.qjl_matrix = np.asarray(qjl_matrix, dtype=np.float32)
        instance.qjl_bits = np.asarray(qjl_bits, dtype=np.uint8)
        instance.residual_norms = np.asarray(residual_norms, dtype=np.float16)
        instance._block_summary_cache = {}
        return instance

    @property
    def nbytes(self) -> int:
        return int(super().nbytes + self.qjl_bits.nbytes + self.residual_norms.nbytes)

    def _residual_estimate(self) -> np.ndarray:
        signs = _unpack_sign_bits(self.qjl_bits, self.qjl_matrix.shape[0]).astype(np.float32) * 2.0 - 1.0
        residual = np.einsum("...d,df->...f", signs, self.qjl_matrix, optimize=True).astype(np.float32)
        residual *= math.sqrt(math.pi / 2.0) / float(self.qjl_matrix.shape[0])
        residual *= self.residual_norms.astype(np.float32)
        return residual.astype(np.float32)

    def score_correction(self, query: np.ndarray, *, head_dim: int, score_weight: float) -> np.ndarray:
        if score_weight <= 0.0:
            return np.zeros((query.shape[0], self.length), dtype=np.float32)
        query = np.asarray(query, dtype=np.float32)
        q_proj = np.einsum("hd,df->hf", query, self.qjl_matrix, optimize=True).astype(np.float32)
        signs = _unpack_sign_bits(self.qjl_bits, self.qjl_matrix.shape[0]).astype(np.float32) * 2.0 - 1.0
        correction = np.einsum("hnd,hd->hn", signs, q_proj, optimize=True).astype(np.float32)
        correction *= math.sqrt(math.pi / 2.0) / float(self.qjl_matrix.shape[0])
        correction *= self.residual_norms.astype(np.float32).squeeze(-1)
        correction /= math.sqrt(float(head_dim))
        correction *= float(score_weight)
        return correction.astype(np.float32)

    def to_float32(self) -> np.ndarray:
        # In this runtime we use QJL as a score correction for K, not as a full-vector decode.
        # Returning the stable 4-bit base avoids injecting high-variance residual noise into V/context.
        return self._base_float32().astype(np.float32)

    def approximate_scores(self, query: np.ndarray, *, head_dim: int) -> np.ndarray:
        """Compute approximate Q·K^T scores using 4-bit base + QJL score correction.

        This combines the base 4-bit approximate scores with the existing QJL
        score correction, producing a better estimate than 4-bit alone.

        Args:
            query: shape (num_heads, head_dim)
            head_dim: dimension per head for scaling

        Returns:
            shape (num_heads, seq_len) approximate attention scores (pre-softmax)
        """
        # Base 4-bit scores
        base_scores = super().approximate_scores(query, head_dim=head_dim)
        # Add QJL score correction (already scaled by 1/sqrt(head_dim) internally)
        correction = self.score_correction(query, head_dim=head_dim, score_weight=0.25)
        return (base_scores + correction).astype(np.float32)

    def materialize_indices(self, indices: np.ndarray) -> np.ndarray:
        """Materialize only selected token positions to float32.

        Uses the stable 4-bit base (no QJL residual for V materialization),
        consistent with to_float32 behavior.

        Args:
            indices: shape (num_heads, topk) integer indices into seq_len dimension

        Returns:
            shape (num_heads, topk, head_dim) float32 values
        """
        return super().materialize_indices(indices)

    def append_compressed(self, values: np.ndarray) -> "_TurboQJLKVArray":
        """Append new token(s) to the compressed array without full materialization.

        Extends the 4-bit base and computes QJL residual for the new tokens.

        Args:
            values: shape (num_heads, new_tokens, head_dim) float32 values

        Returns:
            New _TurboQJLKVArray with appended tokens
        """
        values = np.asarray(values, dtype=np.float32)
        prev_length = self.length
        # Extend 4-bit base
        norms = np.linalg.norm(values, axis=-1, keepdims=True).astype(np.float32)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        unit = values / safe_norms
        rotated = self.rotation.forward(unit)
        quantized = self.codebook.quantize(rotated)
        new_packed = _pack_nibbles(quantized)
        new_norms_fp16 = norms.astype(np.float16)
        # Compute base reconstruction for residual
        base_rotated = self.codebook.dequantize(quantized)
        base_unit = self.rotation.inverse(base_rotated)
        base_reconstructed = base_unit * norms
        # QJL residual
        residual = values - base_reconstructed
        residual_norms = np.linalg.norm(residual, axis=-1, keepdims=True).astype(np.float32)
        safe_residual_norms = np.where(residual_norms > 1e-8, residual_norms, 1.0)
        residual_unit = residual / safe_residual_norms
        projected = np.einsum("...d,df->...f", residual_unit, self.qjl_matrix, optimize=True).astype(np.float32)
        new_qjl_bits = _pack_sign_bits(projected >= 0)
        new_residual_norms = residual_norms.astype(np.float16)
        # Build result
        result = _TurboQJLKVArray.__new__(_TurboQJLKVArray)
        result.rotation = self.rotation
        result.codebook = self.codebook
        result.qjl_matrix = self.qjl_matrix
        result._block_summary_cache = _append_block_summary_cache(
            self._block_summary_cache,
            prev_length=prev_length,
            appended_values=base_rotated * norms.astype(np.float32),
        )
        result.packed = np.concatenate([self.packed, new_packed], axis=1)
        result.norms = np.concatenate([self.norms, new_norms_fp16], axis=1)
        result.qjl_bits = np.concatenate([self.qjl_bits, new_qjl_bits], axis=1)
        result.residual_norms = np.concatenate([self.residual_norms, new_residual_norms], axis=1)
        return result


class _HotWindowKVArray:
    kind = "hot-window"

    def __init__(
        self,
        *,
        cold: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | None,
        hot: np.ndarray,
    ) -> None:
        self.cold = cold
        self.hot = np.asarray(hot, dtype=np.float32)

    @property
    def cold_length(self) -> int:
        if self.cold is None:
            return 0
        if isinstance(self.cold, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
            return self.cold.length
        return int(self.cold.shape[1])

    @property
    def hot_length(self) -> int:
        return int(self.hot.shape[1])

    @property
    def length(self) -> int:
        return self.cold_length + self.hot_length

    @property
    def nbytes(self) -> int:
        cold_bytes = 0
        if self.cold is not None:
            if isinstance(self.cold, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
                cold_bytes = self.cold.nbytes
            else:
                cold_bytes = int(self.cold.nbytes)
        return int(cold_bytes + self.hot.nbytes)

    def to_float32(self) -> np.ndarray:
        cold_values: np.ndarray | None = None
        if self.cold is not None:
            if isinstance(self.cold, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
                cold_values = self.cold.to_float32()
            else:
                cold_values = np.asarray(self.cold, dtype=np.float32)
        if cold_values is None or cold_values.shape[1] == 0:
            return self.hot.astype(np.float32)
        if self.hot.shape[1] == 0:
            return cold_values.astype(np.float32)
        return np.concatenate([cold_values.astype(np.float32), self.hot.astype(np.float32)], axis=1)

    @property
    def supports_selective(self) -> bool:
        """Whether the cold prefix supports approximate scoring."""
        return (
            self.cold is not None
            and isinstance(self.cold, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray))
            and self.cold_length > 0
        )

    def cold_approximate_scores(self, query: np.ndarray, *, head_dim: int) -> np.ndarray:
        """Compute approximate scores for the cold prefix only.

        Args:
            query: shape (num_heads, head_dim)
            head_dim: dimension per head

        Returns:
            shape (num_heads, cold_length) approximate scores
        """
        if not self.supports_selective:
            raise ValueError("cold prefix does not support approximate scoring")
        return self.cold.approximate_scores(query, head_dim=head_dim)

    def cold_approximate_block_scores(self, query: np.ndarray, *, head_dim: int, block_size: int) -> np.ndarray:
        if not self.supports_selective:
            raise ValueError("cold prefix does not support approximate scoring")
        return self.cold.approximate_block_scores(query, head_dim=head_dim, block_size=block_size)

    def cold_materialize_indices(self, indices: np.ndarray) -> np.ndarray:
        """Materialize selected cold prefix positions to float32.

        Args:
            indices: shape (num_heads, topk) integer indices into cold seq_len

        Returns:
            shape (num_heads, topk, head_dim) float32 values
        """
        if not self.supports_selective:
            raise ValueError("cold prefix does not support selective materialization")
        return self.cold.materialize_indices(indices)

    def append_token(self, new_kv: np.ndarray, max_hot: int, store_fn) -> "_HotWindowKVArray":
        """Append a new token to the hot window, spilling to compressed cold if needed.

        Args:
            new_kv: shape (num_heads, 1, head_dim) new token
            max_hot: maximum size of the hot window
            store_fn: callable to compress spilled tokens, e.g. Engine._store_compact_kv_cache

        Returns:
            Updated _HotWindowKVArray instance
        """
        new_kv = np.asarray(new_kv, dtype=np.float32)
        hot = np.concatenate([self.hot, new_kv], axis=1)
        
        if hot.shape[1] <= max_hot:
            return _HotWindowKVArray(cold=self.cold, hot=hot)
            
        # Spill the oldest hot token to cold
        spill_len = hot.shape[1] - max_hot
        spill = hot[:, :spill_len, :]
        new_hot = hot[:, spill_len:, :]
        
        if self.cold is None:
            new_cold = store_fn(spill)
        elif hasattr(self.cold, "append_compressed"):
            new_cold = self.cold.append_compressed(spill)
        else:
            # Fallback for uncompressed cold or unsupported types
            old_cold_fp32 = self.cold if isinstance(self.cold, np.ndarray) else self.cold.to_float32()
            new_cold = store_fn(np.concatenate([old_cold_fp32, spill], axis=1))

        return _HotWindowKVArray(cold=new_cold, hot=new_hot)


def _session_meta_matches_engine(
    meta: dict[str, Any],
    *,
    export_dir: Path,
    kv_cache_precision: str,
    kv_key_precision: str | None = None,
    kv_value_precision: str | None = None,
    kv_quant_seed: int,
    kv_rotation_mode: str,
    kv_hot_window: int,
    kv_topk: int,
    kv_index_refresh_interval: int,
    kv_block_size: int,
    kv_layer_share_stride: int,
    kv_calibration_tokens: int | None = None,
    kv_adaptive_high_kurtosis: float | None = None,
    kv_adaptive_medium_kurtosis: float | None = None,
) -> None:
    if Path(meta["export_dir"]).resolve() != export_dir.resolve():
        raise ValueError("session export_dir does not match this engine")
    expected = {
        "kv_cache_precision": kv_cache_precision,
        "kv_key_precision": kv_key_precision,
        "kv_value_precision": kv_value_precision,
        "kv_quant_seed": int(kv_quant_seed),
        "kv_rotation_mode": kv_rotation_mode,
        "kv_hot_window": int(kv_hot_window),
        "kv_topk": int(kv_topk),
        "kv_index_refresh_interval": int(kv_index_refresh_interval),
        "kv_block_size": int(kv_block_size),
        "kv_layer_share_stride": int(kv_layer_share_stride),
    }
    if kv_calibration_tokens is not None:
        expected["kv_calibration_tokens"] = int(kv_calibration_tokens)
    if kv_adaptive_high_kurtosis is not None:
        expected["kv_adaptive_high_kurtosis"] = float(kv_adaptive_high_kurtosis)
    if kv_adaptive_medium_kurtosis is not None:
        expected["kv_adaptive_medium_kurtosis"] = float(kv_adaptive_medium_kurtosis)
    mismatches = []
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            mismatches.append(f"{key}={meta[key]!r} (session) != {value!r} (engine)")
    if mismatches:
        raise ValueError("session metadata does not match this engine: " + ", ".join(mismatches))


def _array_payload_matches(expected: np.ndarray, actual: np.ndarray, *, atol: float = 1e-6) -> bool:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        return False
    if expected_array.dtype.kind in {"i", "u", "b"} and actual_array.dtype.kind in {"i", "u", "b"}:
        return np.array_equal(expected_array, actual_array)
    return bool(np.allclose(expected_array.astype(np.float32), actual_array.astype(np.float32), atol=atol, rtol=0.0))


def _serialize_adaptive_kurtosis_state(
    state: list[dict[str, _KurtosisAccumulator]] | None,
) -> list[dict[str, dict[str, float | int]]] | None:
    if state is None:
        return None
    return [
        {
            "k": layer_state["k"].to_json(),
            "v": layer_state["v"].to_json(),
        }
        for layer_state in state
    ]


def _deserialize_adaptive_kurtosis_state(
    payload: list[dict[str, Any]] | None,
) -> list[dict[str, _KurtosisAccumulator]] | None:
    if payload is None:
        return None
    state: list[dict[str, _KurtosisAccumulator]] = []
    for layer_payload in payload:
        state.append(
            {
                "k": _KurtosisAccumulator.from_json(dict(layer_payload.get("k", {}))),
                "v": _KurtosisAccumulator.from_json(dict(layer_payload.get("v", {}))),
            }
        )
    return state


def _selective_candidate_topk(
    cold_array: "_TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | None",
    *,
    cold_length: int,
    effective_topk: int,
) -> int:
    if cold_array is None or cold_length <= 0 or effective_topk <= 0:
        return 0
    if isinstance(cold_array, (_Turbo4BitKVArray, _TurboQJLKVArray)):
        return min(cold_length, max(effective_topk * 2, effective_topk + 4))
    return min(cold_length, effective_topk)


def _merge_selective_candidate_indices(
    cached_indices: np.ndarray,
    *,
    cold_length: int,
    new_start: int,
    max_candidates: int,
) -> np.ndarray:
    cached = np.asarray(cached_indices, dtype=np.int64)
    if cached.ndim != 2:
        raise ValueError("cached_indices must have shape (heads, candidates)")
    max_candidates = max(int(max_candidates), 0)
    if max_candidates == 0:
        return np.empty((cached.shape[0], 0), dtype=np.int64)
    new_indices = (
        np.arange(max(int(new_start), 0), int(cold_length), dtype=np.int64)
        if int(cold_length) > int(new_start)
        else np.empty((0,), dtype=np.int64)
    )
    rows: list[np.ndarray] = []
    for head_index in range(cached.shape[0]):
        row = cached[head_index]
        row = row[(row >= 0) & (row < int(cold_length))]
        if new_indices.size:
            row = np.concatenate([row, new_indices], axis=0)
        if row.size == 0:
            rows.append(np.empty((0,), dtype=np.int64))
            continue
        row = np.unique(row)
        if row.size > max_candidates:
            row = row[-max_candidates:]
        rows.append(row.astype(np.int64, copy=False))
    width = max((row.size for row in rows), default=0)
    if width == 0:
        return np.empty((cached.shape[0], 0), dtype=np.int64)
    width = min(width, max_candidates)
    merged = np.empty((cached.shape[0], width), dtype=np.int64)
    for head_index, row in enumerate(rows):
        if row.size < width:
            pad = np.full((width - row.size,), row[-1] if row.size else 0, dtype=np.int64)
            row = np.concatenate([row, pad], axis=0)
        elif row.size > width:
            row = row[-width:]
        merged[head_index] = row
    return merged


def _selected_score_threshold(scores: np.ndarray, *, topk: int) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or int(topk) <= 0:
        return np.full((matrix.shape[0] if matrix.ndim == 2 else 0,), -np.inf, dtype=np.float32)
    kth = min(int(topk), matrix.shape[1])
    return np.partition(matrix, -kth, axis=1)[:, -kth].astype(np.float32)


def _block_shortlist_needs_expansion(
    block_scores: np.ndarray,
    shortlist_indices: np.ndarray,
    shortlist_scores: np.ndarray,
    *,
    effective_topk: int,
    block_size: int,
    relative_margin: float = 0.02,
) -> bool:
    if int(effective_topk) <= 0:
        return False
    scores = np.asarray(block_scores, dtype=np.float32)
    shortlist_indices = np.asarray(shortlist_indices, dtype=np.int64)
    shortlist_scores = np.asarray(shortlist_scores, dtype=np.float32)
    if (
        scores.ndim != 2
        or scores.shape[1] == 0
        or shortlist_indices.ndim != 2
        or shortlist_indices.shape[1] == 0
        or shortlist_scores.ndim != 2
        or shortlist_scores.shape[1] == 0
    ):
        return False
    block_size = max(int(block_size), 1)
    thresholds = _selected_score_threshold(shortlist_scores, topk=effective_topk)
    for head_index in range(scores.shape[0]):
        chosen_blocks = np.unique(shortlist_indices[head_index] // block_size)
        if chosen_blocks.size >= scores.shape[1]:
            continue
        excluded = np.ones(scores.shape[1], dtype=bool)
        excluded[np.clip(chosen_blocks, 0, scores.shape[1] - 1)] = False
        if not np.any(excluded):
            continue
        excluded_upper = float(np.max(scores[head_index, excluded]))
        threshold = float(thresholds[head_index])
        margin = float(relative_margin) * max(1.0, abs(threshold))
        if excluded_upper > (threshold + margin):
            return True
    return False


def _selected_token_set(indices: np.ndarray | None) -> np.ndarray:
    if not isinstance(indices, np.ndarray) or indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    flattened = np.asarray(indices, dtype=np.int64).reshape(-1)
    flattened = flattened[flattened >= 0]
    if flattened.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(flattened).astype(np.int64)


def _new_cross_layer_overlap_stats(num_layers: int) -> dict[str, Any]:
    return {
        "global_samples": 0,
        "global_jaccard_sum": 0.0,
        "global_high_overlap_hits": 0,
        "adjacent_pairs": [
            {
                "pair": [layer_index, layer_index + 1],
                "samples": 0,
                "jaccard_sum": 0.0,
                "high_overlap_hits": 0,
                "intersection_sum": 0,
                "union_sum": 0,
            }
            for layer_index in range(max(int(num_layers) - 1, 0))
        ],
    }


def _record_cross_layer_overlap(
    stats: dict[str, Any],
    *,
    pair_index: int,
    previous_indices: np.ndarray | None,
    current_indices: np.ndarray | None,
    high_overlap_threshold: float = 0.70,
) -> None:
    previous_tokens = _selected_token_set(previous_indices)
    current_tokens = _selected_token_set(current_indices)
    if previous_tokens.size == 0 or current_tokens.size == 0:
        return
    if pair_index < 0 or pair_index >= len(stats.get("adjacent_pairs", [])):
        return
    intersection = int(np.intersect1d(previous_tokens, current_tokens, assume_unique=True).size)
    union = int(np.union1d(previous_tokens, current_tokens).size)
    if union <= 0:
        return
    jaccard = float(intersection) / float(union)
    pair_stats = stats["adjacent_pairs"][pair_index]
    pair_stats["samples"] = int(pair_stats.get("samples", 0)) + 1
    pair_stats["jaccard_sum"] = float(pair_stats.get("jaccard_sum", 0.0)) + jaccard
    pair_stats["intersection_sum"] = int(pair_stats.get("intersection_sum", 0)) + intersection
    pair_stats["union_sum"] = int(pair_stats.get("union_sum", 0)) + union
    if jaccard >= float(high_overlap_threshold):
        pair_stats["high_overlap_hits"] = int(pair_stats.get("high_overlap_hits", 0)) + 1
    stats["global_samples"] = int(stats.get("global_samples", 0)) + 1
    stats["global_jaccard_sum"] = float(stats.get("global_jaccard_sum", 0.0)) + jaccard
    if jaccard >= float(high_overlap_threshold):
        stats["global_high_overlap_hits"] = int(stats.get("global_high_overlap_hits", 0)) + 1


def _summarize_cross_layer_overlap_stats(stats: dict[str, Any]) -> dict[str, Any]:
    global_samples = int(stats.get("global_samples", 0))
    adjacent_pairs = []
    for pair_stats in stats.get("adjacent_pairs", []):
        samples = int(pair_stats.get("samples", 0))
        adjacent_pairs.append(
            {
                "pair": list(pair_stats.get("pair", [])),
                "samples": samples,
                "mean_jaccard": float(pair_stats.get("jaccard_sum", 0.0)) / float(samples) if samples else 0.0,
                "high_overlap_rate": (
                    float(pair_stats.get("high_overlap_hits", 0)) / float(samples) if samples else 0.0
                ),
                "mean_intersection": float(pair_stats.get("intersection_sum", 0)) / float(samples) if samples else 0.0,
                "mean_union": float(pair_stats.get("union_sum", 0)) / float(samples) if samples else 0.0,
            }
        )
    return {
        "global_samples": global_samples,
        "mean_jaccard": float(stats.get("global_jaccard_sum", 0.0)) / float(global_samples) if global_samples else 0.0,
        "high_overlap_rate": (
            float(stats.get("global_high_overlap_hits", 0)) / float(global_samples) if global_samples else 0.0
        ),
        "adjacent_pairs": adjacent_pairs,
    }


def _should_use_selective_attention(
    cold_array: "_TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | None",
    *,
    cold_length: int,
    effective_topk: int,
) -> bool:
    if cold_array is None or cold_length <= 0 or effective_topk <= 0 or effective_topk >= cold_length:
        return False
    # Approximate scoring only starts paying for itself once the cold prefix is
    # materially larger than the exact subset we will attend over. In practice
    # the 4-bit packed-score path needs a longer runway on CPU/NumPy than int8.
    if isinstance(cold_array, _TurboQJLKVArray):
        return cold_length >= max(32, effective_topk * 4)
    if isinstance(cold_array, _Turbo4BitKVArray):
        return cold_length >= max(64, effective_topk * 8)
    return cold_length >= max(24, effective_topk * 4)


_TurboQuantizedKVArray = _TurboInt8KVArray


def _cache_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        "entries": after["entries"],
        "hits": after["hits"] - before["hits"],
        "misses": after["misses"] - before["misses"],
    }


def clear_session_runtime_cache(export_dir: str | Path | None = None) -> None:
    if export_dir is None:
        _SESSION_CACHES.clear()
        return
    _SESSION_CACHES.pop(str(Path(export_dir).resolve()), None)


def _resolve_runtime_cache(
    export_dir: str | Path,
    *,
    cache_mode: str,
    max_tensor_bytes: int = 256 * 1024,
) -> _TensorRuntimeCache:
    if cache_mode == "none":
        return _TensorRuntimeCache(max_tensor_bytes=0)
    if cache_mode == "fresh":
        return _TensorRuntimeCache(max_tensor_bytes=max_tensor_bytes)
    if cache_mode == "session":
        key = str(Path(export_dir).resolve())
        cache = _SESSION_CACHES.get(key)
        if cache is None:
            cache = _TensorRuntimeCache(max_tensor_bytes=max_tensor_bytes)
            _SESSION_CACHES[key] = cache
        return cache
    raise ValueError(f"unsupported cache_mode: {cache_mode}")


def _process_rss_mb() -> float:
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if counters.WorkingSetSize <= 0:
            return float("nan")
        return float(counters.WorkingSetSize) / (1024 * 1024)
    except Exception:
        return float("nan")


def _safe_tensor_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "tensor"


def _is_supported_array(array: np.ndarray) -> bool:
    return array.ndim in (1, 2) and np.issubdtype(array.dtype, np.number)


def _normalize_array(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)
    return np.asarray(array)


def export_tensor_map(
    tensor_map: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    block_rows: int = 256,
    compression_level: int = 6,
    model_ref: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name, array in tensor_map.items():
        arr = np.asarray(array)
        if not _is_supported_array(arr):
            skipped.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "reason": "only 1D and 2D numeric tensors are exported in this prototype",
                }
            )
            continue

        normalized = _normalize_array(arr)
        tensor_dir = output_dir / "tensors" / _safe_tensor_dir(name)
        create_store(
            normalized,
            tensor_dir,
            block_rows=block_rows,
            compression_level=compression_level,
            extra={
                "tensor_name": name,
                "source": "huggingface",
                "original_dtype": str(arr.dtype),
            },
        )
        transpose_path: str | None = None
        if normalized.ndim == 2:
            transpose_dir = tensor_dir / "transpose"
            create_store(
                normalized.T,
                transpose_dir,
                block_rows=block_rows,
                compression_level=compression_level,
                extra={
                    "tensor_name": name,
                    "source": "huggingface",
                    "transpose_of": name,
                    "original_dtype": str(arr.dtype),
                },
            )
            transpose_path = str(transpose_dir.relative_to(output_dir))

        stats = store_stats(tensor_dir)
        exported.append(
            {
                "name": name,
                "path": str(tensor_dir.relative_to(output_dir)),
                "transpose_path": transpose_path,
                "shape": list(normalized.shape),
                "dtype": str(normalized.dtype),
                "compression_ratio": stats["compression_ratio"],
                "raw_bytes": stats["raw_bytes"],
                "compressed_bytes": stats["compressed_bytes"],
            }
        )

    manifest = {
        "format": "helix-proto-hf-manifest",
        "model_ref": model_ref,
        "config": config or {},
        "block_rows": block_rows,
        "compression_level": compression_level,
        "exported": exported,
        "skipped": skipped,
    }
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_local_npz(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    block_rows: int = 256,
) -> dict[str, Any]:
    with np.load(npz_path) as data:
        tensor_map = {key: data[key] for key in data.files}
    return export_tensor_map(
        tensor_map,
        output_dir,
        block_rows=block_rows,
        model_ref=str(npz_path),
        config={"source": "npz"},
    )


def _tensor_to_numpy_exportable(tensor: Any) -> np.ndarray:
    import torch

    cpu_tensor = tensor.detach().cpu()
    try:
        return cpu_tensor.numpy()
    except TypeError:
        if torch.is_floating_point(cpu_tensor):
            return cpu_tensor.to(dtype=torch.float32).numpy()
        if cpu_tensor.dtype == torch.bool:
            return cpu_tensor.to(dtype=torch.uint8).numpy()
        return cpu_tensor.to(dtype=torch.int64).numpy()


def _export_torch_model(
    model: Any,
    config: Any,
    output_dir: str | Path,
    *,
    model_ref: str | None,
    block_rows: int,
    compression_level: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    tensor_map: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            tensor_map[name] = _tensor_to_numpy_exportable(tensor)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return export_tensor_map(
        tensor_map,
        output_dir,
        block_rows=block_rows,
        compression_level=compression_level,
        model_ref=model_ref,
        config=config_dict,
    )


def export_huggingface_model(
    model_ref: str,
    output_dir: str | Path,
    *,
    block_rows: int = 256,
    compression_level: int = 6,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
    except ImportError as exc:
        raise RuntimeError(
            "convert-hf needs optional dependencies: pip install transformers torch"
        ) from exc

    config = AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    last_error: Exception | None = None
    for model_cls in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            model = model_cls.from_pretrained(
                model_ref,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            return _export_torch_model(
                model,
                config,
                output_dir,
                model_ref=model_ref,
                block_rows=block_rows,
                compression_level=compression_level,
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            continue

    raise RuntimeError(f"could not load model {model_ref!r} with available HF loaders") from last_error


def load_manifest(output_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(output_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))


def tensor_store_map(output_dir: str | Path) -> dict[str, Path]:
    manifest = load_manifest(output_dir)
    return {item["name"]: Path(output_dir) / item["path"] for item in manifest["exported"]}


def _layer_norm_last_dim(array: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = array.mean(axis=-1, keepdims=True)
    var = ((array - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (array - mean) / np.sqrt(var + eps)
    return normalized * weight + bias


def _gelu(array: np.ndarray) -> np.ndarray:
    return 0.5 * array * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (array + 0.044715 * (array**3))))


def _softmax(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _streaming_linear_rows(store: Path, hidden_states: np.ndarray, bias: np.ndarray) -> np.ndarray:
    rows = [streaming_matvec(store, row).astype(np.float32) + bias for row in hidden_states]
    return np.stack(rows, axis=0)


def _streaming_right_linear_rows(store: Path, hidden_states: np.ndarray, bias: np.ndarray) -> np.ndarray:
    transpose_store = store / "transpose"
    rows = [streaming_matvec(transpose_store, row).astype(np.float32) + bias for row in hidden_states]
    return np.stack(rows, axis=0)


def _streaming_right_linear_vector(store: Path, hidden_state: np.ndarray, bias: np.ndarray) -> np.ndarray:
    transpose_store = store / "transpose"
    return streaming_matvec(transpose_store, hidden_state).astype(np.float32) + bias


def _bert_embeddings(
    stores: dict[str, Path],
    *,
    token_ids: list[int],
    token_type_ids: list[int],
    eps: float,
) -> np.ndarray:
    word_embeddings = load_tensor_rows(stores["bert.embeddings.word_embeddings.weight"], token_ids)
    pos_embeddings = load_tensor_rows(
        stores["bert.embeddings.position_embeddings.weight"],
        list(range(len(token_ids))),
    )
    type_embeddings = load_tensor_rows(stores["bert.embeddings.token_type_embeddings.weight"], token_type_ids)
    hidden_states = np.stack(
        [
            word_embeddings[position_id] + pos_embeddings[position_id] + type_embeddings[position_id]
            for position_id, _ in enumerate(token_ids)
        ],
        axis=0,
    ).astype(np.float32)
    return _layer_norm_last_dim(
        hidden_states,
        load_full_tensor(stores["bert.embeddings.LayerNorm.weight"]),
        load_full_tensor(stores["bert.embeddings.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)


def _bert_encoder_layer(
    hidden_states: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
) -> np.ndarray:
    layer_prefix = f"bert.encoder.layer.{layer_index}"
    seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    q = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.self.query.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.self.query.bias"]),
    )
    k = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.self.key.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.self.key.bias"]),
    )
    v = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.self.value.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.self.value.bias"]),
    )

    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(float(head_dim))
    probs = _softmax(scores, axis=-1)
    context = np.matmul(probs, v).transpose(1, 0, 2).reshape(seq_len, hidden_size)

    attn_output = _streaming_linear_rows(
        stores[f"{layer_prefix}.attention.output.dense.weight"],
        context,
        load_full_tensor(stores[f"{layer_prefix}.attention.output.dense.bias"]),
    )
    hidden_states = _layer_norm_last_dim(
        attn_output + hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.attention.output.LayerNorm.weight"]),
        load_full_tensor(stores[f"{layer_prefix}.attention.output.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)

    intermediate = _streaming_linear_rows(
        stores[f"{layer_prefix}.intermediate.dense.weight"],
        hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.intermediate.dense.bias"]),
    )
    intermediate = _gelu(intermediate)

    output = _streaming_linear_rows(
        stores[f"{layer_prefix}.output.dense.weight"],
        intermediate,
        load_full_tensor(stores[f"{layer_prefix}.output.dense.bias"]),
    )
    return _layer_norm_last_dim(
        output + hidden_states,
        load_full_tensor(stores[f"{layer_prefix}.output.LayerNorm.weight"]),
        load_full_tensor(stores[f"{layer_prefix}.output.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)


def infer_bert_mlm_logits(
    export_dir: str | Path,
    *,
    token_ids: list[int],
    token_type_ids: list[int] | None = None,
) -> np.ndarray:
    stores = tensor_store_map(export_dir)
    config = load_manifest(export_dir).get("config", {})
    eps = float(config.get("layer_norm_eps", 1e-12))
    num_heads = int(config["num_attention_heads"])
    num_layers = int(config["num_hidden_layers"])

    token_type_ids = token_type_ids or [0] * len(token_ids)
    hidden_states = _bert_embeddings(
        stores,
        token_ids=token_ids,
        token_type_ids=token_type_ids,
        eps=eps,
    )

    for layer_index in range(num_layers):
        hidden_states = _bert_encoder_layer(
            hidden_states,
            stores=stores,
            layer_index=layer_index,
            num_heads=num_heads,
            eps=eps,
        )

    transformed = _streaming_linear_rows(
        stores["cls.predictions.transform.dense.weight"],
        hidden_states,
        load_full_tensor(stores["cls.predictions.transform.dense.bias"]),
    )
    transformed = _gelu(transformed)
    transformed = _layer_norm_last_dim(
        transformed,
        load_full_tensor(stores["cls.predictions.transform.LayerNorm.weight"]),
        load_full_tensor(stores["cls.predictions.transform.LayerNorm.bias"]),
        eps,
    ).astype(np.float32)
    return _streaming_linear_rows(
        stores["cls.predictions.decoder.weight"],
        transformed,
        load_full_tensor(stores["cls.predictions.bias"]),
    )


def infer_zero_layer_bert_mlm_logits(
    export_dir: str | Path,
    *,
    token_id: int,
    position_id: int = 0,
    token_type_id: int = 0,
) -> np.ndarray:
    logits = infer_bert_mlm_logits(
        export_dir,
        token_ids=[token_id],
        token_type_ids=[token_type_id],
    )
    return logits[position_id]


def infer_zero_layer_bert_mlm(
    export_dir: str | Path,
    *,
    token_id: int,
    position_id: int = 0,
    token_type_id: int = 0,
    top_k: int = 5,
) -> dict[str, Any]:
    logits = infer_zero_layer_bert_mlm_logits(
        export_dir,
        token_id=token_id,
        position_id=position_id,
        token_type_id=token_type_id,
    )
    top_indices = np.argsort(logits)[-top_k:][::-1]
    return {
        "token_id": token_id,
        "top_indices": top_indices.tolist(),
        "top_logits": [float(logits[idx]) for idx in top_indices],
        "vocab_size": int(logits.shape[0]),
    }


def infer_one_layer_bert_mlm_logits(
    export_dir: str | Path,
    *,
    token_ids: list[int],
    token_type_ids: list[int] | None = None,
) -> np.ndarray:
    return infer_bert_mlm_logits(export_dir, token_ids=token_ids, token_type_ids=token_type_ids)


def _gpt2_block(
    hidden_states: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
    runtime_cache: _TensorRuntimeCache | None = None,
) -> np.ndarray:
    runtime_cache = runtime_cache or _TensorRuntimeCache()
    prefix = f"transformer.h.{layer_index}"
    seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    ln1 = _layer_norm_last_dim(
        hidden_states,
        runtime_cache.tensor(stores[f"{prefix}.ln_1.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_1.bias"]),
        eps,
    ).astype(np.float32)

    attn_proj = _streaming_right_linear_rows(
        stores[f"{prefix}.attn.c_attn.weight"],
        ln1,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_attn.bias"]),
    )
    q, k, v = np.split(attn_proj, 3, axis=-1)
    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(float(head_dim))
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal_mask[None, :, :], -1e9, scores)
    probs = _softmax(scores, axis=-1)
    context = np.matmul(probs, v).transpose(1, 0, 2).reshape(seq_len, hidden_size)

    attn_output = _streaming_right_linear_rows(
        stores[f"{prefix}.attn.c_proj.weight"],
        context,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_proj.bias"]),
    )
    hidden_states = hidden_states + attn_output

    ln2 = _layer_norm_last_dim(
        hidden_states,
        runtime_cache.tensor(stores[f"{prefix}.ln_2.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_2.bias"]),
        eps,
    ).astype(np.float32)
    mlp = _streaming_right_linear_rows(
        stores[f"{prefix}.mlp.c_fc.weight"],
        ln2,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_fc.bias"]),
    )
    mlp = _gelu(mlp)
    mlp = _streaming_right_linear_rows(
        stores[f"{prefix}.mlp.c_proj.weight"],
        mlp,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_proj.bias"]),
    )
    return hidden_states + mlp


def infer_gpt2_causal_lm_logits(
    export_dir: str | Path,
    *,
    token_ids: list[int],
) -> np.ndarray:
    stores = tensor_store_map(export_dir)
    config = load_manifest(export_dir).get("config", {})
    runtime_cache = _TensorRuntimeCache()
    eps = float(config.get("layer_norm_epsilon", 1e-5))
    num_heads = int(config["n_head"])
    num_layers = int(config["n_layer"])

    wte = runtime_cache.rows(stores["transformer.wte.weight"], token_ids)
    wpe = runtime_cache.rows(stores["transformer.wpe.weight"], list(range(len(token_ids))))
    hidden_states = np.stack(
        [wte[position_id] + wpe[position_id] for position_id, _ in enumerate(token_ids)],
        axis=0,
    ).astype(np.float32)

    for layer_index in range(num_layers):
        hidden_states = _gpt2_block(
            hidden_states,
            stores=stores,
            layer_index=layer_index,
            num_heads=num_heads,
            eps=eps,
            runtime_cache=runtime_cache,
        )

    hidden_states = _layer_norm_last_dim(
        hidden_states,
        runtime_cache.tensor(stores["transformer.ln_f.weight"]),
        runtime_cache.tensor(stores["transformer.ln_f.bias"]),
        eps,
    ).astype(np.float32)

    lm_head_store = stores.get("lm_head.weight", stores["transformer.wte.weight"])
    zeros = np.zeros(load_meta(lm_head_store).shape[0], dtype=np.float32)
    return _streaming_linear_rows(lm_head_store, hidden_states, zeros)


def _gpt2_step_with_kv(
    token_embedding: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
    past_k: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    past_v: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    runtime_cache: _TensorRuntimeCache | None = None,
    qjl_score_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runtime_cache = runtime_cache or _TensorRuntimeCache()
    prefix = f"transformer.h.{layer_index}"
    hidden_size = token_embedding.shape[0]
    head_dim = hidden_size // num_heads

    ln1 = _layer_norm_last_dim(
        token_embedding,
        runtime_cache.tensor(stores[f"{prefix}.ln_1.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_1.bias"]),
        eps,
    ).astype(np.float32)
    attn_proj = _streaming_right_linear_vector(
        stores[f"{prefix}.attn.c_attn.weight"],
        ln1,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_attn.bias"]),
    )
    q, k_new, v_new = np.split(attn_proj, 3)
    q = q.reshape(num_heads, head_dim)
    k_new = k_new.reshape(num_heads, 1, head_dim)
    v_new = v_new.reshape(num_heads, 1, head_dim)

    def _materialize_cache(
        cache: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    ) -> np.ndarray | None:
        if cache is None:
            return None
        if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray, _HotWindowKVArray)):
            return cache.to_float32().astype(np.float32)
        return np.asarray(cache, dtype=np.float32)

    past_k_values = _materialize_cache(past_k)
    past_v_values = _materialize_cache(past_v)

    if past_k_values is None:
        k_all = k_new
        v_all = v_new
    else:
        k_all = np.concatenate([past_k_values, k_new], axis=1)
        v_all = np.concatenate([past_v_values, v_new], axis=1)

    scores = np.einsum("hd,hnd->hn", q, k_all) / np.sqrt(float(head_dim))
    qjl_cache: _TurboQJLKVArray | None = None
    qjl_length = 0
    if isinstance(past_k, _HotWindowKVArray) and isinstance(past_k.cold, _TurboQJLKVArray):
        qjl_cache = past_k.cold
        qjl_length = past_k.cold_length
    elif isinstance(past_k, _TurboQJLKVArray):
        qjl_cache = past_k
        qjl_length = past_k.length
    if qjl_cache is not None and qjl_length > 0:
        scores[:, :qjl_length] += qjl_cache.score_correction(q, head_dim=head_dim, score_weight=qjl_score_weight)
    probs = _softmax(scores, axis=-1)
    context = np.einsum("hn,hnd->hd", probs, v_all).reshape(hidden_size)
    attn_output = _streaming_right_linear_vector(
        stores[f"{prefix}.attn.c_proj.weight"],
        context,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_proj.bias"]),
    )
    hidden = token_embedding + attn_output

    ln2 = _layer_norm_last_dim(
        hidden,
        runtime_cache.tensor(stores[f"{prefix}.ln_2.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_2.bias"]),
        eps,
    ).astype(np.float32)
    mlp = _streaming_right_linear_vector(
        stores[f"{prefix}.mlp.c_fc.weight"],
        ln2,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_fc.bias"]),
    )
    mlp = _gelu(mlp)
    mlp = _streaming_right_linear_vector(
        stores[f"{prefix}.mlp.c_proj.weight"],
        mlp,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_proj.bias"]),
    )
    return hidden + mlp, k_all.astype(np.float32), v_all.astype(np.float32)


def _gpt2_step_with_selective_kv(
    token_embedding: np.ndarray,
    *,
    stores: dict[str, Path],
    layer_index: int,
    num_heads: int,
    eps: float,
    past_k: _HotWindowKVArray,
    past_v: _HotWindowKVArray,
    kv_topk: int,
    kv_block_size: int,
    max_hot_window: int,
    store_k_fn,
    store_v_fn,
    selective_state: dict[str, Any] | None,
    shared_candidate_indices: np.ndarray | None,
    shared_cold_length: int,
    refresh_interval: int,
    selective_stats: dict[str, int] | None = None,
    runtime_cache: _TensorRuntimeCache | None = None,
    qjl_score_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray | _HotWindowKVArray, np.ndarray | _HotWindowKVArray]:
    """GPT2 step with selective attention on compressed cold KV prefix.

    Instead of materializing the entire cold prefix:
    1. Compute approximate scores on the cold compressed prefix
    2. Select top-K most relevant tokens per head
    3. Materialize only those tokens for exact attention
    4. Combine with exact hot window + new token

    This produces a real speedup when kv_topk << cold_length.
    """
    runtime_cache = runtime_cache or _TensorRuntimeCache()
    prefix = f"transformer.h.{layer_index}"
    hidden_size = token_embedding.shape[0]
    head_dim = hidden_size // num_heads

    ln1 = _layer_norm_last_dim(
        token_embedding,
        runtime_cache.tensor(stores[f"{prefix}.ln_1.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_1.bias"]),
        eps,
    ).astype(np.float32)
    attn_proj = _streaming_right_linear_vector(
        stores[f"{prefix}.attn.c_attn.weight"],
        ln1,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_attn.bias"]),
    )
    q, k_new, v_new = np.split(attn_proj, 3)
    q = q.reshape(num_heads, head_dim)
    k_new = k_new.reshape(num_heads, 1, head_dim)
    v_new = v_new.reshape(num_heads, 1, head_dim)

    # --- Selective attention path ---
    cold_length = past_k.cold_length if past_k is not None else 0
    cold_store = past_k.cold if past_k is not None else None
    hot_k = past_k.hot if past_k is not None else np.empty((num_heads, 0, head_dim), dtype=np.float32)
    hot_v = past_v.hot if past_v is not None else np.empty((num_heads, 0, head_dim), dtype=np.float32)
    effective_topk = min(kv_topk, cold_length) if cold_length > 0 else 0
    candidate_topk = _selective_candidate_topk(
        cold_store,
        cold_length=cold_length,
        effective_topk=effective_topk,
    )
    block_size = max(int(kv_block_size), 0)
    use_block_scoring = block_size > 1 and cold_length > block_size and candidate_topk > 0
    selective_stats = selective_stats if selective_stats is not None else {}
    block_scores: np.ndarray | None = None
    block_topk = 0
    block_shortlist: np.ndarray | None = None
    candidate_cap = max(candidate_topk * 8, effective_topk, block_size * 4 if use_block_scoring else 0, 1)
    state_indices = None if selective_state is None else selective_state.get("indices")
    state_cold_length = 0 if selective_state is None else int(selective_state.get("cold_length", 0))
    state_steps_since_full = (
        refresh_interval if selective_state is None else int(selective_state.get("steps_since_full_refresh", refresh_interval))
    )
    can_reuse_shortlist = (
        effective_topk > 0
        and candidate_topk > 0
        and isinstance(state_indices, np.ndarray)
        and state_indices.size > 0
        and cold_length >= state_cold_length
        and (state_steps_since_full + 1) < max(int(refresh_interval), 1)
    )
    can_share_shortlist = (
        effective_topk > 0
        and candidate_topk > 0
        and isinstance(shared_candidate_indices, np.ndarray)
        and shared_candidate_indices.size > 0
        and int(shared_cold_length) == int(cold_length)
    )

    if (
        _should_use_selective_attention(
            cold_store,
            cold_length=cold_length,
            effective_topk=effective_topk,
        )
        and past_k.supports_selective
        and past_v.supports_selective
    ):
        if use_block_scoring:
            block_scores = past_k.cold_approximate_block_scores(q, head_dim=head_dim, block_size=block_size)
            block_topk = min(
                block_scores.shape[1],
                max(int(math.ceil(candidate_topk / float(block_size))) * 4 + 2, 1),
            )
            selected_blocks = np.argpartition(block_scores, -block_topk, axis=-1)[:, -block_topk:]
            block_shortlist = _expand_block_indices(
                selected_blocks,
                cold_length=cold_length,
                block_size=block_size,
            )
            selective_stats["block_pruned_steps"] = int(selective_stats.get("block_pruned_steps", 0)) + 1
            selective_stats["block_rows_scored"] = int(selective_stats.get("block_rows_scored", 0)) + int(
                block_scores.shape[1]
            )

        def _materialize_shortlist(shortlist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            shortlisted_k = past_k.cold_materialize_indices(shortlist)
            shortlisted_v = past_v.cold_materialize_indices(shortlist)
            shortlist_scores_local = np.einsum(
                "hd,hnd->hn", q, shortlisted_k, optimize=True
            ) / np.sqrt(float(head_dim))
            return shortlisted_k, shortlisted_v, shortlist_scores_local.astype(np.float32)

        if can_share_shortlist:
            shortlist_indices = np.asarray(shared_candidate_indices, dtype=np.int64)
            if block_shortlist is not None:
                shortlist_indices = _merge_candidate_index_sets(
                    block_shortlist,
                    shortlist_indices,
                    max_candidates=max(candidate_cap, int(block_shortlist.shape[1])),
                )
            shortlisted_cold_k, shortlisted_cold_v, shortlist_scores = _materialize_shortlist(shortlist_indices)
            selective_stats["cross_layer_share_hits"] = int(selective_stats.get("cross_layer_share_hits", 0)) + 1
            selective_stats["cross_layer_share_candidate_rows"] = int(
                selective_stats.get("cross_layer_share_candidate_rows", 0)
            ) + int(shortlist_indices.shape[1])
        elif can_reuse_shortlist:
            shortlist_indices = _merge_selective_candidate_indices(
                state_indices,
                cold_length=cold_length,
                new_start=state_cold_length,
                max_candidates=candidate_topk,
            )
            if block_shortlist is not None:
                shortlist_indices = _merge_candidate_index_sets(
                    block_shortlist,
                    shortlist_indices,
                    max_candidates=max(candidate_cap, int(block_shortlist.shape[1])),
                )
            shortlisted_cold_k, shortlisted_cold_v, shortlist_scores = _materialize_shortlist(shortlist_indices)
            selective_stats["reuse_hits"] = int(selective_stats.get("reuse_hits", 0)) + 1
            selective_stats["reuse_new_tokens"] = int(selective_stats.get("reuse_new_tokens", 0)) + max(
                cold_length - state_cold_length,
                0,
            )
            selective_stats["reuse_candidate_rows"] = int(selective_stats.get("reuse_candidate_rows", 0)) + int(
                shortlist_indices.shape[1]
            )
        else:
            if block_shortlist is not None:
                shortlist_indices = block_shortlist
            else:
                cold_approx_scores = past_k.cold_approximate_scores(q, head_dim=head_dim)
                shortlist_indices = np.argpartition(
                    cold_approx_scores, -candidate_topk, axis=-1
                )[:, -candidate_topk:]
            shortlisted_cold_k, shortlisted_cold_v, shortlist_scores = _materialize_shortlist(shortlist_indices)
            selective_stats["full_refreshes"] = int(selective_stats.get("full_refreshes", 0)) + 1

        needs_confidence_fallback = False
        if block_scores is not None and _block_shortlist_needs_expansion(
            block_scores,
            shortlist_indices,
            shortlist_scores,
            effective_topk=effective_topk,
            block_size=block_size,
        ):
            expanded_block_topk = min(block_scores.shape[1], max(block_topk * 2, block_topk + 2))
            if expanded_block_topk > block_topk:
                expanded_blocks = np.argpartition(block_scores, -expanded_block_topk, axis=-1)[:, -expanded_block_topk:]
                expanded_shortlist = _expand_block_indices(
                    expanded_blocks,
                    cold_length=cold_length,
                    block_size=block_size,
                )
                shortlist_indices = _merge_candidate_index_sets(
                    expanded_shortlist,
                    shortlist_indices,
                    max_candidates=max(candidate_cap * 2, int(expanded_shortlist.shape[1])),
                )
                shortlisted_cold_k, shortlisted_cold_v, shortlist_scores = _materialize_shortlist(shortlist_indices)
                selective_stats["confidence_expansions"] = int(selective_stats.get("confidence_expansions", 0)) + 1
            needs_confidence_fallback = _block_shortlist_needs_expansion(
                block_scores,
                shortlist_indices,
                shortlist_scores,
                effective_topk=effective_topk,
                block_size=block_size,
            )

        if needs_confidence_fallback:
            if selective_state is not None:
                selective_state["indices"] = None
                selective_state["selected_indices"] = None
                selective_state["share_indices"] = None
                selective_state["cold_length"] = int(cold_length)
                selective_state["steps_since_full_refresh"] = max(int(refresh_interval), 1)
            selective_stats["confidence_fallbacks"] = int(selective_stats.get("confidence_fallbacks", 0)) + 1
            past_k_values = past_k.to_float32() if past_k is not None else None
            past_v_values = past_v.to_float32() if past_v is not None else None
            if past_k_values is None:
                k_all = k_new
                v_all = v_new
            else:
                k_all = np.concatenate([past_k_values, k_new], axis=1)
                v_all = np.concatenate([past_v_values, v_new], axis=1)
            scores = np.einsum("hd,hnd->hn", q, k_all) / np.sqrt(float(head_dim))
            qjl_cache: _TurboQJLKVArray | None = None
            qjl_length = 0
            if isinstance(past_k.cold, _TurboQJLKVArray):
                qjl_cache = past_k.cold
                qjl_length = past_k.cold_length
            if qjl_cache is not None and qjl_length > 0:
                scores[:, :qjl_length] += qjl_cache.score_correction(q, head_dim=head_dim, score_weight=qjl_score_weight)
            probs = _softmax(scores, axis=-1)
            context = np.einsum("hn,hnd->hd", probs, v_all).reshape(hidden_size)
        else:
            cache_topk = min(candidate_topk, shortlist_scores.shape[1]) if shortlist_scores.ndim == 2 else 0
            if cache_topk > 0 and shortlist_scores.shape[1] > cache_topk:
                shortlist_cache_idx = np.argpartition(
                    shortlist_scores, -cache_topk, axis=-1
                )[:, -cache_topk:]
                next_cached_indices = np.take_along_axis(shortlist_indices, shortlist_cache_idx, axis=1)
            else:
                next_cached_indices = shortlist_indices

            if effective_topk > 0 and shortlist_scores.shape[1] > effective_topk:
                shortlist_topk = np.argpartition(
                    shortlist_scores, -effective_topk, axis=-1
                )[:, -effective_topk:]
                selected_cold_indices = np.take_along_axis(shortlist_indices, shortlist_topk, axis=1)
                selected_cold_k = np.take_along_axis(shortlisted_cold_k, shortlist_topk[..., None], axis=1)
                selected_cold_v = np.take_along_axis(shortlisted_cold_v, shortlist_topk[..., None], axis=1)
            else:
                selected_cold_indices = np.asarray(shortlist_indices, dtype=np.int64)
                selected_cold_k = shortlisted_cold_k
                selected_cold_v = shortlisted_cold_v

            if selective_state is not None:
                selective_state["indices"] = np.asarray(next_cached_indices, dtype=np.int64)
                selective_state["selected_indices"] = np.asarray(selected_cold_indices, dtype=np.int64)
                selective_state["share_indices"] = np.asarray(shortlist_indices, dtype=np.int64)
                selective_state["cold_length"] = int(cold_length)
                selective_state["steps_since_full_refresh"] = (
                    state_steps_since_full + 1 if can_reuse_shortlist else 0
                )

            k_exact = np.concatenate([selected_cold_k, hot_k, k_new], axis=1)
            v_exact = np.concatenate([selected_cold_v, hot_v, v_new], axis=1)
            scores = np.einsum("hd,hnd->hn", q, k_exact, optimize=True) / np.sqrt(float(head_dim))
            if isinstance(past_k.cold, _TurboQJLKVArray) and effective_topk > 0:
                pass
            probs = _softmax(scores, axis=-1)
            context = np.einsum("hn,hnd->hd", probs, v_exact, optimize=True).reshape(hidden_size)
    else:
        if selective_state is not None:
            selective_state["indices"] = None
            selective_state["selected_indices"] = None
            selective_state["share_indices"] = None
            selective_state["cold_length"] = int(cold_length)
            selective_state["steps_since_full_refresh"] = max(int(refresh_interval), 1)
        # Fallback to full materialization (cold prefix too small or not supported)
        past_k_values = past_k.to_float32() if past_k is not None else None
        past_v_values = past_v.to_float32() if past_v is not None else None

        if past_k_values is None:
            k_all = k_new
            v_all = v_new
        else:
            k_all = np.concatenate([past_k_values, k_new], axis=1)
            v_all = np.concatenate([past_v_values, v_new], axis=1)

        scores = np.einsum("hd,hnd->hn", q, k_all) / np.sqrt(float(head_dim))
        # QJL score correction on full cold prefix
        qjl_cache: _TurboQJLKVArray | None = None
        qjl_length = 0
        if isinstance(past_k.cold, _TurboQJLKVArray):
            qjl_cache = past_k.cold
            qjl_length = past_k.cold_length
        if qjl_cache is not None and qjl_length > 0:
            scores[:, :qjl_length] += qjl_cache.score_correction(q, head_dim=head_dim, score_weight=qjl_score_weight)
        probs = _softmax(scores, axis=-1)
        context = np.einsum("hn,hnd->hd", probs, v_all).reshape(hidden_size)

    # --- MLP path & Cache update ---
    # Update cache by appending the new token
    if past_k is not None and isinstance(past_k, _HotWindowKVArray):
        next_k = past_k.append_token(k_new, max_hot_window, store_k_fn)
        next_v = past_v.append_token(v_new, max_hot_window, store_v_fn)
    else:
        # Fallback if no valid past cache (first step or mixed modes)
        next_k = np.concatenate([past_k_values, k_new], axis=1) if past_k_values is not None else k_new
        next_v = np.concatenate([past_v_values, v_new], axis=1) if past_v_values is not None else v_new

    attn_output = _streaming_right_linear_vector(
        stores[f"{prefix}.attn.c_proj.weight"],
        context,
        runtime_cache.tensor(stores[f"{prefix}.attn.c_proj.bias"]),
    )
    hidden = token_embedding + attn_output

    ln2 = _layer_norm_last_dim(
        hidden,
        runtime_cache.tensor(stores[f"{prefix}.ln_2.weight"]),
        runtime_cache.tensor(stores[f"{prefix}.ln_2.bias"]),
        eps,
    ).astype(np.float32)
    mlp = _streaming_right_linear_vector(
        stores[f"{prefix}.mlp.c_fc.weight"],
        ln2,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_fc.bias"]),
    )
    mlp = _gelu(mlp)
    mlp = _streaming_right_linear_vector(
        stores[f"{prefix}.mlp.c_proj.weight"],
        mlp,
        runtime_cache.tensor(stores[f"{prefix}.mlp.c_proj.bias"]),
    )
    return hidden + mlp, next_k, next_v


class GPT2StreamingEngine:
    def __init__(
        self,
        export_dir: str | Path,
        *,
        cache_mode: str = "fresh",
        max_tensor_bytes: int = 256 * 1024,
        kv_cache_precision: str = "fp32",
        kv_key_precision: str | None = None,
        kv_value_precision: str | None = None,
        kv_quant_seed: int = 7,
        kv_rotation_mode: str = "hadamard",
        kv_hot_window: int = 0,
        kv_topk: int = 0,
        kv_index_refresh_interval: int = 8,
        kv_block_size: int = 0,
        kv_layer_share_stride: int = 0,
        kv_calibration_tokens: int = 128,
        kv_adaptive_high_kurtosis: float = 10.0,
        kv_adaptive_medium_kurtosis: float = 3.0,
    ) -> None:
        self.export_dir = Path(export_dir)
        self.stores = tensor_store_map(export_dir)
        self.config = load_manifest(export_dir).get("config", {})
        self.eps = float(self.config.get("layer_norm_epsilon", 1e-5))
        self.num_heads = int(self.config["n_head"])
        self.num_layers = int(self.config["n_layer"])
        self.hidden_size = int(self.config["n_embd"])
        self.head_dim = self.hidden_size // self.num_heads
        self.cache_mode = cache_mode
        self.kv_cache_precision = kv_cache_precision
        self.kv_key_precision = kv_key_precision
        self.kv_value_precision = kv_value_precision
        self.kv_quant_seed = int(kv_quant_seed)
        self.kv_rotation_mode = kv_rotation_mode
        self.kv_hot_window = max(int(kv_hot_window), 0)
        self.kv_topk = max(int(kv_topk), 0)
        self.kv_index_refresh_interval = max(int(kv_index_refresh_interval), 1)
        self.kv_block_size = max(int(kv_block_size), 0)
        self.kv_layer_share_stride = max(int(kv_layer_share_stride), 0)
        self.kv_calibration_tokens = max(int(kv_calibration_tokens), 0)
        self.kv_adaptive_high_kurtosis = float(kv_adaptive_high_kurtosis)
        self.kv_adaptive_medium_kurtosis = float(kv_adaptive_medium_kurtosis)
        supported_modes = {"fp32", "turbo-int8", "turbo-4bit", "turbo-qjl", "adaptive"}
        supported_static_modes = {"fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"}
        if self.kv_cache_precision not in supported_modes:
            raise ValueError(f"unsupported kv_cache_precision: {self.kv_cache_precision}")
        if self.kv_key_precision is not None and self.kv_key_precision not in supported_static_modes:
            raise ValueError(f"unsupported kv_key_precision: {self.kv_key_precision}")
        if self.kv_value_precision is not None and self.kv_value_precision not in supported_static_modes:
            raise ValueError(f"unsupported kv_value_precision: {self.kv_value_precision}")
        if self.kv_rotation_mode not in {"qr", "hadamard"}:
            raise ValueError(f"unsupported kv_rotation_mode: {self.kv_rotation_mode}")
        if self.kv_adaptive_medium_kurtosis > self.kv_adaptive_high_kurtosis:
            raise ValueError("kv_adaptive_medium_kurtosis must be <= kv_adaptive_high_kurtosis")
        if self.kv_layer_share_stride == 1:
            raise ValueError("kv_layer_share_stride must be 0 or >= 2")
        if self.kv_cache_precision == "adaptive" and (
            self.kv_key_precision is not None or self.kv_value_precision is not None
        ):
            raise ValueError("adaptive kv_cache_precision does not support kv_key_precision/kv_value_precision")
        configured_kv_modes = {self.kv_cache_precision}
        if self.kv_key_precision is not None:
            configured_kv_modes.add(self.kv_key_precision)
        if self.kv_value_precision is not None:
            configured_kv_modes.add(self.kv_value_precision)
        self._kv_rotation = None
        self._kv_codebook = None
        self._qjl_matrix = None
        self._rebuild_kv_codec_state(configured_kv_modes)
        self._adaptive_mode_enabled = self.kv_cache_precision == "adaptive"
        self._kv_layer_modes: list[str] | None = None
        self._kv_kurtosis_profile: list[dict[str, Any]] | None = None
        self._kv_tokens_profiled = 0
        self._kv_kurtosis_state: list[dict[str, _KurtosisAccumulator]] | None = (
            [{"k": _KurtosisAccumulator(), "v": _KurtosisAccumulator()} for _ in range(self.num_layers)]
            if self._adaptive_mode_enabled
            else None
        )
        self.runtime_cache = _resolve_runtime_cache(
            export_dir,
            cache_mode=cache_mode,
            max_tensor_bytes=max_tensor_bytes,
        )
        self.lm_head_store = self.stores.get("lm_head.weight", self.stores["transformer.wte.weight"])
        self.lm_bias = np.zeros(load_meta(self.lm_head_store).shape[0], dtype=np.float32)
        self._kv_policy: AdaptiveKVPolicy | None = None
        self._kv_policy_phase: str | None = None
        self._kv_policy_allowed_modes: tuple[str, ...] | None = None
        self._kv_mode_trace: list[str] = []
        self._kv_switch_events: list[dict[str, Any]] = []
        self.reset_sequence()

    def _rebuild_kv_codec_state(self, configured_kv_modes: set[str] | None = None) -> None:
        modes = configured_kv_modes or {self.kv_cache_precision}
        if self.kv_key_precision is not None:
            modes.add(self.kv_key_precision)
        if self.kv_value_precision is not None:
            modes.add(self.kv_value_precision)
        self._kv_rotation = (
            _build_kv_rotation(self.head_dim, self.kv_quant_seed, self.kv_rotation_mode)
            if modes != {"fp32"}
            else None
        )
        self._kv_codebook = (
            _compute_lloyd_max_codebook(self._kv_rotation.rotated_dim, 4)
            if modes.intersection({"turbo-4bit", "turbo-qjl", "adaptive"}) and self._kv_rotation is not None
            else None
        )
        self._qjl_matrix = (
            _gaussian_qjl_matrix(self.head_dim, self.kv_quant_seed)
            if "turbo-qjl" in modes
            else None
        )

    def reset_sequence(self) -> None:
        self.caches: list[
            dict[
                str,
                np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
            ]
        ] = [
            {"k": None, "v": None} for _ in range(self.num_layers)
        ]
        self._kv_selective_state = [
            {
                "indices": None,
                "selected_indices": None,
                "share_indices": None,
                "cold_length": 0,
                "steps_since_full_refresh": self.kv_index_refresh_interval,
            }
            for _ in range(self.num_layers)
        ]
        self._kv_selective_stats = {
            "full_refreshes": 0,
            "reuse_hits": 0,
            "reuse_new_tokens": 0,
            "reuse_candidate_rows": 0,
        }
        self._kv_cross_layer_overlap_stats = _new_cross_layer_overlap_stats(self.num_layers)
        self._kv_mode_trace = [self.current_kv_mode]
        self._kv_switch_events = []
        if self._kv_policy is not None:
            self._kv_policy.reset_runtime_state()
            self._kv_policy.record_mode(self.current_kv_mode)

    @property
    def current_kv_mode(self) -> str:
        return _public_kv_mode_name(self.kv_cache_precision)

    def set_kv_policy(
        self,
        policy: AdaptiveKVPolicy | None,
        *,
        phase: str | None = None,
        allowed_modes: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._kv_policy = policy
        self._kv_policy_phase = phase
        self._kv_policy_allowed_modes = (
            tuple(_public_kv_mode_name(mode) for mode in allowed_modes) if allowed_modes else None
        )
        if self._kv_policy is not None:
            self._kv_policy.reset_runtime_state()
            self._kv_policy.record_mode(self.current_kv_mode)

    def switch_kv_mode(self, new_mode: str, *, reason: str | None = None) -> str:
        if self.kv_key_precision is not None or self.kv_value_precision is not None:
            raise ValueError("switch_kv_mode does not support asymmetric K/V modes")
        if self._adaptive_mode_enabled:
            raise ValueError("switch_kv_mode does not support adaptive per-layer mode")
        current_mode = _canonical_switch_mode(self.kv_cache_precision)
        target_mode = _canonical_switch_mode(new_mode)
        if current_mode == target_mode:
            return self.current_kv_mode

        self.kv_cache_precision = target_mode
        self.kv_rotation_mode = "hadamard" if target_mode != "fp32" else self.kv_rotation_mode
        self._rebuild_kv_codec_state({self.kv_cache_precision})
        for layer_index, cache in enumerate(self.caches):
            for cache_name in ("k", "v"):
                materialized = self._materialize_kv_cache(cache[cache_name])
                if materialized is None:
                    continue
                cache[cache_name] = self._store_kv_cache(
                    materialized,
                    layer_index=layer_index,
                    cache_name=cache_name,
                )
        for state in self._kv_selective_state:
            state["indices"] = None
            state["selected_indices"] = None
            state["share_indices"] = None
            state["steps_since_full_refresh"] = self.kv_index_refresh_interval
        event = {
            "from": _public_kv_mode_name(current_mode),
            "to": _public_kv_mode_name(target_mode),
            "reason": str(reason or "manual"),
        }
        self._kv_switch_events.append(event)
        self._kv_mode_trace.append(self.current_kv_mode)
        if self._kv_policy is not None:
            self._kv_policy.record_switch(
                old_mode=_public_kv_mode_name(current_mode),
                new_mode=self.current_kv_mode,
                reason=str(reason or "manual"),
                step_index=len(self._kv_mode_trace) - 1,
            )
            self._kv_policy.record_mode(self.current_kv_mode)
        return self.current_kv_mode

    def _kv_policy_summary(self) -> dict[str, Any] | None:
        if self._kv_policy is None:
            return None
        return {
            "phase": self._kv_policy_phase,
            "allowed_modes": list(self._kv_policy_allowed_modes or ()),
            "baseline_loss": self._kv_policy.current_baseline_loss(),
            "recent_loss": self._kv_policy.current_recent_loss(),
            "mode_trace": list(self._kv_policy._mode_trace),
            "switch_events": list(self._kv_policy._switch_events),
            "mode_histogram": self._kv_policy.mode_histogram(),
        }

    def _layer_kv_precision(self, layer_index: int | None, cache_name: str = "k") -> str:
        if not self._adaptive_mode_enabled:
            if cache_name == "k" and self.kv_key_precision is not None:
                return self.kv_key_precision
            if cache_name == "v" and self.kv_value_precision is not None:
                return self.kv_value_precision
            return self.kv_cache_precision
        if layer_index is None or self._kv_layer_modes is None:
            return "fp32"
        return str(self._kv_layer_modes[layer_index])

    def _qjl_score_weight(self, layer_index: int | None) -> float:
        return 0.25 if self._layer_kv_precision(layer_index, "k") == "turbo-qjl" else 0.0

    def _latest_cache_token(
        self,
        cache: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    ) -> np.ndarray | None:
        if cache is None:
            return None
        if isinstance(cache, _HotWindowKVArray):
            if cache.hot_length > 0:
                return np.asarray(cache.hot[:, -1:, :], dtype=np.float32)
            if cache.cold is not None and cache.cold.length > 0:
                indices = np.full((self.num_heads, 1), cache.cold.length - 1, dtype=np.int64)
                return cache.cold.materialize_indices(indices).astype(np.float32)
            return None
        if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray)):
            indices = np.full((self.num_heads, 1), cache.length - 1, dtype=np.int64)
            return cache.materialize_indices(indices).astype(np.float32)
        array = np.asarray(cache, dtype=np.float32)
        if array.shape[1] == 0:
            return None
        return array[:, -1:, :].astype(np.float32)

    def _collect_adaptive_layer_sample(
        self,
        layer_index: int,
        cache_name: str,
        cache: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    ) -> None:
        if not self._adaptive_mode_enabled or self._kv_layer_modes is not None or self._kv_kurtosis_state is None:
            return
        latest = self._latest_cache_token(cache)
        if latest is None:
            return
        self._kv_kurtosis_state[layer_index][cache_name].update(latest)

    def _profile_to_layer_modes(self) -> tuple[list[str], list[dict[str, Any]]]:
        if self._kv_kurtosis_state is None:
            return [], []
        layer_modes: list[str] = []
        profile: list[dict[str, Any]] = []
        protected_tail_start = max(self.num_layers - max(self.num_layers // 4, 1), 0)
        for layer_index, state in enumerate(self._kv_kurtosis_state):
            k_kurtosis = state["k"].pearson_kurtosis()
            v_kurtosis = state["v"].pearson_kurtosis()
            dominant = max(k_kurtosis, v_kurtosis)
            if dominant >= self.kv_adaptive_high_kurtosis:
                selected_mode = "fp32"
            elif dominant >= self.kv_adaptive_medium_kurtosis:
                selected_mode = "turbo-int8"
            else:
                selected_mode = "turbo-4bit"
            # TailorKV-style caution: the earliest and latest layers tend to be more
            # sensitive than the middle band, so keep a higher-fidelity floor there.
            if self.num_layers > 1 and (layer_index == 0 or layer_index >= protected_tail_start):
                if selected_mode == "turbo-4bit":
                    selected_mode = "turbo-int8"
            layer_modes.append(selected_mode)
            profile.append(
                {
                    "layer_index": layer_index,
                    "k_kurtosis": float(k_kurtosis),
                    "v_kurtosis": float(v_kurtosis),
                    "selected_mode": selected_mode,
                    "protected_terminal_band": bool(
                        self.num_layers > 1 and (layer_index == 0 or layer_index >= protected_tail_start)
                    ),
                }
            )
        return layer_modes, profile

    def _finalize_adaptive_kv_profile(self) -> None:
        if not self._adaptive_mode_enabled or self._kv_layer_modes is not None:
            return
        layer_modes, profile = self._profile_to_layer_modes()
        self._kv_layer_modes = layer_modes
        self._kv_kurtosis_profile = profile
        for layer_index, cache in enumerate(self.caches):
            for name in ("k", "v"):
                value = cache[name]
                if value is None:
                    continue
                materialized = self._materialize_kv_cache(value)
                cache[name] = self._store_kv_cache(materialized, layer_index=layer_index)

    def _maybe_finalize_adaptive_kv_profile(self) -> None:
        if not self._adaptive_mode_enabled or self._kv_layer_modes is not None:
            return
        if self.kv_calibration_tokens <= 0:
            self._finalize_adaptive_kv_profile()
            return
        if self._kv_tokens_profiled >= self.kv_calibration_tokens:
            self._finalize_adaptive_kv_profile()

    def _materialize_kv_cache(
        self,
        cache: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    ) -> np.ndarray | None:
        if cache is None:
            return None
        if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray, _HotWindowKVArray)):
            return cache.to_float32()
        return np.asarray(cache, dtype=np.float32)

    def _store_compact_kv_cache(
        self,
        values: np.ndarray,
        *,
        layer_index: int | None = None,
        cache_name: str = "k",
    ) -> np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray:
        kv_mode = self._layer_kv_precision(layer_index, cache_name)
        if kv_mode == "fp32":
            return np.asarray(values, dtype=np.float32)
        if kv_mode == "turbo-int8":
            return _TurboInt8KVArray(values, rotation=self._kv_rotation)
        if kv_mode == "turbo-4bit":
            return _Turbo4BitKVArray(values, rotation=self._kv_rotation, codebook=self._kv_codebook)
        return _TurboQJLKVArray(
            values,
            rotation=self._kv_rotation,
            codebook=self._kv_codebook,
            qjl_matrix=self._qjl_matrix,
        )

    def _store_kv_cache(
        self,
        values: np.ndarray,
        *,
        layer_index: int | None = None,
        cache_name: str = "k",
    ) -> np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray:
        values = np.asarray(values, dtype=np.float32)
        kv_mode = self._layer_kv_precision(layer_index, cache_name)
        if kv_mode == "fp32" or self.kv_hot_window <= 0:
            return self._store_compact_kv_cache(values, layer_index=layer_index, cache_name=cache_name)
        if values.shape[1] <= self.kv_hot_window:
            return _HotWindowKVArray(cold=None, hot=values)
        cold_length = values.shape[1] - self.kv_hot_window
        return _HotWindowKVArray(
            cold=self._store_compact_kv_cache(
                values[:, :cold_length, :],
                layer_index=layer_index,
                cache_name=cache_name,
            ),
            hot=values[:, cold_length:, :],
        )

    def _deserialize_compact_cache(
        self,
        data: Any,
        prefix: str,
        *,
        layer_index: int | None = None,
        cache_name: str = "k",
    ) -> np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | None:
        kv_mode = self._layer_kv_precision(layer_index, cache_name)
        quant_name = f"{prefix}_q"
        scale_name = f"{prefix}_scales"
        packed_name = f"{prefix}_packed"
        norms_name = f"{prefix}_norms"
        qjl_bits_name = f"{prefix}_qjl_bits"
        qjl_norms_name = f"{prefix}_qjl_norms"
        raw_name = prefix
        if quant_name in data and scale_name in data:
            return _TurboInt8KVArray.from_quantized(
                data[quant_name],
                data[scale_name],
                rotation=self._kv_rotation,
            )
        if (
            packed_name in data
            and norms_name in data
            and kv_mode == "turbo-qjl"
            and qjl_bits_name in data
            and qjl_norms_name in data
        ):
            return _TurboQJLKVArray.from_quantized(
                data[packed_name],
                data[norms_name],
                data[qjl_bits_name],
                data[qjl_norms_name],
                rotation=self._kv_rotation,
                codebook=self._kv_codebook,
                qjl_matrix=self._qjl_matrix,
            )
        if packed_name in data and norms_name in data:
            return _Turbo4BitKVArray.from_quantized(
                data[packed_name],
                data[norms_name],
                rotation=self._kv_rotation,
                codebook=self._kv_codebook,
            )
        if raw_name in data:
            return np.asarray(data[raw_name], dtype=np.float32)
        return None

    def _serialize_cache_arrays(
        self,
        prefix: str,
        value: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray,
        arrays: dict[str, np.ndarray],
    ) -> None:
        if isinstance(value, _HotWindowKVArray):
            arrays[f"{prefix}_hot"] = value.hot.astype(np.float32)
            if value.cold is not None:
                self._serialize_cache_arrays(f"{prefix}_cold", value.cold, arrays)
            return
        if isinstance(value, _TurboInt8KVArray):
            arrays[f"{prefix}_q"] = value.q
            arrays[f"{prefix}_scales"] = value.scales
            return
        if isinstance(value, _TurboQJLKVArray):
            arrays[f"{prefix}_packed"] = value.packed
            arrays[f"{prefix}_norms"] = value.norms
            arrays[f"{prefix}_qjl_bits"] = value.qjl_bits
            arrays[f"{prefix}_qjl_norms"] = value.residual_norms
            return
        if isinstance(value, _Turbo4BitKVArray):
            arrays[f"{prefix}_packed"] = value.packed
            arrays[f"{prefix}_norms"] = value.norms
            return
        arrays[prefix] = np.asarray(value, dtype=np.float32)

    def _session_codec_payload_meta(self) -> dict[str, Any]:
        payload = {
            "format": "compressed-kv-v2",
            "npz_compressed": True,
            "stores_hot_window_exact_tail": bool(self.kv_hot_window > 0),
        }
        if self.kv_topk > 0:
            payload["selective"] = {
                "topk": int(self.kv_topk),
                "index_refresh_interval": int(self.kv_index_refresh_interval),
                "block_size": int(self.kv_block_size),
                "layer_share_stride": int(self.kv_layer_share_stride),
            }
        if self.kv_key_precision is not None or self.kv_value_precision is not None:
            payload["asymmetric"] = {
                "k_mode": self.kv_key_precision or self.kv_cache_precision,
                "v_mode": self.kv_value_precision or self.kv_cache_precision,
            }
        if self._adaptive_mode_enabled:
            payload["adaptive"] = {
                "calibration_tokens": int(self.kv_calibration_tokens),
                "high_kurtosis_threshold": float(self.kv_adaptive_high_kurtosis),
                "medium_kurtosis_threshold": float(self.kv_adaptive_medium_kurtosis),
                "tokens_profiled": int(self._kv_tokens_profiled),
                "ready": self._kv_layer_modes is not None,
            }
        if self._kv_rotation is not None:
            payload["rotation"] = {
                "mode": self.kv_rotation_mode,
                "original_dim": int(self._kv_rotation.original_dim),
                "rotated_dim": int(self._kv_rotation.rotated_dim),
            }
        if self._kv_codebook is not None:
            payload["codebook"] = {
                "bits": int(self._kv_codebook.bits),
                "dim": int(self._kv_codebook.dim),
                "levels": int(self._kv_codebook.levels),
            }
        if self._qjl_matrix is not None:
            payload["qjl"] = {
                "matrix_shape": list(self._qjl_matrix.shape),
            }
        return payload

    def _serialize_session_codec_artifacts(self, arrays: dict[str, np.ndarray]) -> None:
        if isinstance(self._kv_rotation, _DenseOrthogonalRotation):
            arrays["__kv_rotation_matrix"] = self._kv_rotation.matrix.astype(np.float32)
        elif isinstance(self._kv_rotation, _HadamardRotation):
            arrays["__kv_rotation_signs"] = self._kv_rotation.signs.astype(np.int8)
        if self._kv_codebook is not None:
            arrays["__kv_codebook_centroids"] = self._kv_codebook.centroids.astype(np.float32)
            arrays["__kv_codebook_boundaries"] = self._kv_codebook.boundaries.astype(np.float32)
        if self._qjl_matrix is not None:
            arrays["__kv_qjl_matrix"] = self._qjl_matrix.astype(np.float32)

    def _validate_session_codec_artifacts(self, data: Any) -> None:
        if "__kv_rotation_matrix" in data:
            if not isinstance(self._kv_rotation, _DenseOrthogonalRotation):
                raise ValueError("session contains dense rotation payload but engine rotation is different")
            if not _array_payload_matches(self._kv_rotation.matrix, data["__kv_rotation_matrix"]):
                raise ValueError("session dense rotation payload does not match this engine")
        if "__kv_rotation_signs" in data:
            if not isinstance(self._kv_rotation, _HadamardRotation):
                raise ValueError("session contains hadamard rotation payload but engine rotation is different")
            if not _array_payload_matches(self._kv_rotation.signs.astype(np.int8), data["__kv_rotation_signs"]):
                raise ValueError("session hadamard rotation payload does not match this engine")
        if "__kv_codebook_centroids" in data or "__kv_codebook_boundaries" in data:
            if self._kv_codebook is None:
                raise ValueError("session contains codebook payload but engine has no codebook configured")
            if "__kv_codebook_centroids" not in data or "__kv_codebook_boundaries" not in data:
                raise ValueError("session codebook payload is incomplete")
            if not _array_payload_matches(self._kv_codebook.centroids, data["__kv_codebook_centroids"]):
                raise ValueError("session codebook centroids do not match this engine")
            if not _array_payload_matches(self._kv_codebook.boundaries, data["__kv_codebook_boundaries"]):
                raise ValueError("session codebook boundaries do not match this engine")
        if "__kv_qjl_matrix" in data:
            if self._qjl_matrix is None:
                raise ValueError("session contains qjl payload but engine has no qjl matrix configured")
            if not _array_payload_matches(self._qjl_matrix, data["__kv_qjl_matrix"]):
                raise ValueError("session qjl matrix does not match this engine")

    def _kv_cache_length(
        self,
        cache: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    ) -> int:
        if cache is None:
            return 0
        if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray, _HotWindowKVArray)):
            return cache.length
        return int(cache.shape[1])

    def _kv_cache_bytes(
        self,
        cache: np.ndarray | _TurboInt8KVArray | _Turbo4BitKVArray | _TurboQJLKVArray | _HotWindowKVArray | None,
    ) -> int:
        if cache is None:
            return 0
        if isinstance(cache, (_TurboInt8KVArray, _Turbo4BitKVArray, _TurboQJLKVArray, _HotWindowKVArray)):
            return cache.nbytes
        return int(cache.nbytes)

    def save_session(
        self,
        session_dir: str | Path,
        *,
        generated_ids: list[int],
        last_logits: np.ndarray | None = None,
        session_codec: str = "python-npz",
    ) -> Path:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "export_dir": str(self.export_dir),
            "cache_mode": self.cache_mode,
            "generated_ids": list(generated_ids),
            "num_layers": self.num_layers,
            "session_format_version": _SESSION_FORMAT_VERSION,
            "kv_cache_file": "kv_cache.npz",
            "kv_cache_precision": self.kv_cache_precision,
            "kv_key_precision": self.kv_key_precision,
            "kv_value_precision": self.kv_value_precision,
            "kv_quant_seed": self.kv_quant_seed,
            "kv_rotation_mode": self.kv_rotation_mode,
            "kv_hot_window": self.kv_hot_window,
            "kv_topk": self.kv_topk,
            "kv_index_refresh_interval": self.kv_index_refresh_interval,
            "kv_block_size": self.kv_block_size,
            "kv_layer_share_stride": self.kv_layer_share_stride,
            "kv_calibration_tokens": self.kv_calibration_tokens,
            "kv_adaptive_high_kurtosis": self.kv_adaptive_high_kurtosis,
            "kv_adaptive_medium_kurtosis": self.kv_adaptive_medium_kurtosis,
            "kv_tokens_profiled": self._kv_tokens_profiled,
            "kv_layer_modes": self._kv_layer_modes,
            "kv_kurtosis_profile": self._kv_kurtosis_profile,
            "kv_kurtosis_state": _serialize_adaptive_kurtosis_state(self._kv_kurtosis_state),
            "kv_session_payload": self._session_codec_payload_meta(),
            "kv_current_mode": self.current_kv_mode,
            "kv_mode_trace": list(self._kv_mode_trace),
            "kv_switch_events": list(self._kv_switch_events),
            "kv_policy_phase": self._kv_policy_phase,
            "kv_policy_allowed_modes": list(self._kv_policy_allowed_modes or ()),
            "kv_policy_state": None if self._kv_policy is None else self._kv_policy.to_json(),
        }
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        arrays: dict[str, np.ndarray] = {}
        self._serialize_session_codec_artifacts(arrays)
        for layer_index, cache in enumerate(self.caches):
            for name in ("k", "v"):
                value = cache[name]
                if value is None:
                    continue
                self._serialize_cache_arrays(f"layer_{layer_index}_{name}", value, arrays)
        if last_logits is not None:
            arrays["last_logits"] = last_logits
        rust_session.save_session_bundle(
            session_dir,
            meta=meta,
            arrays=arrays,
            session_codec=session_codec,
        )
        return session_dir

    def load_session(self, session_dir: str | Path) -> dict[str, Any]:
        session_dir = Path(session_dir)
        meta, arrays, _ = rust_session.load_session_bundle(session_dir)
        _session_meta_matches_engine(
            meta,
            export_dir=self.export_dir,
            kv_cache_precision=self.kv_cache_precision,
            kv_key_precision=self.kv_key_precision,
            kv_value_precision=self.kv_value_precision,
            kv_quant_seed=self.kv_quant_seed,
            kv_rotation_mode=self.kv_rotation_mode,
            kv_hot_window=self.kv_hot_window,
            kv_topk=self.kv_topk,
            kv_index_refresh_interval=self.kv_index_refresh_interval,
            kv_block_size=self.kv_block_size,
            kv_layer_share_stride=self.kv_layer_share_stride,
            kv_calibration_tokens=self.kv_calibration_tokens,
            kv_adaptive_high_kurtosis=self.kv_adaptive_high_kurtosis,
            kv_adaptive_medium_kurtosis=self.kv_adaptive_medium_kurtosis,
        )
        self._kv_layer_modes = [str(item) for item in meta.get("kv_layer_modes", [])] if meta.get("kv_layer_modes") else None
        self._kv_kurtosis_profile = list(meta.get("kv_kurtosis_profile", []) or []) or None
        self._kv_tokens_profiled = int(meta.get("kv_tokens_profiled", 0))
        self._kv_kurtosis_state = _deserialize_adaptive_kurtosis_state(meta.get("kv_kurtosis_state"))
        policy_state = meta.get("kv_policy_state")
        if self._adaptive_mode_enabled and self._kv_kurtosis_state is None and self._kv_layer_modes is None:
            self._kv_kurtosis_state = [{"k": _KurtosisAccumulator(), "v": _KurtosisAccumulator()} for _ in range(self.num_layers)]
        self.reset_sequence()
        if isinstance(policy_state, dict):
            self._kv_policy = AdaptiveKVPolicy.from_json(policy_state)
            self._kv_policy_phase = meta.get("kv_policy_phase")
            allowed_modes = meta.get("kv_policy_allowed_modes") or []
            self._kv_policy_allowed_modes = tuple(str(item) for item in allowed_modes)
        else:
            self._kv_policy = None
            self._kv_policy_phase = None
            self._kv_policy_allowed_modes = None
        self._kv_mode_trace = [str(item) for item in meta.get("kv_mode_trace", [self.current_kv_mode])]
        self._kv_switch_events = list(meta.get("kv_switch_events", []))
        last_logits: np.ndarray | None = None
        if arrays:
            self._validate_session_codec_artifacts(arrays)
            for layer_index in range(self.num_layers):
                for name in ("k", "v"):
                    prefix = f"layer_{layer_index}_{name}"
                    hot_name = f"{prefix}_hot"
                    if hot_name in arrays:
                        cold = self._deserialize_compact_cache(
                            arrays,
                            f"{prefix}_cold",
                            layer_index=layer_index,
                            cache_name=name,
                        )
                        self.caches[layer_index][name] = _HotWindowKVArray(cold=cold, hot=arrays[hot_name])
                    else:
                        value = self._deserialize_compact_cache(
                            arrays,
                            prefix,
                            layer_index=layer_index,
                            cache_name=name,
                        )
                        if value is not None:
                            self.caches[layer_index][name] = value
            if "last_logits" in arrays:
                last_logits = arrays["last_logits"]
        meta["last_logits"] = last_logits
        return meta

    def _run_step(self, token_id: int, position_id: int) -> np.ndarray:
        token_embed = self.runtime_cache.rows(self.stores["transformer.wte.weight"], [token_id])[0]
        pos_embed = self.runtime_cache.rows(self.stores["transformer.wpe.weight"], [position_id])[0]
        hidden = (token_embed + pos_embed).astype(np.float32)
        for layer_index in range(self.num_layers):
            past_k = self.caches[layer_index]["k"]
            past_v = self.caches[layer_index]["v"]
            use_selective = (
                self.kv_topk > 0
                and isinstance(past_k, _HotWindowKVArray)
                and isinstance(past_v, _HotWindowKVArray)
            )
            if use_selective:
                shared_candidate_indices = None
                shared_cold_length = 0
                if (
                    self.kv_layer_share_stride > 1
                    and layer_index > 0
                    and (layer_index % self.kv_layer_share_stride) != 0
                ):
                    previous_state = self._kv_selective_state[layer_index - 1]
                    previous_share_indices = previous_state.get("share_indices")
                    if isinstance(previous_share_indices, np.ndarray) and previous_share_indices.size > 0:
                        shared_candidate_indices = np.asarray(previous_share_indices, dtype=np.int64)
                        shared_cold_length = int(previous_state.get("cold_length", 0))
                hidden, next_k, next_v = _gpt2_step_with_selective_kv(
                    hidden,
                    stores=self.stores,
                    layer_index=layer_index,
                    num_heads=self.num_heads,
                    eps=self.eps,
                    past_k=past_k,
                    past_v=past_v,
                    kv_topk=self.kv_topk,
                    kv_block_size=self.kv_block_size,
                    max_hot_window=self.kv_hot_window,
                    store_k_fn=lambda values, layer_index=layer_index: self._store_compact_kv_cache(
                        values,
                        layer_index=layer_index,
                        cache_name="k",
                    ),
                    store_v_fn=lambda values, layer_index=layer_index: self._store_compact_kv_cache(
                        values,
                        layer_index=layer_index,
                        cache_name="v",
                    ),
                    selective_state=self._kv_selective_state[layer_index],
                    shared_candidate_indices=shared_candidate_indices,
                    shared_cold_length=shared_cold_length,
                    refresh_interval=self.kv_index_refresh_interval,
                    selective_stats=self._kv_selective_stats,
                    runtime_cache=self.runtime_cache,
                    qjl_score_weight=self._qjl_score_weight(layer_index),
                )
                self.caches[layer_index]["k"] = next_k
                self.caches[layer_index]["v"] = next_v
                if layer_index > 0:
                    _record_cross_layer_overlap(
                        self._kv_cross_layer_overlap_stats,
                        pair_index=layer_index - 1,
                        previous_indices=self._kv_selective_state[layer_index - 1].get("selected_indices"),
                        current_indices=self._kv_selective_state[layer_index].get("selected_indices"),
                    )
            else:
                hidden, next_k, next_v = _gpt2_step_with_kv(
                    hidden,
                    stores=self.stores,
                    layer_index=layer_index,
                    num_heads=self.num_heads,
                    eps=self.eps,
                    past_k=past_k,
                    past_v=past_v,
                    runtime_cache=self.runtime_cache,
                    qjl_score_weight=self._qjl_score_weight(layer_index),
                )
                self.caches[layer_index]["k"] = self._store_kv_cache(next_k, layer_index=layer_index, cache_name="k")
                self.caches[layer_index]["v"] = self._store_kv_cache(next_v, layer_index=layer_index, cache_name="v")
            self._collect_adaptive_layer_sample(layer_index, "k", self.caches[layer_index]["k"])
            self._collect_adaptive_layer_sample(layer_index, "v", self.caches[layer_index]["v"])
        self._kv_tokens_profiled += 1
        self._maybe_finalize_adaptive_kv_profile()
        hidden = _layer_norm_last_dim(
            hidden,
            self.runtime_cache.tensor(self.stores["transformer.ln_f.weight"]),
            self.runtime_cache.tensor(self.stores["transformer.ln_f.bias"]),
            self.eps,
        ).astype(np.float32)
        return streaming_matvec(self.lm_head_store, hidden).astype(np.float32) + self.lm_bias

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
        return self.generate_advanced(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            seed=None,
        )

    def stream_generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
        reset_sequence: bool = True,
    ):
        if reset_sequence:
            self.reset_sequence()
        generated = list(prompt_ids)
        step_logits: list[np.ndarray] = []
        rng = np.random.default_rng(seed) if seed is not None else None

        for position_id, token_id in enumerate(prompt_ids):
            logits = self._run_step(token_id, position_id)
            step_logits.append(logits)
            yield {
                "phase": "prompt",
                "position_id": position_id,
                "token_id": int(token_id),
                "generated_ids": list(generated),
                "last_logits": logits,
            }

        for step_index in range(max_new_tokens):
            next_token = self._sample_next_token(
                step_logits[-1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            self._maybe_apply_kv_policy(source_logits=step_logits[-1], sampled_token=next_token)
            generated.append(next_token)
            logits = self._run_step(next_token, len(generated) - 1)
            step_logits.append(logits)
            yield {
                "phase": "generated",
                "step_index": step_index,
                "position_id": len(generated) - 1,
                "token_id": int(next_token),
                "generated_ids": list(generated),
                "last_logits": logits,
                "current_kv_mode": self.current_kv_mode,
            }

    def _sample_next_token(
        self,
        logits: np.ndarray,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        rng: np.random.Generator | None,
    ) -> int:
        if not do_sample:
            return int(np.argmax(logits))

        scaled = logits.astype(np.float64)
        temp = max(float(temperature), 1e-6)
        scaled = scaled / temp

        if top_k > 0 and top_k < scaled.shape[0]:
            cutoff = np.partition(scaled, -top_k)[-top_k]
            scaled = np.where(scaled >= cutoff, scaled, -1e9)

        probs = _softmax(scaled, axis=-1)

        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            ordered = probs[order]
            cumulative = np.cumsum(ordered)
            keep_mask = cumulative <= top_p
            if not np.any(keep_mask):
                keep_mask[0] = True
            first_exceed = np.argmax(cumulative > top_p)
            if cumulative[first_exceed] > top_p:
                keep_mask[first_exceed] = True
            filtered = np.zeros_like(probs)
            filtered[order[keep_mask]] = probs[order[keep_mask]]
            probs = filtered / filtered.sum()

        rng = rng or np.random.default_rng()
        return int(rng.choice(np.arange(probs.shape[0]), p=probs))

    def _maybe_apply_kv_policy(self, *, source_logits: np.ndarray, sampled_token: int) -> None:
        if self._kv_policy is None:
            return
        decision = self._kv_policy.observe(
            logits=source_logits,
            token_id=int(sampled_token),
            current_mode=self.current_kv_mode,
            allowed_modes=self._kv_policy_allowed_modes,
        )
        target_mode = str(decision.get("target_mode", self.current_kv_mode))
        if target_mode != self.current_kv_mode:
            self.switch_kv_mode(target_mode, reason=str(decision.get("action", "policy")))

    def generate_advanced(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
        reset_sequence: bool = True,
    ) -> dict[str, Any]:
        if reset_sequence:
            self.reset_sequence()
        cache_before = self.runtime_cache.stats()
        generated = list(prompt_ids)
        step_logits: list[np.ndarray] = []
        step_times_ms: list[float] = []
        step_rss_mb: list[float] = []
        rss_before_mb = _process_rss_mb()
        total_start = time.perf_counter()
        rng = np.random.default_rng(seed) if seed is not None else None

        for position_id, token_id in enumerate(prompt_ids):
            step_start = time.perf_counter()
            step_logits.append(self._run_step(token_id, position_id))
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            step_rss_mb.append(_process_rss_mb())

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(
                step_logits[-1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            self._maybe_apply_kv_policy(source_logits=step_logits[-1], sampled_token=next_token)
            generated.append(next_token)
            step_start = time.perf_counter()
            step_logits.append(self._run_step(next_token, len(generated) - 1))
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            step_rss_mb.append(_process_rss_mb())

        total_time_s = time.perf_counter() - total_start
        return {
            "prompt_ids": list(prompt_ids),
            "generated_ids": generated,
            "new_ids": generated[len(prompt_ids) :],
            "num_layers": self.num_layers,
            "cache_mode": self.cache_mode,
            "kv_cache_precision": self.kv_cache_precision,
            "kv_key_precision": self.kv_key_precision,
            "kv_value_precision": self.kv_value_precision,
            "kv_rotation_mode": self.kv_rotation_mode,
            "kv_hot_window": self.kv_hot_window,
            "kv_topk": self.kv_topk,
            "kv_index_refresh_interval": self.kv_index_refresh_interval,
            "kv_block_size": self.kv_block_size,
            "kv_layer_share_stride": self.kv_layer_share_stride,
            "kv_calibration_tokens": self.kv_calibration_tokens,
            "kv_layer_modes": list(self._kv_layer_modes) if self._kv_layer_modes is not None else None,
            "kv_kurtosis_profile": list(self._kv_kurtosis_profile) if self._kv_kurtosis_profile is not None else None,
            "kv_selective_stats": dict(self._kv_selective_stats),
            "kv_cross_layer_overlap": _summarize_cross_layer_overlap_stats(self._kv_cross_layer_overlap_stats),
            "current_kv_mode": self.current_kv_mode,
            "kv_mode_trace": list(self._kv_mode_trace),
            "switch_events": list(self._kv_switch_events),
            "policy_baseline_loss": None if self._kv_policy is None else self._kv_policy.current_baseline_loss(),
            "policy_recent_loss": None if self._kv_policy is None else self._kv_policy.current_recent_loss(),
            "mode_histogram": {} if self._kv_policy is None else self._kv_policy.mode_histogram(),
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": seed,
            "cache_lengths": [
                self._kv_cache_length(c["k"]) for c in self.caches
            ],
            "kv_cache_bytes": int(
                sum(self._kv_cache_bytes(c["k"]) + self._kv_cache_bytes(c["v"]) for c in self.caches)
            ),
            "runtime_cache": _cache_delta(cache_before, self.runtime_cache.stats()),
            "step_times_ms": [round(value, 3) for value in step_times_ms],
            "avg_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
            "total_time_s": total_time_s,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": _process_rss_mb(),
            "rss_peak_mb": max(step_rss_mb) if step_rss_mb else rss_before_mb,
            "last_logits": step_logits[-1] if step_logits else None,
        }

    def resume_advanced(
        self,
        session_dir: str | Path,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> dict[str, Any]:
        meta = self.load_session(session_dir)
        generated_ids = list(meta["generated_ids"])
        if not generated_ids:
            raise ValueError("session contains no generated_ids")
        last_logits = meta.get("last_logits")
        if last_logits is None:
            raise ValueError("session is missing last_logits, cannot resume correctly")

        rng = np.random.default_rng(seed) if seed is not None else None
        next_token = self._sample_next_token(
            np.asarray(last_logits),
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            rng=rng,
        )
        self._maybe_apply_kv_policy(source_logits=np.asarray(last_logits), sampled_token=next_token)

        cache_before = self.runtime_cache.stats()
        step_times_ms: list[float] = []
        step_rss_mb: list[float] = []
        rss_before_mb = _process_rss_mb()
        total_start = time.perf_counter()
        generated = list(generated_ids)
        generated.append(next_token)

        step_start = time.perf_counter()
        step_logits = [self._run_step(next_token, len(generated_ids))]
        step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
        step_rss_mb.append(_process_rss_mb())

        for _ in range(max_new_tokens - 1):
            next_token = self._sample_next_token(
                step_logits[-1],
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                rng=rng,
            )
            self._maybe_apply_kv_policy(source_logits=step_logits[-1], sampled_token=next_token)
            generated.append(next_token)
            step_start = time.perf_counter()
            step_logits.append(self._run_step(next_token, len(generated) - 1))
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
            step_rss_mb.append(_process_rss_mb())

        return {
            "resumed_from": generated_ids,
            "generated_ids": generated,
            "new_ids": generated[len(generated_ids) :],
            "cache_mode": self.cache_mode,
            "kv_cache_precision": self.kv_cache_precision,
            "kv_key_precision": self.kv_key_precision,
            "kv_value_precision": self.kv_value_precision,
            "kv_rotation_mode": self.kv_rotation_mode,
            "kv_hot_window": self.kv_hot_window,
            "kv_topk": self.kv_topk,
            "kv_index_refresh_interval": self.kv_index_refresh_interval,
            "kv_block_size": self.kv_block_size,
            "kv_layer_share_stride": self.kv_layer_share_stride,
            "kv_calibration_tokens": self.kv_calibration_tokens,
            "kv_layer_modes": list(self._kv_layer_modes) if self._kv_layer_modes is not None else None,
            "kv_kurtosis_profile": list(self._kv_kurtosis_profile) if self._kv_kurtosis_profile is not None else None,
            "kv_selective_stats": dict(self._kv_selective_stats),
            "kv_cross_layer_overlap": _summarize_cross_layer_overlap_stats(self._kv_cross_layer_overlap_stats),
            "current_kv_mode": self.current_kv_mode,
            "kv_mode_trace": list(self._kv_mode_trace),
            "switch_events": list(self._kv_switch_events),
            "policy_baseline_loss": None if self._kv_policy is None else self._kv_policy.current_baseline_loss(),
            "policy_recent_loss": None if self._kv_policy is None else self._kv_policy.current_recent_loss(),
            "mode_histogram": {} if self._kv_policy is None else self._kv_policy.mode_histogram(),
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": seed,
            "cache_lengths": [
                self._kv_cache_length(c["k"]) for c in self.caches
            ],
            "kv_cache_bytes": int(
                sum(self._kv_cache_bytes(c["k"]) + self._kv_cache_bytes(c["v"]) for c in self.caches)
            ),
            "runtime_cache": _cache_delta(cache_before, self.runtime_cache.stats()),
            "step_times_ms": [round(value, 3) for value in step_times_ms],
            "avg_step_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
            "total_time_s": time.perf_counter() - total_start,
            "rss_before_mb": rss_before_mb,
            "rss_after_mb": _process_rss_mb(),
            "rss_peak_mb": max(step_rss_mb) if step_rss_mb else rss_before_mb,
            "last_logits": step_logits[-1] if step_logits else np.asarray(last_logits),
        }


def gpt2_generate_greedy(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    cache_mode: str = "fresh",
    kv_cache_precision: str = "fp32",
    kv_key_precision: str | None = None,
    kv_value_precision: str | None = None,
    kv_quant_seed: int = 7,
    kv_rotation_mode: str = "hadamard",
    kv_hot_window: int = 0,
    kv_topk: int = 0,
    kv_index_refresh_interval: int = 8,
    kv_block_size: int = 0,
    kv_layer_share_stride: int = 0,
    kv_calibration_tokens: int = 128,
) -> dict[str, Any]:
    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode=cache_mode,
        kv_cache_precision=kv_cache_precision,
        kv_key_precision=kv_key_precision,
        kv_value_precision=kv_value_precision,
        kv_quant_seed=kv_quant_seed,
        kv_rotation_mode=kv_rotation_mode,
        kv_hot_window=kv_hot_window,
        kv_topk=kv_topk,
        kv_index_refresh_interval=kv_index_refresh_interval,
        kv_block_size=kv_block_size,
        kv_layer_share_stride=kv_layer_share_stride,
        kv_calibration_tokens=kv_calibration_tokens,
    )
    return engine.generate(prompt_ids, max_new_tokens)


def gpt2_generate_sample(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    cache_mode: str = "fresh",
    kv_cache_precision: str = "fp32",
    kv_key_precision: str | None = None,
    kv_value_precision: str | None = None,
    kv_quant_seed: int = 7,
    kv_rotation_mode: str = "hadamard",
    kv_hot_window: int = 0,
    kv_topk: int = 0,
    kv_index_refresh_interval: int = 8,
    kv_block_size: int = 0,
    kv_layer_share_stride: int = 0,
    kv_calibration_tokens: int = 128,
) -> dict[str, Any]:
    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode=cache_mode,
        kv_cache_precision=kv_cache_precision,
        kv_key_precision=kv_key_precision,
        kv_value_precision=kv_value_precision,
        kv_quant_seed=kv_quant_seed,
        kv_rotation_mode=kv_rotation_mode,
        kv_hot_window=kv_hot_window,
        kv_topk=kv_topk,
        kv_index_refresh_interval=kv_index_refresh_interval,
        kv_block_size=kv_block_size,
        kv_layer_share_stride=kv_layer_share_stride,
        kv_calibration_tokens=kv_calibration_tokens,
    )
    return engine.generate_advanced(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


def gpt2_resume_generation(
    export_dir: str | Path,
    session_dir: str | Path,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    cache_mode: str = "session",
    kv_cache_precision: str = "fp32",
    kv_key_precision: str | None = None,
    kv_value_precision: str | None = None,
    kv_quant_seed: int = 7,
    kv_rotation_mode: str = "hadamard",
    kv_hot_window: int = 0,
    kv_topk: int = 0,
    kv_index_refresh_interval: int = 8,
    kv_block_size: int = 0,
    kv_layer_share_stride: int = 0,
    kv_calibration_tokens: int = 128,
) -> dict[str, Any]:
    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode=cache_mode,
        kv_cache_precision=kv_cache_precision,
        kv_key_precision=kv_key_precision,
        kv_value_precision=kv_value_precision,
        kv_quant_seed=kv_quant_seed,
        kv_rotation_mode=kv_rotation_mode,
        kv_hot_window=kv_hot_window,
        kv_topk=kv_topk,
        kv_index_refresh_interval=kv_index_refresh_interval,
        kv_block_size=kv_block_size,
        kv_layer_share_stride=kv_layer_share_stride,
        kv_calibration_tokens=kv_calibration_tokens,
    )
    return engine.resume_advanced(
        session_dir,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


def benchmark_gpt2_generation_cache(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    kv_cache_precision: str = "fp32",
    kv_key_precision: str | None = None,
    kv_value_precision: str | None = None,
    kv_quant_seed: int = 7,
    kv_rotation_mode: str = "hadamard",
    kv_hot_window: int = 0,
    kv_topk: int = 0,
    kv_index_refresh_interval: int = 8,
    kv_block_size: int = 0,
    kv_layer_share_stride: int = 0,
    kv_calibration_tokens: int = 128,
) -> dict[str, Any]:
    clear_session_runtime_cache(export_dir)
    runs = {
        "no_cache": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="none",
            kv_cache_precision=kv_cache_precision,
            kv_key_precision=kv_key_precision,
            kv_value_precision=kv_value_precision,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        ),
        "fresh_cache": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="fresh",
            kv_cache_precision=kv_cache_precision,
            kv_key_precision=kv_key_precision,
            kv_value_precision=kv_value_precision,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        ),
        "session_cold": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="session",
            kv_cache_precision=kv_cache_precision,
            kv_key_precision=kv_key_precision,
            kv_value_precision=kv_value_precision,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        ),
        "session_warm": gpt2_generate_greedy(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            cache_mode="session",
            kv_cache_precision=kv_cache_precision,
            kv_key_precision=kv_key_precision,
            kv_value_precision=kv_value_precision,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        ),
    }
    baseline = runs["no_cache"]["total_time_s"]
    summary: dict[str, Any] = {}
    for name, result in runs.items():
        total_time = float(result["total_time_s"])
        summary[name] = {
            "generated_ids": result["generated_ids"],
            "total_time_s": total_time,
            "avg_step_ms": float(result["avg_step_ms"]),
            "runtime_cache": result["runtime_cache"],
            "kv_cache_precision": result["kv_cache_precision"],
            "kv_key_precision": result["kv_key_precision"],
            "kv_value_precision": result["kv_value_precision"],
            "kv_rotation_mode": result["kv_rotation_mode"],
            "kv_hot_window": result["kv_hot_window"],
            "kv_cache_bytes": int(result["kv_cache_bytes"]),
            "speedup_vs_no_cache": (baseline / total_time) if total_time else float("inf"),
        }
    return {
        "prompt_ids": list(prompt_ids),
        "max_new_tokens": max_new_tokens,
        "kv_cache_precision": kv_cache_precision,
        "kv_key_precision": kv_key_precision,
        "kv_value_precision": kv_value_precision,
        "kv_rotation_mode": kv_rotation_mode,
        "kv_hot_window": kv_hot_window,
        "kv_topk": kv_topk,
        "kv_index_refresh_interval": kv_index_refresh_interval,
        "kv_block_size": kv_block_size,
        "kv_layer_share_stride": kv_layer_share_stride,
        "kv_calibration_tokens": kv_calibration_tokens,
        "runs": summary,
    }


def benchmark_gpt2_generation_suite(
    export_dir: str | Path,
    *,
    prompt_lengths: list[int],
    max_new_tokens: int,
    warm_repeats: int = 2,
    kv_cache_precision: str = "fp32",
    kv_key_precision: str | None = None,
    kv_value_precision: str | None = None,
    kv_quant_seed: int = 7,
    kv_rotation_mode: str = "hadamard",
    kv_hot_window: int = 0,
    kv_topk: int = 0,
    kv_index_refresh_interval: int = 8,
    kv_block_size: int = 0,
    kv_layer_share_stride: int = 0,
    kv_calibration_tokens: int = 128,
) -> dict[str, Any]:
    config = load_manifest(export_dir).get("config", {})
    vocab_size = int(config["vocab_size"])
    suite: dict[str, Any] = {}

    for prompt_length in prompt_lengths:
        prompt_ids = [((idx * 7) + 3) % vocab_size for idx in range(prompt_length)]
        result = benchmark_gpt2_generation_cache(
            export_dir,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            kv_cache_precision=kv_cache_precision,
            kv_key_precision=kv_key_precision,
            kv_value_precision=kv_value_precision,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        )

        clear_session_runtime_cache(export_dir)
        session_engine = GPT2StreamingEngine(
            export_dir,
            cache_mode="session",
            kv_cache_precision=kv_cache_precision,
            kv_key_precision=kv_key_precision,
            kv_value_precision=kv_value_precision,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        )
        warm_runs = [session_engine.generate(prompt_ids, max_new_tokens) for _ in range(warm_repeats)]
        result["session_warm_repeats"] = [
            {
                "total_time_s": run["total_time_s"],
                "avg_step_ms": run["avg_step_ms"],
                "runtime_cache": run["runtime_cache"],
                "generated_ids": run["generated_ids"],
                "kv_cache_precision": run["kv_cache_precision"],
                "kv_key_precision": run["kv_key_precision"],
                "kv_value_precision": run["kv_value_precision"],
                "kv_rotation_mode": run["kv_rotation_mode"],
                "kv_hot_window": run["kv_hot_window"],
                "kv_cache_bytes": int(run["kv_cache_bytes"]),
            }
            for run in warm_runs
        ]
        result["session_warm_avg"] = {
            "total_time_s": float(np.mean([run["total_time_s"] for run in warm_runs])),
            "avg_step_ms": float(np.mean([run["avg_step_ms"] for run in warm_runs])),
            "generated_ids": warm_runs[-1]["generated_ids"],
            "kv_cache_precision": kv_cache_precision,
            "kv_key_precision": kv_key_precision,
            "kv_value_precision": kv_value_precision,
            "kv_rotation_mode": kv_rotation_mode,
            "kv_hot_window": kv_hot_window,
            "kv_topk": kv_topk,
            "kv_index_refresh_interval": kv_index_refresh_interval,
            "kv_block_size": kv_block_size,
            "kv_layer_share_stride": kv_layer_share_stride,
            "kv_cache_bytes": int(np.mean([run["kv_cache_bytes"] for run in warm_runs])),
        }
        suite[str(prompt_length)] = result

    return {
        "prompt_lengths": list(prompt_lengths),
        "max_new_tokens": max_new_tokens,
        "warm_repeats": warm_repeats,
        "kv_cache_precision": kv_cache_precision,
        "kv_key_precision": kv_key_precision,
        "kv_value_precision": kv_value_precision,
        "kv_rotation_mode": kv_rotation_mode,
        "kv_hot_window": kv_hot_window,
        "kv_topk": kv_topk,
        "kv_index_refresh_interval": kv_index_refresh_interval,
        "kv_block_size": kv_block_size,
        "kv_layer_share_stride": kv_layer_share_stride,
        "kv_calibration_tokens": kv_calibration_tokens,
        "suite": suite,
    }


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return shifted - np.log(np.sum(exp))


def _collect_prompt_logits(engine: GPT2StreamingEngine, prompt_ids: list[int]) -> list[np.ndarray]:
    logits_trace: list[np.ndarray] = []
    for item in engine.stream_generate(
        prompt_ids,
        max_new_tokens=0,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        seed=None,
        reset_sequence=True,
    ):
        if item["phase"] == "prompt":
            logits_trace.append(np.asarray(item["last_logits"], dtype=np.float32))
    return logits_trace


def _prompt_perplexity(prompt_ids: list[int], logits_trace: list[np.ndarray]) -> float:
    if len(prompt_ids) < 2 or len(logits_trace) < 2:
        return float("nan")
    losses: list[float] = []
    for index in range(len(prompt_ids) - 1):
        target = prompt_ids[index + 1]
        log_probs = _log_softmax(logits_trace[index])
        losses.append(-float(log_probs[target]))
    return float(np.exp(np.mean(losses)))


def benchmark_gpt2_kv_mode_matrix(
    export_dir: str | Path,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    kv_variants: list[dict[str, Any]],
    kv_quant_seed: int = 7,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    baseline_name: str | None = None
    baseline_logits: np.ndarray | None = None
    baseline_ids: list[int] | None = None

    for variant in kv_variants:
        name = str(variant["name"])
        kv_mode = str(variant["kv_cache_precision"])
        kv_key_precision = variant.get("kv_key_precision")
        kv_value_precision = variant.get("kv_value_precision")
        kv_rotation_mode = str(variant.get("kv_rotation_mode", "hadamard"))
        kv_hot_window = int(variant.get("kv_hot_window", 0))
        kv_topk = int(variant.get("kv_topk", 0))
        kv_index_refresh_interval = int(variant.get("kv_index_refresh_interval", 8))
        kv_block_size = int(variant.get("kv_block_size", 0))
        kv_layer_share_stride = int(variant.get("kv_layer_share_stride", 0))
        kv_calibration_tokens = int(variant.get("kv_calibration_tokens", 128))

        engine = GPT2StreamingEngine(
            export_dir,
            cache_mode="fresh",
            kv_cache_precision=kv_mode,
            kv_key_precision=str(kv_key_precision) if kv_key_precision is not None else None,
            kv_value_precision=str(kv_value_precision) if kv_value_precision is not None else None,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        )
        prompt_logits = _collect_prompt_logits(engine, prompt_ids)
        perplexity = _prompt_perplexity(prompt_ids, prompt_logits)

        engine = GPT2StreamingEngine(
            export_dir,
            cache_mode="fresh",
            kv_cache_precision=kv_mode,
            kv_key_precision=str(kv_key_precision) if kv_key_precision is not None else None,
            kv_value_precision=str(kv_value_precision) if kv_value_precision is not None else None,
            kv_quant_seed=kv_quant_seed,
            kv_rotation_mode=kv_rotation_mode,
            kv_hot_window=kv_hot_window,
            kv_topk=kv_topk,
            kv_index_refresh_interval=kv_index_refresh_interval,
            kv_block_size=kv_block_size,
            kv_layer_share_stride=kv_layer_share_stride,
            kv_calibration_tokens=kv_calibration_tokens,
        )
        run = engine.generate(prompt_ids, max_new_tokens)
        current_last_logits = np.asarray(run["last_logits"], dtype=np.float32)

        metrics = {
            "kv_cache_precision": kv_mode,
            "kv_key_precision": str(kv_key_precision) if kv_key_precision is not None else None,
            "kv_value_precision": str(kv_value_precision) if kv_value_precision is not None else None,
            "kv_rotation_mode": kv_rotation_mode,
            "kv_hot_window": kv_hot_window,
            "kv_calibration_tokens": kv_calibration_tokens,
            "kv_topk": kv_topk,
            "kv_index_refresh_interval": kv_index_refresh_interval,
            "kv_block_size": kv_block_size,
            "kv_layer_share_stride": kv_layer_share_stride,
            "prompt_perplexity": perplexity,
            "total_time_s": float(run["total_time_s"]),
            "avg_step_ms": float(run["avg_step_ms"]),
            "kv_cache_bytes": int(run["kv_cache_bytes"]),
            "generated_ids": list(run["generated_ids"]),
            "cache_lengths": list(run["cache_lengths"]),
            "kv_layer_modes": run.get("kv_layer_modes"),
            "kv_kurtosis_profile": run.get("kv_kurtosis_profile"),
            "kv_selective_stats": run.get("kv_selective_stats"),
            "kv_cross_layer_overlap": run.get("kv_cross_layer_overlap"),
            "current_kv_mode": run.get("current_kv_mode"),
            "kv_mode_trace": run.get("kv_mode_trace"),
            "switch_events": run.get("switch_events"),
            "policy_baseline_loss": run.get("policy_baseline_loss"),
            "policy_recent_loss": run.get("policy_recent_loss"),
            "mode_histogram": run.get("mode_histogram"),
        }

        if baseline_name is None:
            baseline_name = name
            baseline_logits = current_last_logits
            baseline_ids = list(run["generated_ids"])
            metrics["logit_comparison_vs_baseline"] = None
            metrics["generated_match_vs_baseline"] = True
        else:
            assert baseline_logits is not None
            diff = np.abs(current_last_logits - baseline_logits)
            metrics["logit_comparison_vs_baseline"] = {
                "max_abs_err": float(np.max(diff)),
                "mean_abs_err": float(np.mean(diff)),
                "cosine_similarity": float(
                    np.dot(current_last_logits, baseline_logits)
                    / (np.linalg.norm(current_last_logits) * np.linalg.norm(baseline_logits))
                ),
            }
            metrics["generated_match_vs_baseline"] = list(run["generated_ids"]) == baseline_ids

        results[name] = metrics

    return {
        "prompt_ids": list(prompt_ids),
        "max_new_tokens": int(max_new_tokens),
        "kv_quant_seed": int(kv_quant_seed),
        "baseline_variant": baseline_name,
        "variants": results,
    }
