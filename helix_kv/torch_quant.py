from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _maybe_compile(fn):
    if not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, dynamic=True)
    except Exception:
        return fn


def _normalize_scaling_strategy(strategy: str | None, *, default: str = "per-token") -> str:
    lowered = str(strategy or default).strip().lower()
    if lowered in {"token", "per-token", "per_token"}:
        return "per-token"
    if lowered in {"channel", "per-channel", "per_channel"}:
        return "per-channel"
    raise ValueError(f"unsupported scaling strategy: {strategy}")


def _fwht_last_axis_torch(values: torch.Tensor) -> torch.Tensor:
    output = values
    length = int(output.shape[-1])
    step = 1
    while step < length:
        shape = output.shape[:-1] + (length // (step * 2), step * 2)
        output = output.reshape(shape)
        left = output[..., :step]
        right = output[..., step:]
        output = torch.cat((left + right, left - right), dim=-1)
        output = output.reshape(values.shape[:-1] + (length,))
        step *= 2
    return output


def _hadamard_forward(values: torch.Tensor, signs: torch.Tensor, original_dim: int, rotated_dim: int) -> torch.Tensor:
    output = values.to(dtype=torch.float32)
    if rotated_dim != original_dim:
        output = torch.nn.functional.pad(output, (0, rotated_dim - original_dim))
    output = output * signs
    output = _fwht_last_axis_torch(output)
    return output / math.sqrt(float(rotated_dim))


def _hadamard_inverse(values: torch.Tensor, signs: torch.Tensor, original_dim: int, rotated_dim: int) -> torch.Tensor:
    output = _fwht_last_axis_torch(values.to(dtype=torch.float32))
    output = (output / math.sqrt(float(rotated_dim))) * signs
    if rotated_dim != original_dim:
        output = output[..., :original_dim]
    return output


_COMPILED_HADAMARD_FORWARD = _maybe_compile(_hadamard_forward)
_COMPILED_HADAMARD_INVERSE = _maybe_compile(_hadamard_inverse)


def _pack_nibbles_torch(indices: torch.Tensor) -> torch.Tensor:
    values = indices.to(dtype=torch.uint8)
    if values.shape[-1] % 2:
        values = torch.nn.functional.pad(values, (0, 1))
    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).to(dtype=torch.uint8)


def _unpack_nibbles_torch(packed: torch.Tensor, original_length: int) -> torch.Tensor:
    values = packed.to(dtype=torch.uint8)
    unpacked = torch.empty(values.shape[:-1] + (values.shape[-1] * 2,), dtype=torch.uint8, device=values.device)
    unpacked[..., 0::2] = values & 0x0F
    unpacked[..., 1::2] = (values >> 4) & 0x0F
    return unpacked[..., : int(original_length)]


def _quantize_against_centroids(values: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    expanded = values.unsqueeze(-1)
    centroids_view = centroids.view(*([1] * values.ndim), -1)
    distances = torch.abs(expanded - centroids_view)
    return torch.argmin(distances, dim=-1).to(dtype=torch.uint8)


def _fit_codebook(values: torch.Tensor, initial_centroids: torch.Tensor, *, max_iter: int = 5) -> torch.Tensor:
    samples = values.reshape(-1).to(dtype=torch.float32)
    samples = samples[torch.isfinite(samples)]
    if samples.numel() == 0:
        return initial_centroids.to(dtype=torch.float32)
    centroids = initial_centroids.to(device=samples.device, dtype=torch.float32).clone()
    for _ in range(max(int(max_iter), 0)):
        assignments = _quantize_against_centroids(samples, centroids).to(dtype=torch.long)
        updated = centroids.clone()
        for level in range(int(centroids.numel())):
            mask = assignments == level
            if torch.any(mask):
                updated[level] = torch.mean(samples[mask])
        centroids = torch.sort(updated).values
    return centroids


def _pearson_kurtosis(values: torch.Tensor) -> float:
    sample = values.to(dtype=torch.float64).reshape(-1)
    if sample.numel() == 0:
        return 3.0
    centered = sample - torch.mean(sample)
    variance = torch.mean(centered * centered)
    if float(variance.item()) <= 1e-18:
        return 3.0
    fourth = torch.mean(centered**4)
    return float((fourth / (variance * variance)).item())


def _channel_absmax(values: torch.Tensor) -> torch.Tensor:
    return values.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)


def _token_absmax(values: torch.Tensor) -> torch.Tensor:
    return values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)


def _compute_int8_scales(
    rotated: torch.Tensor,
    *,
    scaling_strategy: str,
    reference_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    strategy = _normalize_scaling_strategy(scaling_strategy)
    if strategy == "per-channel":
        base = reference_scales if reference_scales is not None else _channel_absmax(rotated)
    else:
        base = _token_absmax(rotated)
    return (base / 127.0).clamp_min(1e-6)


def _compute_4bit_scales(
    rotated: torch.Tensor,
    *,
    scaling_strategy: str,
    reference_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    strategy = _normalize_scaling_strategy(scaling_strategy)
    if strategy == "per-channel":
        return (reference_scales if reference_scales is not None else _channel_absmax(rotated)).clamp_min(1e-6)
    return _token_absmax(rotated).clamp_min(1e-6)


@dataclass
class TorchRotation:
    mode: str
    original_dim: int
    rotated_dim: int
    device: torch.device
    matrix: torch.Tensor | None = None
    signs: torch.Tensor | None = None

    @classmethod
    def from_legacy(cls, rotation: Any, *, device: torch.device) -> "TorchRotation":
        mode = "hadamard" if hasattr(rotation, "signs") else "qr"
        matrix = None
        signs = None
        if getattr(rotation, "matrix", None) is not None:
            matrix = torch.tensor(np.asarray(rotation.matrix, dtype=np.float32), dtype=torch.float32, device=device)
        if getattr(rotation, "signs", None) is not None:
            signs = torch.tensor(np.asarray(rotation.signs, dtype=np.float32), dtype=torch.float32, device=device)
        return cls(
            mode=mode,
            original_dim=int(getattr(rotation, "original_dim")),
            rotated_dim=int(getattr(rotation, "rotated_dim")),
            device=device,
            matrix=matrix,
            signs=signs,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        array = values.to(device=self.device, dtype=torch.float32)
        if self.mode == "hadamard":
            if self.signs is None:
                raise ValueError("hadamard rotation missing signs")
            if array.is_cuda:
                try:
                    return _COMPILED_HADAMARD_FORWARD(array, self.signs, self.original_dim, self.rotated_dim)
                except Exception:
                    return _hadamard_forward(array, self.signs, self.original_dim, self.rotated_dim)
            return _hadamard_forward(array, self.signs, self.original_dim, self.rotated_dim)
        if self.matrix is None:
            raise ValueError("dense rotation missing matrix")
        return torch.einsum("...d,df->...f", array, self.matrix)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        array = values.to(device=self.device, dtype=torch.float32)
        if self.mode == "hadamard":
            if self.signs is None:
                raise ValueError("hadamard rotation missing signs")
            if array.is_cuda:
                try:
                    return _COMPILED_HADAMARD_INVERSE(array, self.signs, self.original_dim, self.rotated_dim)
                except Exception:
                    return _hadamard_inverse(array, self.signs, self.original_dim, self.rotated_dim)
            return _hadamard_inverse(array, self.signs, self.original_dim, self.rotated_dim)
        if self.matrix is None:
            raise ValueError("dense rotation missing matrix")
        return torch.einsum("...d,df->...f", array, self.matrix.transpose(0, 1))


def _compress_int8_per_token_eager(values: torch.Tensor, rotation: TorchRotation) -> tuple[torch.Tensor, torch.Tensor]:
    rotated = rotation.forward(values)
    safe_scale = _compute_int8_scales(rotated, scaling_strategy="per-token")
    quantized = torch.clamp(torch.round(rotated / safe_scale), -127, 127).to(dtype=torch.int8)
    return quantized, safe_scale.to(dtype=torch.float16)


_COMPRESSED_INT8_PER_TOKEN = _maybe_compile(_compress_int8_per_token_eager)


def _compress_int8(
    values: torch.Tensor,
    *,
    rotation: TorchRotation,
    scaling_strategy: str = "per-token",
    reference_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    strategy = _normalize_scaling_strategy(scaling_strategy)
    if values.is_cuda and rotation.mode == "hadamard" and strategy == "per-token" and reference_scales is None:
        try:
            return _COMPRESSED_INT8_PER_TOKEN(values, rotation)
        except Exception:
            return _compress_int8_per_token_eager(values, rotation)
    rotated = rotation.forward(values)
    safe_scale = _compute_int8_scales(rotated, scaling_strategy=strategy, reference_scales=reference_scales)
    quantized = torch.clamp(torch.round(rotated / safe_scale), -127, 127).to(dtype=torch.int8)
    return quantized, safe_scale.to(dtype=torch.float16)


@dataclass
class TorchInt8KVArray:
    q: torch.Tensor
    scales: torch.Tensor
    rotation: TorchRotation
    scaling_strategy: str = "per-token"

    @classmethod
    def from_values(
        cls,
        values: torch.Tensor,
        *,
        rotation: TorchRotation,
        scaling_strategy: str = "per-token",
        calibration_values: torch.Tensor | None = None,
    ) -> "TorchInt8KVArray":
        strategy = _normalize_scaling_strategy(scaling_strategy)
        reference_scales = None
        if strategy == "per-channel":
            calibration = values if calibration_values is None else calibration_values
            rotated = rotation.forward(calibration.to(dtype=torch.float32))
            reference_scales = (_channel_absmax(rotated) / 127.0).clamp_min(1e-6).to(dtype=torch.float16)
        q, scales = _compress_int8(
            values.to(dtype=torch.float32),
            rotation=rotation,
            scaling_strategy=strategy,
            reference_scales=reference_scales,
        )
        return cls(q=q, scales=scales, rotation=rotation, scaling_strategy=strategy)

    @property
    def length(self) -> int:
        return int(self.q.shape[1])

    @property
    def nbytes(self) -> int:
        return int((self.q.numel() * self.q.element_size()) + (self.scales.numel() * self.scales.element_size()))

    def to_float(self, *, dtype: torch.dtype) -> torch.Tensor:
        restored = self.q.to(dtype=torch.float32) * self.scales.to(dtype=torch.float32)
        return self.rotation.inverse(restored).to(dtype=dtype)

    def append_compressed(self, values: torch.Tensor) -> "TorchInt8KVArray":
        new_q, new_scales = _compress_int8(
            values.to(dtype=torch.float32),
            rotation=self.rotation,
            scaling_strategy=self.scaling_strategy,
            reference_scales=self.scales if self.scaling_strategy == "per-channel" else None,
        )
        combined_scales = self.scales if self.scaling_strategy == "per-channel" else torch.cat([self.scales, new_scales], dim=1)
        return TorchInt8KVArray(
            q=torch.cat([self.q, new_q], dim=1),
            scales=combined_scales,
            rotation=self.rotation,
            scaling_strategy=self.scaling_strategy,
        )


@dataclass
class Torch4BitQuantizer:
    centroids: torch.Tensor
    rotation: TorchRotation
    scaling_strategy: str = "per-channel"
    reference_scales: torch.Tensor | None = None
    max_iter: int = 5

    @property
    def channel_scales(self) -> torch.Tensor | None:
        return self.reference_scales

    @property
    def scale_tensor_name(self) -> str:
        return "channel_scales" if self.scaling_strategy == "per-channel" else "token_scales"

    @classmethod
    def from_calibration(
        cls,
        calibration_values: torch.Tensor,
        *,
        rotation: TorchRotation,
        initial_centroids: torch.Tensor,
        scaling_strategy: str = "per-channel",
        max_iter: int = 5,
    ) -> "Torch4BitQuantizer":
        strategy = _normalize_scaling_strategy(scaling_strategy, default="per-channel")
        rotated = rotation.forward(calibration_values)
        calibration_scales = _compute_4bit_scales(rotated, scaling_strategy=strategy)
        normalized = (rotated / calibration_scales.to(dtype=torch.float32)).clamp(-1.0, 1.0)
        centroids = _fit_codebook(normalized, initial_centroids, max_iter=max_iter)
        reference_scales = None
        if strategy == "per-channel":
            reference_scales = calibration_scales.to(device=calibration_values.device, dtype=torch.float16)
        return cls(
            centroids=centroids.to(device=calibration_values.device, dtype=torch.float32),
            rotation=rotation,
            scaling_strategy=strategy,
            reference_scales=reference_scales,
            max_iter=max_iter,
        )

    def quantize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotated = self.rotation.forward(values)
        scales = _compute_4bit_scales(
            rotated,
            scaling_strategy=self.scaling_strategy,
            reference_scales=self.reference_scales,
        )
        normalized = (rotated / scales.to(dtype=torch.float32)).clamp(-1.0, 1.0)
        indices = _quantize_against_centroids(normalized, self.centroids)
        packed = _pack_nibbles_torch(indices)
        return packed, scales.to(dtype=torch.float16)

    def dequantize(self, packed: torch.Tensor, *, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        unpacked = _unpack_nibbles_torch(packed, self.rotation.rotated_dim).to(dtype=torch.long)
        normalized = self.centroids[unpacked]
        rotated = normalized * scales.to(dtype=torch.float32)
        return self.rotation.inverse(rotated).to(dtype=dtype)


@dataclass
class Torch4BitKVArray:
    packed: torch.Tensor
    scales: torch.Tensor
    quantizer: Torch4BitQuantizer

    @classmethod
    def from_values(cls, values: torch.Tensor, *, quantizer: Torch4BitQuantizer) -> "Torch4BitKVArray":
        packed, scales = quantizer.quantize(values)
        return cls(packed=packed, scales=scales, quantizer=quantizer)

    @property
    def length(self) -> int:
        return int(self.packed.shape[1])

    @property
    def nbytes(self) -> int:
        centroids = self.quantizer.centroids
        return int(
            (self.packed.numel() * self.packed.element_size())
            + (self.scales.numel() * self.scales.element_size())
            + (centroids.numel() * centroids.element_size())
        )

    def to_float(self, *, dtype: torch.dtype) -> torch.Tensor:
        return self.quantizer.dequantize(self.packed, scales=self.scales, dtype=dtype)

    def append_compressed(self, values: torch.Tensor) -> "Torch4BitKVArray":
        packed, scales = self.quantizer.quantize(values)
        combined_scales = self.scales if self.quantizer.scaling_strategy == "per-channel" else torch.cat([self.scales, scales], dim=1)
        return Torch4BitKVArray(
            packed=torch.cat([self.packed, packed], dim=1),
            scales=combined_scales,
            quantizer=self.quantizer,
        )


@dataclass
class TorchHotWindowKVArray:
    cold: torch.Tensor | TorchInt8KVArray | Torch4BitKVArray | None
    hot: torch.Tensor

    @property
    def cold_length(self) -> int:
        if self.cold is None:
            return 0
        if isinstance(self.cold, torch.Tensor):
            return int(self.cold.shape[1])
        return int(self.cold.length)

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
            if isinstance(self.cold, torch.Tensor):
                cold_bytes = int(self.cold.numel() * self.cold.element_size())
            else:
                cold_bytes = int(self.cold.nbytes)
        hot_bytes = int(self.hot.numel() * self.hot.element_size())
        return cold_bytes + hot_bytes

    def to_float(self, *, dtype: torch.dtype) -> torch.Tensor:
        cold_values: torch.Tensor | None = None
        if self.cold is not None:
            if isinstance(self.cold, torch.Tensor):
                cold_values = self.cold.to(dtype=dtype)
            else:
                cold_values = self.cold.to_float(dtype=dtype)
        if cold_values is None or cold_values.shape[1] == 0:
            return self.hot.to(dtype=dtype)
        if self.hot.shape[1] == 0:
            return cold_values.to(dtype=dtype)
        return torch.cat([cold_values.to(dtype=dtype), self.hot.to(dtype=dtype)], dim=1)

    def append_token(self, new_kv: torch.Tensor, max_hot: int, store_fn) -> "TorchHotWindowKVArray":
        hot = torch.cat([self.hot, new_kv], dim=1)
        if hot.shape[1] <= int(max_hot):
            return TorchHotWindowKVArray(cold=self.cold, hot=hot)
        spill_len = hot.shape[1] - int(max_hot)
        spill = hot[:, :spill_len, :]
        new_hot = hot[:, spill_len:, :]
        if self.cold is None:
            new_cold = store_fn(spill)
        elif hasattr(self.cold, "append_compressed"):
            new_cold = self.cold.append_compressed(spill)
        elif isinstance(self.cold, torch.Tensor):
            new_cold = torch.cat([self.cold, spill], dim=1)
        else:
            new_cold = store_fn(torch.cat([self.cold.to_float(dtype=spill.dtype), spill], dim=1))
        return TorchHotWindowKVArray(cold=new_cold, hot=new_hot)
