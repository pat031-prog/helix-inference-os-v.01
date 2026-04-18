"""
HelixLinear — drop-in nn.Linear replacement backed by CDNA v3 compressed storage.

The compressed form IS the executable. Weights are never fully materialized
in persistent GPU memory. Storage is VQ-compressed (codebook + uint8 indices),
with optional sparse sidecar corrections and SVD residual factors.

Memory footprint: ~1/4 of nn.Linear (uint8 indices vs float32 weights).
Compute: codebook gather + matmul per forward (slightly more compute, much less memory).

Usage:
    from helix_substrate.helix_linear import HelixLinear, swap_to_helix, load_cdna_factors

    # Load factors from CDNA v3 directory
    factors = load_cdna_factors("/path/to/cdna_output/")

    # Replace all nn.Linear modules with HelixLinear
    model = swap_to_helix(model, factors)
    model = model.cuda().eval()

Work Order: WO-HELIX-LINEAR-01
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class HelixLinearStats:
    """Compression statistics for a HelixLinear layer."""
    tensor_name: str
    in_features: int
    out_features: int
    rank: int  # SVD rank (0 if VQ-only)
    n_outliers: int
    compression_ratio: float
    cosine_fidelity: float  # best available cosine from stats.json
    storage_bytes: int  # total bytes on disk
    dense_bytes: int  # what nn.Linear would use (out * in * 4)


class HelixLinear(nn.Module):
    """
    Drop-in nn.Linear replacement that stores weights in CDNA v3 format.

    Internal representation:
        W ≈ codebook[indices] + sidecar_deltas + (U * s) @ Vt

    Where:
        codebook: [256] float32 cluster centers

    Instrumentation:
        _last_dispatch_path: "fused" | "naive" | "unknown" — set on every forward() call.
        dispatch_metadata(): returns dict with dispatch path, device, kernel info.
        Emits RuntimeWarning if CUDA input falls to naive path (Triton unavailable).
        indices:  [out, in] uint8 cluster assignments  (4x smaller than float32)
        sidecar:  sparse outlier corrections (precomputed rows/cols/deltas)
        U, s, Vt: optional SVD residual factors

    Forward paths:
        GPU (Triton fused):  Codebook gathered in registers, W never in global memory.
        CPU (tiled naive):   256-row tiles via _dequant_tile(), peak ~8.75 MB temporary.

    Memory: codebook(1KB) + indices(out*in bytes) + sidecar(~few KB) + SVD(small)
    vs nn.Linear: out*in*4 bytes

    Full W is never materialized during forward(). Only bounded tiles exist transiently.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        codebook: torch.Tensor,
        indices: torch.Tensor,
        sidecar_positions: Optional[torch.Tensor] = None,
        sidecar_values: Optional[torch.Tensor] = None,
        svd_U: Optional[torch.Tensor] = None,
        svd_s: Optional[torch.Tensor] = None,
        svd_Vt: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        tensor_name: str = "",
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_name = tensor_name
        self.compute_dtype = compute_dtype
        self._last_dispatch_path: str = "unknown"
        # Cached dispatch decision — frozen at init, updated on .to()/.cuda()
        # Avoids per-forward property evaluation (154 calls/token).
        self._fused_available: bool = self._check_fused_available()
        # Cell-driven SVD skip: when True, SVD correction is bypassed in forward().
        # Set externally by codec_bridge.apply_cell_signal_to_model() when a cell's
        # codebook health indicates SVD is unnecessary (saving compute).
        # Default False = always apply SVD if factors are present.
        self._cell_skip_svd: bool = False
        # Kurtosis gate: runtime SVD gating for Class 2 modules.
        # Attached via attach_kurtosis_gate() or kurtosis_gate.attach_kurtosis_gates().
        # When present, forward() computes input kurtosis and lets the gate
        # decide _cell_skip_svd before the main computation. Zero overhead
        # when None (the common case for 147 of 154 modules).
        self._kurtosis_gate = None
        # Sidecar phase: "fused" | "scatter" | None (auto). Frozen at request start.
        self._sidecar_phase: Optional[str] = None

        # VQ components (read-only buffers, not parameters)
        self.register_buffer("codebook", codebook.contiguous())  # [256]
        # Store indices as uint8 to save memory (4x vs int64).
        # Convert to long only during forward() for the gather operation.
        if indices.dtype != torch.uint8:
            indices = indices.to(torch.uint8)
        self.register_buffer("indices", indices.contiguous())  # [out, in] uint8

        # FP16 codebook for mixed-precision compute path
        if compute_dtype == torch.float16:
            self.register_buffer("codebook_f16", codebook.half().contiguous())
        else:
            self.register_buffer("codebook_f16", None)

        # Sidecar (sparse outlier corrections)
        if sidecar_positions is not None and sidecar_values is not None:
            self.register_buffer("sidecar_positions", sidecar_positions.contiguous())
            self.register_buffer("sidecar_values", sidecar_values.contiguous())
            # Precompute VQ values at sidecar positions for fused kernel
            # (avoids re-gather during forward)
            idx_flat = indices.reshape(-1)
            vq_at_sidecar = codebook[idx_flat[sidecar_positions].long()]
            self.register_buffer("_sidecar_vq_vals", vq_at_sidecar.contiguous())
            # Precompute row/col/delta for chunked naive + fused paths (Phase 4)
            self.register_buffer("_sidecar_rows", (sidecar_positions // in_features).long())
            self.register_buffer("_sidecar_cols", (sidecar_positions % in_features).long())
            self.register_buffer("_sidecar_deltas",
                                 (sidecar_values - vq_at_sidecar).contiguous())
        else:
            self.register_buffer("sidecar_positions", None)
            self.register_buffer("sidecar_values", None)
            self.register_buffer("_sidecar_vq_vals", None)
            self.register_buffer("_sidecar_rows", None)
            self.register_buffer("_sidecar_cols", None)
            self.register_buffer("_sidecar_deltas", None)

        # SVD residual factors
        self.has_svd = svd_U is not None
        if self.has_svd:
            self.register_buffer("svd_U", svd_U.contiguous())  # [out, rank]
            self.register_buffer("svd_s", svd_s.contiguous())  # [rank]
            self.register_buffer("svd_Vt", svd_Vt.contiguous())  # [rank, in]
            self.rank = svd_U.shape[1]
        else:
            self.register_buffer("svd_U", None)
            self.register_buffer("svd_s", None)
            self.register_buffer("svd_Vt", None)
            self.rank = 0

        # Bias
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.register_buffer("bias", None)

    def _apply(self, fn, recurse=True):
        """Override nn.Module._apply to refresh dispatch cache after .to()/.cuda()/.cpu()."""
        result = super()._apply(fn, recurse)
        self._refresh_dispatch_cache()
        return result

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Set compute dtype, creating FP16 codebook buffer if needed."""
        self.compute_dtype = dtype
        if dtype == torch.float16 and self.codebook is not None:
            self.register_buffer("codebook_f16", self.codebook.half().contiguous())
        else:
            self.register_buffer("codebook_f16", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output = x @ W^T + bias without persistent full-size W.

        Two paths:
        - CUDA + Triton: Fused VQ gather-matmul (W never exists in global memory)
        - CPU fallback:  Reconstruct W as temporary, then matmul

        Only the compressed buffers persist in GPU memory.
        Sets _last_dispatch_path to "fused" or "naive" for instrumentation.
        """
        # Kurtosis gate: Class 2 modules decide SVD enable/skip per forward call.
        # Amortized: kurtosis computed every check_interval calls (~0.016ms/call
        # at interval=8), cached decision reused between checks (zero cost).
        # Zero overhead when _kurtosis_gate is None (Class 1 / non-SVD modules).
        if self._kurtosis_gate is not None and self.has_svd:
            with torch.no_grad():
                enable_svd = self._kurtosis_gate.step(x)
                self._cell_skip_svd = not enable_svd

        if self._fused_available and x.is_cuda:
            self._last_dispatch_path = "fused"
            return self._forward_fused(x)
        else:
            self._last_dispatch_path = "naive"
            if x.is_cuda:
                warnings.warn(
                    f"HelixLinear({self.tensor_name}): CUDA input but Triton unavailable, "
                    f"falling back to CPU-style naive path (41x slower). "
                    f"Install triton or check CUDA availability.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._forward_naive(x)

    @staticmethod
    def _check_fused_available() -> bool:
        """One-time check if fused Triton kernel is available."""
        try:
            from helix_substrate.triton_vq_matmul import is_available
            return is_available()
        except (ImportError, RuntimeError):
            return False

    @property
    def _use_fused(self) -> bool:
        """Cached dispatch decision. Updated on device moves via _refresh_dispatch_cache()."""
        return self._fused_available

    def _refresh_dispatch_cache(self) -> None:
        """Refresh cached dispatch decision after device change."""
        self._fused_available = self._check_fused_available() and self.codebook.is_cuda

    def attach_kurtosis_gate(self, threshold: float, hysteresis_n: int = 2,
                              ema_decay: float = 0.5) -> None:
        """Attach a kurtosis gate for runtime SVD gating (Class 2 modules).

        Once attached, forward() computes input kurtosis and lets the gate
        decide whether to enable SVD correction. Only useful on modules
        with has_svd=True.

        Args:
            threshold: Kurtosis above this enables SVD. Calibrate per-module.
            hysteresis_n: Consecutive windows before switching.
            ema_decay: Kurtosis EMA smoothing (lower = more responsive).
        """
        from helix_substrate.kurtosis_gate import KurtosisGate
        self._kurtosis_gate = KurtosisGate(
            threshold=threshold,
            switch_to_svd_after=hysteresis_n,
            recover_to_skip_after=hysteresis_n,
            ema_decay=ema_decay,
        )

    def detach_kurtosis_gate(self) -> None:
        """Remove kurtosis gate, restoring default SVD behavior."""
        self._kurtosis_gate = None
        self._cell_skip_svd = False

    def dispatch_metadata(self) -> dict:
        """Return instrumentation dict for receipt embedding.

        Call after forward() to capture which path was taken.
        """
        return {
            "dispatch_path": self._last_dispatch_path,
            "device": str(self.codebook.device),
            "is_cuda": self.codebook.is_cuda,
            "triton_available": self._fused_available,
            "compute_dtype": str(self.compute_dtype),
            "kernel_metadata": getattr(self, "_last_dispatch", None),
            "cell_skip_svd": self._cell_skip_svd,
            "svd_active": self.has_svd and not self._cell_skip_svd,
            "kurtosis_gate": self._kurtosis_gate.summary() if self._kurtosis_gate else None,
        }

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused VQ gather-matmul via Triton. W never in global memory."""
        from helix_substrate.triton_vq_matmul import fused_vq_matmul

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        dispatch_log = {}
        # Cell-driven SVD gating: skip SVD correction when cell signals it's unnecessary
        skip_svd = self._cell_skip_svd and self.has_svd
        output = fused_vq_matmul(
            x=x_2d,
            codebook=self.codebook,
            indices=self.indices,
            sidecar_rows=self._sidecar_rows,
            sidecar_cols=self._sidecar_cols,
            sidecar_deltas=self._sidecar_deltas,
            svd_U=None if skip_svd else self.svd_U,
            svd_s=None if skip_svd else self.svd_s,
            svd_Vt=None if skip_svd else self.svd_Vt,
            bias=self.bias,
            codebook_f16=self.codebook_f16,
            _dispatch_log=dispatch_log,
            sidecar_phase=self._sidecar_phase,
        )
        self._last_dispatch = dispatch_log

        return output.reshape(*orig_shape[:-1], self.out_features)

    def _dequant_tile(self, start_row: int, end_row: int) -> torch.Tensor:
        """
        Dequantize a tile of weight rows from compressed representation.

        This is the single interface for bounded weight materialization.
        Both the CPU tiled forward path and decode_weight() use this.

        Args:
            start_row: First output row (inclusive)
            end_row:   Last output row (exclusive)

        Returns:
            [end_row - start_row, in_features] float32 tensor with VQ + sidecar + SVD applied.
        """
        # VQ gather for this tile
        tile = self.codebook[self.indices[start_row:end_row].long()]

        # Sidecar correction (precomputed rows/cols/deltas)
        if self._sidecar_rows is not None:
            mask = (self._sidecar_rows >= start_row) & (self._sidecar_rows < end_row)
            if mask.any():
                tile = tile.clone()
                local_rows = self._sidecar_rows[mask] - start_row
                local_cols = self._sidecar_cols[mask]
                tile[local_rows, local_cols] += self._sidecar_deltas[mask]

        # SVD residual correction (gated by cell signal)
        if self.has_svd and not self._cell_skip_svd:
            scaled_U = self.svd_U[start_row:end_row] * self.svd_s.unsqueeze(0)
            tile = tile + scaled_U @ self.svd_Vt

        return tile

    def _forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled CPU path: process 256 output rows at a time via _dequant_tile().

        Peak temporary: 256 * max_in * 4 bytes (~8.75 MB worst case)
        vs full W:      out * in * 4 bytes (~52.5 MB worst case).
        """
        CHUNK = 256
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).float()
        N = x_2d.shape[0]
        output = torch.zeros(N, self.out_features, device=x.device, dtype=torch.float32)

        # Optional FP16 compute (matmul in FP16, accumulate in FP32)
        use_fp16 = self.compute_dtype == torch.float16 and x.is_cuda
        x_compute = x_2d.half() if use_fp16 else x_2d

        for i in range(0, self.out_features, CHUNK):
            end = min(i + CHUNK, self.out_features)
            W_tile = self._dequant_tile(i, end)

            if use_fp16:
                output[:, i:end] = (x_compute @ W_tile.half().t()).float()
            else:
                output[:, i:end] = x_compute @ W_tile.t()

        # Bias
        if self.bias is not None:
            output += self.bias.unsqueeze(0)

        return output.reshape(*orig_shape[:-1], self.out_features)

    def decode_weight(self) -> torch.Tensor:
        """Reconstruct the full weight tensor (for debugging/validation).

        Uses _dequant_tile() over the full row range — same code path as forward,
        just without the tiled loop. Only call this for validation, never in forward().
        """
        with torch.no_grad():
            return self._dequant_tile(0, self.out_features)

    def memory_savings(self) -> dict:
        """Report memory usage vs equivalent nn.Linear."""
        dense_bytes = self.out_features * self.in_features * 4  # float32
        compressed = (
            self.codebook.numel() * 4  # codebook: 256 * 4 = 1024
            + self.indices.numel() * 1  # uint8 indices
        )
        if self.codebook_f16 is not None:
            compressed += self.codebook_f16.numel() * 2  # float16
        if self.sidecar_positions is not None:
            compressed += self.sidecar_positions.numel() * 8  # int64
            compressed += self.sidecar_values.numel() * 4  # float32
        if self._sidecar_rows is not None:
            compressed += self._sidecar_rows.numel() * 8   # int64
            compressed += self._sidecar_cols.numel() * 8   # int64
            compressed += self._sidecar_deltas.numel() * 4  # float32
        if self.has_svd:
            compressed += self.svd_U.numel() * 4
            compressed += self.svd_s.numel() * 4
            compressed += self.svd_Vt.numel() * 4
        return {
            "dense_bytes": dense_bytes,
            "compressed_bytes": compressed,
            "ratio": round(dense_bytes / max(1, compressed), 2),
            "savings_pct": round(100 * (1 - compressed / dense_bytes), 1),
        }

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
        ]
        if self.has_svd:
            parts.append(f"svd_rank={self.rank}")
        if self.sidecar_positions is not None:
            parts.append(f"n_outliers={self.sidecar_positions.numel()}")
        if self.bias is not None:
            parts.append("bias=True")
        savings = self.memory_savings()
        parts.append(f"compression={savings['ratio']}x")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# CDNA v3 Loader
# ---------------------------------------------------------------------------

def load_helix_linear_from_cdnav3(
    tensor_dir: Path,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> HelixLinear:
    """
    Load a single HelixLinear from a .cdnav3 directory.

    Args:
        tensor_dir: Path to the {name}.cdnav3/ directory
        bias: Optional bias tensor (from original nn.Linear)

    Returns:
        HelixLinear module ready for .cuda() or .eval()
    """
    tensor_dir = Path(tensor_dir)
    meta = json.loads((tensor_dir / "meta.json").read_text())
    rows, cols = meta["shape"]

    # Codebook: [256] float32
    codebook = torch.from_numpy(
        np.load(tensor_dir / "codebook.npy").astype(np.float32)
    )

    # Indices: [rows, cols] uint8 — stored as uint8 to save memory
    raw_indices = np.fromfile(tensor_dir / "indices.bin", dtype=np.uint8)
    indices = torch.from_numpy(raw_indices.reshape(rows, cols).copy())

    # Sidecar: optional outlier corrections
    sidecar_positions = None
    sidecar_values = None
    sidecar_path = tensor_dir / "sidecar.npz"
    if sidecar_path.exists():
        sidecar_data = np.load(sidecar_path)
        sidecar_positions = torch.from_numpy(
            sidecar_data["positions"].astype(np.int64).copy()
        )
        sidecar_values = torch.from_numpy(
            sidecar_data["values"].astype(np.float32).copy()
        )

    # SVD residual factors: optional
    svd_U = svd_s = svd_Vt = None
    if (tensor_dir / "svd_U.npy").exists():
        svd_U = torch.from_numpy(
            np.load(tensor_dir / "svd_U.npy").astype(np.float32).copy()
        )
        svd_s = torch.from_numpy(
            np.load(tensor_dir / "svd_s.npy").astype(np.float32).copy()
        )
        svd_Vt = torch.from_numpy(
            np.load(tensor_dir / "svd_Vt.npy").astype(np.float32).copy()
        )

    return HelixLinear(
        in_features=cols,
        out_features=rows,
        codebook=codebook,
        indices=indices,
        sidecar_positions=sidecar_positions,
        sidecar_values=sidecar_values,
        svd_U=svd_U,
        svd_s=svd_s,
        svd_Vt=svd_Vt,
        bias=bias,
        tensor_name=meta.get("tensor_name", ""),
        compute_dtype=compute_dtype,
    )


def load_cdna_factors(
    cdna_dir: Path,
    model: Optional[nn.Module] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[str, HelixLinear]:
    """
    Load all CDNA v3 tensors from a directory into HelixLinear modules.

    Scans for all .cdnav3/ subdirectories, loads each as HelixLinear,
    and maps them to HuggingFace-style tensor names.

    Args:
        cdna_dir: Path containing .cdnav3/ subdirectories
        model: Optional model to extract biases from original nn.Linear modules

    Returns:
        Dict mapping HF tensor names → HelixLinear modules
        e.g. {"model.layers.0.self_attn.q_proj": HelixLinear(...), ...}
    """
    cdna_dir = Path(cdna_dir)
    result: Dict[str, HelixLinear] = {}

    # Collect biases from original model if provided
    biases: Dict[str, torch.Tensor] = {}
    if model is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                biases[name] = module.bias.data.clone()

    # Scan for .cdnav3 directories
    for tensor_path in sorted(cdna_dir.glob("*.cdnav3")):
        if not tensor_path.is_dir():
            continue

        meta_path = tensor_path / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        tensor_name = meta["tensor_name"]

        # Skip non-2D (norms stored as .npy, not .cdnav3)
        if meta.get("storage_mode") == "exact":
            continue

        # Convert tensor name to module path
        # "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj"
        module_name = _tensor_name_to_module_path(tensor_name)

        bias = biases.get(module_name)
        helix_mod = load_helix_linear_from_cdnav3(tensor_path, bias=bias, compute_dtype=compute_dtype)
        result[module_name] = helix_mod

    return result


def swap_to_helix(
    model: nn.Module,
    helix_modules: Dict[str, HelixLinear],
    compute_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Replace nn.Linear modules in a model with HelixLinear equivalents.

    This is a one-shot surgery: walks model.named_modules(), replaces any
    nn.Linear whose path matches a key in helix_modules.

    Args:
        model: PyTorch model (e.g., AutoModelForCausalLM)
        helix_modules: Dict from load_cdna_factors()

    Returns:
        Modified model (same object, modules replaced in-place)

    Example:
        factors = load_cdna_factors("/path/to/cdna/", model)
        model = swap_to_helix(model, factors)
        model = model.cuda().eval()
    """
    replaced = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if name in helix_modules:
            # Only replace nn.Linear modules — skip Embedding, Conv1d, etc.
            if not isinstance(module, nn.Linear):
                skipped += 1
                continue

            new_mod = helix_modules[name]
            if compute_dtype != torch.float32:
                new_mod.set_compute_dtype(compute_dtype)

            # Verify shape compatibility
            assert new_mod.in_features == module.in_features, (
                f"{name}: in_features mismatch "
                f"({new_mod.in_features} vs {module.in_features})"
            )
            assert new_mod.out_features == module.out_features, (
                f"{name}: out_features mismatch "
                f"({new_mod.out_features} vs {module.out_features})"
            )

            # Replace module in parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_mod)
            replaced += 1

    return model


def freeze_sidecar_phase(model: nn.Module, phase: Optional[str]) -> int:
    """Freeze sidecar routing across all HelixLinear modules.

    Args:
        model: Model containing HelixLinear modules.
        phase: "fused" (decode, N<=16), "scatter" (prefill, N>16), or None (auto).

    Returns:
        Number of HelixLinear modules updated.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, HelixLinear):
            module._sidecar_phase = phase
            count += 1
    return count


def swap_summary(model: nn.Module) -> dict:
    """Report how many modules are HelixLinear vs nn.Linear."""
    helix_count = 0
    linear_count = 0
    helix_bytes = 0
    dense_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, HelixLinear):
            helix_count += 1
            savings = module.memory_savings()
            helix_bytes += savings["compressed_bytes"]
            dense_bytes += savings["dense_bytes"]
        elif isinstance(module, nn.Linear):
            linear_count += 1
            dense_bytes += module.weight.numel() * 4

    return {
        "helix_modules": helix_count,
        "linear_modules": linear_count,
        "total_linear": helix_count + linear_count,
        "compressed_bytes": helix_bytes,
        "dense_equivalent_bytes": dense_bytes,
        "overall_ratio": round(dense_bytes / max(1, helix_bytes), 2),
    }


# ---------------------------------------------------------------------------
# Tensor name mapping helpers
# ---------------------------------------------------------------------------

# HuggingFace → module path mapping
_HF_WEIGHT_SUFFIX = ".weight"

# GGUF → HF name mapping (for cross-format compatibility)
_GGUF_TO_HF = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def _tensor_name_to_module_path(tensor_name: str) -> str:
    """
    Convert a tensor name (from meta.json) to a PyTorch module path.

    Handles both HuggingFace and GGUF naming conventions:
        "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj"
        "blk.0.attn_q.weight" → "model.layers.0.self_attn.q_proj"
    """
    if tensor_name.startswith("model.language_model."):
        suffix = tensor_name[len("model.language_model.") :]
        if suffix.endswith(_HF_WEIGHT_SUFFIX):
            suffix = suffix[: -len(_HF_WEIGHT_SUFFIX)]
        return f"model.{suffix}"

    # HuggingFace format: strip .weight suffix
    if tensor_name.startswith("model.layers."):
        if tensor_name.endswith(_HF_WEIGHT_SUFFIX):
            return tensor_name[: -len(_HF_WEIGHT_SUFFIX)]
        return tensor_name

    # GGUF format: blk.N.module.weight
    if tensor_name.startswith("blk."):
        parts = tensor_name.split(".")
        # blk.N.module.weight → layer_idx=N, module=parts[2]
        layer_idx = int(parts[1])
        module_key = parts[2]
        if len(parts) > 3 and parts[3] != "weight":
            module_key = f"{parts[2]}_{parts[3]}"

        hf_module = _GGUF_TO_HF.get(module_key, module_key)
        return f"model.layers.{layer_idx}.{hf_module}"

    # Special tensors
    if tensor_name in (
        "token_embd.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ):
        return "model.embed_tokens"
    if tensor_name in ("output.weight", "lm_head.weight", "model.language_model.lm_head.weight"):
        return "lm_head"
    if tensor_name == "model.language_model.norm.weight":
        return "model.norm"

    # Fallback: strip .weight if present
    if tensor_name.endswith(_HF_WEIGHT_SUFFIX):
        return tensor_name[: -len(_HF_WEIGHT_SUFFIX)]
    return tensor_name
