"""
CDNA v3 sidecar generation.

Provides outlier detection and sidecar writing for the v3 tensor format.
Supports two detection modes:
  - Percentile-based (always available, no activations needed)
  - Contribution-error-aware (requires calibration activations)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def find_outliers_percentile(
    tensor: np.ndarray,
    percentile: float = 99.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find outlier positions using percentile thresholding.

    Values beyond the given percentile (both tails) are flagged as outliers.

    Args:
        tensor: 2D tensor (float32)
        percentile: Percentile threshold (e.g., 99.9 means top/bottom 0.1%)

    Returns:
        (positions, values) where positions are flat indices into the tensor
    """
    flat = tensor.astype(np.float32).ravel()
    lo = np.percentile(flat, 100.0 - percentile)
    hi = np.percentile(flat, percentile)
    mask = (flat < lo) | (flat > hi)
    positions = np.where(mask)[0].astype(np.int64)
    values = flat[positions].astype(np.float32)
    return positions, values


def find_outliers_contribution(
    original: np.ndarray,
    quantized: np.ndarray,
    activations: np.ndarray,
    top_k: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find outliers using contribution error: max_activation[col] * |weight_error[row, col]|.

    This selects outliers that actually matter for inference, not just statistically
    extreme values.

    Args:
        original: Original 2D weight tensor
        quantized: Quantized 2D weight tensor (same shape)
        activations: Calibration activations [batch, input_dim] -- max over batch used
        top_k: Number of outliers to select (default: 0.1% of elements)

    Returns:
        (positions, values) sorted by position
    """
    original = original.astype(np.float32)
    quantized = quantized.astype(np.float32)

    weight_error = np.abs(original - quantized)

    # max activation per column (input dimension)
    max_act = np.max(np.abs(activations), axis=0).astype(np.float32)

    # Broadcast: contribution_error[row, col] = max_act[col] * weight_error[row, col]
    contribution_error = weight_error * max_act[np.newaxis, :]

    if top_k is None:
        top_k = max(1, int(original.size * 0.001))  # 0.1% default

    flat_ce = contribution_error.ravel()
    # Partial sort for efficiency
    top_indices = np.argpartition(flat_ce, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(flat_ce[top_indices])[::-1]]

    # Convert to sorted positions
    positions = np.sort(top_indices).astype(np.int64)
    values = original.ravel()[positions].astype(np.float32)
    return positions, values


def write_sidecar_npz(
    positions: np.ndarray,
    values: np.ndarray,
    output_path: Path,
) -> dict:
    """
    Write sidecar in npz format.

    Args:
        positions: Flat indices (int64)
        values: Original values at those positions (float32)
        output_path: Path to write .npz file

    Returns:
        Receipt dict with stats
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        positions=positions.astype(np.int64),
        values=values.astype(np.float16),
    )

    return {
        "format": "npz",
        "num_corrections": len(positions),
        "size_bytes": output_path.stat().st_size,
    }


def read_sidecar_v3(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a v3 sidecar file. Auto-detects format (.npz or .hxzo).

    Returns:
        (positions, values) -- positions as int64, values as float32
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path)
        positions = data["positions"].astype(np.int64)
        values = data["values"].astype(np.float32)
        return positions, values

    if path.suffix == ".hxzo":
        from helix_substrate.sidecar import read_outlier_sidecar
        positions, values, _meta = read_outlier_sidecar(str(path))
        return positions.astype(np.int64), values.astype(np.float32)

    raise ValueError(f"Unknown sidecar format: {path.suffix}")
