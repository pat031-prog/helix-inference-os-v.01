"""
helix_cdc/regrow/cdna_encoder.py
================================

Simple CDNA encoder for kernel syscall usage.

Encodes a tensor to CDNA format using k-means quantization.

CDNA Format:
  - Magic: b"HXZC" (4 bytes)
  - Version: 1 (4 bytes, uint32 LE)
  - Rows: M (4 bytes, uint32 LE)
  - Cols: K (4 bytes, uint32 LE)
  - Codebook: [256] float32 (1024 bytes)
  - Indices: [M, K] uint8 (M*K bytes, zlib compressed)
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

# CDNA format constants
CDNA_MAGIC = b"HXZC"
CDNA_VERSION = 1
CODEBOOK_SIZE = 256


def encode_tensor_to_cdna(
    tensor: np.ndarray,
    output_path: Path,
    tensor_name: str = "tensor",
    n_clusters: int = 256,
    max_iters: int = 10,
) -> dict:
    """
    Encode a tensor to CDNA format using k-means quantization.

    Args:
        tensor: Input tensor (2D float32)
        output_path: Path to write CDNA shard
        tensor_name: Name for logging
        n_clusters: Number of codebook entries (default: 256)
        max_iters: Max k-means iterations (default: 10)

    Returns:
        dict with encoding stats
    """
    # Ensure 2D
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    # Ensure float32
    tensor = tensor.astype(np.float32)
    rows, cols = tensor.shape

    # Flatten for k-means
    flat = tensor.flatten()

    # Simple k-means (sklearn-free)
    codebook, indices_flat = _simple_kmeans(flat, n_clusters, max_iters)

    # Reshape indices back to 2D
    indices = indices_flat.reshape(rows, cols).astype(np.uint8)

    # Compress indices
    indices_bytes = indices.tobytes()
    indices_compressed = zlib.compress(indices_bytes, level=6)

    # Write CDNA file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Header
        f.write(CDNA_MAGIC)
        f.write(struct.pack("<I", CDNA_VERSION))
        f.write(struct.pack("<I", rows))
        f.write(struct.pack("<I", cols))

        # Codebook (always 256 entries, padded if needed)
        codebook_full = np.zeros(256, dtype=np.float32)
        codebook_full[:len(codebook)] = codebook
        f.write(codebook_full.tobytes())

        # Compressed indices
        f.write(struct.pack("<I", len(indices_bytes)))  # Uncompressed size
        f.write(struct.pack("<I", len(indices_compressed)))  # Compressed size
        f.write(indices_compressed)

    return {
        "rows": rows,
        "cols": cols,
        "codebook_size": len(codebook),
        "indices_bytes": len(indices_bytes),
        "compressed_bytes": len(indices_compressed),
        "compression_ratio": len(indices_bytes) / max(1, len(indices_compressed)),
    }


def _simple_kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means without sklearn dependency.

    Args:
        data: 1D array of values
        n_clusters: Number of clusters
        max_iters: Maximum iterations

    Returns:
        (centroids, assignments)
    """
    n_clusters = min(n_clusters, len(np.unique(data)))

    # Initialize centroids using percentiles
    percentiles = np.linspace(0, 100, n_clusters)
    centroids = np.percentile(data, percentiles).astype(np.float32)

    # K-means iterations
    for _ in range(max_iters):
        # Assign points to nearest centroid
        dists = np.abs(data[:, np.newaxis] - centroids)
        assignments = np.argmin(dists, axis=1).astype(np.uint8)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            mask = assignments == i
            if np.any(mask):
                new_centroids[i] = np.mean(data[mask])
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, assignments


def decode_cdna_to_tensor(cdna_path: Path) -> np.ndarray:
    """
    Decode a CDNA shard back to tensor.

    Args:
        cdna_path: Path to CDNA shard

    Returns:
        Decoded tensor (2D float32)
    """
    with open(cdna_path, "rb") as f:
        # Read header
        magic = f.read(4)
        if magic != CDNA_MAGIC:
            raise ValueError(f"Invalid CDNA magic: {magic}")

        version = struct.unpack("<I", f.read(4))[0]
        if version != CDNA_VERSION:
            raise ValueError(f"Unsupported CDNA version: {version}")

        rows = struct.unpack("<I", f.read(4))[0]
        cols = struct.unpack("<I", f.read(4))[0]

        # Read codebook
        codebook = np.frombuffer(f.read(256 * 4), dtype=np.float32)

        # Read compressed indices
        uncompressed_size = struct.unpack("<I", f.read(4))[0]
        compressed_size = struct.unpack("<I", f.read(4))[0]
        indices_compressed = f.read(compressed_size)

        # Decompress
        indices_bytes = zlib.decompress(indices_compressed)
        indices = np.frombuffer(indices_bytes, dtype=np.uint8).reshape(rows, cols)

        # Dequantize
        tensor = codebook[indices]

    return tensor
