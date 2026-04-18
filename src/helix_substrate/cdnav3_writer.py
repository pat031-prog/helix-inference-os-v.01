"""
CDNA v3 tensor writer.

Writes per-tensor directories with the v3 layout:
  {safe_name}.cdnav3/
    meta.json       - shape, name, policy, codebook hash
    codebook.npy    - [N] float32 codebook
    indices.bin     - [M*K] uint8 raw indices
    sidecar.npz     - outlier positions + values (optional)
    stats.json      - encoding fidelity metrics
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from helix_substrate.tensor_policy import (
    TensorClass,
    TensorPolicy,
    classify_tensor,
    get_default_policy,
)
from helix_substrate.generate_sidecars_v3 import (
    find_outliers_percentile,
    write_sidecar_npz,
)

# Max elements to subsample for k-means (prevents OOM)
_KMEANS_MAX_SAMPLES = 500_000
# Chunk size for assignment pass
_ASSIGN_CHUNK_SIZE = 1_024 * 1_024


class CDNAv3Writer:
    """Write tensors in CDNA v3 directory format."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_tensor(
        self,
        tensor: np.ndarray,
        tensor_name: str,
        policy: Optional[TensorPolicy] = None,
        source_artifact: str = "",
    ) -> dict:
        """
        Write a tensor in v3 format.

        Args:
            tensor: Input tensor (1D or 2D, float32)
            tensor_name: Canonical tensor name
            policy: Compression policy (auto-assigned if None)
            source_artifact: Source file path for provenance

        Returns:
            Stats dict with encoding metrics
        """
        tensor = tensor.astype(np.float32)

        if policy is None:
            tc = classify_tensor(tensor_name, shape=tensor.shape)
            policy = get_default_policy(tc)

        # Morpho codec: grow from seed (handles 1D and 2D)
        if policy.storage_mode == "morpho":
            return self._write_morpho(tensor, tensor_name, policy)

        # 1D tensors: save exact as .npy
        if tensor.ndim == 1 or policy.storage_mode == "exact":
            return self._write_exact(tensor, tensor_name)

        # 2D tensors: quantize with optional sidecar
        if tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])

        return self._write_quantized(tensor, tensor_name, policy, source_artifact)

    def _write_exact(self, tensor: np.ndarray, tensor_name: str) -> dict:
        """Write a tensor exactly as .npy with a companion meta file."""
        safe = _safe_name(tensor_name)
        out_path = self.base_dir / f"{safe}.npy"
        np.save(out_path, tensor)

        # Write companion meta so manifest can recover the original name
        meta_path = self.base_dir / f"{safe}.npy.meta.json"
        meta_path.write_text(json.dumps({
            "tensor_name": tensor_name,
            "shape": list(tensor.shape),
            "storage_mode": "exact",
        }))

        return {
            "tensor_name": tensor_name,
            "storage_mode": "exact",
            "shape": list(tensor.shape),
            "original_bytes": tensor.nbytes,
            "compressed_bytes": out_path.stat().st_size,
        }

    def _write_morpho(
        self,
        tensor: np.ndarray,
        tensor_name: str,
        policy: TensorPolicy,
    ) -> dict:
        """Write a tensor using morpho codec (wave-grown from seed).

        If policy.morpho_min_cosine > 0, applies a quality gate:
        falls back to exact storage if the morpho fit doesn't meet threshold.
        """
        from helix_substrate.morpho_codec import morpho_encode

        safe = _safe_name(tensor_name)
        out_dir = self.base_dir / f"{safe}.cdnav3"

        stats = morpho_encode(
            tensor=tensor,
            tensor_name=tensor_name,
            output_dir=out_dir,
            steps=policy.morpho_growth_steps,
            target_se=policy.morpho_target_se,
            c=policy.morpho_c,
            gamma=policy.morpho_gamma,
            geometry=policy.morpho_geometry,
            fit=policy.morpho_fit,
            n_codons=policy.morpho_n_codons,
            max_iter=policy.morpho_max_iter,
            verbose=policy.morpho_verbose,
            spectral=policy.morpho_spectral,
            n_harmonics=policy.morpho_n_harmonics,
        )

        # Quality gate: fall back to exact if below threshold
        if policy.morpho_min_cosine > 0 and stats.get("cosine", 0) < policy.morpho_min_cosine:
            import shutil
            if out_dir.exists():
                shutil.rmtree(out_dir)
            exact_stats = self._write_exact(tensor, tensor_name)
            exact_stats["morpho_fallback"] = True
            exact_stats["morpho_cosine"] = stats.get("cosine", 0)
            return exact_stats

        return stats

    def _write_quantized(
        self,
        tensor: np.ndarray,
        tensor_name: str,
        policy: TensorPolicy,
        source_artifact: str,
    ) -> dict:
        """Write a quantized tensor with optional sidecar."""
        safe = _safe_name(tensor_name)
        out_dir = self.base_dir / f"{safe}.cdnav3"
        out_dir.mkdir(parents=True, exist_ok=True)

        rows, cols = tensor.shape
        flat = tensor.ravel()

        # --- Build codebook ---
        if policy.use_kmeans:
            codebook = self._build_kmeans_codebook(flat, policy.n_clusters)
        else:
            codebook = self._build_uniform_codebook(flat, policy.n_clusters)

        # --- Assign indices (chunked to avoid OOM) ---
        indices = self._chunked_assign(flat, codebook)
        indices_2d = indices.reshape(rows, cols)

        # --- Compute baseline fidelity (before sidecar) ---
        reconstructed = codebook[indices]
        cosine_no_sidecar = float(_cosine(flat, reconstructed))
        max_abs_diff = float(np.max(np.abs(flat - reconstructed)))

        # --- Sidecar ---
        num_outliers = 0
        sidecar_bytes = 0
        cosine_with_sidecar = cosine_no_sidecar

        if policy.sidecar_enabled:
            positions, values = find_outliers_percentile(tensor, policy.percentile)
            if len(positions) > 0:
                if policy.max_corrections > 0 and len(positions) > policy.max_corrections:
                    # Keep only top-N by absolute value
                    top_idx = np.argsort(np.abs(values))[::-1][:policy.max_corrections]
                    positions = np.sort(positions[top_idx])
                    values = tensor.ravel()[positions]

                sidecar_path = out_dir / "sidecar.npz"
                receipt = write_sidecar_npz(positions, values, sidecar_path)
                num_outliers = receipt["num_corrections"]
                sidecar_bytes = receipt["size_bytes"]

                # Compute fidelity with sidecar
                patched = reconstructed.copy()
                patched[positions] = values
                cosine_with_sidecar = float(_cosine(flat, patched))

        # --- Write codebook ---
        codebook_path = out_dir / "codebook.npy"
        np.save(codebook_path, codebook)
        codebook_sha256 = hashlib.sha256(codebook.tobytes()).hexdigest()

        # --- Write indices ---
        indices_path = out_dir / "indices.bin"
        indices.astype(np.uint8).tofile(indices_path)

        # --- SVD residual correction ---
        svd_bytes = 0
        cosine_with_svd = cosine_with_sidecar
        svd_rank_actual = 0

        if policy.svd_residual_rank > 0:
            # Build VQ+sidecar reconstruction as 2D
            if num_outliers > 0:
                # patched is the flat VQ+sidecar array (already computed above)
                recon_2d = patched.reshape(rows, cols)
            else:
                recon_2d = reconstructed.reshape(rows, cols)

            residual = tensor - recon_2d
            svd_rank_actual = min(policy.svd_residual_rank, min(rows, cols))
            U, s, Vt = np.linalg.svd(residual, full_matrices=False)
            U_r = U[:, :svd_rank_actual].astype(np.float32)
            s_r = s[:svd_rank_actual].astype(np.float32)
            Vt_r = Vt[:svd_rank_actual, :].astype(np.float32)

            np.save(out_dir / "svd_U.npy", U_r)
            np.save(out_dir / "svd_s.npy", s_r)
            np.save(out_dir / "svd_Vt.npy", Vt_r)

            svd_bytes = sum(
                (out_dir / f).stat().st_size
                for f in ("svd_U.npy", "svd_s.npy", "svd_Vt.npy")
            )

            corrected = recon_2d + (U_r * s_r[None, :]) @ Vt_r
            cosine_with_svd = float(_cosine(flat, corrected.ravel()))

        # --- Write meta.json ---
        meta = {
            "format_version": "cdna_v3",
            "tensor_name": tensor_name,
            "shape": [rows, cols],
            "dtype": "float32",
            "storage_mode": policy.storage_mode,
            "tensor_class": policy.tensor_class.value,
            "n_clusters": policy.n_clusters,
            "percentile": policy.percentile,
            "use_kmeans": policy.use_kmeans,
            "sidecar_enabled": policy.sidecar_enabled,
            "block_rows": policy.block_rows,
            "codebook_sha256": codebook_sha256,
            "svd_residual_rank": svd_rank_actual,
            "source_artifact": source_artifact,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # --- Write stats.json ---
        indices_bytes = indices_path.stat().st_size
        codebook_bytes = codebook_path.stat().st_size
        compressed_bytes = indices_bytes + codebook_bytes + sidecar_bytes + svd_bytes

        stats = {
            "tensor_name": tensor_name,
            "shape": [rows, cols],
            "original_bytes": tensor.nbytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(tensor.nbytes / max(1, compressed_bytes), 2),
            "cosine_no_sidecar": round(cosine_no_sidecar, 6),
            "cosine_with_sidecar": round(cosine_with_sidecar, 6),
            "cosine_with_svd": round(cosine_with_svd, 6),
            "max_abs_diff": round(max_abs_diff, 6),
            "num_outliers": num_outliers,
            "sidecar_bytes": sidecar_bytes,
            "svd_residual_rank": svd_rank_actual,
            "svd_bytes": svd_bytes,
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        return stats

    def _build_kmeans_codebook(
        self, flat: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Build codebook using k-means on subsampled data."""
        from helix_substrate.cdna_encoder import _simple_kmeans

        # Subsample to avoid OOM in k-means
        if len(flat) > _KMEANS_MAX_SAMPLES:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(flat), _KMEANS_MAX_SAMPLES, replace=False)
            sample = flat[sample_idx]
        else:
            sample = flat

        codebook, _ = _simple_kmeans(sample, n_clusters, max_iters=10)
        return codebook.astype(np.float32)

    def _build_uniform_codebook(
        self, flat: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Build uniform linear codebook."""
        vmin, vmax = float(flat.min()), float(flat.max())
        return np.linspace(vmin, vmax, n_clusters).astype(np.float32)

    def _chunked_assign(
        self, flat: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        """Assign codebook indices in chunks to avoid OOM."""
        indices = np.empty(len(flat), dtype=np.uint8)
        for start in range(0, len(flat), _ASSIGN_CHUNK_SIZE):
            end = min(start + _ASSIGN_CHUNK_SIZE, len(flat))
            chunk = flat[start:end]
            dists = np.abs(chunk[:, np.newaxis] - codebook)
            indices[start:end] = np.argmin(dists, axis=1).astype(np.uint8)
        return indices


def _safe_name(tensor_name: str) -> str:
    """Convert tensor name to filesystem-safe string."""
    return tensor_name.replace("/", "_").replace(".", "_")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat arrays."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
