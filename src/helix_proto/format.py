from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


META_FILE = "meta.json"
BLOCKS_DIR = "blocks"


@dataclass
class BlockRecord:
    block_id: int
    row_start: int
    row_end: int
    sha256: str
    compressed_size: int
    raw_size: int


@dataclass
class StoreMeta:
    shape: tuple[int, ...]
    dtype: str
    block_rows: int
    order: str
    blocks: list[BlockRecord]
    extra: dict[str, object] | None = None


def _store_path(path: str | Path) -> Path:
    return Path(path)


def _block_file(base: Path, block_id: int) -> Path:
    return base / BLOCKS_DIR / f"block_{block_id:06d}.bin"


def _encode_block(array: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    return contiguous.tobytes(order="C")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_tensor(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim < 1:
        raise ValueError("expected a tensor with at least 1 dimension")
    return tensor


def create_store(
    matrix: np.ndarray,
    output_dir: str | Path,
    *,
    block_rows: int = 256,
    compression_level: int = 6,
    extra: dict[str, object] | None = None,
) -> StoreMeta:
    matrix = _ensure_tensor(np.asarray(matrix))
    output_dir = _store_path(output_dir)
    blocks_dir = output_dir / BLOCKS_DIR
    blocks_dir.mkdir(parents=True, exist_ok=True)

    records: list[BlockRecord] = []
    total_blocks = math.ceil(matrix.shape[0] / block_rows)

    for block_id in range(total_blocks):
        row_start = block_id * block_rows
        row_end = min(row_start + block_rows, matrix.shape[0])
        block = np.ascontiguousarray(matrix[row_start:row_end])
        raw = _encode_block(block)
        compressed = zlib.compress(raw, level=compression_level)
        _block_file(output_dir, block_id).write_bytes(compressed)
        records.append(
            BlockRecord(
                block_id=block_id,
                row_start=row_start,
                row_end=row_end,
                sha256=_hash_bytes(raw),
                compressed_size=len(compressed),
                raw_size=len(raw),
            )
        )

    meta = StoreMeta(
        shape=tuple(int(dim) for dim in matrix.shape),
        dtype=str(matrix.dtype),
        block_rows=block_rows,
        order="C",
        blocks=records,
        extra=extra or {},
    )
    (output_dir / META_FILE).write_text(
        json.dumps(
            {
                "shape": list(meta.shape),
                "dtype": meta.dtype,
                "block_rows": meta.block_rows,
                "order": meta.order,
                "extra": meta.extra,
                "blocks": [asdict(record) for record in meta.blocks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta


def load_meta(store_dir: str | Path) -> StoreMeta:
    store_dir = _store_path(store_dir)
    payload = json.loads((store_dir / META_FILE).read_text(encoding="utf-8"))
    return StoreMeta(
        shape=tuple(payload["shape"]),
        dtype=payload["dtype"],
        block_rows=payload["block_rows"],
        order=payload["order"],
        extra=payload.get("extra", {}),
        blocks=[BlockRecord(**record) for record in payload["blocks"]],
    )


def iter_blocks(store_dir: str | Path) -> Iterable[tuple[BlockRecord, np.ndarray]]:
    store_dir = _store_path(store_dir)
    meta = load_meta(store_dir)
    dtype = np.dtype(meta.dtype)

    for record in meta.blocks:
        compressed = _block_file(store_dir, record.block_id).read_bytes()
        raw = zlib.decompress(compressed)
        digest = _hash_bytes(raw)
        if digest != record.sha256:
            raise ValueError(
                f"block {record.block_id} hash mismatch: expected {record.sha256}, got {digest}"
            )
        rows = record.row_end - record.row_start
        block_shape = (rows, *meta.shape[1:])
        block = np.frombuffer(raw, dtype=dtype).reshape(block_shape, order=meta.order)
        yield record, block


def verify_store(store_dir: str | Path) -> list[tuple[int, bool]]:
    results: list[tuple[int, bool]] = []
    for record, _ in iter_blocks(store_dir):
        results.append((record.block_id, True))
    return results


def streaming_matvec(store_dir: str | Path, vector: np.ndarray) -> np.ndarray:
    meta = load_meta(store_dir)
    if len(meta.shape) != 2:
        raise ValueError("streaming_matvec only supports 2D tensors")
    vector = np.asarray(vector, dtype=np.dtype(meta.dtype))
    if vector.ndim != 1:
        raise ValueError("expected a 1D vector")
    if vector.shape[0] != meta.shape[1]:
        raise ValueError(
            f"vector length mismatch: expected {meta.shape[1]}, got {vector.shape[0]}"
        )

    result = np.empty(meta.shape[0], dtype=np.result_type(vector.dtype, np.float32))
    for record, block in iter_blocks(store_dir):
        result[record.row_start : record.row_end] = block @ vector
    return result


def load_full_tensor(store_dir: str | Path) -> np.ndarray:
    parts = [block.copy() for _, block in iter_blocks(store_dir)]
    if not parts:
        raise ValueError("store contains no blocks")
    return np.concatenate(parts, axis=0)


def load_tensor_rows(store_dir: str | Path, indices: list[int] | tuple[int, ...]) -> np.ndarray:
    meta = load_meta(store_dir)
    if not indices:
        raise ValueError("indices must not be empty")

    normalized = [int(idx) for idx in indices]
    for idx in normalized:
        if idx < 0 or idx >= meta.shape[0]:
            raise IndexError(f"row index {idx} out of bounds for tensor with first dim {meta.shape[0]}")

    lookup = {idx: [] for idx in normalized}
    for out_pos, idx in enumerate(normalized):
        lookup[idx].append(out_pos)

    result = np.empty((len(normalized), *meta.shape[1:]), dtype=np.dtype(meta.dtype))
    remaining = set(lookup)

    for record, block in iter_blocks(store_dir):
        block_hits = [idx for idx in remaining if record.row_start <= idx < record.row_end]
        if not block_hits:
            continue
        for idx in block_hits:
            local_idx = idx - record.row_start
            for out_pos in lookup[idx]:
                result[out_pos] = block[local_idx]
            remaining.remove(idx)
        if not remaining:
            break

    if remaining:
        missing = ", ".join(str(idx) for idx in sorted(remaining))
        raise ValueError(f"could not resolve requested rows: {missing}")
    return result


def store_stats(store_dir: str | Path) -> dict[str, float]:
    meta = load_meta(store_dir)
    raw = sum(record.raw_size for record in meta.blocks)
    compressed = sum(record.compressed_size for record in meta.blocks)
    ratio = raw / compressed if compressed else float("inf")
    return {
        "rows": meta.shape[0],
        "cols": meta.shape[1] if len(meta.shape) > 1 else 1,
        "ndim": len(meta.shape),
        "blocks": len(meta.blocks),
        "raw_bytes": raw,
        "compressed_bytes": compressed,
        "compression_ratio": ratio,
    }
