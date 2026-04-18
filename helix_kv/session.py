from __future__ import annotations

from pathlib import Path

from helix_kv.cache import CompressedKVCache
from helix_kv.config import KVConfig


def save_cache(cache: CompressedKVCache, path: str | Path) -> Path:
    return cache.save(path)


def load_cache(
    path: str | Path,
    *,
    export_dir: str | Path | None = None,
    config: KVConfig | None = None,
    cache_mode: str = "session",
    kv_quant_seed: int = 7,
) -> CompressedKVCache:
    return CompressedKVCache.load(
        path,
        export_dir=export_dir,
        config=config,
        cache_mode=cache_mode,
        kv_quant_seed=kv_quant_seed,
    )
