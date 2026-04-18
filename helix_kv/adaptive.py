from __future__ import annotations

from helix_kv.config import KVConfig


def build_adaptive_config(
    *,
    calibration_tokens: int = 128,
    hot_window: int = 0,
    topk: int = 0,
    index_refresh_interval: int = 8,
    block_size: int = 0,
    layer_share_stride: int = 0,
    high_kurtosis: float = 10.0,
    medium_kurtosis: float = 3.0,
) -> KVConfig:
    return KVConfig(
        mode="adaptive",
        hot_window=hot_window,
        topk=topk,
        index_refresh_interval=index_refresh_interval,
        block_size=block_size,
        layer_share_stride=layer_share_stride,
        calibration_tokens=calibration_tokens,
        adaptive_high_kurtosis=high_kurtosis,
        adaptive_medium_kurtosis=medium_kurtosis,
    )
