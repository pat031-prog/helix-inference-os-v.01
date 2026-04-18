from __future__ import annotations

from pathlib import Path

import numpy as np

from helix_kv import rust_session
from helix_kv.layer_bridge import run_mock_airllm_loop


def test_mock_airllm_bridge_injects_one_layer_at_a_time(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    arrays = {
        "layer_0_k": np.ones((2, 2), dtype=np.float32),
        "layer_0_v": np.ones((2, 2), dtype=np.float32),
        "layer_1_k": np.ones((2, 2), dtype=np.float32),
        "layer_1_v": np.ones((2, 2), dtype=np.float32),
    }
    meta = {
        "format": "bridge-test",
        "helix_layer_slices": {
            "format": "helix-layer-slices-v0",
            "layers": [
                {
                    "layer_index": index,
                    "layer_name": f"layer.{index}",
                    "arrays": [
                        {"name": f"layer_{index}_k", "layer_index": index, "cache_kind": "key_cache"},
                        {"name": f"layer_{index}_v", "layer_index": index, "cache_kind": "value_cache"},
                    ],
                }
                for index in range(2)
            ],
        },
    }
    rust_session.save_session_bundle(session_dir, meta=meta, arrays=arrays, session_codec="rust-hlx-buffered")

    result = run_mock_airllm_loop(session_dir=session_dir, layer_indices=[0, 1])
    inject_events = [event for event in result["timeline"] if event["event"] == "inject_layer_cache"]

    assert result["all_layer_injections_hit"] is True
    assert [event["array_count"] for event in inject_events] == [2, 2]
    assert result["total_injected_arrays"] == 4
    assert result["bridge_mode"] == "mock-airllm-layer-lifecycle"
