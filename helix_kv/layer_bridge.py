from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix_kv import rust_session


@dataclass(frozen=True)
class LayerEvent:
    event: str
    layer_index: int
    elapsed_ms: float
    array_count: int = 0
    bytes_read: int = 0
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "layer_index": self.layer_index,
            "elapsed_ms": self.elapsed_ms,
            "array_count": self.array_count,
            "bytes_read": self.bytes_read,
            "detail": self.detail,
        }


class LayerLifecycleAdapter:
    def activate_layer(self, layer_index: int) -> LayerEvent:
        raise NotImplementedError

    def run_layer(self, layer_index: int, arrays: dict[str, Any]) -> LayerEvent:
        raise NotImplementedError

    def unload_layer(self, layer_index: int) -> LayerEvent:
        raise NotImplementedError


class MockLayerLifecycleAdapter(LayerLifecycleAdapter):
    """Dependency-free AirLLM-like lifecycle used for local bridge validation."""

    def __init__(self) -> None:
        self.events: list[LayerEvent] = []

    def activate_layer(self, layer_index: int) -> LayerEvent:
        start = time.perf_counter()
        event = LayerEvent("activate_layer", int(layer_index), (time.perf_counter() - start) * 1000.0)
        self.events.append(event)
        return event

    def run_layer(self, layer_index: int, arrays: dict[str, Any]) -> LayerEvent:
        start = time.perf_counter()
        bytes_seen = int(sum(int(getattr(array, "nbytes", 0)) for array in arrays.values()))
        event = LayerEvent(
            "run_layer",
            int(layer_index),
            (time.perf_counter() - start) * 1000.0,
            array_count=len(arrays),
            bytes_read=bytes_seen,
            detail="mock compute over injected layer cache",
        )
        self.events.append(event)
        return event

    def unload_layer(self, layer_index: int) -> LayerEvent:
        start = time.perf_counter()
        event = LayerEvent("unload_layer", int(layer_index), (time.perf_counter() - start) * 1000.0)
        self.events.append(event)
        return event


class LayerCacheInjector:
    def __init__(self, session_dir: str | Path, *, verify_policy: str = "receipt-only") -> None:
        self.session_dir = Path(session_dir)
        self.verify_policy = str(verify_policy)
        self.events: list[dict[str, Any]] = []

    def inject_layer_cache(self, layer_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        start = time.perf_counter()
        _, arrays, receipt = rust_session.load_hlx_layer_slice(
            self.session_dir,
            int(layer_index),
            verify_policy=self.verify_policy,
        )
        layer_info = dict(receipt.get("layer_slice") or {})
        event = {
            "event": "inject_layer_cache",
            "layer_index": int(layer_index),
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "array_count": len(arrays),
            "bytes_read": int(layer_info.get("bytes_read") or 0),
            "status": layer_info.get("status", "unknown"),
            "selected_array_names": layer_info.get("selected_array_names", []),
        }
        self.events.append(event)
        return arrays, event


def run_mock_airllm_loop(
    *,
    session_dir: str | Path,
    layer_indices: list[int],
    lifecycle: LayerLifecycleAdapter | None = None,
    verify_policy: str = "receipt-only",
) -> dict[str, Any]:
    adapter = lifecycle or MockLayerLifecycleAdapter()
    injector = LayerCacheInjector(session_dir, verify_policy=verify_policy)
    timeline: list[dict[str, Any]] = []
    for layer_index in [int(item) for item in layer_indices]:
        timeline.append(adapter.activate_layer(layer_index).to_dict())
        arrays, inject_event = injector.inject_layer_cache(layer_index)
        timeline.append(inject_event)
        timeline.append(adapter.run_layer(layer_index, arrays).to_dict())
        timeline.append(adapter.unload_layer(layer_index).to_dict())
    injected = [event for event in timeline if event.get("event") == "inject_layer_cache"]
    return {
        "timeline": timeline,
        "layer_indices": [int(item) for item in layer_indices],
        "all_layer_injections_hit": all(event.get("status") == "hit" for event in injected),
        "total_injected_arrays": int(sum(int(event.get("array_count") or 0) for event in injected)),
        "total_bytes_read": int(sum(int(event.get("bytes_read") or 0) for event in injected)),
        "bridge_mode": "mock-airllm-layer-lifecycle",
        "dependency": "none",
    }


__all__ = [
    "LayerCacheInjector",
    "LayerEvent",
    "LayerLifecycleAdapter",
    "MockLayerLifecycleAdapter",
    "run_mock_airllm_loop",
]
