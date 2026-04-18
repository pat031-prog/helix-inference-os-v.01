from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import shutil
import statistics
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix_kv import rust_session  # noqa: E402
from helix_kv.layer_bridge import LayerCacheInjector  # noqa: E402
from helix_kv.memory_catalog import MemoryCatalog  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(pct)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
            psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = kernel.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        multiplier = 1024 if sys.platform != "darwin" else 1
        return int(usage.ru_maxrss) * multiplier
    except Exception:
        return None


def _layer_fixture(layer_count: int = 4) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {"global_guard": np.array([777], dtype=np.int64)}
    layers: list[dict[str, Any]] = []
    for layer_index in range(int(layer_count)):
        key_name = f"layer_{layer_index}_k"
        value_name = f"layer_{layer_index}_v"
        arrays[key_name] = (np.arange(16, dtype=np.float32).reshape(4, 4) + layer_index)
        arrays[value_name] = (np.arange(16, 32, dtype=np.float32).reshape(4, 4) + layer_index)
        layers.append(
            {
                "layer_index": layer_index,
                "layer_name": f"transformer.h.{layer_index}",
                "block_type": "transformer",
                "architecture": "transformer",
                "arrays": [
                    {
                        "name": key_name,
                        "layer_index": layer_index,
                        "layer_name": f"transformer.h.{layer_index}",
                        "cache_kind": "key_cache",
                    },
                    {
                        "name": value_name,
                        "layer_index": layer_index,
                        "layer_name": f"transformer.h.{layer_index}",
                        "cache_kind": "value_cache",
                    },
                ],
            }
        )
    return arrays, {"format": "helix-layer-slices-v0", "architecture": "transformer", "layers": layers}


def _read_hlx_manifest(session_dir: Path) -> tuple[dict[str, Any], int]:
    with (session_dir / "kv_cache.hlx").open("rb") as handle:
        magic = handle.read(len(rust_session.HLX_MAGIC))
        if magic != rust_session.HLX_MAGIC:
            raise ValueError("invalid .hlx magic")
        manifest_len = struct.unpack("<Q", handle.read(8))[0]
        manifest = json.loads(handle.read(int(manifest_len)).decode("utf-8"))
    return manifest, len(rust_session.HLX_MAGIC) + 8 + int(manifest_len)


def _tamper_hlx_array(session_dir: Path, array_name: str) -> dict[str, Any]:
    manifest, data_start = _read_hlx_manifest(session_dir)
    target = next((entry for entry in manifest.get("arrays", []) if entry.get("name") == array_name), None)
    if target is None:
        raise KeyError(f"array not found in .hlx manifest: {array_name}")
    offset = data_start + int(target["offset"]) + max(0, int(target["byte_length"]) // 2)
    with (session_dir / "kv_cache.hlx").open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)
        if not original:
            raise ValueError("cannot tamper empty .hlx payload")
        handle.seek(offset)
        handle.write(bytes([original[0] ^ 0xFF]))
    return {
        "array_name": array_name,
        "layer_index": int(array_name.split("_")[1]) if array_name.startswith("layer_") else None,
        "file_offset": int(offset),
        "original_byte": int(original[0]),
        "tampered_byte": int(original[0] ^ 0xFF),
    }


def run_memory_concurrency(output_dir: Path, *, workers: int = 20, writes_per_worker: int = 5) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-memory-concurrency-") as temp:
        db_path = Path(temp) / "memory.sqlite"
        init_catalog = MemoryCatalog.open(db_path)
        init_stats = init_catalog.stats()
        init_catalog.close()
        errors: list[dict[str, Any]] = []

        def worker(worker_id: int) -> dict[str, Any]:
            catalog = MemoryCatalog.open(db_path)
            try:
                for item_index in range(int(writes_per_worker)):
                    suffix = f"{worker_id:02d}-{item_index:02d}"
                    catalog.observe(
                        project="helix",
                        agent_id=f"agent-{worker_id % 5}",
                        session_id=f"session-{worker_id}",
                        observation_id=f"obs-concurrency-{suffix}",
                        content=f"Concurrency observation {suffix}: writer and reader touched SQLite WAL safely.",
                        summary=f"Concurrency observation {suffix}",
                        tags=["concurrency", "wal"],
                    )
                    catalog.remember(
                        project="helix",
                        agent_id=f"agent-{worker_id % 5}",
                        session_id=f"session-{worker_id}",
                        memory_id=f"mem-concurrency-{suffix}",
                        memory_type="working",
                        summary=f"Concurrency memory {suffix}",
                        content="SQLite WAL should keep HeliX recall writes from colliding under agent load.",
                        importance=5 + (item_index % 5),
                        tags=["concurrency", "wal"],
                    )
                    catalog.search(project="helix", agent_id=f"agent-{worker_id % 5}", query="SQLite WAL concurrency", limit=3)
                    catalog.build_context(
                        project="helix",
                        agent_id=f"agent-{worker_id % 5}",
                        query="SQLite WAL",
                        mode="search",
                        budget_tokens=120,
                    )
                return {"worker_id": worker_id, "ok": True}
            finally:
                catalog.close()

        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(worker, worker_id) for worker_id in range(int(workers))]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append({"error": str(exc)})

        final_catalog = MemoryCatalog.open(db_path)
        final_stats = final_catalog.stats()
        sample_context = final_catalog.build_context(
            project="helix",
            agent_id="agent-0",
            query="SQLite WAL concurrency",
            mode="search",
            budget_tokens=160,
        )
        final_catalog.close()

    expected = int(workers) * int(writes_per_worker)
    payload = {
        "title": "HeliX MemoryCatalog Concurrency",
        "benchmark_kind": "session-os-memory-catalog-concurrency-v1",
        "status": "completed" if not errors else "failed",
        "worker_count": int(workers),
        "writes_per_worker": int(writes_per_worker),
        "expected_observations": expected,
        "expected_memories": expected,
        "actual_observations": int(final_stats["observation_count"]),
        "actual_memories": int(final_stats["memory_count"]),
        "lost_observations": max(0, expected - int(final_stats["observation_count"])),
        "lost_memories": max(0, expected - int(final_stats["memory_count"])),
        "write_errors": len(errors),
        "errors": errors[:5],
        "journal_mode": final_stats.get("journal_mode"),
        "busy_timeout_ms": final_stats.get("busy_timeout_ms"),
        "fts_enabled": final_stats.get("fts_enabled"),
        "sample_context_tokens": sample_context["tokens"],
        "claim_boundary": "This validates one MemoryCatalog connection per worker under SQLite WAL; it is not a distributed database claim.",
        "initial_stats": init_stats,
    }
    _write_json(output_dir / "local-memory-catalog-concurrency.json", payload)
    return payload


def run_memory_decay(output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-memory-decay-") as temp:
        catalog = MemoryCatalog.open(Path(temp) / "memory.sqlite")
        critical_ids: list[str] = []
        noise_ids: list[str] = []
        for index in range(5):
            item = catalog.remember(
                project="helix",
                agent_id="release-captain",
                memory_id=f"mem-critical-{index}",
                memory_type="semantic",
                summary=f"Critical release invariant {index}",
                content="Keep pending checkpoints separate from verified cryptographic audit in every public claim.",
                importance=10,
                decay_score=1.0,
                tags=["critical", "claims", "audit"],
            )
            critical_ids.append(item.memory_id)
        for index in range(95):
            item = catalog.remember(
                project="helix",
                agent_id="release-captain",
                memory_id=f"mem-noise-{index}",
                memory_type="working",
                summary=f"Scratch note {index}",
                content=f"Low-priority scratchpad detail {index} with little relevance to release safety.",
                importance=1 + (index % 3),
                decay_score=0.05 + ((index % 5) * 0.02),
                tags=["scratch"],
            )
            noise_ids.append(item.memory_id)
        context = catalog.build_context(
            project="helix",
            agent_id="release-captain",
            mode="summary",
            budget_tokens=260,
            limit=100,
        )
        catalog.close()

    selected = set(context["memory_ids"])
    critical_retained = [memory_id for memory_id in critical_ids if memory_id in selected]
    noise_selected = [memory_id for memory_id in noise_ids if memory_id in selected]
    payload = {
        "title": "HeliX MemoryCatalog Decay Selection",
        "benchmark_kind": "session-os-memory-decay-selection-v1",
        "status": "completed",
        "memory_count": 100,
        "critical_memory_count": len(critical_ids),
        "selected_memory_count": len(context["memory_ids"]),
        "context_tokens": context["tokens"],
        "budget_tokens": 260,
        "critical_retained_count": len(critical_retained),
        "critical_retained_all": len(critical_retained) == len(critical_ids),
        "noise_selected_count": len(noise_selected),
        "selected_memory_ids": context["memory_ids"],
        "claim_boundary": "This is deterministic lexical/priority selection, not embedding-quality evaluation.",
    }
    _write_json(output_dir / "local-memory-decay-selection.json", payload)
    return payload


def run_hlx_chaos(output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-hlx-chaos-") as temp:
        session_dir = Path(temp) / "session"
        arrays, layer_meta = _layer_fixture(layer_count=4)
        rust_session.save_session_bundle(
            session_dir,
            meta={"format": "chaos-session", "helix_layer_slices": layer_meta},
            arrays=arrays,
            session_codec="rust-hlx-buffered",
        )
        before_arrays, before_event = LayerCacheInjector(session_dir, verify_policy="receipt-only").inject_layer_cache(2)
        tamper = _tamper_hlx_array(session_dir, "layer_2_k")
        receipt_only_after: dict[str, Any]
        try:
            _, receipt_only_after = LayerCacheInjector(session_dir, verify_policy="receipt-only").inject_layer_cache(2)
        except Exception as exc:
            receipt_only_after = {"status": "raised", "error": str(exc)}
        try:
            LayerCacheInjector(session_dir, verify_policy="full").inject_layer_cache(2)
            blocked = False
            full_error = None
        except Exception as exc:
            blocked = True
            full_error = str(exc)

    payload = {
        "title": "HeliX .hlx Layer Slice Chaos",
        "benchmark_kind": "session-os-hlx-layer-chaos-v1",
        "status": "completed" if blocked else "failed",
        "tamper": tamper,
        "receipt_only_before_tamper": before_event,
        "receipt_only_after_tamper": receipt_only_after,
        "pre_tamper_array_count": len(before_arrays),
        "tamper_detected": bool(blocked),
        "full_verify_blocked_injection": bool(blocked),
        "full_verify_error": full_error,
        "claim_boundary": "receipt-only is a hot-path mode and does not prove integrity; full verification is required before integrity claims.",
    }
    _write_json(output_dir / "local-hlx-layer-chaos.json", payload)
    return payload


def run_ffi_soak(output_dir: Path, *, duration_seconds: float = 60.0) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="helix-layer-soak-") as temp:
        session_dir = Path(temp) / "session"
        arrays, layer_meta = _layer_fixture(layer_count=3)
        start_time = time.perf_counter()
        deadline = start_time + max(0.0, float(duration_seconds))
        rss_start = _rss_bytes()
        rss_peak = rss_start
        save_times: list[float] = []
        load_times: list[float] = []
        errors: list[str] = []
        iteration = 0
        while iteration == 0 or time.perf_counter() < deadline:
            iteration += 1
            try:
                save_start = time.perf_counter()
                rust_session.save_session_bundle(
                    session_dir,
                    meta={"format": "soak-session", "iteration": iteration, "helix_layer_slices": layer_meta},
                    arrays=arrays,
                    session_codec="rust-hlx-buffered",
                )
                save_times.append((time.perf_counter() - save_start) * 1000.0)
                load_start = time.perf_counter()
                LayerCacheInjector(session_dir, verify_policy="receipt-only").inject_layer_cache(iteration % 3)
                load_times.append((time.perf_counter() - load_start) * 1000.0)
                current_rss = _rss_bytes()
                if current_rss is not None:
                    rss_peak = current_rss if rss_peak is None else max(int(rss_peak), int(current_rss))
            except Exception as exc:
                errors.append(str(exc))
                break
        rss_end = _rss_bytes()

    rss_delta = None if rss_start is None or rss_end is None else int(rss_end) - int(rss_start)
    rss_growth_pct = None if rss_start in {None, 0} or rss_delta is None else (float(rss_delta) / float(rss_start)) * 100.0
    payload = {
        "title": "HeliX Rust/Python Layer Slice Soak",
        "benchmark_kind": "session-os-rust-python-layer-slice-soak-v1",
        "status": "completed" if not errors else "failed",
        "duration_seconds_requested": float(duration_seconds),
        "duration_seconds_actual": time.perf_counter() - start_time,
        "iteration_count": int(iteration),
        "error_count": len(errors),
        "errors": errors[:5],
        "rss_start_bytes": rss_start,
        "rss_end_bytes": rss_end,
        "rss_peak_bytes": rss_peak,
        "rss_delta_bytes": rss_delta,
        "rss_growth_pct": rss_growth_pct,
        "save_time_ms_p50": _percentile(save_times, 0.50),
        "save_time_ms_p95": _percentile(save_times, 0.95),
        "load_time_ms_p50": _percentile(load_times, 0.50),
        "load_time_ms_p95": _percentile(load_times, 0.95),
        "mean_load_time_ms": statistics.fmean(load_times) if load_times else None,
        "claim_boundary": "This is a local process soak for obvious RSS drift; it is not a formal allocator proof.",
    }
    _write_json(output_dir / "local-rust-python-layer-slice-soak.json", payload)
    return payload


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = str(args.suite)
    random.seed(7)
    results: dict[str, Any] = {}
    if suite in {"all", "memory-concurrency"}:
        results["memory_concurrency"] = run_memory_concurrency(output_dir)
    if suite in {"all", "memory-decay"}:
        results["memory_decay"] = run_memory_decay(output_dir)
    if suite in {"all", "hlx-chaos"}:
        results["hlx_chaos"] = run_hlx_chaos(output_dir)
    if suite in {"all", "ffi-soak"}:
        results["ffi_soak"] = run_ffi_soak(output_dir, duration_seconds=float(args.duration_seconds))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local HeliX Session OS reliability gauntlet.")
    parser.add_argument("--suite", choices=["all", "memory-concurrency", "memory-decay", "hlx-chaos", "ffi-soak"], default="all")
    parser.add_argument("--output-dir", default="verification")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_suite(args)
    print(json.dumps({"status": "completed", "suites": sorted(results)}, indent=2))


if __name__ == "__main__":
    main()
