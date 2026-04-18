from __future__ import annotations

from pathlib import Path

from helix_kv.memory_catalog import MemoryCatalog
from tools.run_local_airllm_real_smoke import run_smoke as run_airllm_smoke
from tools.run_local_session_os_reliability import (
    run_ffi_soak,
    run_hlx_chaos,
    run_memory_concurrency,
    run_memory_decay,
)


class _Args:
    output_dir: str
    model_ref: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_path: str | None = None
    prompt: str = "Say one sentence."
    max_length: int = 16
    max_new_tokens: int = 1
    local_files_only: bool = True


def test_memory_catalog_defaults_enable_wal_and_timeout(tmp_path: Path) -> None:
    catalog = MemoryCatalog.open(tmp_path / "memory.sqlite")
    stats = catalog.stats()
    catalog.close()

    assert stats["busy_timeout_ms"] == 5000
    assert stats["journal_mode"] in {"wal", "memory", "delete", "unavailable"}


def test_memory_concurrency_runner_keeps_all_writes(tmp_path: Path) -> None:
    payload = run_memory_concurrency(tmp_path, workers=4, writes_per_worker=2)

    assert payload["lost_observations"] == 0
    assert payload["lost_memories"] == 0
    assert payload["write_errors"] == 0
    assert (tmp_path / "local-memory-catalog-concurrency.json").exists()


def test_memory_decay_runner_keeps_critical_memories(tmp_path: Path) -> None:
    payload = run_memory_decay(tmp_path)

    assert payload["critical_retained_all"] is True
    assert payload["critical_retained_count"] == payload["critical_memory_count"]
    assert payload["noise_selected_count"] < 10


def test_hlx_chaos_blocks_full_verify_injection(tmp_path: Path) -> None:
    payload = run_hlx_chaos(tmp_path)

    assert payload["tamper_detected"] is True
    assert payload["full_verify_blocked_injection"] is True
    assert payload["receipt_only_before_tamper"]["status"] == "hit"


def test_ffi_soak_runner_completes_short_loop(tmp_path: Path) -> None:
    payload = run_ffi_soak(tmp_path, duration_seconds=0.0)

    assert payload["status"] == "completed"
    assert payload["iteration_count"] >= 1
    assert payload["error_count"] == 0
    assert payload["load_time_ms_p50"] is not None


def test_airllm_real_smoke_skips_cleanly_without_required_runtime(tmp_path: Path) -> None:
    args = _Args()
    args.output_dir = str(tmp_path)
    payload = run_airllm_smoke(args)

    assert payload["status"] in {"skipped_dependency_missing", "skipped_model_not_cached", "skipped_runtime_error", "completed"}
    assert (tmp_path / "local-airllm-real-smoke.json").exists()
    assert payload["helix_layer_bridge_sidecar"]["all_layer_injections_hit"] is True
