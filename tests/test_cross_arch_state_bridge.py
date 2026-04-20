from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import run_cross_arch_state_bridge_v1 as runner


pytestmark = pytest.mark.skipif(
    os.environ.get("HELIX_ENABLE_CROSS_ARCH_STATE_BRIDGE") != "1",
    reason=(
        "Cross-arch state bridge runner requires GPT-2 and Zamba2 weights cached "
        "locally and is multi-minute. Set HELIX_ENABLE_CROSS_ARCH_STATE_BRIDGE=1 to run."
    ),
)


def test_cross_arch_state_bridge_runs_end_to_end(tmp_path: Path) -> None:
    if not runner._ref_cached(runner.GPT2_REF):
        pytest.skip(f"GPT-2 weights not cached: {runner.GPT2_REF}")
    if not runner._ref_cached(runner.ZAMBA_REF):
        pytest.skip(f"Zamba2 weights not cached: {runner.ZAMBA_REF}")

    args = runner.build_parser().parse_args([
        "--output-dir", str(tmp_path),
        "--tokens-per-round", "8",
        "--compression-tokens", "1",
        "--no-measure-compression",
    ])
    artifact = runner.run_cross_arch_state_bridge(args)

    assert artifact["artifact"] == "local-cross-arch-state-bridge-v1"
    assert artifact["status"] == "completed"
    assert artifact["cross_arch_bridge_kind"] == "tokens+signed_hmem"
    assert artifact["per_arch_bit_identity_ok"] is True

    rounds = artifact["rounds"]
    assert [r["round"] for r in rounds] == [1, 2, 3, 4, 5]
    assert rounds[0]["model"] == "gpt2"
    assert rounds[1]["action"] == "signed_hmem_bridge"
    assert rounds[2]["model"] == "zamba"
    assert rounds[3]["action"] == "restore_from_r1+continue"
    assert rounds[3]["r1_hlx_restore_check"]["bit_identity_post_restore"] is True
    assert rounds[4]["action"] == "regression_probe"
    assert rounds[4]["r1_prompt_replay"] is True

    for r in (rounds[0], rounds[2]):
        assert r["hlx"]["bit_identity"] is True

    memory_chain = artifact["signed_memory_chain"]
    assert len(memory_chain) >= 4
    for entry in memory_chain:
        assert entry.get("memory_id")
        assert entry.get("receipt_signing_mode") == "ephemeral_preregistered"
