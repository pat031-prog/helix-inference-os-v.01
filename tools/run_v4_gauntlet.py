from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


TEST_MAP = {
    "memory-contamination-triad": "tests/test_v4_memory_contamination_triad.py",
    "lineage-forgery-gauntlet": "tests/test_v4_lineage_forgery_gauntlet.py",
    "provider-substitution-longitudinal": "tests/test_v4_provider_substitution_longitudinal.py",
    "zero-day-osint-backtest": "tests/test_v4_zero_day_osint_backtest.py",
    "signed-receipt-provenance": "tests/test_v4_signed_receipts.py",
    "indirect-prompt-injection-memory": "tests/test_v4_indirect_prompt_injection_memory.py",
    "byzantine-shared-hmem": "tests/test_v4_byzantine_shared_hmem.py",
    "hybrid-rerank-ab": "tests/test_v4_hybrid_rerank_ab.py",
    "cross-provider-behavioral-triangulation": "tests/test_v4_cross_provider_triangulation.py",
    "stack-upgrade-integration": "tests/test_stack_upgrade_integration.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HeliX v4 falsable gauntlets.")
    parser.add_argument("--test-id", default="all", choices=["all", *TEST_MAP])
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    targets = list(TEST_MAP.values()) if args.test_id == "all" else [TEST_MAP[args.test_id]]
    pytest_args = args.pytest_args or ["-q"]
    if len(pytest_args) == 1 and " " in pytest_args[0]:
        pytest_args = shlex.split(pytest_args[0])
    cmd = [sys.executable, "-m", "pytest", *pytest_args, *targets]
    print("[helix] Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
