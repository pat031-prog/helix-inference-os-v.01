# Reproducing HeliX Public Claims

This file is the practical entrypoint for reproducing the current public claims and checking the current repo state.

## No API Required: Local Mechanical Claims

These are the fastest local commands for the strongest mechanical claims:

```powershell
python -m pytest -q tests/test_v4_lineage_forgery_gauntlet.py
python -m pytest -q tests/test_v4_signed_receipts.py tests/test_memory_catalog.py
python -m pytest -q tests/test_infinite_depth_memory_suite.py
python -m pytest -q tests/test_multi_agent_concurrency_suite.py
```

These should complete locally without external provider keys.

For the bounded multi-agent methodology lane, you can also regenerate the raw artifact directly:

```powershell
python tools\run_multi_agent_concurrency_suite_v1.py --output-dir verification\nuclear-methodology\multi-agent-concurrency --run-id public-multi-agent-concurrency-20260423 --case all
```

That lane is deterministic by default and should be cited as a local stale-parent race / quarantine methodology artifact, not as distributed consensus or a production multi-writer benchmark.

## Public Claim Batch

Current public-claim batch used for this docs pass:

```powershell
python -m pytest -q `
  tests/test_memory_catalog.py `
  tests/test_v4_signed_receipts.py `
  tests/test_v4_lineage_forgery_gauntlet.py `
  tests/test_v4_memory_contamination_triad.py `
  tests/test_provider_integrity_observatory.py `
  tests/test_v4_provider_substitution_longitudinal.py `
  tests/test_infinite_depth_memory_suite.py `
  tests/test_active_memory_ab_trial.py `
  tests/test_cloud_amnesia_derby.py `
  tests/test_ghost_in_the_shell_live.py `
  tests/test_ghost_in_the_shell_live_v2.py `
  tests/test_v4_cross_provider_triangulation.py
```

Observed on 2026-04-23 in this repo state:

```text
36 passed, 2 failed, 2 skipped in 6.28s
```

Known failures in that batch:

- `tests/test_active_memory_ab_trial.py`
- `tests/test_cloud_amnesia_derby.py`

Known skips in that batch:

- `tests/test_ghost_in_the_shell_live.py`
- `tests/test_ghost_in_the_shell_live_v2.py`

## Token-Required / Live Suites

These lanes may require provider credentials, optional local model installs, or extra runtime setup:

- `tests/test_ghost_in_the_shell_live.py`
- `tests/test_ghost_in_the_shell_live_v2.py`
- `/cert nuclear-methodology`
- `/cert post-nuclear-methodology`
- `/cert long-horizon-checkpoints`

Provider costs vary by model and token pricing. Record prompt/output token counts and observed cost notes in the resulting artifact instead of hardcoding stable dollar claims in docs.

## Public Evidence Lint

Validate the curated public layer:

```powershell
python tools\helix_claim_lint.py --public-evidence evidence\index.json --public-docs .
```

This should fail if:

- a curated evidence entry points to a missing file
- a recorded SHA-256 no longer matches the cited source artifact
- a public claim entry is missing key boundary fields
- the canonical public docs stop existing or stop linking the public layer

`verification/` remains a raw archive and historical lab tree, so a whole-tree lint there is intentionally noisier and should not be treated as the canonical public acceptance gate for this docs-first pass.
