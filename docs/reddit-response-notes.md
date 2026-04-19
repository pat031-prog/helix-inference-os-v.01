# Reddit response notes

Use these as short replies when someone points out valid weaknesses in the
original wording. The tone should be calm: acknowledge the correction, narrow the
claim, and point to reproducible evidence.

## r/LLMDevs style reply

You are right that OpenAI-compatible APIs often return a `model` field. The
measured claim is not that HeliX magically infers model identity. The measured
claim is narrower: HeliX preserves `requested_model`, provider-returned
`actual_model`, prompt/output digests, latency and lineage so mismatches are
auditable after the fact.

The relevant artifact is:

`verification/local-provider-substitution-ledger-20260418-154218.json`

The detector is simply:

```text
requested_model != actual_model
```

The DAG adds tamper-evident preservation and replayable audit context; it is not
the mismatch detector by itself.

## r/cybersecurity style reply

Fair criticism: the original wording was too hot. The security-relevant claim is
not provider intent. It is evidence handling.

HeliX records:

- requested model
- provider-returned model
- prompt digest
- output digest
- latency
- call id
- state/retrieval events
- artifact SHA-256

That makes model-metadata mismatches and memory-contamination failures reviewable
instead of anecdotal.

## If someone says the provider could hide the returned model

Agreed. If a provider deliberately returned a false `model` field, this test
would not identify the true model. The current claim is limited to
provider-returned metadata. Stronger claims require independent behavioral
fingerprinting or cross-provider equivalence tests.

## If someone says the memory test only shows prompt injection

That is exactly why the v4 direction includes contamination arms:

- `memory_off`
- `memory_on`
- `memory_wrong`
- `memory_poisoned`

The failed Ghost v2 run is kept as a negative finding:

`verification/local-ghost-in-the-shell-live-v2-20260418-160448.json`

Raw retrieval was contaminated and the model selected the wrong action. The
fix is not to hide it; the fix is to require receipt adjudication before making
the decision.

## If someone says the post reads like generated marketing

That is fair feedback. The repo should be judged from the minimal reproducer and
artifacts, not from the launch copy.

Minimal checks:

```powershell
python -m pytest -q tests/test_verification_hardening.py
python -m pytest -q tests/test_v4_provider_substitution_longitudinal.py
python tools/helix_claim_lint.py verification --scope batch-20260418
```

Main reviewer entry point:

`verification/README-reviewer.md`
