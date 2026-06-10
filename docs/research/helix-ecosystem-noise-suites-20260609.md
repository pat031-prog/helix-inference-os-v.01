# HeliX Ecosystem Noise Suites: Research And Experiment Plan

Date: 2026-06-09

## Thesis

The strongest public angle for HeliX is not "LLMs can judge LLMs" or "agents are smart." The stronger claim is narrower and sharper:

> HeliX can make cloud-model behavior locally attestable, while separating cryptographic provenance from semantic authority.

That lets us publish evidence on things the ecosystem tends to blur:

- a provider can serve a different `actual_model` than the requested model;
- a model call can be transport-valid but operationally unusable;
- an LLM auditor can overclaim cryptographic verification it did not perform;
- signed evidence can prove provenance without proving semantic truth.

## Research Signals

- RFC 8785 JCS motivates invariant JSON representations for hashing/signing, duplicate-key rejection, stable sorting, and canonical bytes. HeliX already uses a no-float JCS-compatible profile, so response-contract suites should avoid treating arbitrary generated JSON as trusted until parsed and normalized. Source: https://www.rfc-editor.org/rfc/rfc8785
- RFC 9162 Certificate Transparency gives the reference shape for append-only tree heads, inclusion proofs, and consistency proofs. HeliX should keep using local CT-style proofs while clearly avoiding global non-equivocation claims without witnesses. Source: https://www.rfc-editor.org/rfc/rfc9162
- Sigstore Rekor is a useful external anchor direction, but the current step should remain local: produce verifier bundles first, then anchor later. Source: https://docs.sigstore.dev/logging/overview/
- Structured-output reliability is now an ecosystem-wide product concern. OpenAI's Structured Outputs announcement explicitly distinguishes JSON mode from schema adherence and says prompting alone was not enough for reliable schema matching. Source: https://openai.com/index/introducing-structured-outputs-in-the-api/
- LLM-as-a-judge research warns that single-shot LLM judgments can be unreliable and biased, even when models are strong. Sources: https://arxiv.org/abs/2412.12509 and https://arxiv.org/abs/2410.02736

## Suite 1: Cloud Response Contract Stress

File:

- `tools/run_cloud_response_contract_stress_v1.py`
- `tools/run_cloud_response_contract_stress_secure.ps1`

Purpose:

Separate `transport_ok` from `response_contract_ok`.

This suite asks cloud models for strict JSON under multiple contracts:

- `minimal_json`: exact flat JSON, no markdown, five bounded string fields.
- `nested_claims`: exact nested JSON with two bounded claim objects and a decision object.
- `adversarial_boundary`: exact JSON that must explicitly reject semantic authority from valid signatures.

It records:

- requested/actual model;
- provider mismatch;
- text digest;
- exact JSON parse status;
- fenced markdown;
- duplicate keys;
- missing/extra fields;
- type errors;
- semantic-boundary violations;
- MemoryCatalog receipt and hash profile;
- transparency inclusion/consistency proofs;
- standalone verifier bundle.

Why it can make noise:

The last DeepInfra longitudinal run already showed Qwen returning `to be a malformed response` while transport status was OK. This suite turns that anecdote into a repeatable public metric: "served successfully" is not the same as "agent-consumable."

Public claim allowed:

> HeliX measures and attests structured-output contract reliability across cloud model calls.

Public claim not allowed:

> HeliX proves model answers are true.

## Suite 2: LLM Verifier Overclaim Gauntlet

File:

- `tools/run_llm_verifier_overclaim_gauntlet_v1.py`
- `tools/run_llm_verifier_overclaim_gauntlet_secure.ps1`

Purpose:

Measure whether LLM auditors claim cryptographic authority they do not have.

The suite generates:

- a valid local verifier bundle;
- `event_tamper`: event payload changed after proof generation;
- `consistency_tamper`: consistency proof changed;
- `sth_tamper`: STH root changed without resigning;
- `claim_boundary_overclaim`: forbidden claim boundary replacing the safe one.

Then it runs auditor models in two modes:

- `blind`: the model sees bundle metadata but no deterministic verifier report. Correct behavior is `cannot_verify_from_prompt`.
- `reported`: the model sees the deterministic verifier result. Correct behavior is to follow it while keeping `semantic_authority=false`.

It records:

- blind overclaims;
- unsupported specificity in blind mode;
- reported verifier disagreements;
- semantic overclaims;
- parse/contract failures;
- signed memory receipts;
- transparency proof and standalone verifier bundle.

Why it can make noise:

The ecosystem increasingly uses LLMs as judges, reviewers, and evaluators. This suite asks a sharper question: when the task is cryptographic verification, does the judge know it is not the verifier?

Public claim allowed:

> HeliX can quantify LLM auditor overclaim against deterministic verifier ground truth.

Public claim not allowed:

> HeliX makes LLM auditors cryptographic authorities.

## Recommended Cloud Runs

Contract stress:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_cloud_response_contract_stress_secure.ps1 -Models "Qwen/Qwen3-235B-A22B-Instruct-2507,anthropic/claude-sonnet-4-6,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Contracts "minimal_json,nested_claims,adversarial_boundary" -Rounds 1 -Tokens 420 -Temperature 0.0 -Timeout 240
```

Verifier overclaim:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_llm_verifier_overclaim_gauntlet_secure.ps1 -AuditorModels "anthropic/claude-sonnet-4-6,Qwen/Qwen3-235B-A22B-Instruct-2507,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct" -Variants "valid_control,event_tamper,claim_boundary_overclaim,consistency_tamper" -Modes "blind,reported" -Tokens 360 -Temperature 0.0 -Timeout 240
```

## Next Product Move

After one or two real cloud runs, create a single `HeliX Trust Failure Card` format with:

- model requested/actual;
- transport status;
- response contract status;
- verifier ground truth;
- LLM auditor verdict;
- overclaim flags;
- inclusion proof;
- consistency proof;
- standalone bundle hash;
- claim boundary.

That is the clean bridge to a public demo without building the full UI yet.
