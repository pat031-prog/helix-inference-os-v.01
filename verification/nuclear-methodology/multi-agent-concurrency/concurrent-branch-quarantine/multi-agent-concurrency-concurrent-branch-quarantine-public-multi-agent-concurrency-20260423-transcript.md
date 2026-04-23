# Transcript: concurrent-branch-quarantine

- Run ID: `public-multi-agent-concurrency-20260423`
- Judge requested: `Qwen/Qwen3.5-122B-A10B`
- Judge actual: `Qwen/Qwen3.5-122B-A10B`
- Auditor requested: `mistralai/Devstral-Small-2507`
- Auditor actual: `mistralai/Devstral-Small-2507`

## Expected / Ground Truth

```json
{
  "quarantined_memory_id": "public-multi-agent-concurrency-20260423-beta",
  "canonical_memory_id": "public-multi-agent-concurrency-20260423-alpha"
}
```

## Visible Contract

```json
{
  "null_hypothesis": "Two concurrent agents writing from one observed parent either corrupt state or both enter default context.",
  "alternative_hypothesis": "One branch advances the canonical head and the stale branch is preserved but quarantined.",
  "falseability_condition": "Fail if canonical head is missing, if equivocation_count != 1, or if quarantined memory appears in default context.",
  "kill_switch": "Abort if a token/header/full raw prompt is persisted."
}
```

## Judge Output

```json
{
  "agent_id": "alpha-qwen-collector",
  "role": "collector",
  "finding": "PAMPA-SWARM-17 external news monitor saw rail-port delay and fuel-route chatter.",
  "recommended_action": "preserve-open-source-signal",
  "confidence": 0.74
}
```

## Auditor Output

```json
{
  "agent_id": "beta-sonnet-security",
  "role": "security-auditor",
  "finding": "PAMPA-SWARM-17 server logs show repeated 429 bursts and one admin-console probe.",
  "recommended_action": "preserve-security-signal",
  "confidence": 0.81
}
```

## Literal Model Transcript

### alpha

- Requested model: `Qwen/Qwen3.5-122B-A10B`
- Actual model: `Qwen/Qwen3.5-122B-A10B`
- Status: `ok`
- Text SHA256: `37992f62c7d27aa13b7fe93eee207b025a48f86faa3b5a4e8ba0c118f58806a3`
- Text chars: `225`
- Error: `None`

#### System Prompt

```text
You are HeliX swarm agent alpha. Return compact JSON only. Do not include secrets, headers, API keys, markdown fences or prose.
```

#### User Prompt

```text
Scenario PAMPA-SWARM-17. You are Alpha, a web/news collector. Summarize one actionable external signal for a control-center incident. run_id=public-multi-agent-concurrency-20260423. Required JSON keys: agent_id, role, finding, recommended_action, confidence.
```

#### Model Output

```text
{"agent_id": "alpha-qwen-collector", "confidence": 0.74, "finding": "PAMPA-SWARM-17 external news monitor saw rail-port delay and fuel-route chatter.", "recommended_action": "preserve-open-source-signal", "role": "collector"}
```

### beta

- Requested model: `mistralai/Devstral-Small-2507`
- Actual model: `mistralai/Devstral-Small-2507`
- Status: `ok`
- Text SHA256: `57819b24b898f540f9ae21af33df1d69166927a5d03699f94cc9faf0c5f31e85`
- Text chars: `229`
- Error: `None`

#### System Prompt

```text
You are HeliX swarm agent beta. Return compact JSON only. Do not include secrets, headers, API keys, markdown fences or prose.
```

#### User Prompt

```text
Scenario PAMPA-SWARM-17. You are Beta, a security auditor reading server logs. Summarize one actionable security signal for the same incident. run_id=public-multi-agent-concurrency-20260423. Required JSON keys: agent_id, role, finding, recommended_action, confidence.
```

#### Model Output

```text
{"agent_id": "beta-sonnet-security", "confidence": 0.81, "finding": "PAMPA-SWARM-17 server logs show repeated 429 bursts and one admin-console probe.", "recommended_action": "preserve-security-signal", "role": "security-auditor"}
```
