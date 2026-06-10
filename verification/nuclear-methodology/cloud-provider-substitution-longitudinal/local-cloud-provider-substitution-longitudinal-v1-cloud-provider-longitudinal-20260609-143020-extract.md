# HeliX Cloud Provider Substitution Longitudinal: cloud-provider-longitudinal-20260609-143020

## Verdict

- Status: `completed`
- Score: `1.0`
- Rounds: `3`
- Models: `Qwen/Qwen3-235B-A22B-Instruct-2507, anthropic/claude-sonnet-4-6, deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.3-70B-Instruct`
- Tree size: `19`
- Final root: `5534f5c2b8afb487f6d1df3e9128b94ec48b2ae28a52cd692c33abf0e9be9565`

## Findings

- Provider substitution detected: `True`
- Actual model drift detected: `False`
- Output digest drift detected: `True`
- Availability errors: `0`

## Failing Gates

- None

## Model Summaries

- `Qwen/Qwen3-235B-A22B-Instruct-2507`: ok `3/3`, actual `Qwen/Qwen3-235B-A22B-Instruct-2507`, substitutions `0`, output digests `3`, avg latency `5422ms`
- `anthropic/claude-sonnet-4-6`: ok `3/3`, actual `anthropic/claude-sonnet-4-6`, substitutions `0`, output digests `3`, avg latency `6551ms`
- `deepseek-ai/DeepSeek-V3`: ok `3/3`, actual `deepseek-ai/DeepSeek-V3`, substitutions `0`, output digests `3`, avg latency `22812ms`
- `meta-llama/Llama-3.3-70B-Instruct`: ok `3/3`, actual `meta-llama/Llama-3.3-70B-Instruct-Turbo`, substitutions `3`, output digests `3`, avg latency `5776ms`

## Claim Boundary

Local DeepInfra evidence is auditable; semantic truth and global non-equivocation still require external witnesses.
