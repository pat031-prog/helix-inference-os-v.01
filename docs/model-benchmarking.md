# Local model benchmarking

This repo now includes a local benchmark flow for the cached models you already have on disk.

## What it measures

- Quality proxy score per role: `general`, `code`, `legal`
- Tokens per second
- RSS peak memory
- Per-prompt outputs for manual review

The prompt suite lives in `benchmarks/local_assistant_prompts.json`.

## Run the baseline benchmark

```powershell
$env:PYTHONPATH='src'
python -m helix_proto.cli benchmark-local-models `
  --output ".\benchmark-output\local-models"
```

By default it uses:

- `Qwen/Qwen2.5-0.5B`
- `Qwen/Qwen2.5-1.5B`
- `sshleifer/tiny-gpt2`
- `facebook/opt-350m`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `microsoft/phi-2`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

and runs in `local_files_only` mode, so it will not download anything.

Artifacts:

- `benchmark-output/local-models/benchmark_report.json`
- `benchmark-output/local-models/benchmark_summary.md`

## Restrict the run to a subset

```powershell
$env:PYTHONPATH='src'
python -m helix_proto.cli benchmark-local-models `
  "Qwen/Qwen2.5-0.5B" `
  "HuggingFaceTB/SmolLM2-1.7B-Instruct" `
  --max-new-tokens 48 `
  --output ".\benchmark-output\quick-pass"
```

## Prepare assistants from the benchmark result

This command reads the benchmark report, prepares only the recommended models, and configures the assistant router with specialized system prompts.

```powershell
$env:PYTHONPATH='src'
python -m helix_proto.cli prepare-best-assistants `
  --report ".\benchmark-output\local-models\benchmark_report.json" `
  --workspace-root ".\workspace" `
  --block-rows 256
```

Notes:

- The command shares one prepared workspace alias per unique model, so if the same model wins two roles it is not duplicated on disk.
- Assistant aliases are configured after the prepare step.

## Evaluate a fine-tuned model

If you have a merged local model directory from Colab:

```powershell
$env:PYTHONPATH='src'
python -m helix_proto.cli eval-finetuned-model `
  "C:\models\qwen-legal-ft" `
  --baseline-report ".\benchmark-output\local-models\benchmark_report.json" `
  --baseline-model-ref "Qwen/Qwen2.5-1.5B" `
  --output ".\benchmark-output\legal-ft"
```

Artifacts:

- `benchmark-output/legal-ft/benchmark_report.json`
- `benchmark-output/legal-ft/benchmark_summary.md`
- `benchmark-output/legal-ft/benchmark_comparison.json`
- `benchmark-output/legal-ft/benchmark_comparison.md`
