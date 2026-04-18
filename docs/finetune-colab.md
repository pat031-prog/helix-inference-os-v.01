# QLoRA fine-tuning in Colab

This scaffolding is meant for Google Colab with a T4 GPU. It does not assume local GPU training.

## Recommended base models for your current setup

- `Qwen/Qwen2.5-1.5B`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Those are the best balance between capability and Colab-friendly size for a first legal fine-tune.

## Files included

- `datasets/legal_ar_template.jsonl`
- `datasets/legal_ar_seed.jsonl`
- `datasets/ops_linux_seed.jsonl`
- `finetune/train_qlora_trl.py`
- `finetune/train_qlora_unsloth.py`
- `finetune/merge_peft_adapter.py`
- `finetune/run_job.py`
- `finetune/prepare_job_bundle.py`
- `finetune/jobs/legal_qwen15b_unsloth.json`
- `finetune/jobs/ops_qwen05b_unsloth.json`

## Install packages in Colab

TRL route:

```python
!pip install -U transformers datasets peft trl accelerate bitsandbytes safetensors
```

Unsloth route:

```python
!pip install -U unsloth transformers datasets trl peft accelerate bitsandbytes safetensors
```

## Dataset format

The scripts expect JSONL rows with:

- `instruction`
- `response` or `output`

Optional fields:

- `system`
- `input`
- `context`

Example row:

```json
{"instruction":"Una persona compro online un electrodomestico y quiere arrepentirse a los 7 dias.","output":"En terminos generales...","system":"Sos un asistente de informacion juridica general para Argentina."}
```

## Train with TRL

```python
!python finetune/train_qlora_trl.py \
  --model-name "Qwen/Qwen2.5-1.5B" \
  --dataset-path "datasets/legal_ar_template.jsonl" \
  --output-dir "/content/out/legal-qwen-trl"
```

## Train with Unsloth

```python
!python finetune/train_qlora_unsloth.py \
  --model-name "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --dataset-path "datasets/legal_ar_template.jsonl" \
  --output-dir "/content/out/legal-smollm-unsloth"
```

## Recommended one-command jobs

Legal fine-tune, recommended base:

```python
!python finetune/run_job.py --job-config finetune/jobs/legal_qwen15b_unsloth.json --print-colab
!python finetune/run_job.py --job-config finetune/jobs/legal_qwen15b_unsloth.json --execute
```

Ops fine-tune, recommended base:

```python
!python finetune/run_job.py --job-config finetune/jobs/ops_qwen05b_unsloth.json --print-colab
!python finetune/run_job.py --job-config finetune/jobs/ops_qwen05b_unsloth.json --execute
```

If you prefer the stock HF route:

```python
!python finetune/run_job.py --job-config finetune/jobs/legal_qwen15b_trl.json --execute
!python finetune/run_job.py --job-config finetune/jobs/ops_qwen05b_trl.json --execute
```

The Unsloth script saves:

- adapter weights in `adapter/`
- a merged full model in `merged_16bit/`

The TRL script saves:

- adapter weights in `adapter/`

If you train with TRL and want a full model directory, merge the adapter after training.

## Merge adapter into full weights

```python
!python finetune/merge_peft_adapter.py \
  --base-model "Qwen/Qwen2.5-1.5B" \
  --adapter-dir "/content/out/legal-qwen-trl/adapter" \
  --output-dir "/content/out/legal-qwen-merged"
```

This produces a plain Transformers directory that `helix-proto` can ingest directly.

## Create a portable bundle before going to Colab

This writes the dataset, commands and return-path files into one folder:

```powershell
python finetune/prepare_job_bundle.py --job-config finetune/jobs/legal_qwen15b_unsloth.json
python finetune/prepare_job_bundle.py --job-config finetune/jobs/ops_qwen05b_unsloth.json
```

Each bundle contains:

- copied dataset
- `job.json`
- `train_colab.txt`
- `merge_colab.txt` when needed
- `import_to_helix.txt`
- `eval_in_helix.txt`
- `summary.json`

## Bring the fine-tuned model back into helix-proto

Once you download or mount the merged model directory on your machine:

```powershell
$env:PYTHONPATH='src'
python -m helix_proto.cli prepare-model `
  "C:\ruta\legal-qwen-merged" `
  --alias legal-ft `
  --workspace-root ".\workspace" `
  --local-files-only
```

Then benchmark it against the same prompt suite:

```powershell
$env:PYTHONPATH='src'
python -m helix_proto.cli eval-finetuned-model `
  "C:\ruta\legal-qwen-merged" `
  --baseline-report ".\benchmark-output\local-models\benchmark_report.json" `
  --baseline-model-ref "Qwen/Qwen2.5-1.5B" `
  --output ".\benchmark-output\legal-ft"
```

## Practical notes

- Keep the first dataset tiny and clean before scaling.
- Use the benchmark first to choose the best base model before spending Colab time.
- Prefer an instruct model for the legal assistant if the benchmark is close.
- For this repo, the cleanest return path is: Colab fine-tune -> merged Transformers directory -> `prepare-model`.

## Official references used for this scaffold

- Unsloth fine-tuning guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- Unsloth merged save flow: https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide
- Hugging Face TRL `SFTTrainer`: https://huggingface.co/docs/trl/main/en/sft_trainer
- Hugging Face Transformers bitsandbytes QLoRA notes: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Hugging Face PEFT merge guidance: https://huggingface.co/docs/peft/en/developer_guides/checkpoint
