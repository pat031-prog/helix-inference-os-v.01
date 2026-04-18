$env:PYTHONPATH='src'
python -m helix_proto.cli eval-finetuned-model C:\ruta\modelo-merged --baseline-report benchmark-output/candidate-four/benchmark_report.json --baseline-model-ref Qwen/Qwen2.5-0.5B --output .\benchmark-output\ops-qwen05b-unsloth
