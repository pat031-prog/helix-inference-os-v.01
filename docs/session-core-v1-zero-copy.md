# HeliX Session Core v1: Zero-Copy PyBuffer

## Resumen

Esta pasada agrega `rust-hlx-buffered`, un codec que evita el staging a archivos raw
temporales. Python pasa buffers C-contiguous a Rust via `memoryview`/PyBuffer, y Rust
escribe `kv_cache.hlx` directamente.

El objetivo era medir el techo real antes de prometer `5ms`. El resultado local es
mejor que el path viejo, pero todavia no llega a `5ms`.

## Comandos

```powershell
python tools\run_local_session_core.py --models gpt2,qwen --profile laptop-12gb --quick --codec rust-hlx-buffered --transformer-only --repeats 20 --output-dir verification --local-files-only --timeout-seconds 900
python tools\run_local_agent_hypervisor.py --scenario pr-war-room --model qwen --agents 5 --rounds 1 --timeslice-tokens 4 --codec rust-hlx-buffered --output-dir verification --local-files-only --timeout-seconds 900
python tools\run_local_agent_hypervisor.py --scenario hybrid-cameo --model zamba --agents 2 --rounds 1 --timeslice-tokens 1 --codec rust-hlx-buffered --output-dir verification --local-files-only --timeout-seconds 900
```

## Artifacts

- `verification/local-session-core-zero-copy-summary.json`
- `verification/local-agent-hypervisor-pr-war-room.json`
- `verification/local-agent-hypervisor-zamba-cameo.json`

## Resultados locales

`gpt2` con `rust-hlx-buffered`:

- `hash_match=true`
- `generated_ids_match=true`
- `top1_match_all=true`
- `max_abs_logit_delta=0.0`
- `save_time_ms_p50=50.42`
- `save_time_ms_p95=57.97`
- `load_time_ms_p50=9.80`
- `verify_time_ms_p50=9.11`
- `rust_hash_time_ms_p50=1.26`
- `rust_write_time_ms_p50=5.87`

`Qwen/Qwen2.5-1.5B-Instruct` con `rust-hlx-buffered`:

- `hash_match=true`
- `generated_ids_match=true`
- `top1_match_all=true`
- `max_abs_logit_delta=0.0`
- `save_time_ms_p50=54.30`
- `save_time_ms_p95=80.64`
- `load_time_ms_p50=31.10`
- `verify_time_ms_p50=8.91`
- `rust_hash_time_ms_p50=1.01`
- `rust_write_time_ms_p50=6.44`

## Lectura correcta

Claim permitido:

- HeliX ya tiene un codec `.hlx` PyBuffer-backed que evita staging raw en disco y mantiene equivalencia deterministica en GPT-2 y Qwen.
- El path buffered mejora fuertemente el orden de magnitud respecto al bridge/staging previo.
- El writer Rust interno ya tiene hashing y escritura en pocos milisegundos para caches chicos.

Claim no permitido todavia:

- No decir que HeliX alcanzo `5ms` end-to-end. El p50 local queda alrededor de `50-55ms`.
- No decir que esto es zero-copy desde tensor GPU. El path actual usa arrays CPU C-contiguous.
- No decir que Zamba entra en el objetivo `5ms`; Zamba queda como cameo hibrido de integridad.

## Proximo cuello

El costo restante parece estar fuera del hash/write puro de Rust. Las sospechas principales son:

- overhead de muchos buffers/arrays chicos cruzando PyO3;
- preparacion Python del manifest/specs;
- metadata JSON y receipt;
- conversion previa tensor CPU -> ndarray en el serializador.

El siguiente experimento razonable no es `tensor.data_ptr()` directo todavia. Primero conviene agrupar arrays por storage/layer o pasar una tabla compacta de specs para reducir overhead de llamadas y objetos Python.
