# HeliX Local Session Core v0

## Resumen

Esta pasada agrega un nucleo local Rust para sesiones de inferencia y un scheduler tipo
hypervisor local, manteniendo el scope chico para esta laptop.

El objetivo no fue prometer velocidad antes de medirla. El objetivo fue separar tres cosas:

- un formato binario `.hlx` verificable con Merkle root;
- una interfaz Python via PyO3 o bridge CLI;
- pruebas rapidas sobre Transformer-only y un carril hibrido Zamba2 corto.

## Que se implemento

- Nuevo crate Rust: `crates/helix-state-core`.
- Formato `kv_cache.hlx` con header, indice JSON, arrays raw C-contiguous, hashes por array y Merkle por chunks de `1 MiB`.
- Modulo Python `_helix_state_core` via PyO3 compilado con `stable-x86_64-pc-windows-gnullvm`.
- Wrapper Python: `helix_kv.rust_session`.
- `session_codec="python-npz"|"rust-hlx"|"auto"` en los paths de save/load.
- Runner rapido: `tools/run_local_session_core.py`.
- Hypervisor local secuencial: `tools/run_local_agent_hypervisor.py`.
- Playback en `helix-watch` para `local-session-core-summary.json` y `local-agent-hypervisor-demo.json`.

## Comandos

```powershell
$env:PATH="$env:USERPROFILE\.cargo\bin;$env:PATH"
cargo +stable-x86_64-pc-windows-gnullvm test --manifest-path crates\helix-state-core\Cargo.toml --target x86_64-pc-windows-gnullvm --features python
pytest -q tests\test_rust_session_core.py tests\test_local_session_core.py tests\test_agent_hypervisor.py
python tools\run_local_session_core.py --models gpt2,qwen,zamba --profile laptop-12gb --quick --codec rust-hlx --output-dir verification --local-files-only
python tools\run_local_agent_hypervisor.py --model gpt2 --agents 5 --rounds 2 --timeslice-tokens 1 --codec rust-hlx --output-dir verification --local-files-only
```

Para abrir la telemetria:

```powershell
.\crates\helix-watch\target\x86_64-pc-windows-gnullvm\release\helix-watch.exe .\verification\local-session-core-summary.json
.\crates\helix-watch\target\x86_64-pc-windows-gnullvm\release\helix-watch.exe .\verification\local-agent-hypervisor-demo.json
```

## Resultados locales

Artifacts principales:

- `verification/local-session-core-summary.json`
- `verification/local-session-core-gpt2.json`
- `verification/local-session-core-qwen.json`
- `verification/local-session-core-zamba.json`
- `verification/local-session-core-toolchain.json`
- `verification/local-agent-hypervisor-demo.json`

La corrida rapida completo los tres carriles:

- `gpt2`: Transformer-only rapido, `hash_match=true`, `generated_ids_match=true`, `top1_match_all=true`, delta de logits `0.0`.
- `Qwen/Qwen2.5-1.5B-Instruct`: Transformer moderno cached, `hash_match=true`, `generated_ids_match=true`, `top1_match_all=true`, delta de logits `0.0`.
- `Zyphra/Zamba2-1.2B-Instruct-v2`: hibrido corto, `hash_match=true`, `generated_ids_match=true`, `top1_match_all=true`, delta de logits `0.0`.

El hypervisor local corrio 5 agentes, 2 rondas, 1 token por timeslice, con un solo modelo cargado y sesiones guardadas/restauradas entre turnos. Los 10 restores de timeslice reportaron `hash_match=true`.

## Lectura correcta

Claim verificado:

- HeliX puede guardar/restaurar caches de inferencia Transformer y un cache hibrido corto Zamba2 usando un bundle `.hlx` verificable.
- El snapshot restaurado puede producir la misma continuacion deterministica en estas corridas cortas: mismos tokens, mismo top-1 y delta de logits `0.0`.
- El hypervisor v0 demuestra context switching local secuencial de sesiones, no multitarea GPU ni daemon distribuido.

Claim no verificado todavia:

- `.hlx` todavia no es mas rapido que `.npz` en estas corridas chicas.
- El save/load sigue pagando staging Python de arrays; el siguiente paso de performance es pasar buffers/tensores directo a Rust, sin raw staging intermedio.
- Zamba se midio como carril hibrido corto local, no como stress largo ni HXQ.

## Caveat de performance

PyO3 esta activo (`pyo3_module_available=true`), pero el path v0 aun serializa arrays desde Python antes de invocar Rust. Por eso el valor diferencial actual es integridad, formato, Merkle verify y una interfaz binaria lista; la aceleracion fuerte requiere mover el empaquetado de buffers al lado Rust.
