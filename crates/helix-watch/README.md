# helix-watch

`helix-watch` is the Rust playback TUI for HeliX stress telemetry. It does not run inference in v1; it reads JSON/JSONL artifacts produced by the Python stress runner.

## Build

```powershell
cargo +stable-x86_64-pc-windows-gnullvm build --release --target x86_64-pc-windows-gnullvm --manifest-path crates\helix-watch\Cargo.toml
```

## Open The Stress Dashboard

```powershell
.\crates\helix-watch\target\x86_64-pc-windows-gnullvm\release\helix-watch.exe .\verification\local-zamba2-stress-dashboard.json
```

## Open A Single Mission

```powershell
.\crates\helix-watch\target\x86_64-pc-windows-gnullvm\release\helix-watch.exe .\verification\local-zamba2-stress-restore-equivalence.json
```

## What It Shows

- Long-Context Coder: runtime-cache ratio, speedup, answer preview, identifier recall.
- State Juggler: session hash match, save/load timing, session bytes, answer previews.
- Context Switcher: promoted-block surge, logits finitude, fallback behavior.
- Restore Equivalence: hash match, token match, top-1 match, logit delta, session bytes.
