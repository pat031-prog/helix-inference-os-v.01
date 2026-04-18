$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = "src"
python -m helix_proto.cli serve-api --workspace-root workspace --host 127.0.0.1 --port 8000
