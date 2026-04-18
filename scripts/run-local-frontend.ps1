$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python -m http.server 3000 --bind 127.0.0.1 --directory frontend
