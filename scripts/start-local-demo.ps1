$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendScript = Join-Path $repoRoot "scripts\run-local-backend.ps1"
$frontendScript = Join-Path $repoRoot "scripts\run-local-frontend.ps1"

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $backendScript
)

Start-Sleep -Seconds 1

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $frontendScript
)
