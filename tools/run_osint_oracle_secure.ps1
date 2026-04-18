param(
    [switch]$Live,
    [switch]$FixtureOnly,
    [string]$Model = "Qwen/Qwen3.5-122B-A10B",
    [string]$PyArgs = "--fixture-only --output-dir verification"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $verificationDir "local-zero-day-osint-oracle-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-zero-day-osint-oracle-$stamp.json"

function Convert-SecureStringToPlainText {
    param([Parameter(Mandatory = $true)][securestring]$Secure)
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
}

$previousToken = $env:DEEPINFRA_API_TOKEN
$startedAt = [DateTimeOffset]::Now
$exitCode = 1
try {
    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if ([string]::IsNullOrWhiteSpace($plainToken)) {
        throw "DeepInfra token cannot be empty."
    }
    $env:DEEPINFRA_API_TOKEN = $plainToken
    $plainToken = $null
    Write-Host "[helix] Token loaded for this oracle process only. It will not be written to disk."

    $modeArgs = if ($Live) { "--live" } elseif ($FixtureOnly) { "--fixture-only" } else { $PyArgs }
    $commandArgs = @($modeArgs -split '\s+' | Where-Object { $_ })
    $commandArgs += @("--llm-mode", "deepinfra", "--model", $Model, "--output-dir", "verification")
    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Running OSINT Oracle with model: $Model"
    & python tools\run_live_zero_day_oracle.py @commandArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
} finally {
    $endedAt = [DateTimeOffset]::Now
    $hash = $null
    $bytes = 0
    if (Test-Path -LiteralPath $logPath) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $bytes = (Get-Item -LiteralPath $logPath).Length
    }
    $suspiciousShortLog = ($exitCode -eq 0 -and $bytes -lt 5000)
    if ($suspiciousShortLog) {
        Write-Host "[helix] Suspicious short success log detected ($bytes bytes < 5000). Marking run failed."
        $exitCode = 2
    }
    [ordered]@{
        artifact = "local-zero-day-osint-oracle-run"
        generated_by = "tools/run_osint_oracle_secure.ps1"
        mode = $(if ($Live) { "live" } else { "fixture" })
        model = $Model
        started_at = $startedAt.ToString("o")
        ended_at = $endedAt.ToString("o")
        duration_s = [Math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        log_path = $logPath
        log_sha256 = $hash
        log_bytes = $bytes
        suspicious_short_log = $suspiciousShortLog
        token_handling = [ordered]@{
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        claim_boundary = "Conservative OSINT correlation only; no confirmed zero-day claims."
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"
    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }
}

exit $exitCode
