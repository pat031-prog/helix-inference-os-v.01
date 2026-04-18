param(
    [switch]$Synthetic,
    [string]$PytestArgs = "tests/test_llm_workloads.py -v -s"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$pytestArgList = @($PytestArgs -split '\s+' | Where-Object { $_ })
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $verificationDir "local-deepinfra-workloads-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-deepinfra-workloads-$stamp-run.json"

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
    if ($Synthetic) {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
        Write-Host "[helix] Running LLM workloads in synthetic mode."
    } else {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        Write-Host "[helix] Token loaded for this pytest process only. It will not be written to disk."
    }

    Write-Host "[helix] Evidence log: $logPath"
    python -m pytest @pytestArgList 2>&1 | Tee-Object -FilePath $logPath
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
        artifact = "local-deepinfra-workloads-run"
        generated_by = "tools/run_deepinfra_workloads_secure.ps1"
        mode = $(if ($Synthetic) { "synthetic" } else { "real" })
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
            token_prompt_hidden = (-not [bool]$Synthetic)
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        claim_boundary = "Cloud workload logs are evidence receipts, not proof of model identity unless requested/actual model fields are recorded."
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"

    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }
}

exit $exitCode
