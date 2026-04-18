param(
    [switch]$Synthetic,
    [switch]$EnableThinking,
    [string]$Model = "",
    [string]$FastModel = "meta-llama/Llama-3.2-3B-Instruct",
    [string]$ThinkModel = "meta-llama/Meta-Llama-3-8B-Instruct",
    [string]$PytestArgs = "tests/test_frontier_workloads.py -q -s"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$pytestArgList = @($PytestArgs -split '\s+' | Where-Object { $_ })
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $verificationDir "local-frontier-workloads-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-frontier-workloads-$stamp-run.json"

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
$previousFastModel = $env:HELIX_FRONTIER_FAST_MODEL
$previousThinkModel = $env:HELIX_FRONTIER_THINK_MODEL
$previousDisableThinking = $env:HELIX_FRONTIER_DISABLE_THINKING
$startedAt = [DateTimeOffset]::Now
$exitCode = 1

try {
    if (-not [string]::IsNullOrWhiteSpace($Model)) {
        $FastModel = $Model
        $ThinkModel = $Model
    }

    $env:HELIX_FRONTIER_FAST_MODEL = $FastModel
    $env:HELIX_FRONTIER_THINK_MODEL = $ThinkModel
    if ($EnableThinking) {
        $env:HELIX_FRONTIER_DISABLE_THINKING = "0"
    } else {
        $env:HELIX_FRONTIER_DISABLE_THINKING = "1"
    }

    if ($Synthetic) {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
        Write-Host "[helix] Running Frontier Workloads in synthetic mode."
        Write-Host "[helix] Model env set but no cloud call will be made."
    } else {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        Write-Host "[helix] Token loaded for this pytest process only. It will not be written to disk."
        Write-Host "[helix] Running Frontier Workloads."
        Write-Host "[helix] Fast model: $FastModel"
        Write-Host "[helix] Think model: $ThinkModel"
        Write-Host "[helix] Disable thinking: $(-not $EnableThinking)"
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
        artifact = "local-frontier-workloads-run"
        generated_by = "tools/run_frontier_workloads_secure.ps1"
        mode = $(if ($Synthetic) { "synthetic" } else { "real" })
        fast_model = $FastModel
        think_model = $ThinkModel
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
        claim_boundary = "Frontier workload runs require requested/actual model receipts before public model-identity claims."
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"

    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousFastModel) {
        $env:HELIX_FRONTIER_FAST_MODEL = $previousFastModel
    } else {
        Remove-Item Env:\HELIX_FRONTIER_FAST_MODEL -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousThinkModel) {
        $env:HELIX_FRONTIER_THINK_MODEL = $previousThinkModel
    } else {
        Remove-Item Env:\HELIX_FRONTIER_THINK_MODEL -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousDisableThinking) {
        $env:HELIX_FRONTIER_DISABLE_THINKING = $previousDisableThinking
    } else {
        Remove-Item Env:\HELIX_FRONTIER_DISABLE_THINKING -ErrorAction SilentlyContinue
    }
}

exit $exitCode
