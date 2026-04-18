param(
    [switch]$Synthetic,
    [switch]$RequireQualityGate,
    [switch]$EnableThinking,
    [string]$Model = "Qwen/Qwen3.5-122B-A10B",
    [int]$MaxTokens = 512,
    [string]$PytestArgs = "tests/test_cloud_amnesia_derby.py -q -s"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$pytestArgList = @($PytestArgs -split '\s+' | Where-Object { $_ })
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $verificationDir "local-cloud-amnesia-derby-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-cloud-amnesia-derby-$stamp-run.json"

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
$previousModel = $env:HELIX_DERBY_MODEL
$previousMaxTokens = $env:HELIX_DERBY_MAX_TOKENS
$previousRequireQuality = $env:HELIX_DERBY_REQUIRE_QUALITY
$previousDisableThinking = $env:HELIX_DERBY_DISABLE_THINKING
$startedAt = [DateTimeOffset]::Now
$exitCode = 1

try {
    $env:HELIX_DERBY_MODEL = $Model
    $env:HELIX_DERBY_MAX_TOKENS = [string]$MaxTokens
    if ($RequireQualityGate) {
        $env:HELIX_DERBY_REQUIRE_QUALITY = "1"
    } else {
        $env:HELIX_DERBY_REQUIRE_QUALITY = "0"
    }
    if ($EnableThinking) {
        $env:HELIX_DERBY_DISABLE_THINKING = "0"
    } else {
        $env:HELIX_DERBY_DISABLE_THINKING = "1"
    }

    if ($Synthetic) {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
        Write-Host "[helix] Running Cloud Amnesia Derby in synthetic mode."
        Write-Host "[helix] Model env set to '$Model' but no cloud call will be made."
    } else {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        Write-Host "[helix] Token loaded for this pytest process only. It will not be written to disk."
        Write-Host "[helix] Running Cloud Amnesia Derby with model: $Model"
        Write-Host "[helix] Max tokens: $MaxTokens | Require quality gate: $RequireQualityGate | Disable thinking: $(-not $EnableThinking)"
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
        artifact = "local-cloud-amnesia-derby-run"
        generated_by = "tools/run_cloud_amnesia_derby_secure.ps1"
        mode = $(if ($Synthetic) { "synthetic" } else { "real" })
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
            token_prompt_hidden = (-not [bool]$Synthetic)
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        claim_boundary = "Synthetic deltas are template-completion deltas; real deltas require cloud model audit."
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"

    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousModel) {
        $env:HELIX_DERBY_MODEL = $previousModel
    } else {
        Remove-Item Env:\HELIX_DERBY_MODEL -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousMaxTokens) {
        $env:HELIX_DERBY_MAX_TOKENS = $previousMaxTokens
    } else {
        Remove-Item Env:\HELIX_DERBY_MAX_TOKENS -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousRequireQuality) {
        $env:HELIX_DERBY_REQUIRE_QUALITY = $previousRequireQuality
    } else {
        Remove-Item Env:\HELIX_DERBY_REQUIRE_QUALITY -ErrorAction SilentlyContinue
    }

    if ($null -ne $previousDisableThinking) {
        $env:HELIX_DERBY_DISABLE_THINKING = $previousDisableThinking
    } else {
        Remove-Item Env:\HELIX_DERBY_DISABLE_THINKING -ErrorAction SilentlyContinue
    }
}

exit $exitCode
