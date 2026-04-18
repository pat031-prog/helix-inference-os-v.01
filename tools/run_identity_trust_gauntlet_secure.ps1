param(
    [switch]$Synthetic,
    [switch]$EnableThinking,
    [string]$Model = "Qwen/Qwen3.5-122B-A10B",
    [int]$OuroborosCycles = 20,
    [string]$PytestArgs = "tests/test_identity_trust_gauntlet.py -v -s"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $verificationDir "local-identity-trust-gauntlet-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-identity-trust-gauntlet-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-identity-trust-gauntlet.log"
$pytestArgList = @($PytestArgs -split '\s+' | Where-Object { $_ })

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
$previousModel = $env:HELIX_IDENTITY_MODEL
$previousThinking = $env:HELIX_IDENTITY_DISABLE_THINKING
$previousCycles = $env:HELIX_OUROBOROS_CYCLES
$startedAt = [DateTimeOffset]::Now
$exitCode = 1
$mode = "real"

try {
    $env:HELIX_IDENTITY_MODEL = $Model
    $env:HELIX_OUROBOROS_CYCLES = [string]$OuroborosCycles
    if ($EnableThinking) {
        $env:HELIX_IDENTITY_DISABLE_THINKING = "0"
    } else {
        $env:HELIX_IDENTITY_DISABLE_THINKING = "1"
    }

    if ($Synthetic) {
        $mode = "synthetic"
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
        Write-Host "[helix] Running Identity Trust Gauntlet in synthetic mode."
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
    Write-Host "[helix] Identity model: $Model"
    Write-Host "[helix] Ouroboros cycles: $OuroborosCycles"
    Write-Host "[helix] Disable thinking: $(-not $EnableThinking)"
    python -m pytest @pytestArgList 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
} finally {
    $endedAt = [DateTimeOffset]::Now
    $hash = $null
    $bytes = 0
    if (Test-Path -LiteralPath $logPath) {
        Copy-Item -LiteralPath $logPath -Destination $stableLogPath -Force
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $bytes = (Get-Item -LiteralPath $logPath).Length
    }
    $evidenceArtifacts = @(
        (Join-Path $verificationDir "local-governance-accountability-ledger.json"),
        (Join-Path $verificationDir "local-ouroboros-20-self-modeling.json"),
        (Join-Path $verificationDir "local-multi-agent-trust-network.json"),
        (Join-Path $verificationDir "local-cross-test-research-memo.json")
    )
    $artifactBytes = 0
    foreach ($artifactPath in $evidenceArtifacts) {
        if (Test-Path -LiteralPath $artifactPath) {
            $artifactBytes += (Get-Item -LiteralPath $artifactPath).Length
        }
    }
    $suspiciousShortLog = ($exitCode -eq 0 -and $bytes -lt 5000 -and $artifactBytes -lt 5000)
    if ($suspiciousShortLog) {
        $exitCode = 2
        Write-Host "[helix] suspicious_short_log: passed run produced less than 5000 log bytes and less than 5000 artifact bytes."
    }
    [ordered]@{
        artifact = "local-identity-trust-gauntlet-run"
        generated_by = "tools/run_identity_trust_gauntlet_secure.ps1"
        mode = $mode
        model = $Model
        ouroboros_cycles = $OuroborosCycles
        started_at = $startedAt.ToString("o")
        ended_at = $endedAt.ToString("o")
        duration_s = [Math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        suspicious_short_log = $suspiciousShortLog
        log_path = $logPath
        stable_log_path = $stableLogPath
        log_sha256 = $hash
        log_bytes = $bytes
        evidence_artifact_bytes = $artifactBytes
        evidence_artifacts_checked = $evidenceArtifacts
        token_handling = [ordered]@{
            token_prompt_hidden = (-not [bool]$Synthetic)
            token_written_to_disk = $false
            api_key_persisted = $false
        }
        claim_boundary = "Verified self-modeling behavior and accountability; not consciousness or regulatory compliance."
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"

    if ($null -ne $previousToken) {
        $env:DEEPINFRA_API_TOKEN = $previousToken
    } else {
        Remove-Item Env:\DEEPINFRA_API_TOKEN -ErrorAction SilentlyContinue
    }
    if ($null -ne $previousModel) { $env:HELIX_IDENTITY_MODEL = $previousModel } else { Remove-Item Env:\HELIX_IDENTITY_MODEL -ErrorAction SilentlyContinue }
    if ($null -ne $previousThinking) { $env:HELIX_IDENTITY_DISABLE_THINKING = $previousThinking } else { Remove-Item Env:\HELIX_IDENTITY_DISABLE_THINKING -ErrorAction SilentlyContinue }
    if ($null -ne $previousCycles) { $env:HELIX_OUROBOROS_CYCLES = $previousCycles } else { Remove-Item Env:\HELIX_OUROBOROS_CYCLES -ErrorAction SilentlyContinue }
}

exit $exitCode
