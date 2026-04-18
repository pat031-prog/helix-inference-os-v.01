param(
    [string]$TestId = "memory-contamination-triad",
    [switch]$EnableThinking
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo "verification"
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
$runId = "v4-gauntlet-$TestId-$stamp"
$logPath = Join-Path $verificationDir "local-v4-gauntlet-$TestId-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-v4-gauntlet-$TestId-$stamp-run.json"

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

$previous = @{
    DEEPINFRA_API_TOKEN = $env:DEEPINFRA_API_TOKEN
    HELIX_RUN_STARTED_AT_UTC = $env:HELIX_RUN_STARTED_AT_UTC
    HELIX_RUN_DATE_UTC = $env:HELIX_RUN_DATE_UTC
    HELIX_RUN_TIMEZONE = $env:HELIX_RUN_TIMEZONE
    HELIX_RUN_ID = $env:HELIX_RUN_ID
    HELIX_OBSERVATORY_DISABLE_THINKING = $env:HELIX_OBSERVATORY_DISABLE_THINKING
}

$exitCode = 1
try {
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $runId
    $env:HELIX_OBSERVATORY_DISABLE_THINKING = $(if ($EnableThinking) { "0" } else { "1" })

    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if (-not [string]::IsNullOrWhiteSpace($plainToken)) {
        $env:DEEPINFRA_API_TOKEN = $plainToken
    }
    $plainToken = $null
    Write-Host "[helix] Token loaded for this process only if provided. It will not be written to disk."
    Write-Host "[helix] Running v4 gauntlet: $TestId"
    Write-Host "[helix] Evidence log: $logPath"

    python tools\run_v4_gauntlet.py --test-id $TestId "--pytest-args=-q -s" 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
} finally {
    $endedAtUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    $logHash = $null
    $logBytes = 0
    if (Test-Path -LiteralPath $logPath) {
        $logHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $logBytes = (Get-Item -LiteralPath $logPath).Length
    }
    $expectedArtifacts = @()
    if ($TestId -eq "all" -or $TestId -eq "memory-contamination-triad") {
        $expectedArtifacts += (Join-Path $verificationDir "local-v4-memory-contamination-triad.json")
    }
    if ($TestId -eq "all" -or $TestId -eq "lineage-forgery-gauntlet") {
        $expectedArtifacts += (Join-Path $verificationDir "local-v4-lineage-forgery-gauntlet.json")
    }
    if ($TestId -eq "all" -or $TestId -eq "zero-day-osint-backtest") {
        $expectedArtifacts += (Join-Path $verificationDir "local-v4-zero-day-osint-backtest.json")
    }
    if ($TestId -eq "all" -or $TestId -eq "provider-substitution-longitudinal") {
        $expectedArtifacts += (Join-Path $verificationDir "provider-substitution-longitudinal\provider-substitution-longitudinal-fixture.json")
    }
    $artifactBytes = 0
    foreach ($artifactPath in $expectedArtifacts) {
        if (Test-Path -LiteralPath $artifactPath) {
            $artifactBytes += (Get-Item -LiteralPath $artifactPath).Length
        }
    }
    $suspiciousShortLog = ($exitCode -eq 0 -and $logBytes -lt 5000 -and $artifactBytes -lt 5000)
    if ($suspiciousShortLog) {
        $exitCode = 2
        Write-Host "[helix] suspicious_short_log: passed run produced less than 5000 log bytes and less than 5000 artifact bytes."
    }
    $manifest = [ordered]@{
        artifact = "local-v4-gauntlet-run"
        generated_by = "tools/run_v4_gauntlet_secure.ps1"
        test_id = $TestId
        run_id = $runId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        failure_reason = $(if ($suspiciousShortLog) { "suspicious_short_log" } else { $null })
        suspicious_short_log = $suspiciousShortLog
        log_path = $logPath
        log_sha256 = $logHash
        log_bytes = $logBytes
        evidence_artifact_bytes = $artifactBytes
        evidence_artifacts_checked = $expectedArtifacts
        token_handling = [ordered]@{
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        claim_boundary = "v4 gauntlet artifacts are falsable tests with preregistered hypotheses and bounded claim ladders."
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "[helix] Evidence manifest: $manifestPath"

    foreach ($key in $previous.Keys) {
        if ($null -ne $previous[$key]) {
            Set-Item -Path "Env:\$key" -Value $previous[$key]
        } else {
            Remove-Item -Path "Env:\$key" -ErrorAction SilentlyContinue
        }
    }
}

exit $exitCode
