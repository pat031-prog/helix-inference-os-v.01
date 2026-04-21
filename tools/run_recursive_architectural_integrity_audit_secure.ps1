param(
    [string]$Case = "all",
    [string]$ProposerModel = "Qwen/Qwen3.6-35B-A3B",
    [string]$RedTeamModel = "anthropic/claude-4-sonnet",
    [int]$Tokens = 3600,
    [string]$OutputDir = "verification/nuclear-methodology/recursive-architectural-integrity-audit",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$allowedCases = @(
    "all",
    "evidence-lineage-ingestion",
    "meta-architectural-recursion"
)
if ($allowedCases -notcontains $Case) {
    throw "Unsupported Case '$Case'. Allowed: $($allowedCases -join ', ')"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "recursive-architectural-integrity-audit-$stamp"
}
$logPath = Join-Path $verificationDir "local-recursive-architectural-integrity-audit-suite-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-recursive-architectural-integrity-audit-suite-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-recursive-architectural-integrity-audit-suite.log"
$stableManifestPath = Join-Path $verificationDir "local-recursive-architectural-integrity-audit-suite-run.json"

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

function Assert-DeepInfraModelRef {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -notmatch "^[^/\s]+/[^/\s]+$") {
        throw "$Name must be a full DeepInfra model ref like 'owner/model'. Got: '$Value'"
    }
}

Assert-DeepInfraModelRef -Name "ProposerModel" -Value $ProposerModel
Assert-DeepInfraModelRef -Name "RedTeamModel" -Value $RedTeamModel

$previous = @{
    DEEPINFRA_API_TOKEN = $env:DEEPINFRA_API_TOKEN
    HELIX_RUN_STARTED_AT_UTC = $env:HELIX_RUN_STARTED_AT_UTC
    HELIX_RUN_DATE_UTC = $env:HELIX_RUN_DATE_UTC
    HELIX_RUN_TIMEZONE = $env:HELIX_RUN_TIMEZONE
    HELIX_RUN_ID = $env:HELIX_RUN_ID
}

$exitCode = 1
$artifactPath = $null
$artifactHash = $null
$artifactBytes = 0

try {
    if ($Case -ne "evidence-lineage-ingestion") {
        $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
        $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
        if ([string]::IsNullOrWhiteSpace($plainToken)) {
            throw "DeepInfra token cannot be empty."
        }
        $env:DEEPINFRA_API_TOKEN = $plainToken
        $plainToken = $null
        Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."
    } else {
        Write-Host "[helix] Evidence-only case selected; no token required."
    }

    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Case: $Case"
    Write-Host "[helix] Proposer model: $ProposerModel"
    Write-Host "[helix] Red-team model: $RedTeamModel"
    Write-Host "[helix] Tokens: $Tokens"
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_recursive_architectural_integrity_audit_v1.py",
        "--output-dir", $verificationDir,
        "--proposer-model", $ProposerModel,
        "--red-team-model", $RedTeamModel,
        "--tokens", [string]$Tokens,
        "--run-id", $RunId,
        "--case", $Case
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @pyArgs 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
} finally {
    $endedAtUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    $logHash = $null
    $logBytes = 0
    if (Test-Path -LiteralPath $logPath) {
        Copy-Item -LiteralPath $logPath -Destination $stableLogPath -Force
        $logHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $logPath).Hash.ToLowerInvariant()
        $logBytes = (Get-Item -LiteralPath $logPath).Length
    }

    $candidate = Join-Path $verificationDir "local-recursive-architectural-integrity-audit-suite-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-recursive-architectural-integrity-audit-suite-run"
        generated_by = "tools/run_recursive_architectural_integrity_audit_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        case = $Case
        models = [ordered]@{
            proposer_requested = $ProposerModel
            red_team_requested = $RedTeamModel
        }
        tokens = $Tokens
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        log_path = $logPath
        stable_log_path = $stableLogPath
        log_sha256 = $logHash
        log_bytes = $logBytes
        artifact_path = $artifactPath
        artifact_sha256 = $artifactHash
        artifact_bytes = $artifactBytes
        token_handling = [ordered]@{
            token_required = ($Case -ne "evidence-lineage-ingestion")
            token_prompt_hidden = ($Case -ne "evidence-lineage-ingestion")
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        transcript_sidecars = "Each case artifact records transcript_exports.jsonl_path and transcript_exports.md_path."
        claim_boundary = "Meta-architecture audit over recent artifacts, not production readiness."
    }
    $manifestJson = $manifest | ConvertTo-Json -Depth 8
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($manifestPath, $manifestJson, $utf8NoBom)
    Copy-Item -LiteralPath $manifestPath -Destination $stableManifestPath -Force
    Write-Host "[helix] Evidence manifest: $manifestPath"
    if ($null -ne $artifactPath) {
        Write-Host "[helix] Artifact: $artifactPath"
    }

    foreach ($key in $previous.Keys) {
        if ($null -ne $previous[$key]) {
            Set-Item -Path "Env:\$key" -Value $previous[$key]
        } else {
            Remove-Item -Path "Env:\$key" -ErrorAction SilentlyContinue
        }
    }
}

exit $exitCode
