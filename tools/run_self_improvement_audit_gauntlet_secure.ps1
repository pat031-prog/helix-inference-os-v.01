param(
    [string]$Models = "anthropic/claude-4-sonnet,Qwen/Qwen3.6-35B-A3B,google/gemma-4-31B-it,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct,mistralai/Mixtral-8x7B-Instruct-v0.1",
    [string]$AnalystModel = "Qwen/Qwen3.6-35B-A3B",
    [string]$AuditorModel = "anthropic/claude-4-sonnet",
    [int]$ProposalsPerModel = 3,
    [int]$MaxBacklogItems = 8,
    [int]$MaxEvidenceChars = 1400,
    [int]$ReviewerTokens = 2200,
    [int]$AnalysisTokens = 4200,
    [int]$AuditorTokens = 1200,
    [string]$OutputDir = "verification/nuclear-methodology/self-improvement-audit-gauntlet",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo
$verificationDir = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "self-improvement-audit-$stamp"
}
$logPath = Join-Path $verificationDir "local-self-improvement-audit-gauntlet-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-self-improvement-audit-gauntlet-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-self-improvement-audit-gauntlet.log"
$stableManifestPath = Join-Path $verificationDir "local-self-improvement-audit-gauntlet-run.json"

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

foreach ($model in $Models.Split(",")) {
    Assert-DeepInfraModelRef -Name "Models entry" -Value $model.Trim()
}
Assert-DeepInfraModelRef -Name "AnalystModel" -Value $AnalystModel
Assert-DeepInfraModelRef -Name "AuditorModel" -Value $AuditorModel

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
    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if ([string]::IsNullOrWhiteSpace($plainToken)) {
        throw "DeepInfra token cannot be empty."
    }
    $env:DEEPINFRA_API_TOKEN = $plainToken
    $plainToken = $null
    Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."

    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run date UTC: $runDateUtc"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Models: $Models"
    Write-Host "[helix] Analyst model: $AnalystModel"
    Write-Host "[helix] Auditor model: $AuditorModel"
    Write-Host "[helix] Proposals per model: $ProposalsPerModel"
    Write-Host "[helix] Max backlog items: $MaxBacklogItems"
    Write-Host "[helix] Max evidence chars: $MaxEvidenceChars"
    Write-Host "[helix] Reviewer tokens: $ReviewerTokens"
    Write-Host "[helix] Analysis tokens: $AnalysisTokens"
    Write-Host "[helix] Auditor tokens: $AuditorTokens"
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_self_improvement_audit_gauntlet_v1.py",
        "--output-dir", $verificationDir,
        "--models", $Models,
        "--analyst-model", $AnalystModel,
        "--auditor-model", $AuditorModel,
        "--proposals-per-model", [string]$ProposalsPerModel,
        "--max-backlog-items", [string]$MaxBacklogItems,
        "--max-evidence-chars", [string]$MaxEvidenceChars,
        "--reviewer-tokens", [string]$ReviewerTokens,
        "--analysis-tokens", [string]$AnalysisTokens,
        "--auditor-tokens", [string]$AuditorTokens,
        "--run-id", $RunId
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

    $candidate = Join-Path $verificationDir "local-self-improvement-audit-gauntlet-$RunId.json"
    if (Test-Path -LiteralPath $candidate) {
        $artifactPath = $candidate
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $candidate).Length
    }

    $manifest = [ordered]@{
        artifact = "local-self-improvement-audit-gauntlet-run"
        generated_by = "tools/run_self_improvement_audit_gauntlet_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        models = $Models
        analyst_model = $AnalystModel
        auditor_model = $AuditorModel
        proposals_per_model = $ProposalsPerModel
        max_backlog_items = $MaxBacklogItems
        max_evidence_chars = $MaxEvidenceChars
        reviewer_tokens = $ReviewerTokens
        analysis_tokens = $AnalysisTokens
        auditor_tokens = $AuditorTokens
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
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        claim_boundary = "Evidence-cited candidate backlog only. No patches are applied by this gauntlet."
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
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
