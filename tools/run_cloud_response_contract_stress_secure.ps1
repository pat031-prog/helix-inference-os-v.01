param(
    [string]$Models = "Qwen/Qwen3-235B-A22B-Instruct-2507,anthropic/claude-sonnet-4-6,deepseek-ai/DeepSeek-V3,meta-llama/Llama-3.3-70B-Instruct",
    [string]$Contracts = "minimal_json,nested_claims,adversarial_boundary",
    [int]$Rounds = 1,
    [int]$Tokens = 420,
    [double]$Temperature = 0.0,
    [int]$Timeout = 240,
    [string]$RunId = "",
    [string]$OutputDir = "verification\nuclear-methodology\cloud-response-contract-stress"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $repo

if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $verificationDir = $OutputDir
} else {
    $verificationDir = Join-Path $repo $OutputDir
}
New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runStartedUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$runDateUtc = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-dd")
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "cloud-contract-stress-$stamp"
}

$modelList = @($Models -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($modelList.Count -lt 2) {
    throw "At least two DeepInfra model refs are required."
}
foreach ($model in $modelList) {
    if ($model -notmatch "^[^/\s,]+/[^,\s]+$") {
        throw "Models entry must be a full DeepInfra model ref like 'owner/model'. Got: '$model'"
    }
}

$logPath = Join-Path $verificationDir "local-cloud-response-contract-stress-$stamp.log"
$manifestPath = Join-Path $verificationDir "local-cloud-response-contract-stress-$stamp-run.json"
$stableLogPath = Join-Path $verificationDir "local-cloud-response-contract-stress.log"
$stableManifestPath = Join-Path $verificationDir "local-cloud-response-contract-stress-run.json"
$artifactSlug = "local-cloud-response-contract-stress-v1"
$artifactPath = Join-Path $verificationDir "$artifactSlug-$RunId.json"
$extractPath = Join-Path $verificationDir "$artifactSlug-$RunId-extract.md"
$transcriptPath = Join-Path $verificationDir "$artifactSlug-$RunId-transcript.md"
$bundlePath = Join-Path $verificationDir "$artifactSlug-$RunId-bundle.json"

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
}
$exitCode = 1

try {
    $secureToken = Read-Host "Paste a freshly rotated DeepInfra token (input hidden)" -AsSecureString
    $plainToken = Convert-SecureStringToPlainText -Secure $secureToken
    if ([string]::IsNullOrWhiteSpace($plainToken)) {
        throw "DeepInfra token cannot be empty."
    }

    $env:DEEPINFRA_API_TOKEN = $plainToken
    $plainToken = $null
    $env:HELIX_RUN_STARTED_AT_UTC = $runStartedUtc
    $env:HELIX_RUN_DATE_UTC = $runDateUtc
    $env:HELIX_RUN_TIMEZONE = "America/Buenos_Aires"
    $env:HELIX_RUN_ID = $RunId

    Write-Host "[helix] Token loaded for this python process only. It will not be written to disk."
    Write-Host "[helix] Evidence log: $logPath"
    Write-Host "[helix] Run ID: $RunId"
    Write-Host "[helix] Models: $($modelList -join ',')"
    Write-Host "[helix] Contracts: $Contracts"
    Write-Host "[helix] Rounds: $Rounds"
    Write-Host "[helix] Tokens per call: $Tokens"
    Write-Host "[helix] Temperature: $Temperature"
    Write-Host "[helix] Output dir: $verificationDir"

    $pyArgs = @(
        "tools\run_cloud_response_contract_stress_v1.py",
        "--output-dir", $verificationDir,
        "--models", ($modelList -join ","),
        "--contracts", $Contracts,
        "--rounds", [string]$Rounds,
        "--tokens", [string]$Tokens,
        "--temperature", [string]$Temperature,
        "--timeout", [string]$Timeout,
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

    $artifactHash = $null
    $artifactBytes = 0
    $bundleHash = $null
    if (Test-Path -LiteralPath $artifactPath) {
        $artifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifactPath).Hash.ToLowerInvariant()
        $artifactBytes = (Get-Item -LiteralPath $artifactPath).Length
    }
    if (Test-Path -LiteralPath $bundlePath) {
        $bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundlePath).Hash.ToLowerInvariant()
    }

    $manifest = [ordered]@{
        artifact = "local-cloud-response-contract-stress-run"
        generated_by = "tools/run_cloud_response_contract_stress_secure.ps1"
        run_id = $RunId
        run_started_at_utc = $runStartedUtc
        run_ended_at_utc = $endedAtUtc
        run_date_utc = $runDateUtc
        run_timezone = "America/Buenos_Aires"
        models = $modelList
        contracts = $Contracts
        rounds = $Rounds
        tokens_per_call = $Tokens
        temperature_milli = [int][Math]::Round($Temperature * 1000)
        timeout_s = $Timeout
        exit_code = $exitCode
        passed = ($exitCode -eq 0)
        log_path = $logPath
        stable_log_path = $stableLogPath
        log_sha256 = $logHash
        log_bytes = $logBytes
        artifact_path = $(if (Test-Path -LiteralPath $artifactPath) { $artifactPath } else { $null })
        artifact_sha256 = $artifactHash
        artifact_bytes = $artifactBytes
        extract_markdown_path = $(if (Test-Path -LiteralPath $extractPath) { $extractPath } else { $null })
        transcript_markdown_path = $(if (Test-Path -LiteralPath $transcriptPath) { $transcriptPath } else { $null })
        standalone_bundle_path = $(if (Test-Path -LiteralPath $bundlePath) { $bundlePath } else { $null })
        standalone_bundle_sha256 = $bundleHash
        token_handling = [ordered]@{
            token_prompt_hidden = $true
            token_written_to_disk = $false
            api_key_persisted = $false
            headers_recorded = $false
        }
        claim_boundary = "Observed structured-output reliability only. No semantic truth or global non-equivocation claim."
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Copy-Item -LiteralPath $manifestPath -Destination $stableManifestPath -Force
    Write-Host "[helix] Evidence manifest: $manifestPath"
    if (Test-Path -LiteralPath $artifactPath) {
        Write-Host "[helix] Artifact: $artifactPath"
    }
    if (Test-Path -LiteralPath $bundlePath) {
        Write-Host "[helix] Bundle: $bundlePath"
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
