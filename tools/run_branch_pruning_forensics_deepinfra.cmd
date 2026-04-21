@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_branch_pruning_forensics_suite_secure.ps1" -Case "all" -Depth 5000 -BranchDepth 4 -BytesPerNode 2048 -Repeats 7 -MaxAnchorMs 25.0 -UseDeepInfra -SolverModel "Qwen/Qwen3.6-35B-A3B" -AuditorModel "anthropic/claude-4-sonnet" -Tokens 2200
exit /b %ERRORLEVEL%
