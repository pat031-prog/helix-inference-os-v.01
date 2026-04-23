@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_multi_agent_concurrency_suite_secure.ps1" -Case "all" -UseDeepInfra -AlphaModel "Qwen/Qwen3.5-122B-A10B" -BetaModel "mistralai/Devstral-Small-2507" -GammaModel "anthropic/claude-4-sonnet" -MaxTokens 512
exit /b %ERRORLEVEL%
