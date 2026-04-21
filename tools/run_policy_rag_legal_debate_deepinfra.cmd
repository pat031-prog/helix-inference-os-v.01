@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_policy_rag_legal_debate_suite_secure.ps1" -Case "all" -SourceRepo "C:\Users\Big Duck\Desktop\TestGS\rag_polizas" -Repeats 3 -MinChunks 100 -MinPdfCount 2 -UseDeepInfra -ClientModel "Qwen/Qwen3.6-35B-A3B" -InsurerModel "meta-llama/Llama-3.3-70B-Instruct" -AuditorModel "anthropic/claude-4-sonnet" -Tokens 2400
exit /b %ERRORLEVEL%
