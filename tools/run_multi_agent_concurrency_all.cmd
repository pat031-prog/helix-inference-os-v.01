@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_multi_agent_concurrency_suite_secure.ps1" -Case "all" -MaxTokens 512
exit /b %ERRORLEVEL%
