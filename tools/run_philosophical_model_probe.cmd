@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_philosophical_model_probe_secure.ps1" -MaxTokens 650 -Temperature 0
exit /b %ERRORLEVEL%
