@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_long_horizon_checkpoint_suite_secure.ps1" -Case "all" -Tokens 3600 -ChainLength 48
exit /b %ERRORLEVEL%
