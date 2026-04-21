@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_hard_anchors_suite_secure.ps1" -Case "all" -Tokens 3600
exit /b %ERRORLEVEL%
