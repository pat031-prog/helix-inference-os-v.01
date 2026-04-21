@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_post_nuclear_methodology_suite_secure.ps1" -Case "all" -Tokens 3200
exit /b %ERRORLEVEL%
