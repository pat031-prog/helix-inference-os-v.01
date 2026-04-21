@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_nuclear_methodology_suite_secure.ps1" -Case "rollback-fence-replay" -Tokens 2800
exit /b %ERRORLEVEL%
