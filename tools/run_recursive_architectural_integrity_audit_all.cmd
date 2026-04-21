@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_recursive_architectural_integrity_audit_secure.ps1" -Case "all" -Tokens 3600
exit /b %ERRORLEVEL%
