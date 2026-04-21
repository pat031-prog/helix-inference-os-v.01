@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_memory_fork_forensics_secure.ps1" -ForensicModel "Qwen/Qwen3.6-35B-A3B" -AuditorModel "zai-org/GLM-5.1" -Tokens 2400
exit /b %ERRORLEVEL%
