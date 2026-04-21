@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0run_infinite_depth_memory_suite_secure.ps1" -Case "all" -Depth 5000 -SmallDepth 128 -MidDepth 1024 -Repeats 7 -BudgetTokens 800 -Limit 5
exit /b %ERRORLEVEL%
