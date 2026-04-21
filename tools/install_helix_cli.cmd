@echo off
setlocal EnableExtensions

for /f "usebackq delims=" %%I in (`python -c "import sysconfig; print(sysconfig.get_path('scripts'))"`) do set "HELIX_SCRIPTS=%%I"
for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable)"`) do set "HELIX_PYTHON=%%I"
if not defined HELIX_SCRIPTS (
  echo [helix] Could not resolve the active Python Scripts directory.
  exit /b 1
)
if not defined HELIX_PYTHON (
  echo [helix] Could not resolve the active Python executable.
  exit /b 1
)

set "HELIX_REPO=%~dp0.."
set "HELIX_LAUNCHER=%HELIX_SCRIPTS%\helix.cmd"

> "%HELIX_LAUNCHER%" echo @echo off
>> "%HELIX_LAUNCHER%" echo setlocal
>> "%HELIX_LAUNCHER%" echo set "HELIX_REPO=%HELIX_REPO%"
>> "%HELIX_LAUNCHER%" echo set "PYTHONPATH=%%HELIX_REPO%%\src;%%HELIX_REPO%%;%%PYTHONPATH%%"
>> "%HELIX_LAUNCHER%" echo "%HELIX_PYTHON%" -m helix_proto.helix_cli %%*
>> "%HELIX_LAUNCHER%" echo exit /b %%ERRORLEVEL%%

echo [helix] Installed launcher:
echo   %HELIX_LAUNCHER%
echo [helix] If this directory is on PATH for your Anaconda env, run:
echo   helix
