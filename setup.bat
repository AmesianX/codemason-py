@echo off
setlocal enableextensions
rem Bootstrap codemason on THIS Windows machine: (re)creates .venv and installs deps.
rem
rem WHY: a virtualenv is machine-specific -- .venv\Scripts\python.exe points at the
rem local interpreter and packages may be linked from a local cache. Copying .venv
rem to another box breaks it. So NEVER ship .venv; ship the source and run this.
rem (Windows twin of setup.sh -- .bat runs in cmd.exe with no ExecutionPolicy hurdle.)

cd /d "%~dp0"
echo [setup] (re)creating .venv in %CD%
if exist ".venv" rmdir /s /q ".venv"

where uv >nul 2>nul
if %errorlevel%==0 (
    echo [setup] uv found -- creating .venv ^(python 3.11^)
    uv venv --python 3.11 .venv || uv venv .venv
    set "VIRTUAL_ENV=%CD%\.venv"
    uv pip install -e .
    if errorlevel 1 goto :fail
    goto :done
)

rem No uv -- find a Python interpreter (py launcher preferred, then python)
set "PY="
py -3.11 --version >nul 2>nul && set "PY=py -3.11"
if not defined PY ( py -3 --version >nul 2>nul && set "PY=py -3" )
if not defined PY ( python --version >nul 2>nul && set "PY=python" )
if not defined PY (
    echo [setup] ERROR: no Python found. Install Python 3.10+ from https://python.org
    exit /b 1
)
echo [setup] using %PY%
%PY% -m venv .venv
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -U pip -q
".venv\Scripts\python.exe" -m pip install -e . -q
if errorlevel 1 goto :fail

:done
echo.
echo [setup] done. run:  .venv\Scripts\codemason.exe
echo         ^(or add "%CD%\.venv\Scripts" to your PATH, then just: codemason^)
exit /b 0

:fail
echo [setup] install FAILED (see errors above)
exit /b 1
