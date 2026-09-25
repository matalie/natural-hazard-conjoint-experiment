@echo off
setlocal

rem Run the Snakemake workflow from the repository root.
rem Usage examples:
rem   run_workflow.bat
rem   run_workflow.bat main_model
rem   run_workflow.bat -n
rem
rem The script:
rem   1. loads optional machine-specific settings,
rem   2. activates the shared project environment,
rem   3. configures the external MSYS2 compiler required by PyTensor,
rem   4. limits nested numerical-library threading,
rem   5. starts Snakemake and forwards all command-line arguments.

cd /d "%~dp0"


rem ------------------------------------------------------------
rem Optional machine-specific settings
rem ------------------------------------------------------------
if exist "%~dp0local_settings.bat" (
    call "%~dp0local_settings.bat"
)

rem ------------------------------------------------------------
rem Defaults
rem ------------------------------------------------------------
if not defined PROJECT_ENV (
    set "PROJECT_ENV=natural-hazard-solidarity"
)

if not defined SNAKEMAKE_CORES (
    set "SNAKEMAKE_CORES=4"
)

if not defined MSYS2_UCRT64 (
    set "MSYS2_UCRT64=C:\msys64\ucrt64"
)

if not defined PYTENSOR_CXX (
    set "PYTENSOR_CXX=C:/msys64/ucrt64/bin/g++.exe"
)

rem ------------------------------------------------------------
rem Project environment
rem ------------------------------------------------------------
call "%LOCALAPPDATA%\miniforge3\Scripts\activate.bat" "%PROJECT_ENV%"

if errorlevel 1 (
    echo ERROR: Could not activate conda environment "%PROJECT_ENV%".
    echo Create it first with:
    echo     conda env create -f environment.yaml
    exit /b 1
)

rem Make project's src/ package importable in Python processes
if defined PYTHONPATH (
    set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%CD%\src"
)

rem ------------------------------------------------------------
rem PyTensor compiler and validation
rem ------------------------------------------------------------
set "PATH=%MSYS2_UCRT64%\bin;%PATH%"
set "PYTENSOR_FLAGS=cxx=%PYTENSOR_CXX%"

rem Prevent numerical libraries from starting additional thread pools.
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"

rem Validataion
if not exist "%MSYS2_UCRT64%\bin\g++.exe" (
    echo ERROR: g++.exe not found:
    echo        %MSYS2_UCRT64%\bin\g++.exe
    echo.
    echo Check local_settings.bat.
    exit /b 1
)

"%MSYS2_UCRT64%\bin\g++.exe" --version >nul 2>&1

if errorlevel 1 (
    echo ERROR: g++.exe exists but could not be executed.
    echo        Check the MSYS2 installation and PATH.
    exit /b 1
)

rem ------------------------------------------------------------
rem Run workflow
rem ------------------------------------------------------------
snakemake ^
    -s workflow/Snakefile ^
    --cores %SNAKEMAKE_CORES% ^
    %*

endlocal