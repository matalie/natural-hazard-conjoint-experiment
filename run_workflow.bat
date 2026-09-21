@echo off
setlocal
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
if not defined SNAKEMAKE_ENV (
    set "SNAKEMAKE_ENV=snakemake"
)

if not defined SNAKEMAKE_CORES (
    set "SNAKEMAKE_CORES=4"
)

if not defined SNAKEMAKE_CONDA_PREFIX (
    set "SNAKEMAKE_CONDA_PREFIX=%USERPROFILE%\snakemake_envs"
)

if not defined MSYS2_UCRT64 (
    set "MSYS2_UCRT64=C:\msys64\ucrt64"
)

if not defined PYTENSOR_CXX (
    set "PYTENSOR_CXX=C:/msys64/ucrt64/bin/g++.exe"
)

rem ------------------------------------------------------------
rem Activate Snakemake
rem ------------------------------------------------------------
call "%USERPROFILE%\AppData\Local\miniforge3\Scripts\activate.bat" "%SNAKEMAKE_ENV%"

if errorlevel 1 (
    echo ERROR: Could not activate conda environment "%SNAKEMAKE_ENV%".
    exit /b 1
)

rem ------------------------------------------------------------
rem External compiler used by PyTensor
rem ------------------------------------------------------------
set "PATH=%MSYS2_UCRT64%\bin;%PATH%"
set "PYTENSOR_FLAGS=cxx=%PYTENSOR_CXX%"

rem Avoid thread oversubscription
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"

rem ------------------------------------------------------------
rem Fail early if compiler is unavailable
rem ------------------------------------------------------------
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
    --sdm conda ^
    --conda-prefix "%SNAKEMAKE_CONDA_PREFIX%" ^
    %*

endlocal