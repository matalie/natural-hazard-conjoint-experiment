@echo off

rem Copy this file to local_settings.bat and change only machine-specific values.
set "SNAKEMAKE_ENV=snakemake"
set "SNAKEMAKE_CORES=1"
set "SNAKEMAKE_CONDA_PREFIX=%USERPROFILE%\snakemake_envs"

set "MSYS2_UCRT64=C:\msys64\ucrt64"
set "PYTENSOR_CXX=C:/msys64/ucrt64/bin/g++.exe"
