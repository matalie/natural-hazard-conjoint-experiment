@echo off
setlocal

call "%USERPROFILE%\AppData\Local\miniforge3\Scripts\activate.bat" snakemake

set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set "PYTENSOR_FLAGS=cxx=C:/msys64/ucrt64/bin/g++.exe"

snakemake ^
  -s workflow/Snakefile ^
  --cores 4 ^
  --sdm conda ^
  --conda-prefix "%USERPROFILE%\snakemake_envs" ^
  %*

endlocal