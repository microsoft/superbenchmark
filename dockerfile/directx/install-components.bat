REM Copyright (c) Microsoft Corporation - All rights reserved
REM Licensed under the MIT License

curl -s -L https://aka.ms/vs/17/release/vs_BuildTools.exe -o "vs_BuildTools.exe"
start /b /wait vs_BuildTools.exe --config  %SB_HOME%\dockerfile\directx\mini_vsconfig.json --wait --quiet --norestart --nocache
if %errorlevel% neq 0 (
  exit /b %errorlevel%
)
del "vs_BuildTools.exe"
