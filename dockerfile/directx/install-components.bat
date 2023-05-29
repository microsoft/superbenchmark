start /wait vs_BuildTools.exe --config  %SB_HOME%\dockerfile\directx\mini_vsconfig.json --wait --quiet --norestart > nul
if %errorlevel% neq 0 (
  exit /b %errorlevel%
)
