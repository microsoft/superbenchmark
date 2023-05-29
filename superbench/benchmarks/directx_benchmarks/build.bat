@echo off
setlocal enabledelayedexpansion

REM Iterate through all .sln files in the current directory and its subdirectories
for /r %%F in (*.sln) do (
    :: Process the .sln file here
    set SLN_PATH=%%~dpF%%~nxF
    SET CURRENT_PATH=%~dp0
    SET MSBUILD="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
    :: Download dependencise
    %MSBUILD% %SLN_PATH% -t:restore -p:RestorePackagesConfig=true
    :: Build project
    %MSBUILD% %SLN_PATH%  /p:Configuration=Release /p:AdditionalLibraryDirectories="%WindowsSDKDir%\Lib" /p:AdditionalIncludeDirectories="%WindowsSDKDir%\Include" /p:OutDir="%SB_MICRO_PATH%\bin"
)

endlocal
