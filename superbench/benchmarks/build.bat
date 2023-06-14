@echo off
REM Copyright (c) Microsoft Corporation - All rights reserved
REM Licensed under the MIT License


SETLOCAL EnableDelayedExpansion

for /r %%F in (*.sln) do (
    echo Found .sln file: %%~dpF%%~nxF
    SET "SLN_PATH=%%~dpF%%~nxF"
    SET "MSBUILD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
    REM Download dependencies
    "!MSBUILD!" "!SLN_PATH!" -t:restore -p:RestorePackagesConfig=true
    REM Build project
    "!MSBUILD!" "!SLN_PATH!" /p:Configuration=Release /p:AdditionalLibraryDirectories="%WindowsSDKDir%\Lib" /p:AdditionalIncludeDirectories="%WindowsSDKDir%\Include" /p:OutDir="%SB_MICRO_PATH%\bin"
)

endlocal
