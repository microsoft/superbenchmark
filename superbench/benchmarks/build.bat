@echo off
REM Copyright (c) Microsoft Corporation - All rights reserved
REM Licensed under the MIT License


SETLOCAL EnableDelayedExpansion

for /r %%F in (*.vcxproj) do (
    echo Found .vcxproj file: %%~dpF%%~nxF
    SET "PROJ_PATH=%%~dpF%%~nxF"
    SET "MSBUILD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
    REM Download dependencies
    "!MSBUILD!" "!PROJ_PATH!" -t:restore -p:RestorePackagesConfig=true
    REM Build project
    "!MSBUILD!" "!PROJ_PATH!" /p:Configuration=Release /p:Platform=x64 /p:AdditionalLibraryDirectories="%WindowsSDKDir%\Lib" /p:AdditionalIncludeDirectories="%WindowsSDKDir%\Include" /p:OutDir="%SB_MICRO_PATH%\bin"
)

endlocal
