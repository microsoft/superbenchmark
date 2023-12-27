FROM mcr.microsoft.com/windows:2004


# Install Python and additional packages
# Download Python
ADD https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe python-installer.exe
# Install Python
RUN python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 && DEL python-installer.exe
# Verify Python Was Successfully Installed
RUN python --version && \
    python -m ensurepip --upgrade

# Install choco and install some necessary packages
RUN powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; \
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
RUN choco install -y vcredist-all vim git make

# Retrieve the DirectX runtime files required by the Unreal Engine, since even the full Windows base image does not include them
RUN mkdir C:\GatheredDlls
RUN curl -s -L "https://download.microsoft.com/download/8/4/A/84A35BF1-DAFE-4AE8-82AF-AD2AE20B6B14/directx_Jun2010_redist.exe" --output %TEMP%\directx_redist.exe && \
    start /wait %TEMP%\directx_redist.exe /Q /T:%TEMP%\DirectX && \
    expand %TEMP%\DirectX\APR2007_xinput_x64.cab -F:xinput1_3.dll C:\GatheredDlls\ && \
    expand %TEMP%\DirectX\Feb2010_X3DAudio_x64.cab -F:X3DAudio1_7.dll C:\GatheredDlls\ && \
    expand %TEMP%\DirectX\Jun2010_D3DCompiler_43_x64.cab -F:D3DCompiler_43.dll C:\GatheredDlls\ && \
    expand %TEMP%\DirectX\Jun2010_XAudio_x64.cab -F:XAudio2_7.dll C:\GatheredDlls\ && \
    expand %TEMP%\DirectX\Jun2010_XAudio_x64.cab -F:XAPOFX1_5.dll C:\GatheredDlls\ && \
    break

# Retrieve the DirectX shader compiler files needed for DirectX Raytracing (DXR)
RUN curl -s -L "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.6.2104/dxc_2021_04-20.zip" --output %TEMP%\dxc.zip && \
    powershell -Command "Expand-Archive -Path \"$env:TEMP\dxc.zip\" -DestinationPath $env:TEMP" && \
    xcopy /y %TEMP%\bin\x64\dxcompiler.dll C:\GatheredDlls\ && \
    xcopy /y %TEMP%\bin\x64\dxil.dll C:\GatheredDlls\ && \
    break

# Copy the required DLLs to System32 dir
RUN xcopy C:\GatheredDlls\* C:\windows\System32\ /i

ENV SB_HOME="C:/superbench" \
    SB_MICRO_PATH="C:/superbench" \
    WindowsSDKDir="\\Program Files (x86)\\Windows Kits\\10\\"

RUN setx INCLUDE "%include%;%WindowsSDKDir%\\Include" /M && \
    setx LIB "%lib%;%WindowsSDKDir%\\Lib" /M && \
    setx PATH "%path%;%SB_MICRO_PATH%\\bin" /M

WORKDIR ${SB_HOME}
COPY ./ ${SB_HOME}

# Download vs_BuildTools.exe if not already present
RUN mkdir "%SB_MICRO_PATH%/bin"
RUN curl -s -L https://dist.nuget.org/win-x86-commandline/latest/nuget.exe -o "%SB_MICRO_PATH%/bin/nuget.exe"
# Run the setup script to install the visual studio components
RUN "%SB_HOME%\\dockerfile\\directx\\install-components.bat"

RUN powershell -Command "Set-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1;"
RUN git config --system core.longpaths true
# Install Superbench
RUN python -m pip install setuptools==65.0.0 && \
    python -m pip install --no-cache-dir .[amdworker] && \
    make directxbuild

ADD third_party third_party
RUN make -C third_party directx_amd

# Run the entrypoint script for enabling vendor-specific graphics APIs
RUN powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine -Force"
ENTRYPOINT [ "python", "dockerfile/directx/enable-graphics-apis.py" ]
CMD [ "cmd.exe" ]
