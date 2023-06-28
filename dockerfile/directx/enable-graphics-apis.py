# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Enables graphics APIs in the Windows container."""
# Reference to
# https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Extras/Containers/Dockerfiles/windows/runtime/enable-graphics-apis.ps1

import os
import shutil
import glob


def copy_to_system32(source_directory, filenames, rename=None):
    """Copies the specified files from the source directory to the system32 directory."""
    for filename in filenames:
        source = os.path.join(source_directory, filename)
        destination = os.path.join('C:\\Windows\\System32', filename)
        if rename and filename in rename:
            renamed = rename[filename]
            destination = os.path.join('C:\\Windows\\System32', renamed)
        try:
            print(f'Copying {source} to {destination}')
            shutil.copy2(source, destination)
        except Exception as e:
            print(f'Warning: failed to copy file {filename}. Reason: {str(e)}')


# Attempt to locate the NVIDIA Display Driver directory in the host system's driver store
nvidia_sentinel_file = glob.glob('C:\\Windows\\System32\\HostDriverStore\\FileRepository\\nv*.inf_amd64_*\\nvapi64.dll')
if nvidia_sentinel_file:
    nvidia_directory = os.path.dirname(nvidia_sentinel_file[0])
    print(f'Found NVIDIA Display Driver directory: {nvidia_directory}')

    print('\nEnabling NVIDIA NVAPI support:')
    copy_to_system32(nvidia_directory, ['nvapi64.dll'])

    print('\nEnabling NVIDIA NVENC support:')
    copy_to_system32(nvidia_directory, ['nvEncodeAPI64.dll', 'nvEncMFTH264x.dll', 'nvEncMFThevcx.dll'])

    print('\nEnabling NVIDIA CUVID/NVDEC support:')
    copy_to_system32(
        nvidia_directory, ['nvcuvid64.dll', 'nvDecMFTMjpeg.dll', 'nvDecMFTMjpegx.dll'],
        {'nvcuvid64.dll': 'nvcuvid.dll'}
    )

    print('\nEnabling NVIDIA CUDA support:')
    copy_to_system32(
        nvidia_directory, ['nvcuda64.dll', 'nvcuda_loader64.dll', 'nvptxJitCompiler64.dll'],
        {'nvcuda_loader64.dll': 'nvcuda.dll'}
    )

    print('\n')

# Attempt to locate the AMD Display Driver directory in the host system's driver store
amd_sentinel_file = glob.glob('C:\\Windows\\System32\\HostDriverStore\\FileRepository\\u*.inf_amd64_*\\*\\aticfx64.dll')
if amd_sentinel_file:
    amd_directory = os.path.dirname(amd_sentinel_file[0])
    print(f'Found AMD Display Driver directory: {amd_directory}')

    print('\nCopying AMD DirectX driver files:')
    copy_to_system32(amd_directory, ['aticfx64.dll', 'atidxx64.dll'])

    print('\nEnabling AMD Display Library (ADL) support:')
    copy_to_system32(amd_directory, ['atiadlxx.dll', 'atiadlxy.dll'])

    print('\nEnabling AMD Advanced Media Framework (AMF) support:')
    copy_to_system32(amd_directory, ['amfrt64.dll', 'amfrtdrv64.dll', 'amdihk64.dll'])

    print('\n')
