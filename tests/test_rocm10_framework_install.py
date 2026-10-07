# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Guard the ROCm10-only source installer against unpinned or CUDA wheel substitutions."""

from pathlib import Path
import re
import subprocess


INSTALLER = Path(__file__).resolve().parents[1] / 'dockerfile/etc/install-rocm10-frameworks.sh'


def test_framework_source_pins():
    """Every framework checkout must use an immutable source revision."""
    source = INSTALLER.read_text()
    for name in ['TE_COMMIT', 'AITER_COMMIT', 'MEGATRON_COMMIT']:
        assert re.search(r'^readonly {}=[0-9a-f]{{40}}$'.format(name), source, re.MULTILINE)
    assert 'https://github.com/ROCm/TransformerEngine.git' in source
    assert 'https://github.com/ROCm/aiter.git' in source
    assert '--no-build-isolation --no-deps' in source
    assert 'pip install transformer_engine' not in source


def test_framework_sdk_and_architecture_guards():
    """Preserve the installed SDK/Triton and compile the requested local targets."""
    source = INSTALLER.read_text()
    assert 'AITER_USE_SYSTEM_TRITON=1' in source
    assert 'PIP_CONSTRAINT=' in source
    assert '"torch": "2.13.0+rocm10.0.0"' in source
    assert '"triton": "3.8.0+git4cff872c.rocm10.0.0"' in source
    assert '"rocm-sdk-core": "10.0.0"' in source
    assert '"rocm-sdk-devel": "10.0.0"' in source
    assert 'ROCM_PATH="$(rocm-sdk path --root)"' in source
    assert "NVTE_ROCM_ARCH='gfx942;gfx950;gfx1250'" in source
    assert 'MAX_JOBS > 16' in source
    assert 'megatron_core_0.19.2_rocm10.patch' in source


def test_framework_installer_shell_syntax():
    """Check shell syntax without installing packages or requiring a GPU."""
    subprocess.run(['bash', '-n', str(INSTALLER)], check=True)
