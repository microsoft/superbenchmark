# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the experimental ROCm 10 image scaffold."""

import os
import subprocess
import unittest
from pathlib import Path


class Rocm10DockerfileTestCase(unittest.TestCase):
    """Check package selection and executable architecture routing without Docker."""

    @classmethod
    def setUpClass(cls):
        """Load the Dockerfile and extract RCCL test architecture handling."""
        root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (root / 'dockerfile' / 'rocm10.0.x.dockerfile').read_text(encoding='utf-8')
        start = cls.dockerfile.index('rccl_architectures=$(printf')
        end = cls.dockerfile.index('    cmake -S /tmp/rocm-systems', start)
        cls.routing = cls.dockerfile[start:end] + 'printf \'%s\\n\' "$rccl_architectures"\n'

    def run_routing(self, targets):
        """Execute the real Dockerfile routing block with a supplied target list."""
        env = os.environ.copy()
        env['AMDGPU_TARGETS'] = targets
        return subprocess.run(
            ['/bin/bash', '-e', '-o', 'pipefail', '-c', self.routing],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )

    def test_architecture_routing(self):
        """Normalize single, multiple, and whitespace-separated architectures."""
        for targets, expected in (
            ('gfx942 gfx950', 'gfx942;gfx950'),
            ('gfx942', 'gfx942'),
            ('gfx90a gfx942 gfx950 gfx1250', 'gfx90a;gfx942;gfx950;gfx1250'),
            ('  gfx942\t gfx950\n', 'gfx942;gfx950'),
        ):
            with self.subTest(targets=targets):
                result = self.run_routing(targets)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_empty_architectures_fail_explicitly(self):
        """Do not fall back to the build host's detected GPU architecture."""
        for targets in ('', ' \t\n'):
            with self.subTest(targets=targets):
                result = self.run_routing(targets)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('AMDGPU_TARGETS must contain at least one GPU architecture.', result.stderr)

    def test_all_device_sdk_uses_base_package_ecosystem(self):
        """Keep all-device wheels rather than installing a second native SDK."""
        self.assertIn('rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:', self.dockerfile)
        self.assertIn('"rocm[devel,libraries,device-all]==${ROCM_VERSION}"', self.dockerfile)
        self.assertIn('rocm-sdk init', self.dockerfile)
        self.assertIn('ln -s "$(rocm-sdk path --root)" "${ROCM_PATH}"', self.dockerfile)
        self.assertNotIn('apt install -y amdrocm', self.dockerfile)

    def test_rccl_uses_packaged_library_and_matched_test_sources(self):
        """Build only test clients and retain runtime library search paths."""
        self.assertIn('test -r "${ROCM_PATH}/lib/librccl.so"', self.dockerfile)
        self.assertIn('ARG ROCM_SYSTEMS_COMMIT=6b0e43f341195e203754e08f850e437ff2fc09f9', self.dockerfile)
        self.assertIn('-DGPU_TARGETS="${rccl_architectures}"', self.dockerfile)
        self.assertIn('-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', self.dockerfile)
        self.assertNotIn('https://github.com/ROCm/rccl.git', self.dockerfile)
        self.assertNotIn('https://github.com/ROCmSoftwarePlatform/rccl.git', self.dockerfile)
        self.assertNotIn('LD_PRELOAD=', self.dockerfile)

    def test_legacy_mpi_and_scaffold_boundaries(self):
        """Keep the MPI prefix while avoiding unqualified legacy dependencies."""
        self.assertIn('MPI_HOME=/usr/local/mpi', self.dockerfile)
        self.assertIn('ARG OPENMPI_VERSION=4.1.8', self.dockerfile)
        self.assertNotIn('--with-rocm=', self.dockerfile)
        self.assertIn('make -C third_party fio rocm_perftest', self.dockerfile)
        self.assertNotIn('ROCM_VER=rocm-5.5.0', self.dockerfile)
        self.assertNotIn('https://github.com/ROCm/TransformerEngine.git', self.dockerfile)
        self.assertIn("pip install -c /tmp/rocm10-constraints.txt '.[amdworker]'", self.dockerfile)
        self.assertIn('python3 -m pip check', self.dockerfile)

    def test_extras_do_not_install_another_sdk(self):
        """Pin the relocatable RVS archive and expose its bundled TransferBench."""
        self.assertIn('ARG RVS_VERSION=1.6.122-708', self.dockerfile)
        self.assertIn('sha256sum --check --strict', self.dockerfile)
        self.assertIn('tar -xzf /tmp/rvs.tar.gz -C "${EXTRAS_PATH}"', self.dockerfile)
        self.assertIn('test -x "${EXTRAS_PATH}/bin/TransferBench"', self.dockerfile)
        self.assertIn('LD_LIBRARY_PATH="${EXTRAS_PATH}/lib:${LD_LIBRARY_PATH}"', self.dockerfile)
        self.assertNotIn('apt install -y amdrocm10-rvs', self.dockerfile)

    def test_amdsmi_bindings_share_the_core_library(self):
        """Avoid independent AMD SMI initialization states in PyTorch and SuperBench."""
        self.assertIn(
            'distribution("rocm-sdk-core").locate_file("_rocm_sdk_core/lib/libamd_smi.so.27")',
            self.dockerfile,
        )
        self.assertIn('cmp "${ROCM_PATH}/lib/libamd_smi.so" "${smi_library}"', self.dockerfile)
        self.assertIn('ln -sf "${smi_library}" "${ROCM_PATH}/lib/libamd_smi.so"', self.dockerfile)
        self.assertIn('ln -sf "${smi_library}" "${ROCM_PATH}/lib/libamd_smi.so.27"', self.dockerfile)


if __name__ == '__main__':
    unittest.main()
