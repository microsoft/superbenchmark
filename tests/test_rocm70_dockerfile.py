# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ROCm 7.0 Dockerfile build configuration."""

import os
import subprocess
import unittest
from pathlib import Path


class Rocm70DockerfileTestCase(unittest.TestCase):
    """Test ROCm 7.0 architecture routing without building an image."""
    @classmethod
    def setUpClass(cls):
        """Load the Dockerfile and extract its architecture-routing shell blocks."""
        repo_root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (repo_root / 'dockerfile' / 'rocm7.0.x.dockerfile').read_text(encoding='utf-8')
        cls.hipblaslt_script = cls._extract_script(
            'hipblaslt_architectures=$(printf',
            '    mkdir -p build/release && cd build/release',
            'printf \'%s\\n\' "$@"\n',
        )
        cls.transformer_engine_script = cls._extract_script(
            'transformer_engine_architectures=$(printf',
            '    git clone --recursive -b v2.4_rocm https://github.com/ROCm/TransformerEngine.git',
            'printf \'%s\\n\' "$transformer_engine_architectures" "$nvte_fused_attn_aotriton"\n',
        )

    @classmethod
    def _extract_script(cls, start_marker, end_marker, result_command):
        """Extract an executable shell block between two Dockerfile markers."""
        start = cls.dockerfile.index(start_marker)
        end = cls.dockerfile.index(end_marker, start)
        return cls.dockerfile[start:end] + result_command

    def _run_script(self, script, targets):
        """Run an extracted routing block for the requested targets."""
        env = os.environ.copy()
        env.update({'AMDGPU_TARGETS': targets, 'NUM_MAKE_JOBS': '64'})
        result = subprocess.run(
            ['/bin/sh', '-c', script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
        )
        return result.stdout.splitlines()

    def _expected_hipblaslt_command(self, architectures):
        """Return the expected hipblaslt-bench CMake configure command."""
        return [
            'cmake',
            f'-DAMDGPU_TARGETS={architectures}',
            '-DHIPBLASLT_USE_ROCROLLER=OFF',
            '-DBUILD_CLIENTS_BENCHMARKS=ON',
            '-DBUILD_CLIENTS_TESTS=OFF',
            '-DBUILD_CLIENTS_SAMPLES=OFF',
            '-DTensile_SKIP_BUILD=ON',
            '-DCMAKE_PREFIX_PATH=/opt/rocm;/usr/local',
            '-DCMAKE_BUILD_TYPE=Release',
            '../..',
        ]

    def test_architecture_routing(self):
        """Test default, single, lower, mixed, and whitespace-normalized target routing."""
        configurations = (
            ('gfx942 gfx950', 'gfx942;gfx950', ['gfx942;gfx950', '1']),
            ('gfx942', 'gfx942', ['gfx942', '1']),
            ('gfx950', 'gfx950', ['gfx950', '1']),
            ('gfx90a', 'gfx90a', ['gfx90a', '0']),
            ('gfx90a gfx942 gfx950', 'gfx90a;gfx942;gfx950', ['gfx90a;gfx942;gfx950', '0']),
            ('  gfx942\t  gfx950  ', 'gfx942;gfx950', ['gfx942;gfx950', '1']),
            ('', '', ['', '0']),
        )

        for targets, expected_architectures, expected_transformer_engine in configurations:
            with self.subTest(targets=targets):
                self.assertEqual(
                    self._expected_hipblaslt_command(expected_architectures),
                    self._run_script(self.hipblaslt_script, targets),
                )
                self.assertEqual(
                    expected_transformer_engine,
                    self._run_script(self.transformer_engine_script, targets),
                )

    def test_transformer_engine_receives_routed_configuration(self):
        """Test that TransformerEngine receives the computed architecture settings."""
        self.assertIn('ARG AMDGPU_TARGETS="gfx942 gfx950"', self.dockerfile)
        self.assertIn('NVTE_FUSED_ATTN_AOTRITON="${nvte_fused_attn_aotriton}"', self.dockerfile)
        self.assertIn('NVTE_ROCM_ARCH="${transformer_engine_architectures}"', self.dockerfile)


if __name__ == '__main__':
    unittest.main()
