# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ROCm 6.4 Dockerfile build configuration."""

import os
import subprocess
import unittest
from pathlib import Path


class Rocm64DockerfileTestCase(unittest.TestCase):
    """Test ROCm 6.4 architecture routing without building an image."""
    @classmethod
    def setUpClass(cls):
        """Load the Dockerfile and extract its architecture-routing shell blocks."""
        repo_root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (repo_root / 'dockerfile' / 'rocm6.4.x.dockerfile').read_text(encoding='utf-8')
        cls.hipblaslt_script = cls._extract_script(
            'hipblaslt_architectures=$(printf',
            '    cd hipBLASLt && "$@"',
            'printf \'%s\\n\' "$@"\n',
        )
        cls.transformer_engine_script = cls._extract_script(
            'transformer_engine_architectures=$(printf',
            '    git clone --recursive https://github.com/ROCm/TransformerEngine.git',
            'printf \'%s\\n\' "$transformer_engine_architectures" "$nvte_fused_attn_aotriton"\n',
        )

    @classmethod
    def _extract_script(cls, start_marker, end_marker, result_command):
        """Extract an executable shell block between two Dockerfile markers."""
        start = cls.dockerfile.index(start_marker)
        end = cls.dockerfile.index(end_marker, start)
        return cls.dockerfile[start:end].replace('\\\n', '\n') + result_command

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

    def test_architecture_routing(self):
        """Test default, lower, mixed, and whitespace-normalized target routing."""
        configurations = (
            (
                'gfx942',
                ['./install.sh', '-dc', '-j', '64', '-a', 'gfx942', '--logic-yaml-filter', 'aquavanjaram/gfx942/**/*'],
                ['gfx942', '1'],
            ),
            (
                'gfx90a',
                ['./install.sh', '-dc', '-j', '64', '-a', 'gfx90a', '--logic-yaml-filter', 'aldebaran/**/*'],
                ['gfx90a', '0'],
            ),
            (
                'gfx908',
                ['./install.sh', '-dc', '-j', '64', '-a', 'gfx908', '--logic-yaml-filter', 'arcturus/**/*'],
                ['gfx908', '0'],
            ),
            (
                'gfx908 gfx90a gfx942',
                ['./install.sh', '-dc', '-j', '64', '-a', 'gfx908;gfx90a;gfx942'],
                ['gfx908;gfx90a;gfx942', '0'],
            ),
            (
                '  gfx908   gfx90a\tgfx942  ',
                ['./install.sh', '-dc', '-j', '64', '-a', 'gfx908;gfx90a;gfx942'],
                ['gfx908;gfx90a;gfx942', '0'],
            ),
        )

        for targets, expected_hipblaslt, expected_transformer_engine in configurations:
            with self.subTest(targets=targets):
                self.assertEqual(expected_hipblaslt, self._run_script(self.hipblaslt_script, targets))
                self.assertEqual(
                    expected_transformer_engine,
                    self._run_script(self.transformer_engine_script, targets),
                )

    def test_transformer_engine_receives_routed_configuration(self):
        """Test that TransformerEngine receives the computed architecture settings."""
        self.assertIn('ARG AMDGPU_TARGETS="gfx942"', self.dockerfile)
        self.assertIn('NVTE_FUSED_ATTN_AOTRITON="${nvte_fused_attn_aotriton}"', self.dockerfile)
        self.assertIn('NVTE_ROCM_ARCH="${transformer_engine_architectures}"', self.dockerfile)


if __name__ == '__main__':
    unittest.main()
