# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for CUDA 13.3 image build configuration."""

import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class Cuda133DockerfileTestCase(unittest.TestCase):
    """Test toolkit routing without downloading dependencies or building an image."""
    @classmethod
    def setUpClass(cls):
        """Locate the build recipes."""
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.makefile = cls.repo_root / 'third_party' / 'Makefile'

    @unittest.skipUnless(shutil.which('make') and shutil.which('bc'), 'CUTLASS recipe tests require GNU Make and bc.')
    def test_cutlass_toolkit_routing(self):
        """Keep CUDA 13.3's CUTLASS change separate from other toolkit versions."""
        configurations = (
            ('11.1', None, '70;75;80;86'),
            ('11.8', None, '70;75;80;86;89;90'),
            ('12.8', 'v3.9.2', '90;100'),
            ('12.9', 'v4.1.0', '100;103'),
            ('13.0', 'v4.1.0', '100;103'),
            ('13.2', 'v4.1.0', '100;103'),
            ('13.3', 'v4.7.0', '100;103'),
            ('13.4', 'v4.1.0', '100;103;107'),
            ('13.5', 'v4.1.0', '100;103;107'),
        )
        with tempfile.TemporaryDirectory() as directory:
            cutlass = Path(directory) / 'cutlass'
            cutlass.mkdir()
            (cutlass / 'CMakeLists.txt').touch()
            for version, tag, architectures in configurations:
                with self.subTest(cuda=version):
                    result = subprocess.run(
                        [
                            'make', '--dry-run', '--no-print-directory', '-f',
                            str(self.makefile), 'cuda_cutlass', 'CUDA_VER=' + version, 'ROCM_VER=0.0', 'NUM_MAKE_JOBS=2'
                        ],
                        cwd=directory,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        timeout=30,
                    )
                    self.assertEqual(result.stderr, '')
                    self.assertIn('-DCUTLASS_NVCC_ARCHS=' + architectures, shlex.split(result.stdout))
                    clones = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('git clone ')]
                    if tag is None:
                        self.assertEqual(clones, [])
                    else:
                        self.assertEqual(len(clones), 1)
                        self.assertEqual(clones[0][2:4], ['--branch', tag])
                    self.assertTrue((cutlass / 'CMakeLists.txt').is_file())
                    self.assertFalse((cutlass / 'build').exists())

    def test_mlc_matches_existing_cuda_image(self):
        """Reuse the existing CUDA image's checked MLC download and installation."""
        commands = []
        for version in ('13.0', '13.3'):
            path = self.repo_root / 'dockerfile' / ('cuda' + version + '.dockerfile')
            dockerfile = path.read_text(encoding='utf-8')
            block = dockerfile.split('# Install Intel MLC\n', 1)[1].split('# Install AOCC compiler\n', 1)[0]
            commands.append(shlex.split(block.replace('\\\n', ' ')))
        self.assertEqual(commands[0], commands[1])


if __name__ == '__main__':
    unittest.main()
