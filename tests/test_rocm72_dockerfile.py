# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ROCm 7.2 Dockerfile build configuration."""

import os
import subprocess
import unittest
from pathlib import Path


class Rocm72DockerfileTestCase(unittest.TestCase):
    """Test ROCm 7.2 architecture routing without building an image."""
    @classmethod
    def setUpClass(cls):
        """Load the Dockerfile and extract its architecture-routing shell blocks."""
        repo_root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (repo_root / 'dockerfile' / 'rocm7.2.x.dockerfile').read_text(encoding='utf-8')
        cls.standalone_cmake_path = repo_root / 'dockerfile' / 'etc' / 'hipblaslt-bench-standalone.cmake'
        cls.hipblaslt_script = cls._extract_script(
            'hipblaslt_architectures=$(printf',
            '    mkdir -p build && cd build',
            'printf \'%s\\n\' "$@"\n',
        )
        cls.transformer_engine_script = cls._extract_script(
            'transformer_engine_architectures=$(printf',
            '    python3 -m pip install onnxscript==0.7.2 onnx==1.23.0',
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
        env.update({'AMDGPU_TARGETS': targets, 'NUM_MAKE_JOBS': '64', 'ROCM_PATH': '/opt/rocm'})
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
        """Return the expected standalone hipblaslt-bench CMake configure command."""
        architecture_flags = [f'-DCMAKE_HIP_ARCHITECTURES={architectures}'] if architectures else []
        return [
            'cmake',
            '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
            '-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++',
            *architecture_flags,
            '-DCMAKE_PREFIX_PATH=/opt/rocm;/usr/local',
            '-DBLAS_LIBRARIES=/usr/local/lib/libblas.a',
            '-DLAPACK_LIBRARIES=/usr/local/lib/liblapack.a',
            '-DCMAKE_BUILD_TYPE=Release',
            '..',
        ]

    def test_architecture_routing(self):
        """Test default, single, lower, mixed, whitespace-normalized, and empty target routing.

        An empty target list must omit -DCMAKE_HIP_ARCHITECTURES entirely, because CMake rejects a
        defined-but-empty value and only auto-detects when the variable is undefined.
        """
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

    def test_standalone_hipblaslt_cmake_is_used(self):
        """Test that the standalone hipblaslt-bench CMake script is shipped and copied into the build."""
        self.assertTrue(self.standalone_cmake_path.is_file())
        self.assertIn(
            'COPY dockerfile/etc/hipblaslt-bench-standalone.cmake /tmp/hipblaslt-bench-standalone.cmake',
            self.dockerfile,
        )
        self.assertIn('cp /tmp/hipblaslt-bench-standalone.cmake hipBLASLt/CMakeLists.txt', self.dockerfile)

    def test_rocm_sources_pinned_to_base_image_release(self):
        """Test that ROCm library sources are pinned to the tag matching the base image."""
        self.assertIn('rocm/pytorch:rocm7.2.4_', self.dockerfile)
        self.assertIn('git clone -b rocm-7.2.4 https://github.com/ROCmSoftwarePlatform/rccl.git', self.dockerfile)
        self.assertIn(
            'git clone --depth 1 -b rocm-7.2.4 https://github.com/ROCmSoftwarePlatform/hipBLASLt.git',
            self.dockerfile,
        )
        self.assertIn('ROCBLAS_BRANCH=rocm-7.2.4', self.dockerfile)
        self.assertNotIn('release/rocm-rel-7.2', self.dockerfile)

    def test_mlc_checksum_verification(self):
        """Test MLC download is checksum-verified and installed as an executable."""
        self.assertIn('wget -q https://downloadmirror.intel.com/926327/mlc_v3.13.tgz -O mlc.tgz', self.dockerfile)
        self.assertIn(
            'echo "a8537e8ff3fad626d75a383fabc224ccc4cc98a0111c9989f7fb26b639f12019  mlc.tgz" | sha256sum -c -',
            self.dockerfile,
        )
        self.assertIn('install -m 755 ./Linux/mlc /usr/local/bin/', self.dockerfile)

    def test_rocm_build_refreshes_package_index(self):
        """Test APT metadata is refreshed before installing ROCm dependencies."""
        build_command = 'RUN apt-get update && \\\n    make RCCL_HOME=/opt/rccl/build/ '
        self.assertIn(build_command, self.dockerfile)
        self.assertEqual(self.dockerfile.count('make RCCL_HOME='), 1)


if __name__ == '__main__':
    unittest.main()
