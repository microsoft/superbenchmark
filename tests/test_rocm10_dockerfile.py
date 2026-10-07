# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the experimental ROCm 10 image scaffold."""

import os
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path


class Rocm10DockerfileTestCase(unittest.TestCase):
    """Check package selection and executable architecture routing without Docker."""

    @classmethod
    def setUpClass(cls):
        """Load the Dockerfile and extract RCCL test architecture handling."""
        root = Path(__file__).resolve().parents[1]
        cls.dockerfile = (root / 'dockerfile' / 'rocm10.0.x.dockerfile').read_text(encoding='utf-8')
        env_start = cls.dockerfile.index("printf 'export ROCM_PATH=%q")
        env_end = cls.dockerfile.index('\n\n', env_start)
        cls.environment_script = cls.dockerfile[env_start:env_end]
        registration_start = cls.dockerfile.index("RUN python3 -c 'import sysconfig;")
        registration_end = cls.dockerfile.index('\n\n', registration_start)
        cls.registration_script = cls.dockerfile[registration_start + len('RUN '):registration_end]
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
            ('gfx942 gfx950 gfx1250', 'gfx942;gfx950;gfx1250'),
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

    def test_default_build_parallelism(self):
        """Default native builds to 64 jobs while retaining the framework build cap."""
        self.assertIn('ARG NUM_MAKE_JOBS=64', self.dockerfile)
        self.assertIn('CMAKE_BUILD_PARALLEL_LEVEL="${NUM_MAKE_JOBS}"', self.dockerfile)
        self.assertIn('if [ "${framework_jobs}" -gt 16 ]; then framework_jobs=16; fi', self.dockerfile)

    def test_all_device_sdk_uses_base_package_ecosystem(self):
        """Discover the installed wheel SDK without an alias or global loader override."""
        self.assertIn('rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:', self.dockerfile)
        self.assertIn('"rocm[devel,libraries,device-all]==${ROCM_VERSION}"', self.dockerfile)
        self.assertIn('rocm-sdk init', self.dockerfile)
        self.assertIn('ROCM_PATH="$(rocm-sdk path --root)"', self.dockerfile)
        self.assertNotIn('/opt/rocm/wheel', self.dockerfile)
        self.assertNotIn('ln -s "$(rocm-sdk path --root)"', self.dockerfile)
        self.assertNotIn('/etc/ld.so.conf.d/rocm.conf', self.dockerfile)
        self.assertIn('ENV BASH_ENV=/etc/profile.d/rocm10-env.sh', self.dockerfile)
        self.assertIn('ENTRYPOINT ["/bin/bash", "-e", "-c", "exec \\"$@\\"", "--"]', self.dockerfile)
        self.assertIn('CMD ["/bin/bash"]', self.dockerfile)

    def test_sdk_environment_uses_discovered_paths(self):
        """Activate the SDK for shell builds and arbitrary container commands."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk_root = str(root / 'sdk with spaces')
            bin_dir = root / 'venv' / 'bin'
            bin_dir.mkdir(parents=True)
            sdk = bin_dir / 'rocm-sdk'
            sdk.write_text('#!/bin/sh\n[ "$1" = path ] && [ "$2" = --root ] || exit 2\n'
                           'printf "%s\\n" "$TEST_SDK_ROOT"\n')
            sdk.chmod(0o755)
            env = os.environ.copy()
            env.pop('BASH_ENV', None)
            env.update({
                'VIRTUAL_ENV': str(root / 'venv'),
                'TEST_SDK_ROOT': sdk_root,
                'MPI_HOME': '/vendor/mpi',
                'SB_MICRO_PATH': '/superbench',
                'EXTRAS_PATH': '/vendor/extras',
                'ROCM_PATH': '/stale/sdk',
                'ROCM_HOME': '/stale/sdk',
                'HIP_PATH': '/stale/sdk',
                'RCCL_HOME': '/stale/sdk',
                'LD_LIBRARY_PATH': '/stale/sdk/lib',
                'PATH': f'{bin_dir}:/usr/bin:/bin',
            })
            activation = root / 'rocm10-env.sh'
            script = 'ROCM_PATH="$(rocm-sdk path --root)" && ' + self.environment_script.replace(
                '/etc/profile.d/rocm10-env.sh', str(activation)
            )
            subprocess.run(['/bin/bash', '-e', '-c', script], env=env, check=True)
            sdk.unlink()
            env.update({'BASH_ENV': str(activation), 'PATH': '/usr/bin:/bin'})
            result = subprocess.run(
                ['/bin/bash', '-e', '-c', 'exec "$@"', '--', '/usr/bin/env'],
                env=env, check=True, capture_output=True, text=True,
            )
            actual = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
            for name in ('ROCM_PATH', 'ROCM_HOME', 'HIP_PATH', 'RCCL_HOME'):
                self.assertEqual(actual[name], sdk_root)
            self.assertEqual(
                actual['PATH'], f'/vendor/extras/bin:/vendor/mpi/bin:/superbench/bin:{sdk_root}/bin:/usr/bin:/bin'
            )
            self.assertEqual(
                actual['LD_LIBRARY_PATH'],
                f'/vendor/extras/lib:/vendor/mpi/lib:{sdk_root}/lib:{sdk_root}/lib/llvm/lib:'
                f'{sdk_root}/lib/host-math/lib:{sdk_root}/lib/rocm_sysdeps/lib',
            )

    def test_sdk_discovery_failure_stops_environment_generation(self):
        """Never publish stale SDK paths if build-time discovery fails."""
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / 'bin'
            bin_dir.mkdir()
            sdk = bin_dir / 'rocm-sdk'
            sdk.write_text('#!/bin/sh\necho "SDK unavailable" >&2\nexit 7\n')
            sdk.chmod(0o755)
            env = os.environ.copy()
            env.update({'PATH': f'{bin_dir}:/usr/bin:/bin'})
            env.pop('BASH_ENV', None)
            activation = Path(directory) / 'rocm10-env.sh'
            script = 'ROCM_PATH="$(rocm-sdk path --root)" && ' + self.environment_script.replace(
                '/etc/profile.d/rocm10-env.sh', str(activation)
            )
            result = subprocess.run(
                ['/bin/bash', '-e', '-c', script],
                env=env, check=False, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 7)
            self.assertIn('SDK unavailable', result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertFalse(activation.exists())

    def test_rccl_uses_packaged_library_and_matched_test_sources(self):
        """Build only test clients and retain runtime library search paths."""
        self.assertIn('ARG AMDGPU_TARGETS="gfx942 gfx950 gfx1250"', self.dockerfile)
        self.assertIn('test -r "${ROCM_PATH}/lib/librccl.so"', self.dockerfile)
        self.assertIn('ARG ROCM_SYSTEMS_COMMIT=6b0e43f341195e203754e08f850e437ff2fc09f9', self.dockerfile)
        self.assertIn('-DGPU_TARGETS="${rccl_architectures}"', self.dockerfile)
        self.assertIn('-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', self.dockerfile)
        self.assertNotIn('https://github.com/ROCm/rccl.git', self.dockerfile)
        self.assertNotIn('https://github.com/ROCmSoftwarePlatform/rccl.git', self.dockerfile)
        self.assertNotIn('LD_PRELOAD=', self.dockerfile)

    def test_doca_uses_only_the_userspace_profile(self):
        """Use the matching Ubuntu release without installing host kernel drivers."""
        self.assertIn('ARG DOCA_VERSION=3.4.0', self.dockerfile)
        self.assertIn('ARG DOCA_PACKAGE_VERSION=3.4.0-085000', self.dockerfile)
        self.assertIn('-26.04-ubuntu2604_amd64.deb', self.dockerfile)
        self.assertIn('"doca-ofed-userspace=${DOCA_PACKAGE_VERSION}"', self.dockerfile)
        self.assertIn('"${DOCA_BUNDLE_SHA256}" /tmp/doca-host.deb | sha256sum --check --strict', self.dockerfile)
        self.assertIn('apt-get purge -y doca-host', self.dockerfile)
        self.assertNotIn('apt-get install -y doca-ofed ', self.dockerfile)
        self.assertNotIn('amdgpu-dkms', self.dockerfile)

    def test_mpi_reuses_the_doca_package(self):
        """Install vendor MPI before removing its repository, without a source build."""
        self.assertIn('ARG DOCA_OPENMPI_VERSION=5.0.10rc2.2605121430-1.b99be7132e', self.dockerfile)
        self.assertIn('"openmpi=${DOCA_OPENMPI_VERSION}"', self.dockerfile)
        self.assertIn("dpkg-query -L openmpi | sed -n 's#/bin/mpirun$##p'", self.dockerfile)
        self.assertIn('test -x "${mpi_prefix}/bin/mpicc"', self.dockerfile)
        self.assertIn('ln -s "${mpi_prefix}" "${MPI_HOME}"', self.dockerfile)
        self.assertLess(
            self.dockerfile.index('"openmpi=${DOCA_OPENMPI_VERSION}"'),
            self.dockerfile.index('apt-get purge -y doca-host'),
        )
        self.assertNotIn('ARG OPENMPI_VERSION=', self.dockerfile)
        self.assertNotIn('https://github.com/open-mpi/ompi.git', self.dockerfile)
        self.assertNotIn('--disable-oshmem', self.dockerfile)

    def test_prebuilt_blas_clients_keep_their_kernel_packs(self):
        """Keep helper kernels and SDK dependency paths without extracting another runtime."""
        self.assertIn('therock-dist-linux-multiarch-tests-${ROCM_VERSION}.tar.gz', self.dockerfile)
        self.assertIn("'./bin/rocblas-bench' './bin/hipblaslt-bench' './.kpack/blas_test_*.kpack'", self.dockerfile)
        for arch in ('gfx942', 'gfx950', 'gfx1250'):
            self.assertIn(f'test -s "${{SB_HOME}}/.kpack/blas_test_{arch}.kpack"', self.dockerfile)
        self.assertIn(
            '${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib',
            self.environment_script,
        )

    def test_transferbench_backend_is_inherited_by_ssh(self):
        """Preserve existing YAML while making the image's backend choice explicit."""
        self.assertIn('SB_ROCM_MEM_BW_BACKEND=transferbench', self.dockerfile)
        self.assertEqual(self.dockerfile.count('> /etc/environment'), 1)
        self.assertIn('"SB_ROCM_MEM_BW_BACKEND=${SB_ROCM_MEM_BW_BACKEND}"', self.dockerfile)

    def test_installation_order_and_prerequisites(self):
        """Follow the older images' layout while retaining ROCm 10 prerequisites."""
        steps = (
            'RUN apt-get update',
            'ARG DOCKER_VERSION=',
            'RUN mkdir -p /root/.ssh',
            'ARG DOCA_VERSION=',
            '"rocm[devel,libraries,device-all]==${ROCM_VERSION}"',
            "RUN python3 -c 'import sysconfig;",
            '"amdrocm10-rvs=${RVS_VERSION}"',
            'WORKDIR ${SB_HOME}',
            'COPY third_party third_party',
            'RUN git init /tmp/rocm-systems',
            'make -C third_party fio rocm_perftest',
            'ARG ROCM_TESTS_SHA256=',
            'COPY . .',
            "pip install -c /tmp/rocm10-constraints.txt '.[torch]'",
            'python3 -m pip install pytest pytest-timeout vcrpy',
            'bash dockerfile/etc/install-rocm10-frameworks.sh',
            '> /etc/environment',
            'CXX="${ROCM_PATH}/bin/hipcc" make cppbuild',
            'make postinstall',
            'RUN python3 -m pip check',
            'sb --help',
            'rm -rf .git',
        )
        positions = [self.dockerfile.index(step) for step in steps]
        self.assertEqual(positions, sorted(positions))

    def test_framework_runtime_configuration_survives_build_layers(self):
        """Keep the selected Megatron path and precise Triton FP32 mode at runtime."""
        self.assertIn('SB_MEGATRON_PATH=/opt/rocm10-frameworks/Megatron-LM', self.dockerfile)
        self.assertIn('AITER_USE_SYSTEM_TRITON=1', self.dockerfile)
        self.assertIn('TRITON_F32_DEFAULT=ieee', self.dockerfile)
        self.assertIn('bash dockerfile/etc/install-rocm10-frameworks.sh', self.dockerfile)
        for name in (
            'ROCM_PATH', 'ROCM_HOME', 'HIP_PATH', 'RCCL_HOME',
            'SB_MEGATRON_PATH', 'AITER_USE_SYSTEM_TRITON', 'TRITON_F32_DEFAULT', 'MAX_JOBS',
            'CUDA_DEVICE_MAX_CONNECTIONS', 'NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN',
            'NVTE_FUSED_ATTN_CK', 'NVTE_FUSED_ATTN_AOTRITON'
        ):
            self.assertIn(f'"{name}=${{{name}}}"', self.dockerfile)
        self.assertIn('CUDA_DEVICE_MAX_CONNECTIONS=1', self.dockerfile)
        self.assertIn('NVTE_FUSED_ATTN_CK=1', self.dockerfile)
        self.assertIn('NVTE_FUSED_ATTN_AOTRITON=0', self.dockerfile)
        self.assertLess(
            self.dockerfile.index("RUN python3 -c 'import sysconfig;"),
            self.dockerfile.index('bash dockerfile/etc/install-rocm10-frameworks.sh'),
        )

    def test_mpi_prefix_and_scaffold_boundaries(self):
        """Keep the MPI prefix while avoiding unqualified legacy dependencies."""
        self.assertIn('MPI_HOME=/usr/local/mpi', self.dockerfile)
        self.assertNotIn('--with-rocm=', self.dockerfile)
        self.assertIn('make -C third_party fio rocm_perftest', self.dockerfile)
        self.assertNotIn('ROCM_VER=rocm-5.5.0', self.dockerfile)
        self.assertNotIn('https://github.com/ROCm/TransformerEngine.git', self.dockerfile)
        self.assertIn("pip install -c /tmp/rocm10-constraints.txt '.[torch]'", self.dockerfile)
        self.assertIn('python3 -m pip check', self.dockerfile)

    def test_rvs_uses_signed_apt_repositories(self):
        """Install RVS and matching dependencies with APT, not a tarball or forced extraction."""
        self.assertIn('ARG RVS_VERSION=1.6.122-708', self.dockerfile)
        self.assertIn('ARG ROCM_APT_VERSION=10.0.0-4', self.dockerfile)
        self.assertIn('https://stable.repo.amd.com/rocm/extras/rvs/packages/ubuntu2604/', self.dockerfile)
        self.assertIn('https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/', self.dockerfile)
        self.assertIn('Signed-By: /etc/apt/keyrings/amdrocm.gpg', self.dockerfile)
        self.assertIn('"amdrocm10-rvs=${RVS_VERSION}"', self.dockerfile)
        for component in ('runtime', 'base', 'amdsmi', 'blas', 'rand', 'llvm'):
            self.assertIn(f'"amdrocm-{component}=${{ROCM_APT_VERSION}}"', self.dockerfile)
        self.assertIn('EXTRAS_PATH=/opt/rocm/extras-10', self.dockerfile)
        self.assertIn('test -x "${EXTRAS_PATH}/bin/TransferBench"', self.dockerfile)
        self.assertIn('LD_LIBRARY_PATH="${EXTRAS_PATH}/lib:', self.environment_script)
        self.assertNotIn('/rvs/tarball/', self.dockerfile)
        self.assertNotIn('RVS_SHA256', self.dockerfile)
        self.assertNotIn('/tmp/rvs.tar.gz', self.dockerfile)

    def test_perftest_uses_the_configured_sdk(self):
        """Honor the wheel SDK prefix while preserving other images' /opt/rocm default."""
        root = Path(__file__).resolve().parents[1]
        makefile = (root / 'third_party' / 'Makefile').read_text(encoding='utf-8')
        self.assertIn('--enable-rocm --with-rocm="$(ROCM_PATH)"', makefile)
        self.assertIn('ROCM_PATH ?= /opt/rocm', makefile)

    def test_amdsmi_uses_the_core_bindings_in_place(self):
        """Preserve the SDK's binding/library layout without copying or patching either."""
        self.assertIn('https://github.com/ROCm/TheRock/issues/3054', self.dockerfile)
        self.assertIn('https://github.com/ROCm/TheRock/pull/8424', self.dockerfile)
        self.assertIn('Path(sysconfig.get_path("purelib"), "amdsmi.pth")', self.registration_script)
        self.assertIn('.write_text("_rocm_sdk_core/share/amd_smi\\n")', self.registration_script)
        self.assertNotIn('--editable', self.dockerfile)
        self.assertNotIn("'.[amdworker]'", self.dockerfile)
        self.assertNotIn('version("amdsmi")', self.dockerfile)
        self.assertNotIn('pip install "${ROCM_PATH}/share/amd_smi"', self.dockerfile)
        self.assertNotIn('smi_library=', self.dockerfile)
        self.assertNotIn('ln -sf', self.dockerfile)

    def test_amdsmi_registration_makes_bundled_module_importable(self):
        """Execute the real build step in an isolated interpreter with an SDK payload."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv.EnvBuilder(with_pip=False).create(root)
            python = root / 'bin' / 'python3'
            env = os.environ.copy()
            for name in ('PYTHONPATH', 'PYTHONHOME', 'BASH_ENV'):
                env.pop(name, None)
            env['PATH'] = f'{python.parent}:/usr/bin:/bin'
            purelib = Path(subprocess.check_output(
                [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
                env=env, text=True,
            ).strip())
            module = purelib / '_rocm_sdk_core' / 'share' / 'amd_smi' / 'amdsmi'
            module.mkdir(parents=True)
            (module / '__init__.py').write_text('VERSION = "sdk-bundled"\n')
            subprocess.run(['/bin/bash', '-e', '-c', self.registration_script], env=env, check=True)
            self.assertEqual((purelib / 'amdsmi.pth').read_text(), '_rocm_sdk_core/share/amd_smi\n')
            imported = subprocess.check_output(
                [str(python), '-c', 'import amdsmi; print(amdsmi.__file__)'], env=env, text=True,
            ).strip()
            self.assertEqual(Path(imported), module / '__init__.py')


if __name__ == '__main__':
    unittest.main()
