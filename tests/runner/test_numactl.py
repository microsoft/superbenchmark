# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for numactl command builder."""

import unittest

from omegaconf import OmegaConf

from superbench.runner.numactl import get_local_numactl_command


class NumactlTestCase(unittest.TestCase):
    """A class for numactl command builder test cases."""
    def test_get_local_numactl_command_without_config(self):
        """Test no numactl command is generated without config."""
        mode = OmegaConf.create({
            'name': 'local',
            'proc_num': 2,
            'proc_rank': 1,
        })

        self.assertEqual(get_local_numactl_command(mode), ('', ''))

    def test_get_local_numactl_command_gpu_affinity(self):
        """Test GPU affinity generates setup command and node bindings."""
        mode = OmegaConf.create(
            {
                'name': 'local',
                'proc_num': 2,
                'proc_rank': 1,
                'numactl': {
                    'cpunodebind': 'gpu_affinity',
                    'membind': 'gpu_affinity',
                },
            }
        )

        self.assertEqual(
            get_local_numactl_command(mode), (
                'SB_GPU_NUMA_AFFINITY=$(sb node topo --get gpu-numa-affinity --gpu-id 1)',
                'numactl -N ${SB_GPU_NUMA_AFFINITY} -m ${SB_GPU_NUMA_AFFINITY}',
            )
        )

    def test_get_local_numactl_command_template_values(self):
        """Test template values are formatted with local process variables."""
        mode = OmegaConf.create(
            {
                'name': 'local',
                'proc_num': 8,
                'proc_rank': 6,
                'numactl': {
                    'cpunodebind': '$(({proc_rank}/2))',
                    'membind': '$(({proc_num}/4))',
                    'physcpubind': '$(({proc_rank}*16))-$(({proc_rank}*16+15))',
                },
            }
        )

        self.assertEqual(
            get_local_numactl_command(mode), ('', 'numactl -N $((6/2)) -m $((8/4)) -C $((6*16))-$((6*16+15))')
        )

    def test_get_local_numactl_command_list_values(self):
        """Test list values are formatted as numactl node and CPU lists."""
        mode = OmegaConf.create(
            {
                'name': 'local',
                'proc_num': 8,
                'proc_rank': 6,
                'numactl': {
                    'cpunodebind': [0, 1],
                    'membind': ['{proc_rank}', 7],
                    'physcpubind': ['0-15', '32-47'],
                },
            }
        )

        self.assertEqual(get_local_numactl_command(mode), ('', 'numactl -N 0,1 -m 6,7 -C 0-15,32-47'))

    def test_get_local_numactl_command_disabled_values(self):
        """Test disabled values do not generate numactl options."""
        mode = OmegaConf.create(
            {
                'name': 'local',
                'proc_num': 2,
                'proc_rank': 1,
                'numactl': {
                    'cpunodebind': 'none',
                    'membind': False,
                    'physcpubind': None,
                },
            }
        )

        self.assertEqual(get_local_numactl_command(mode), ('', ''))

    def test_get_local_numactl_command_rejects_gpu_affinity_for_physcpubind(self):
        """Test physcpubind rejects GPU affinity."""
        mode = OmegaConf.create(
            {
                'name': 'local',
                'proc_num': 2,
                'proc_rank': 1,
                'numactl': {
                    'physcpubind': 'gpu_affinity',
                },
            }
        )

        with self.assertRaisesRegex(ValueError, 'gpu_affinity is not supported for numactl.physcpubind'):
            get_local_numactl_command(mode)
