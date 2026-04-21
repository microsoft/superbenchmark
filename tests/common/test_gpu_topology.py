# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPU topology utilities."""

import json
import unittest
from unittest import mock

from superbench.common.utils.gpu_topology import (
    get_gpu_numa_affinity,
    get_gpu_numa_map,
)


class GpuTopologyTest(unittest.TestCase):
    """Test GPU topology utilities."""
    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_map(self, mock_run_command):
        """Test get_gpu_numa_map parses hy-smi output."""
        mock_run_command.return_value.returncode = 0
        mock_run_command.return_value.stdout = json.dumps(
            {
                'card0': {
                    '(Topology) Numa Node': '3',
                    '(Topology) Numa Affinity': '3',
                },
                'card1': {
                    '(Topology) Numa Node': '1',
                    '(Topology) Numa Affinity': '1,2',
                },
                'card2': {
                    '(Topology) Numa Node': '2',
                    '(Topology) Numa Affinity': '2-3',
                },
            }
        )

        self.assertEqual(
            get_gpu_numa_map(), {
                0: {
                    'numa_node': '3',
                    'numa_affinity': '3',
                },
                1: {
                    'numa_node': '1',
                    'numa_affinity': '1,2',
                },
                2: {
                    'numa_node': '2',
                    'numa_affinity': '2-3',
                },
            }
        )
        mock_run_command.assert_called_once_with('hy-smi --showtoponuma --json', quiet=True)

    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_map_command_failure(self, mock_run_command):
        """Test get_gpu_numa_map command failure."""
        mock_run_command.return_value.returncode = 1
        mock_run_command.return_value.stdout = 'hy-smi failed'

        with self.assertRaisesRegex(RuntimeError, 'Failed to get GPU NUMA topology from hy-smi'):
            get_gpu_numa_map()

    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_map_parse_failure(self, mock_run_command):
        """Test get_gpu_numa_map parse failure."""
        mock_run_command.return_value.returncode = 0
        mock_run_command.return_value.stdout = json.dumps({'card0': {}})

        with self.assertRaisesRegex(RuntimeError, 'Failed to parse GPU NUMA topology from hy-smi'):
            get_gpu_numa_map()

    @mock.patch('superbench.common.utils.gpu_topology.run_command')
    def test_get_gpu_numa_map_invalid_affinity(self, mock_run_command):
        """Test get_gpu_numa_map rejects invalid NUMA affinity."""
        mock_run_command.return_value.returncode = 0
        mock_run_command.return_value.stdout = json.dumps(
            {
                'card0': {
                    '(Topology) Numa Node': '0',
                    '(Topology) Numa Affinity': '0,a',
                },
            }
        )

        with self.assertRaisesRegex(RuntimeError, 'invalid NUMA node list'):
            get_gpu_numa_map()

    @mock.patch('superbench.common.utils.gpu_topology.get_gpu_numa_map')
    def test_get_gpu_numa_affinity(self, mock_get_gpu_numa_map):
        """Test get_gpu_numa_affinity returns GPU NUMA affinity."""
        mock_get_gpu_numa_map.return_value = {
            1: {
                'numa_node': '0',
                'numa_affinity': '1',
            },
        }

        self.assertEqual(get_gpu_numa_affinity(1), '1')
