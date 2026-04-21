# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI node handler test."""

import io
import unittest
import contextlib
from unittest import mock

from knack.util import CLIError

import superbench.cli._node_handler as node_handler


class CLINodeHandlerTestCase(unittest.TestCase):
    """A class for node handler test cases."""
    @mock.patch('superbench.cli._node_handler.get_gpu_numa_map')
    def test_topo_command_handler_gpu_numa_map(self, mock_get_gpu_numa_map):
        """Test topo command handler gets GPU NUMA map."""
        mock_get_gpu_numa_map.return_value = {
            1: {
                'numa_node': '0',
                'numa_affinity': '1',
            },
        }
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            node_handler.topo_command_handler(get='gpu-numa-map')

        self.assertEqual(stdout.getvalue(), '{"1": {"numa_node": "0", "numa_affinity": "1"}}\n')

    @mock.patch('superbench.cli._node_handler.get_gpu_numa_affinity')
    def test_topo_command_handler_gpu_numa_affinity(self, mock_get_gpu_numa_affinity):
        """Test topo command handler gets GPU NUMA affinity."""
        mock_get_gpu_numa_affinity.return_value = '1'
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            node_handler.topo_command_handler(get='gpu-numa-affinity', gpu_id=1)

        self.assertEqual(stdout.getvalue(), '1\n')
        mock_get_gpu_numa_affinity.assert_called_once_with(1)

    def test_topo_command_handler_invalid_get(self):
        """Test topo command handler rejects invalid get value."""
        with self.assertRaises(CLIError):
            node_handler.topo_command_handler(get='invalid', gpu_id=1)

    def test_topo_command_handler_missing_gpu_id(self):
        """Test topo command handler requires gpu_id."""
        with self.assertRaises(CLIError):
            node_handler.topo_command_handler(get='gpu-numa-affinity')
