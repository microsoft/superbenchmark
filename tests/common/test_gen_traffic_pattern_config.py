# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for traffic pattern config generation module."""
import argparse
import unittest
import tempfile

from tests.helper import decorator
from superbench.common.utils import gen_traffic_pattern_host_groups


class GenConfigTest(unittest.TestCase):
    """Test the utils for generating config."""
    @decorator.load_data('tests/data/mpi_pattern.txt')    # noqa: C901
    @decorator.load_data('tests/data/ib_traffic_topo_aware_hostfile')    # noqa: C901
    def test_gen_traffic_pattern_host_group(self, expected_mpi_pattern, tp_hostfile):
        """Test the function of generating traffic pattern config from specified mode."""
        # Test for all-nodes pattern
        test_config_file = tempfile.NamedTemporaryFile()
        test_config_path = test_config_file.name
        test_benchmark_name = 'test_benchmark'
        hostx = ['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='all-nodes',
        )
        parser.add_argument(
            '--mpi_pattern',
            type=bool,
            default=True,
        )
        pattern, _ = parser.parse_known_args()
        expected_host_group = [[['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']]]
        self.assertEqual(
            gen_traffic_pattern_host_groups(hostx, pattern, test_config_path, test_benchmark_name), expected_host_group
        )

        # Test for pair-wise pattern
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='pair-wise',
        )
        parser.add_argument(
            '--mpi_pattern',
            type=bool,
            default=True,
        )
        pattern, _ = parser.parse_known_args()
        expected_host_group = [
            [['node0', 'node7'], ['node1', 'node6'], ['node2', 'node5'], ['node3', 'node4']],
            [['node0', 'node1'], ['node2', 'node7'], ['node3', 'node6'], ['node4', 'node5']],
            [['node0', 'node2'], ['node3', 'node1'], ['node4', 'node7'], ['node5', 'node6']],
            [['node0', 'node3'], ['node4', 'node2'], ['node5', 'node1'], ['node6', 'node7']],
            [['node0', 'node4'], ['node5', 'node3'], ['node6', 'node2'], ['node7', 'node1']],
            [['node0', 'node5'], ['node6', 'node4'], ['node7', 'node3'], ['node1', 'node2']],
            [['node0', 'node6'], ['node7', 'node5'], ['node1', 'node4'], ['node2', 'node3']]
        ]
        self.assertEqual(
            gen_traffic_pattern_host_groups(hostx, pattern, test_config_path, test_benchmark_name), expected_host_group
        )

        # Test for k-batch pattern
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='k-batch',
        )
        parser.add_argument(
            '--batch',
            type=int,
            default=3,
        )
        parser.add_argument(
            '--mpi_pattern',
            type=bool,
            default=True,
        )
        pattern, _ = parser.parse_known_args()
        expected_host_group = [[['node0', 'node1', 'node2'], ['node3', 'node4', 'node5']]]
        self.assertEqual(
            gen_traffic_pattern_host_groups(hostx, pattern, test_config_path, test_benchmark_name), expected_host_group
        )

        # Test for topo-aware pattern
        tp_ibstat_path = 'tests/data/ib_traffic_topo_aware_ibstat.txt'
        tp_ibnetdiscover_path = 'tests/data/ib_traffic_topo_aware_ibnetdiscover.txt'
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='topo-aware',
        )
        parser.add_argument(
            '--ibstat',
            type=str,
            default=tp_ibstat_path,
        )
        parser.add_argument(
            '--ibnetdiscover',
            type=str,
            default=tp_ibnetdiscover_path,
        )
        parser.add_argument(
            '--min_dist',
            type=int,
            default=2,
        )
        parser.add_argument(
            '--max_dist',
            type=int,
            default=6,
        )
        parser.add_argument(
            '--mpi_pattern',
            type=bool,
            default=True,
        )
        hostx = tp_hostfile.split()
        pattern, _ = parser.parse_known_args()
        expected_host_group = [
            [
                ['vma414bbc00005I', 'vma414bbc00005J'], ['vma414bbc00005K', 'vma414bbc00005L'],
                ['vma414bbc00005M', 'vma414bbc00005N'], ['vma414bbc00005O', 'vma414bbc00005P'],
                ['vma414bbc00005Q', 'vma414bbc00005R']
            ],
            [
                ['vma414bbc00005I', 'vma414bbc00005K'], ['vma414bbc00005J', 'vma414bbc00005L'],
                ['vma414bbc00005O', 'vma414bbc00005Q'], ['vma414bbc00005P', 'vma414bbc00005R']
            ],
            [
                ['vma414bbc00005I', 'vma414bbc00005O'], ['vma414bbc00005J', 'vma414bbc00005P'],
                ['vma414bbc00005K', 'vma414bbc00005Q'], ['vma414bbc00005L', 'vma414bbc00005R']
            ]
        ]
        self.assertEqual(
            gen_traffic_pattern_host_groups(hostx, pattern, test_config_path, test_benchmark_name), expected_host_group
        )

        # Test for mpi_pattern file
        with open(test_config_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, expected_mpi_pattern)
        test_config_file.close()

        # Test for invalid pattern
        hostx = ['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='invalid pattern',
        )
        parser.add_argument(
            '--mpi_pattern',
            type=bool,
            default=True,
        )
        pattern, _ = parser.parse_known_args()
        gen_traffic_pattern_host_groups(hostx, pattern, test_config_path, test_benchmark_name)
