# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for traffic pattern config generation module."""
import argparse
import unittest

from tests.helper import decorator
from superbench.common.utils import gen_traffic_pattern_host_group


class GenConfigTest(unittest.TestCase):
    """Test the utils for generating config."""
    @decorator.load_data('tests/data/ib_traffic_topo_aware_hostfile')    # noqa: C901
    def test_gen_traffic_pattern_host_group(self, tp_hostfile):
        """Test the function of generating traffic pattern config from specified mode."""
        # Test for all-nodes pattern
        hostx = ['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='all-nodes',
        )
        pattern, _ = parser.parse_known_args()
        expected_host_group = [[['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']]]
        self.assertEqual(gen_traffic_pattern_host_group(hostx, pattern), expected_host_group)

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
        self.assertEqual(gen_traffic_pattern_host_group(hostx, pattern), expected_host_group)

        # Test for invalid pattern
        hostx = ['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='invalid pattern',
        )
        pattern, _ = parser.parse_known_args()
        gen_traffic_pattern_host_group(hostx, pattern)
