# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for traffic pattern config generation module."""
import argparse
import unittest

from superbench.common.utils import gen_traffic_pattern_host_group


class GenConfigTest(unittest.TestCase):
    """Test the utils for generating config."""
    def test_gen_traffic_pattern_host_group(self):
        """Test the function of generating traffic pattern config from specified mode."""
        # test under 8 nodes
        # test all-nodes pattern
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
        # test pair-wise pattern
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='pair-wise',
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
        self.assertEqual(gen_traffic_pattern_host_group(hostx, pattern), expected_host_group)
        # test k-batch pattern
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--type',
            type=str,
            default='k-batch',
        )
        parser.add_argument(
            '--scale',
            type=int,
            default=3,
        )
        pattern, _ = parser.parse_known_args()
        expected_host_group = [[['node0', 'node1', 'node2'], ['node3', 'node4', 'node5']]]
        self.assertEqual(gen_traffic_pattern_host_group(hostx, pattern), expected_host_group)
