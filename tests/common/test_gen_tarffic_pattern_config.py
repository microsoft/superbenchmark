# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for traffic pattern config generation module."""
import argparse
import unittest

from superbench.common.utils import gen_tarffic_pattern_host_group


class GenConfigTest(unittest.TestCase):
    """Test the utils for generating config."""
    def test_gen_tarffic_pattern_host_group(self):
        """Test the function of generating traffic pattern config from specified mode."""
        # test under 8 nodes
        hostx = ['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']
        parser = argparse.ArgumentParser(
            add_help=False,
            usage=argparse.SUPPRESS,
            allow_abbrev=False,
        )
        parser.add_argument(
            '--pattern',
            type=str,
            default='all-nodes',
            required=False,
        )
        args, _ = parser.parse_known_args()
        expected_host_group = [[['node0', 'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7']]]
        self.assertEqual(gen_tarffic_pattern_host_group(hostx, args), expected_host_group)
