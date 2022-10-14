# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for traffic config generation module."""
import unittest
import uuid

from superbench.common.utils import gen_all_nodes_config, gen_pair_wise_config, gen_k_batch_config


def gen_hostlist(hostlist, num):
    """Generate a fake list of specified number of hosts."""
    hostlist.clear()
    for i in range(0, num):
        hostlist.append(str(uuid.uuid4()))


class GenConfigTest(unittest.TestCase):
    """Test the utils for generating config."""
    def test_generate_config(self):
        """Test the function of generating traffic pattern config."""
        hostlist = []
        # test under 8 nodes
        gen_hostlist(hostlist, 8)
        expected_all_node_config = ['0,1,2,3,4,5,6,7']
        self.assertEqual(gen_all_nodes_config(len(hostlist)), expected_all_node_config)

        expected_pair_wise_config = [
            '0,7;1,6;2,5;3,4', '0,1;2,7;3,6;4,5', '0,2;3,1;4,7;5,6', '0,3;4,2;5,1;6,7', '0,4;5,3;6,2;7,1',
            '0,5;6,4;7,3;1,2', '0,6;7,5;1,4;2,3'
        ]
        self.assertEqual(gen_pair_wise_config(len(hostlist)), expected_pair_wise_config)

        expected_k_batch_config = [
            ['0;1;2;3;4;5;6;7'], ['0,1;2,3;4,5;6,7'], ['0,1,2;3,4,5'], ['0,1,2,3;4,5,6,7'], ['0,1,2,3,4'],
            ['0,1,2,3,4,5'], ['0,1,2,3,4,5,6'], ['0,1,2,3,4,5,6,7']
        ]

        for k in range(1, len(hostlist) + 1):
            self.assertEqual(gen_k_batch_config(k, len(hostlist)), expected_k_batch_config[k - 1])
