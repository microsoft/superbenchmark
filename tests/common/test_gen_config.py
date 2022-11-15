# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for traffic config generation module."""
import unittest
import uuid

from superbench.common.utils import gen_all_nodes_config


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
