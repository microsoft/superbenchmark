# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner test."""

import unittest
import shutil
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from superbench.runner import SuperBenchRunner


class RunnerTestCase(unittest.TestCase):
    """A class for runner test cases.

    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    """
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        default_config_file = Path(__file__).parent / '../../superbench/config/default.yaml'
        self.default_config = OmegaConf.load(str(default_config_file))
        self.output_dir = tempfile.mkdtemp()

        self.runner = SuperBenchRunner(self.default_config, None, None, self.output_dir)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.output_dir)

    def test_set_logger(self):
        """Test log file exists."""
        expected_log_file = Path(self.runner._output_dir) / 'sb-run.log'
        self.assertTrue(expected_log_file.is_file())
