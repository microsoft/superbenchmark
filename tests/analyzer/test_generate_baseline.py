# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BaselineGeneration module."""

import unittest
import json
from pathlib import Path

from superbench.analyzer import BaselineGeneration
import superbench.analyzer.file_handler as file_handler


class TestBaselineGeneration(unittest.TestCase):
    """Test for BaselineGeneration class."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        self.parent_path = Path(__file__).parent
        self.test_raw_data = str(self.parent_path / 'test_generate_baseline_results.jsonl')
        self.test_summary_rule_file = str(self.parent_path / 'test_generate_baseline_summary_rules.yaml')
        self.test_diagnosis_rule_file = str(self.parent_path / 'test_generate_baseline_diagnosis_rules.yaml')
        self.output_baseline_file = str(self.parent_path / 'baseline.json')
        self.pre_baseline_file = str(self.parent_path / 'pre_baseline.json')

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        for file in [self.output_baseline_file, self.pre_baseline_file]:
            p = Path(file)
            if p.is_file():
                p.unlink()

    def test_baseline_generation_run(self):
        """Test for the run process of result generate-baseline."""
        # Test - generate baseline without previous baseline
        BaselineGeneration().run(
            self.test_raw_data,
            self.test_summary_rule_file,
            self.test_diagnosis_rule_file,
            None,
            'fix_threshold',
            str(self.parent_path),
            digit=2
        )
        baseline = file_handler.read_baseline(self.output_baseline_file)
        expected_baseline = {
            'kernel-launch/event_time': 0.0055,
            'kernel-launch/wall_time': 0.009,
            'mem-bw/d2h_bw': 26.22,
            'mem-bw/h2d_bw': 26.07
        }
        assert (expected_baseline == baseline)

        # Test - generate baseline with previous baseline
        pre_baseline = {'gemm-flops/FP32': 18318.4, 'gemm-flops/FP16': 33878}
        with open(self.pre_baseline_file, 'w') as f:
            json.dump(pre_baseline, f)

        BaselineGeneration().run(
            self.test_raw_data,
            self.test_summary_rule_file,
            self.test_diagnosis_rule_file,
            self.pre_baseline_file,
            'fix_threshold',
            str(self.parent_path),
            digit=2
        )
        baseline = file_handler.read_baseline(self.output_baseline_file)
        expected_baseline = {
            'kernel-launch/event_time': 0.0055,
            'kernel-launch/wall_time': 0.009,
            'mem-bw/d2h_bw': 26.22,
            'mem-bw/h2d_bw': 26.07,
            'gemm-flops/FP32': 18318.4,
            'gemm-flops/FP16': 33878
        }
        assert (expected_baseline == baseline)
