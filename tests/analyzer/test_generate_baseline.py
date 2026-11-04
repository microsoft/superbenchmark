# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BaselineGeneration module."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from superbench.analyzer import BaselineGeneration
import superbench.analyzer.file_handler as file_handler


@pytest.mark.parametrize('algorithm', ['mean', 'fix_threshold'])
@pytest.mark.parametrize('existing_value', [0.0, 42.0])
def test_existing_baseline_overlap(algorithm, existing_value):
    """Preserve overlapping and unrelated values while generating every new metric."""
    fixtures = Path(__file__).parent
    inputs = [str(fixtures / name) for name in (
        'test_generate_baseline_results.jsonl', 'test_generate_baseline_summary_rules.yaml',
        'test_generate_baseline_diagnosis_rules.yaml'
    )]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'baseline.json'
        BaselineGeneration().run(*inputs, None, algorithm, directory, digit=2)
        control = json.loads(output.read_text())
        previous = {'mem-bw/h2d_bw': existing_value, 'unrelated/metric': 7.0}
        previous_file = Path(directory) / 'previous.json'
        previous_file.write_text(json.dumps(previous))
        BaselineGeneration().run(*inputs, str(previous_file), algorithm, directory, digit=2)
        assert json.loads(output.read_text()) == {**control, **previous}


@pytest.mark.parametrize('existing_value', [0.0, 42.0])
def test_existing_threshold_returns_baseline_and_data(existing_value):
    """Keep the documented tuple shape and unchanged series for an existing baseline."""
    series = pd.Series([7.0, 9.0])
    rule = {'metrics': {'metric': existing_value}}
    baseline, normal_data = BaselineGeneration().fix_threshold_outlier_detection(
        series, rule, 'metric', lambda *args: 0
    )
    assert baseline == existing_value
    pd.testing.assert_series_equal(normal_data, series)


def test_baseline_generation_failure_propagates():
    """Surface a generation error without replacing a previously valid output."""
    fixtures = Path(__file__).parent
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'baseline.json'
        sentinel = '{"previous": 1}\n'
        output.write_text(sentinel)
        with patch.object(BaselineGeneration, 'generate_baseline', side_effect=ValueError('controlled failure')), \
                pytest.raises(ValueError, match='controlled failure'):
            BaselineGeneration().run(
                str(fixtures / 'test_generate_baseline_results.jsonl'),
                str(fixtures / 'test_generate_baseline_summary_rules.yaml'),
                None, None, 'mean', directory,
            )
        assert output.read_text() == sentinel


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
