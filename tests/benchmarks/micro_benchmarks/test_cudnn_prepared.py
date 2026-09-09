"""CPU-only tests for the optional cuDNN prepared execution boundary."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from superbench.benchmarks.micro_benchmarks.cudnn_function import CudnnBenchmark


class CudnnPreparedTests(unittest.TestCase):
    """Check prepared configuration and output without native GPU work."""

    def setUp(self):
        """Prepare the Python wrapper without launching native GPU work."""
        logger_patch = patch('superbench.benchmarks.micro_benchmarks.cudnn_function.logger')
        logger_patch.start()
        self.addCleanup(logger_patch.stop)
        self.benchmark = CudnnBenchmark('cudnn-function')
        self.benchmark._args = SimpleNamespace(
            execution_mode='prepared', workspace_limit_mib=1024, enable_auto_algo=False,
            num_steps=2, num_warmup=1, num_in_step=10, random_seed=33931, config_json_str=None,
            bin_dir='/tmp', log_raw_data=False,
        )
        self.benchmark._result = Mock()
        self.benchmark._curr_run_index = 0
        with patch('superbench.benchmarks.micro_benchmarks.MicroBenchmarkWithInvoke._preprocess', return_value=True):
            self.assertTrue(self.benchmark._preprocess())
        self.metadata = {
            'execution_mode': 'prepared', 'policy': 'deterministic-v1', 'plan': {'engine': 17, 'knobs': []},
            'setup_ms': 3, 'plan_build_ms': 2, 'first_call_ms': 1, 'benchmark_ms': 6,
        }

    def output(self, samples='0.01,0.02,'):
        """Return a representative native prepared record."""
        return '[prepared_plan]: ' + json.dumps(self.metadata) + '\n[raw_data]: ' + samples

    def test_prepared_defaults_and_legacy_identity(self):
        """Limit the prepared defaults without changing legacy commands."""
        self.assertEqual(len(self.benchmark._commands), 6)
        self.assertFalse(self.benchmark._args.tolerant_fail)
        for command in self.benchmark._commands:
            config = json.loads(command.split('--config_json ')[1][1:-1])
            self.assertEqual(config['executionMode'], 'prepared')
            self.assertEqual(config['planPolicy'], 'deterministic-v1')
            self.assertNotIn('algo', config)
        self.benchmark._args.execution_mode = 'legacy'
        self.benchmark._commands = []
        with patch('superbench.benchmarks.micro_benchmarks.MicroBenchmarkWithInvoke._preprocess', return_value=True):
            self.assertTrue(self.benchmark._preprocess())
        self.assertEqual(len(self.benchmark._commands), 18)
        self.assertTrue(self.benchmark._args.tolerant_fail)
        self.assertTrue(all('executionMode' not in command for command in self.benchmark._commands))

    def test_prepared_metrics_include_plan_and_costs(self):
        """Keep steady timing separate from setup and legacy metric identity."""
        self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
        calls = self.benchmark._result.add_result.call_args_list
        self.assertEqual(len(calls), 5)
        self.assertIn('_executionmode_prepared_', calls[0][0][0])
        self.assertIn('_plan_', calls[0][0][0])
        self.assertEqual(calls[0][0][1], 15)
        self.assertEqual({call[0][1] for call in calls[1:]}, {1000, 2000, 3000, 6000})

    def test_missing_or_invalid_output_never_publishes_positive_timing(self):
        """Reject old binaries, malformed samples and late native failures."""
        outputs = ['[raw_data]: 0.01,0.02,', self.output() + '\nError: failure',
                   self.output() + '\n[raw_data]: 0.01,0.02,', self.output('0.01,'),
                   self.output('nan,0.02,'), self.output('0,0.02,'), self.output('0.01,0.02')]
        for output in outputs:
            with self.subTest(output=output):
                self.benchmark._result.reset_mock()
                self.assertFalse(self.benchmark._process_raw_result(0, output))
                self.assertEqual(self.benchmark._result.add_result.call_args[0][1], -1)
                self.assertEqual(self.benchmark._result.add_result.call_count, 1)

    def test_plan_policy_and_setup_validation(self):
        """Reject metadata that does not establish the requested path."""
        for change in ({'policy': 'different'}, {'plan': {}}, {'setup_ms': float('nan')}, {'first_call_ms': -1}):
            with self.subTest(change=change):
                original = dict(self.metadata)
                self.metadata.update(change)
                self.benchmark._result.reset_mock()
                self.assertFalse(self.benchmark._process_raw_result(0, self.output()))
                self.assertEqual(self.benchmark._result.add_result.call_count, 1)
                self.metadata = original

    def test_plan_serialization_order_does_not_change_metric(self):
        """Canonicalize the plan before calculating its metric fingerprint."""
        self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
        metric = self.benchmark._result.add_result.call_args_list[0][0][0]
        self.metadata['plan'] = {'knobs': [], 'engine': 17}
        self.benchmark._result.reset_mock()
        self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
        self.assertEqual(metric, self.benchmark._result.add_result.call_args_list[0][0][0])

    def test_unsupported_configuration_is_explicit(self):
        """Do not dispatch an unsupported direction or legacy auto selection."""
        config = json.loads(self.benchmark._commands[0].split('--config_json ')[1][1:-1])
        with self.assertRaises(ValueError):
            self.benchmark._execution_config(dict(config, name='cudnnConvolutionForward'))
        self.benchmark._args.enable_auto_algo = True
        with self.assertRaises(ValueError):
            self.benchmark._execution_config(config)
