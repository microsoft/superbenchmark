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
            'execution_mode': 'prepared', 'policy': 'screened-v1', 'input_type': 0,
            'verification': {
                'policy': 'full-output-v1', 'passed': True, 'inputs': 5, 'checked_elements': 5 * 32 * 128 * 3 * 3,
                'reference': 'cuBLAS-FP64-with-CPU-crosschecks', 'atol': 0.0005, 'rtol': 0.0005,
                'maximum_tolerance_ratio': 0.5,
            },
            'plan': {
                'schemaVersion': 4, 'cudnnVersion': 91200,
                'engine': {'engineId': 17, 'smVersion': 1030, 'knobChoices': {'workspace': 0}},
                'operationGraph': {'tensors': [{'dataType': 'CUDNN_DATA_FLOAT', 'dim': [2, 8, 5, 5]}]},
                'GPUProfile': {'cudaDeviceId': 0, 'pciDeviceId': 0, 'smClockRateKHz': 2070000},
            },
            'setup_ms': 3, 'plan_build_ms': 2, 'postcheck_call_ms': 1, 'benchmark_ms': 6,
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
            self.assertEqual(config['planPolicy'], 'screened-v1')
            self.assertNotIn('algo', config)
        self.benchmark._args.execution_mode = 'legacy'
        self.benchmark._commands = []
        with patch('superbench.benchmarks.micro_benchmarks.MicroBenchmarkWithInvoke._preprocess', return_value=True):
            self.assertTrue(self.benchmark._preprocess())
        self.assertEqual(len(self.benchmark._commands), 18)
        self.assertTrue(self.benchmark._args.tolerant_fail)
        self.assertTrue(all('executionMode' not in command for command in self.benchmark._commands))
        self.assertTrue(all('algo' in json.loads(command.split('--config_json ')[1][1:-1])
                            for command in self.benchmark._commands))

    def test_prepared_rejects_legacy_constraints(self):
        """Reject explicit algorithms, unsupported directions and auto selection."""
        config = json.loads(self.benchmark._commands[0].split('--config_json ')[1][1:-1])
        for change in ({'algo': 1}, {'name': 'cudnnConvolutionForward'}, {'planPolicy': 'unknown'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.benchmark._execution_config(dict(config, **change))
        self.benchmark._args.enable_auto_algo = True
        with self.assertRaises(ValueError):
            self.benchmark._execution_config(config)

    def test_prepared_result_requires_qualification(self):
        """Publish identified timing only for a qualified plan and valid samples."""
        output = self.output()
        self.assertTrue(self.benchmark._process_raw_result(0, output))
        calls = self.benchmark._result.add_result.call_args_list
        self.assertEqual(len(calls), 5)
        self.assertIn('_executionmode_prepared_', calls[0][0][0])
        self.assertIn('_actualinputtype_0_plan_', calls[0][0][0])
        self.assertEqual(calls[0][0][1], 15)
        self.assertEqual({call[0][1] for call in calls[1:]}, {1000, 2000, 3000, 6000})

        self.metadata['verification']['passed'] = False
        for invalid in (self.output(), '[raw_data]: 0.01,0.02,', output.replace('0.01,0.02,', 'nan,0.02,')):
            with self.subTest(output=invalid):
                self.benchmark._result.reset_mock()
                self.assertFalse(self.benchmark._process_raw_result(0, invalid))
                self.assertEqual(self.benchmark._result.add_result.call_args[0][1], -1)
                self.assertEqual(self.benchmark._result.add_result.call_count, 1)
