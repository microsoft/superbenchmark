"""CPU-only tests for the optional cuDNN prepared execution boundary."""

import json
import os
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.helper import decorator
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
            'execution_mode': 'prepared', 'policy': 'screened-v1',
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
        for change in ({'policy': 'different'}, {'plan': {}}, {'setup_ms': float('nan')}, {'postcheck_call_ms': -1}):
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
        self.metadata['plan'] = dict(reversed(list(self.metadata['plan'].items())))
        self.benchmark._result.reset_mock()
        self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
        self.assertEqual(metric, self.benchmark._result.add_result.call_args_list[0][0][0])

    def test_gpu_profile_does_not_change_execution_identity(self):
        """Keep device visibility and nominal clocks out of metric keys."""
        self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
        metric = self.benchmark._result.add_result.call_args_list[0][0][0]
        self.metadata['plan']['GPUProfile'] = {'cudaDeviceId': 3, 'pciDeviceId': 99, 'smClockRateKHz': 1800000}
        self.benchmark._result.reset_mock()
        output = self.output()
        self.assertTrue(self.benchmark._process_raw_result(0, output))
        self.assertEqual(metric, self.benchmark._result.add_result.call_args_list[0][0][0])
        self.assertEqual(self.benchmark._result.add_raw_data.call_args_list[0][0][1], output)

    def test_numerical_screen_is_required_before_publishing(self):
        """Never accept missing, partial, weakened or failed qualification."""
        original = dict(self.metadata['verification'])
        changes = [
            {'passed': False}, {'inputs': 1}, {'checked_elements': 128}, {'atol': 0.01}, {'rtol': 0.01},
            {'maximum_tolerance_ratio': 1.01}, {'maximum_tolerance_ratio': float('nan')},
            {'reference': 'legacy-agreement'},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.metadata['verification'] = dict(original, **change)
                self.benchmark._result.reset_mock()
                self.assertFalse(self.benchmark._process_raw_result(0, self.output()))
                self.assertEqual(self.benchmark._result.add_result.call_args_list[0][0][1], -1)
                self.assertEqual(self.benchmark._result.add_result.call_count, 1)
        self.metadata.pop('verification')
        self.benchmark._result.reset_mock()
        self.assertFalse(self.benchmark._process_raw_result(0, self.output()))
        self.assertEqual(self.benchmark._result.add_result.call_count, 1)

    def test_computation_changes_preserve_distinct_metrics(self):
        """Distinguish architectures, algorithms, versions, shapes and precision."""
        self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
        metric = self.benchmark._result.add_result.call_args_list[0][0][0]
        original = dict(self.metadata['plan'])
        changes = [
            {'schemaVersion': 5}, {'cudnnVersion': 91300},
            {'engine': dict(original['engine'], engineId=18)},
            {'engine': dict(original['engine'], smVersion=1000)},
            {'engine': dict(original['engine'], knobChoices={'workspace': 1})},
            {'operationGraph': {'tensors': [{'dataType': 'CUDNN_DATA_HALF', 'dim': [2, 8, 5, 5]}]}},
            {'operationGraph': {'tensors': [{'dataType': 'CUDNN_DATA_FLOAT', 'dim': [4, 8, 5, 5]}]}},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.metadata['plan'] = dict(original, **change)
                self.benchmark._result.reset_mock()
                self.assertTrue(self.benchmark._process_raw_result(0, self.output()))
                self.assertNotEqual(metric, self.benchmark._result.add_result.call_args_list[0][0][0])

    def test_unsupported_configuration_is_explicit(self):
        """Do not dispatch an unsupported direction or legacy auto selection."""
        config = json.loads(self.benchmark._commands[0].split('--config_json ')[1][1:-1])
        with self.assertRaises(ValueError):
            self.benchmark._execution_config(dict(config, name='cudnnConvolutionForward'))
        with self.assertRaises(ValueError):
            self.benchmark._execution_config(dict(config, planPolicy='deterministic-v1'))
        self.benchmark._args.enable_auto_algo = True
        with self.assertRaises(ValueError):
            self.benchmark._execution_config(config)

    def test_missing_architecture_identity_is_rejected(self):
        """Do not combine plans that omit architecture from execution identity."""
        self.metadata['plan']['engine'].pop('smVersion')
        self.assertFalse(self.benchmark._process_raw_result(0, self.output()))
        self.assertEqual(self.benchmark._result.add_result.call_count, 1)
        self.assertEqual(self.benchmark._result.add_result.call_args_list[0][0][1], -1)


@decorator.cuda_test
def test_cudnn_native_regressions():
    """Run installed native precision and full-shape prepared checks in CUDA test jobs."""
    environment = os.environ.copy()
    search_path = environment.get('PATH', '')
    if environment.get('SB_MICRO_PATH'):
        root = environment['SB_MICRO_PATH']
        search_path = os.path.join(root, 'bin') + os.pathsep + search_path
        environment['LD_LIBRARY_PATH'] = os.path.join(root, 'lib') + os.pathsep + environment.get('LD_LIBRARY_PATH', '')
    for name in ('cudnn_data_types_test', 'cudnn_prepared_test'):
        binary = shutil.which(name, path=search_path)
        assert binary, 'Build and install cuDNN with BUILD_TESTING=ON before running CUDA tests: ' + name
        result = subprocess.run(
            [binary], env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900
        )
        if name == 'cudnn_prepared_test' and result.returncode == 77:
            raise unittest.SkipTest(result.stdout.strip())
        assert result.returncode == 0, result.stdout
    binary = shutil.which('cudnn_benchmark', path=search_path)
    assert binary, 'Install cudnn_benchmark before running CUDA tests.'
    config = {
        'name': 'cudnnConvolutionBackwardFilter', 'executionMode': 'prepared', 'planPolicy': 'screened-v1',
        'workspaceLimitMiB': 0, 'inputDims': [32, 128, 14, 14], 'inputStride': [25088, 196, 14, 1],
        'outputDims': [32, 32, 14, 14], 'outputStride': [6272, 196, 14, 1], 'filterDims': [32, 128, 3, 3],
        'inputType': 2, 'convType': 0, 'tensorOp': True, 'arrayLength': 2, 'mode': 1,
        'padA': [1, 1], 'filterStrideA': [1, 1], 'dilationA': [1, 1],
    }
    result = subprocess.run(
        [binary, '--num_test', '1', '--warm_up', '1', '--num_in_step', '1', '--random_seed', '33931',
         '--config_json', json.dumps(config)],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    if result.returncode == 0:
        metadata = json.loads(next(line.split(': ', 1)[1] for line in result.stdout.splitlines()
                                   if line.startswith('[prepared_plan]: ')))
        assert metadata['verification']['passed'] is True
    else:
        assert 'unsupported under screened-v1 policy' in result.stdout, result.stdout
        assert '[raw_data]:' not in result.stdout
