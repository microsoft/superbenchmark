"""CPU-only tests for cuDNN timing output validation."""

import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks.cudnn_function import CudnnBenchmark


class CudnnRawResultTests(unittest.TestCase):
    """Test parsing without preprocessing or launching the native benchmark."""

    def setUp(self):
        """Provide only the state consumed by the timing parser."""
        self.benchmark = CudnnBenchmark.__new__(CudnnBenchmark)
        self.benchmark._name = 'cudnn-function'
        self.benchmark._curr_run_index = 0
        self.benchmark._args = SimpleNamespace(num_steps=3, log_raw_data=False)
        self.benchmark._result = Mock(spec=['add_raw_data', 'add_result'])
        configuration = {
            'tensorOp': True,
            'name': 'cudnnConvolutionForward',
            'inputDims': [1, 2, 3, 3],
            'algo': 1,
        }
        self.benchmark._commands = ["cudnn_benchmark --config_json '{}'".format(json.dumps(configuration))]
        self.metric = 'name_cudnnconvolutionforward_algo_1_inputdims_[1_2_3_3]_tensorop_true_time'
        self.valid_row = '[raw_data]: 0.001,0.002,0.003,'

    def assert_valid_output(self, raw_output, samples, mean_time):
        """Check successful metrics, units, and unmodified raw output."""
        self.benchmark._result.reset_mock()
        self.assertIs(self.benchmark._process_raw_result(0, raw_output), True)
        self.benchmark._result.add_result.assert_called_once_with(self.metric, mean_time)
        self.assertEqual(
            self.benchmark._result.add_raw_data.call_args_list,
            [
                call('raw_output_0', raw_output, self.benchmark._args.log_raw_data),
                call(self.metric, samples, self.benchmark._args.log_raw_data),
            ]
        )

    def assert_invalid_output(self, raw_output):
        """Require failure without any partial timing data or good metric."""
        self.benchmark._result.reset_mock()
        self.assertIs(self.benchmark._process_raw_result(0, raw_output), False)
        self.benchmark._result.add_result.assert_called_once_with(self.metric, -1)
        self.benchmark._result.add_raw_data.assert_called_once_with(
            'raw_output_0', raw_output, self.benchmark._args.log_raw_data
        )

    def test_valid_native_output(self):
        """Keep naming, milliseconds-to-microseconds scaling, and logging flags."""
        raw_output = 'cuDNN benchmark\n{}\nBenchmark completed\n'.format(self.valid_row)
        for log_raw_data in (False, True):
            with self.subTest(log_raw_data=log_raw_data):
                self.benchmark._args.log_raw_data = log_raw_data
                self.assert_valid_output(raw_output, [0.001, 0.002, 0.003], 2.0)

    def test_valid_without_trailing_comma(self):
        """Do not drop the last sample when the optional comma is absent."""
        self.assert_valid_output('[raw_data]: 0.001,0.002,0.003', [0.001, 0.002, 0.003], 2.0)

    def test_valid_with_whitespace(self):
        """Allow whitespace around the native values and trailing comma."""
        self.assert_valid_output('[raw_data]: \t1e-3, 2e-3, 3e-3, \t\n', [0.001, 0.002, 0.003], 2.0)

    def test_absent_timing_record(self):
        """Empty or informational output cannot report success."""
        for raw_output in ('', '\n \n', 'cuDNN benchmark completed\n'):
            with self.subTest(raw_output=raw_output):
                self.assert_invalid_output(raw_output)

    def test_empty_timing_record(self):
        """A timing marker alone is not a measurement."""
        for raw_output in ('[raw_data]:', '[raw_data]: \n', '[raw_data]: ,', '[raw_data]: \t, \t'):
            with self.subTest(raw_output=raw_output):
                self.assert_invalid_output(raw_output)

    def test_non_finite_samples(self):
        """Reject NaN, infinity, and numbers that overflow float parsing."""
        for sample in ('nan', 'NaN', 'inf', '-inf', '+Inf', '1e309', '-1e309'):
            with self.subTest(sample=sample):
                self.assert_invalid_output('[raw_data]: 0.001,{},0.003,'.format(sample))

    def test_non_positive_samples(self):
        """Every sample must be positive, including after float underflow."""
        for sample in ('0', '0.0', '-0.0', '-0.001', '1e-400'):
            with self.subTest(sample=sample):
                self.assert_invalid_output('[raw_data]: 0.001,{},0.003,'.format(sample))

    def test_wrong_sample_count(self):
        """Require exactly num_steps samples, with or without a final comma."""
        for samples in ('0.001,', '0.001,0.002,', '0.001,0.002,0.003,0.004,', '0.001,0.002,0.003,0.004'):
            with self.subTest(samples=samples):
                self.assert_invalid_output('[raw_data]: ' + samples)

    def test_non_positive_num_steps(self):
        """Zero or negative requested steps cannot produce a valid record."""
        for num_steps in (0, -1):
            for raw_output in ('[raw_data]: ', self.valid_row):
                with self.subTest(num_steps=num_steps, raw_output=raw_output):
                    self.benchmark._args.num_steps = num_steps
                    self.assert_invalid_output(raw_output)

    def test_duplicate_timing_records(self):
        """Reject a second record without retaining the first measurement."""
        for second_row in (self.valid_row, '[raw_data]: 0.004,0.005,0.006,', '[raw_data]: '):
            with self.subTest(second_row=second_row):
                self.assert_invalid_output(self.valid_row + '\n' + second_row)

    def test_malformed_timing_record(self):
        """Do not discard invalid tokens or extra delimiters as a final comma."""
        for raw_output in (
            '[raw_data] 0.001,0.002,0.003,',
            '[raw_data]: 0.001,invalid,0.003,',
            '[raw_data]: 0.001,,0.003,',
            '[raw_data]: ,0.001,0.002,0.003,',
            '[raw_data]: 0.001,0.002,0.003,,',
            '[raw_data]: 0.001,0.002,0.003, ,',
            '[raw_data]: 0.001,0.002,0.003,garbage',
        ):
            with self.subTest(raw_output=raw_output):
                self.assert_invalid_output(raw_output)

    def test_error_before_or_after_valid_record(self):
        """An error anywhere in stdout invalidates otherwise valid timings."""
        for raw_output in (
            'Error: CUDNN_STATUS_NOT_SUPPORTED\n' + self.valid_row,
            self.valid_row + '\nError: cuDNN call failed',
        ):
            with self.subTest(raw_output=raw_output):
                self.assert_invalid_output(raw_output)

    def test_error_without_timing_record(self):
        """Keep the existing failure marker when only an error is reported."""
        self.assert_invalid_output('Error: CUDNN_STATUS_NOT_SUPPORTED\n')

    def test_unit_conversion_overflow(self):
        """Finite input samples must not produce an infinite scaled metric."""
        self.assert_invalid_output('[raw_data]: 1e306,1e306,1e306,')

    def test_large_finite_scaled_metric(self):
        """Do not reject large timings whose scaled mean remains finite."""
        self.assert_valid_output('[raw_data]: 1e300,1e300,1e300,', [1e300, 1e300, 1e300], 1e300 * 1000)

    def test_tolerant_invoke_keeps_failure_and_later_successful_cases(self):
        """A native rejection must not erase valid timings or become overall success."""
        self.benchmark._args.tolerant_fail = True
        self.benchmark._args.log_flushing = False
        self.benchmark._args.bin_dir = '/tmp'
        self.benchmark._result = Mock(spec=['add_raw_data', 'add_result', 'set_return_code'])
        self.benchmark._commands *= 3
        outcomes = [subprocess.CompletedProcess('binary', 0, self.valid_row),
                    subprocess.CompletedProcess('binary', 255, 'Error: CUDNN_STATUS_NOT_SUPPORTED'),
                    subprocess.CompletedProcess('binary', 0, self.valid_row)]
        with patch('superbench.benchmarks.micro_benchmarks.micro_base.run_command', side_effect=outcomes) as execute:
            self.assertFalse(self.benchmark._benchmark())
        self.assertEqual(execute.call_count, 3)
        self.benchmark._result.set_return_code.assert_called_once_with(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
        self.assertEqual(self.benchmark._result.add_result.call_args_list, [call(self.metric, 2.0)] * 2)

    def test_zero_exit_empty_output_sets_parsing_failure(self):
        """An empty successful process is not a successful benchmark measurement."""
        self.benchmark._args.tolerant_fail = True
        self.benchmark._args.log_flushing = False
        self.benchmark._args.bin_dir = '/tmp'
        self.benchmark._result = Mock(spec=['add_raw_data', 'add_result', 'set_return_code'])
        with patch('superbench.benchmarks.micro_benchmarks.micro_base.run_command',
                   return_value=subprocess.CompletedProcess('binary', 0, '')):
            self.assertFalse(self.benchmark._benchmark())
        self.benchmark._result.set_return_code.assert_called_once_with(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
        self.benchmark._result.add_result.assert_called_once_with(self.metric, -1)


if __name__ == '__main__':
    unittest.main()
