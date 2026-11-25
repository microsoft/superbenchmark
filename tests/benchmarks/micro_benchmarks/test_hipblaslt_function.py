# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for hipblaslt-bench benchmark."""

import shlex
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.micro_benchmarks import hipblaslt_function
from superbench.benchmarks.result import BenchmarkResult

_MXFP4_PARAMETERS = '--in_types mxfp4 --shapes 64,128,256 --transA T --transB N --initialization maf2_0'
_MXFP4_HELP_OPTIONS = (
    '--scaleA <value> 0 = None, 3 = B32E8, 4 = B16E8', '--scaleB <value> 0 = None, 3 = B32E8, 4 = B16E8',
    '--initialization <value> hpl, trig_float, maf2_0'
)


class HipblasLtBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for hipblaslt-bench benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'hipblaslt-gemm'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/hipblaslt-bench'])

    def get_benchmark(self):
        """Get Benchmark."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.ROCM)
        return benchmark_cls(self.benchmark_name, parameters='')

    def preprocess_mxfp4(
        self, parameters=_MXFP4_PARAMETERS, arch='gfx1250:sramecc+:xnack-', help_result=None, device_error=None
    ):
        """Preprocess with synthetic ROCm device and client capability information."""
        if help_result is None:
            help_result = SimpleNamespace(returncode=0, stdout='\n'.join(_MXFP4_HELP_OPTIONS))
        benchmark = hipblaslt_function.HipBlasLtBenchmark(self.benchmark_name, parameters=parameters)
        properties = Mock(return_value=SimpleNamespace(gcnArchName=arch), side_effect=device_error)
        cuda = SimpleNamespace(get_device_properties=properties)
        with patch.object(hipblaslt_function, 'torch', SimpleNamespace(cuda=cuda)), \
                patch.object(hipblaslt_function, 'run_command', return_value=help_result):
            result = benchmark._preprocess()
        return benchmark, result

    def make_mxfp4_result_benchmark(self):
        """Make a fresh result collector for one requested shape without probing a device or client."""
        benchmark = hipblaslt_function.HipBlasLtBenchmark(self.benchmark_name)
        benchmark._args = SimpleNamespace(log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)
        benchmark._shapes_to_run = [(64, 128, 256, 0, 'mxfp4')]
        benchmark._precision_in_commands = ['mxfp4']
        return benchmark

    def assert_invalid_mxfp4(self, output):
        """Check that a rejected row publishes neither correctness nor performance metrics."""
        benchmark = self.make_mxfp4_result_benchmark()
        self.assertFalse(benchmark._process_raw_result(0, output))
        self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
        self.assertEqual(set(benchmark.result), {'return_code'})

    def mxfp4_output(self, overrides=None, missing=(), reverse=False):
        """Build synthetic CSV, not a captured GPU result."""
        row = {
            'transA': 'T',
            'transB': 'N',
            'grouped_gemm': '0',
            'batch_count': '1',
            'm': '64',
            'n': '128',
            'k': '256',
            'alpha': '1',
            'beta': '0',
            'a_type': 'f4_r',
            'b_type': 'f4_r',
            'c_type': 'f32_r',
            'd_type': 'f32_r',
            'compute_type': 'f32_r',
            'scaleA': '3',
            'scaleB': '3',
            'use_gpu_timer': '1',
            'hipblaslt-Gflops': '1048.576',
            'us': '4',
            'norm_error': '1e-6',
            'atol': '0.0001',
            'rtol': '0.0001',
        }
        row.update(overrides or {})
        for column in missing:
            del row[column]
        columns = list(row)
        if reverse:
            columns.reverse()
        return '[0]:' + ','.join(columns) + '\n' + ','.join(row[column] for column in columns) + \
            '\n    --kernel name:    synthetic_mxfp4_kernel\n'

    def test_hipblaslt_gemm_cls(self):
        """Test hipblaslt-bench benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.ROCM:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_hipblaslt_gemm_command_generation(self):
        """Test hipblaslt-bench benchmark command generation."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.ROCM)
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters='--batch 4:2:-1 --shapes 2,4,8 --in_types fp16 fp32 fp64 int8',
        )
        self.assertFalse(benchmark._preprocess())
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=' --shapes 2,4,8 --in_types fp16 fp32 fp64 int8',
        )
        self.assertFalse(benchmark._preprocess())
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=' --shapes 2:4,4:8 --in_types fp16 fp32',
        )
        self.assertFalse(benchmark._preprocess())
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters='--shapes 2:4,4:8,8:32 2:4,4:8,8:32:+4 --in_types fp16 fp32 bf16 fp8',
        )
        self.assertTrue(benchmark._preprocess())
        self.assertEqual((2 * 2 * 3 + 2 * 2 * 7) * len(benchmark._args.in_types), len(benchmark._commands))

        def cmd(t, b, m, n, k):
            if b == 0:
                return f'{benchmark._HipBlasLtBenchmark__bin_path} ' + \
                    f'-m {m} -n {n} -k {k} -j 20 -i 50 {benchmark._in_type_map[t]}' + \
                    f' --transA {benchmark._args.transA} --transB {benchmark._args.transB}' + \
                    f' --initialization {benchmark._args.initialization}'
            else:
                return f'{benchmark._HipBlasLtBenchmark__bin_path} ' + \
                    f'-m {m} -n {n} -k {k} -j 20 -i 50 {benchmark._in_type_map[t]} -b {b}' + \
                    f' --transA {benchmark._args.transA} --transB {benchmark._args.transB}' + \
                    f' --initialization {benchmark._args.initialization}'

        for _t in ['fp16', 'fp32', 'bf16', 'fp8']:
            for _m in [2, 4]:
                for _n in [4, 8]:
                    for _k in [8, 16, 32]:
                        self.assertIn(cmd(_t, 0, _m, _n, _k), benchmark._commands)
                    for _k in [8, 12, 16, 20, 24, 28, 32]:
                        self.assertIn(cmd(_t, 0, _m, _n, _k), benchmark._commands)

    def test_mxfp4_command_generation(self):
        """Check MXFP4 scaling, layout, and standalone verification and timing switches."""
        benchmark, result = self.preprocess_mxfp4()
        self.assertTrue(result)
        self.assertEqual(len(benchmark._commands), 1)
        tokens = shlex.split(benchmark._commands[0])
        for option, value in [
            ('-m', '64'), ('-n', '128'), ('-k', '256'), ('--a_type', 'f4_r'), ('--b_type', 'f4_r'),
            ('--c_type', 'f32_r'), ('--d_type', 'f32_r'), ('--compute_type', 'f32_r'), ('--scaleA', '3'),
            ('--scaleB', '3'), ('--alpha', '1'), ('--beta', '0'), ('--algo_method', 'heuristic'),
            ('--requested_solution', '1'), ('--transA', 'T'), ('--transB', 'N'), ('--initialization', 'maf2_0')
        ]:
            with self.subTest(option=option):
                self.assertEqual(tokens.count(option), 1)
                self.assertEqual(tokens[tokens.index(option) + 1], value)
        self.assertNotIn('--verify', tokens)
        for option in ['--verify_strict', '--use_gpu_timer', '--print_kernel_info']:
            with self.subTest(option=option):
                self.assertEqual(tokens.count(option), 1)
                next_index = tokens.index(option) + 1
                self.assertTrue(
                    next_index == len(tokens) or tokens[next_index].startswith('-'),
                    '{} must not consume a value'.format(option),
                )
        self.assertFalse(any(token.startswith(('--scaleABlock', '--scaleBBlock')) for token in tokens))
        self.assertEqual(benchmark.raw_data['mxfp4_gpu_arch'][0], 'gfx1250')
        self.assertNotIn('mxfp4_client_help', benchmark.raw_data)

    def test_mxfp4_invalid_arguments(self):
        """Reject unsupported layouts, initialization, shapes, or counts."""
        valid = _MXFP4_PARAMETERS
        invalid = [
            valid.replace('--transA T', '--transA N'),
            valid.replace('--transB N', '--transB T'),
            valid.replace('--initialization maf2_0', '--initialization rand_int'),
            valid.replace('--in_types mxfp4', '--in_types fp16'),
            valid.replace('--in_types mxfp4', '--in_types mxfp4 fp16'),
            valid.replace('64,128,256', '64,128,257'),
            valid.replace('64,128,256', '0,128,256'),
            valid + ' --batch 1',
            valid + ' --num_warmup -1',
            valid + ' --num_steps 0',
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                benchmark, result = self.preprocess_mxfp4(parameters)
                self.assertFalse(result)
                self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)
                self.assertEqual(benchmark._commands, [])

    def test_mxfp4_unsupported_devices(self):
        """Reject older GPUs and unqualified future architectures without generating commands."""
        for arch in ['gfx942', 'gfx1251']:
            with self.subTest(arch=arch):
                benchmark, result = self.preprocess_mxfp4(arch=arch)
                self.assertFalse(result)
                self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE)
                self.assertEqual(benchmark._commands, [])
        benchmark, result = self.preprocess_mxfp4(device_error=AssertionError('Invalid device id'))
        self.assertFalse(result)
        self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
        self.assertEqual(benchmark._commands, [])

    def test_mxfp4_client_capabilities(self):
        """Reject old client options and explicit capability-query failures."""
        help_text = '\n'.join(_MXFP4_HELP_OPTIONS)
        with patch.object(BenchmarkResult, 'add_raw_data', autospec=True) as record:
            benchmark, ready = self.preprocess_mxfp4(_MXFP4_PARAMETERS + ' --log_raw_data')
            self.assertTrue(ready)
            record.assert_any_call(benchmark._result, 'mxfp4_client_help', help_text, True)

        for output in [
            SimpleNamespace(returncode=0, stdout=help_text.replace('3 = B32E8', '3 = block')),
            SimpleNamespace(returncode=0, stdout=help_text.replace('3 = B32E8', '3 = B16E8', 1)),
            SimpleNamespace(returncode=0, stdout=help_text.replace('maf2_0', 'maf2_0_unit_scale')),
            SimpleNamespace(returncode=1, stdout='Cannot load hipBLASLt'),
        ]:
            with self.subTest(output=output):
                benchmark, result = self.preprocess_mxfp4(help_result=output)
                self.assertFalse(result)
                expected = ReturnCode.INVALID_ARGUMENT if output.returncode == 0 else \
                    ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE
                self.assertEqual(benchmark.return_code, expected)
                self.assertEqual(benchmark._commands, [])
                self.assertEqual(benchmark.raw_data['mxfp4_client_help'], [output.stdout])

        help_result = SimpleNamespace(returncode=0, stdout=help_text.replace('3 = B32E8', '3=b32e8'))
        self.assertTrue(self.preprocess_mxfp4(help_result=help_result)[1])
        for missing in _MXFP4_HELP_OPTIONS:
            with self.subTest(missing=missing):
                help_result = SimpleNamespace(
                    returncode=0, stdout='\n'.join(option for option in _MXFP4_HELP_OPTIONS if option != missing)
                )
                benchmark, result = self.preprocess_mxfp4(help_result=help_result)
                self.assertFalse(result)
                self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)

    def test_mxfp4_subprocess_status(self):
        """A nonzero client exit must suppress metrics even when it prints a valid row."""
        benchmark, result = self.preprocess_mxfp4()
        self.assertTrue(result)
        output = SimpleNamespace(returncode=2, stdout=self.mxfp4_output())
        with patch('superbench.benchmarks.micro_benchmarks.micro_base.run_command', return_value=output):
            self.assertFalse(benchmark._benchmark())
        self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
        self.assertEqual(set(benchmark.result), {'return_code'})

    def test_mxfp4_result_parsing(self):
        """Check reordered columns, exact metric units, and the norm threshold boundary."""
        for reverse, norm in [(False, '1e-6'), (True, '9.999e-6')]:
            with self.subTest(reverse=reverse, norm=norm):
                benchmark = self.make_mxfp4_result_benchmark()
                output = self.mxfp4_output({'norm_error': norm}, reverse=reverse)
                column_count = len(output.splitlines()[0].split(','))
                footer = 'diagnostic: ' + ','.join(['detail'] * column_count)
                self.assertTrue(benchmark._process_raw_result(0, output + footer))
                self.assertAlmostEqual(benchmark.result['mxfp4_1_64_128_256_flops'][0], 1.048576)
                self.assertEqual(benchmark.result['mxfp4_1_64_128_256_time'], [4])
                self.assertEqual(benchmark.result['mxfp4_1_64_128_256_norm_error'], [float(norm)])
                self.assertEqual(benchmark.result['mxfp4_1_64_128_256_correctness'], [1])
                self.assertEqual(benchmark.return_code, ReturnCode.SUCCESS)
        self.assert_invalid_mxfp4(self.mxfp4_output({'m': '128', 'n': '64'}))

    def test_mxfp4_result_requires_valid_command_index(self):
        """Do not report wrapper bookkeeping errors as malformed client output."""
        for index, error in [(1, IndexError), (None, TypeError)]:
            with self.subTest(index=index), self.assertRaises(error):
                self.make_mxfp4_result_benchmark()._process_raw_result(index, self.mxfp4_output())

    def test_mxfp4_result_coefficients(self):
        """Accept equivalent printed scalars, but never report metrics for a different GEMM."""
        benchmark = self.make_mxfp4_result_benchmark()
        self.assertTrue(benchmark._process_raw_result(0, self.mxfp4_output({'alpha': '1.00e+0', 'beta': '-0.0'})))

        for column, value in [
            ('alpha', '0'), ('beta', '1'), ('alpha', '1.0000000000000001'), ('beta', '1e-400'), ('alpha', 'nan'),
            ('beta', 'inf'), ('alpha', 'invalid')
        ]:
            with self.subTest(column=column, value=value):
                self.assert_invalid_mxfp4(self.mxfp4_output({column: value}))

    def test_mxfp4_invalid_results(self):
        """Reject wrong cases, fallback dtypes, invalid metrics, and failed CPU verification."""
        for column, value in [
            ('m', '256'), ('n', '256'), ('k', '128'), ('batch_count', '2'), ('transA', 'N'), ('transB', 'T'),
            ('grouped_gemm', '1'), ('a_type', 'f16_r'), ('b_type', 'f16_r'), ('c_type', 'f16_r'), ('d_type', 'f16_r'),
            ('compute_type', 'f16_r'), ('scaleA', '0'), ('scaleB', '1'), ('use_gpu_timer', '0'), ('us', '0'),
            ('us', '-1'), ('us', 'nan'), ('us', 'inf'), ('hipblaslt-Gflops', '0'), ('hipblaslt-Gflops', '-1'),
            ('hipblaslt-Gflops', 'nan'), ('hipblaslt-Gflops', 'inf'), ('hipblaslt-Gflops', '10'),
            ('norm_error', '1e-5'), ('norm_error', '-1'), ('norm_error', 'nan'), ('norm_error', 'inf'),
            ('atol', 'failed'), ('rtol', 'failed'), ('atol', 'nan'), ('rtol', 'inf'), ('rtol', '-1')
        ]:
            with self.subTest(column=column, value=value):
                self.assert_invalid_mxfp4(self.mxfp4_output({column: value}))

    def test_mxfp4_incomplete_results(self):
        """Reject absent verification columns, malformed CSV, and multiple algorithm results."""
        output = self.mxfp4_output()
        header, row, kernel = output.splitlines()
        invalid = [
            self.mxfp4_output(missing=[column])
            for column in ['alpha', 'beta', 'a_type', 'scaleA', 'norm_error', 'atol', 'rtol', 'us', 'use_gpu_timer']
        ]
        invalid.extend(
            [
                header + '\n',
                header + '\n' + row + ',extra\n' + kernel,
                header + ',m\n' + row + ',128\n' + kernel,
                header + '\n' + row + '\n',
                output.replace('synthetic_mxfp4_kernel', '   '),
                output + output,
                output + row + '\n',
                'No supported solutions found',
            ]
        )
        for raw_output in invalid:
            with self.subTest(raw_output=raw_output):
                self.assert_invalid_mxfp4(raw_output)

    def test_hipblaslt_gemm_result_parsing(self):
        """Test hipblaslt-bench benchmark result parsing."""
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['896,896,896'], in_types=['fp16'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # Old format (hipBLASLt v600, 23 columns)
        example_raw_output = """
hipBLASLt version: 600
hipBLASLt git version: 52776da
Query device success: there are 1 devices
-------------------------------------------------------------------------------
Device ID 0 : AMD Radeon Graphics gfx942:sramecc+:xnack-
with 206.1 GB memory, max. SCLK 2100 MHz, max. MCLK 1300 MHz, compute capability 9.4
maxGridDimX 2147483647, sharedMemPerBlock 65.5 KB, maxThreadsPerBlock 1024, warpSize 64
-------------------------------------------------------------------------------

Is supported 1 / Total solutions: 1
[0]transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,d_type,compute_type,activation_type,bias_vector,hipblaslt-Gflops,us
N,N,0,1,896,896,896,1,896,802816,0,896,802816,896,802816,896,802816,fp16_r,f32_r,none,0, 58624.5, 24.54
"""
        # Positive case - valid raw output (old format)
        self.assertTrue(benchmark._process_raw_result(0, example_raw_output))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertIn('fp16_1_896_896_896_flops', benchmark.result)
        self.assertAlmostEqual(58.6245, benchmark.result['fp16_1_896_896_896_flops'][0], places=4)

        # Negative case - invalid raw output
        self.assertFalse(benchmark._process_raw_result(1, 'HipBLAS API failed'))

    def test_hipblaslt_gemm_result_parsing_new_format(self):
        """Test hipblaslt-bench benchmark result parsing with new 34-column format (hipBLASLt v1500+)."""
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['4096,4096,4096'], in_types=['fp16'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # New format (hipBLASLt v1500, 34 columns) - includes a_type, b_type, c_type, d_type,
        # scaleA, scaleB, scaleC, scaleD, amaxD, bias_type, aux_type, and hipblaslt-GB/s columns
        example_raw_output_new = """
hipBLASLt version: 1500
hipBLASLt git version: 8c69191d
Query device success: there are 1 devices. (Target device ID is 0)
Device ID 0 :  gfx942:sramecc+:xnack-
with 205.6 GB memory, max. SCLK 2100 MHz, max. MCLK 1300 MHz, compute capability 9.4
maxGridDimX 2147483647, sharedMemPerBlock 65.5 KB, maxThreadsPerBlock 1024, warpSize 64

Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,aux_type,hipblaslt-Gflops,hipblaslt-GB/s,us
    N,N,0,1,4096,4096,4096,1,4096,16777216,0,4096,16777216,4096,16777216,4096,16777216,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,none,0,f16_r,f16_r,678209,462.62,202.65
"""
        # Positive case - valid raw output (new format)
        self.assertTrue(benchmark._process_raw_result(0, example_raw_output_new))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertIn('fp16_1_4096_4096_4096_flops', benchmark.result)
        self.assertAlmostEqual(678.209, benchmark.result['fp16_1_4096_4096_4096_flops'][0], places=3)

    def test_hipblaslt_gemm_result_parsing_future_format_with_inserted_column(self):
        """Test that the parser is forward-compatible when a new column is inserted before batch_count.

        This proves the metric key is built purely from header-named columns, not fixed
        positions, so reordering or inserting columns in a future hipBLASLt release does
        not silently produce a wrong metric key.
        """
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['4096,4096,4096'], in_types=['fp16'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # Synthetic future format: a new column 'fake_new_col' is inserted before batch_count,
        # and the data values for the key fields are padded with whitespace to confirm that
        # individual field values are stripped before being used to build the metric key.
        # A minimal header is used so the padded data line stays within the 120-column limit.
        example_raw_output_future = """
hipBLASLt version: 9999
Is supported 1 / Total solutions: 1
[0]:transA,transB,fake_new_col,batch_count,m,n,k,hipblaslt-Gflops,us
N,N,FAKE, 1 , 4096 , 4096 , 4096 ,678209,202.65
"""
        self.assertTrue(benchmark._process_raw_result(0, example_raw_output_future))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        # The correct, header-driven key must be present with the correct value.
        self.assertIn('fp16_1_4096_4096_4096_flops', benchmark.result)
        self.assertAlmostEqual(678.209, benchmark.result['fp16_1_4096_4096_4096_flops'][0], places=3)

        # No key derived from the wrong (positional) field should leak through.
        for key in benchmark.result:
            self.assertNotIn('FAKE', key)
            self.assertNotIn('fake_new_col', key)

    def test_hipblaslt_gemm_result_parsing_missing_required_column(self):
        """Test that the parser fails loudly when a required key column (e.g. batch_count) is missing.

        Failing surfaces unknown output formats explicitly instead of silently producing
        a wrong metric key.
        """
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['896,896,896'], in_types=['fp16'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # batch_count is removed from both the header and the data line.
        example_raw_output_missing_col = """
hipBLASLt version: 600
Is supported 1 / Total solutions: 1
[0]transA,transB,grouped_gemm,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,d_type,compute_type,activation_type,bias_vector,hipblaslt-Gflops,us
N,N,0,896,896,896,1,896,802816,0,896,802816,896,802816,896,802816,fp16_r,f32_r,none,0, 58624.5, 24.54
"""
        self.assertFalse(benchmark._process_raw_result(0, example_raw_output_missing_col))
        self.assertEqual(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE, benchmark.return_code)
