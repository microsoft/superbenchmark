# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mem-bw benchmark."""

import numbers
import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.micro_benchmarks.rocm_memory_bw_performance import RocmMemBwBenchmark


class RocmMemBwTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for rocm mem-bw benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/hipBusBandwidth', 'bin/TransferBench'])

    def setUp(self):
        """Keep the legacy default independent of the image's backend selection."""
        self.environment = mock.patch.dict(os.environ, {'SB_ROCM_MEM_BW_BACKEND': 'hipbusbandwidth'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        module = 'superbench.benchmarks.micro_benchmarks.rocm_memory_bw_performance'
        libc_patch = mock.patch(module + '.ctypes.CDLL')
        self.libc = libc_patch.start()
        self.addCleanup(libc_patch.stop)
        self.libc.return_value.sched_getcpu.return_value = 0
        nodes_patch = mock.patch(module + '.Path.iterdir', autospec=True)
        self.cpu_nodes = nodes_patch.start()
        self.addCleanup(nodes_patch.stop)
        self.cpu_nodes.return_value = [Path('/sys/devices/system/cpu/cpu0/node0')]

    def _transferbench(self, parameters=''):
        """Prepare the explicitly selected backend with mock executables."""
        with mock.patch.dict(os.environ, {'SB_ROCM_MEM_BW_BACKEND': 'transferbench'}):
            benchmark = RocmMemBwBenchmark('mem-bw', parameters)
        self.assertTrue(benchmark._preprocess())
        return benchmark

    @staticmethod
    def _fixture(memory='pinned', direction='htod'):
        """Read captured TransferBench v1.69.00 GPU2 output, logical device zero."""
        return Path('tests/data/rocm_transferbench_{}_{}.log'.format(memory, direction)).read_text()

    def _report(self, size, durations, direction='htod', memory='pinned'):
        """Build variable-size reports using the measured CLI format, including misleading summaries."""
        header = self._fixture(memory, direction).split(' Executor:')[0]
        header = re.sub(r'(NUM_ITERATIONS\s*,\s*)3,', r'\g<1>{},'.format(len(durations)), header)
        host = 'P0' if memory == 'pinned' else 'H0'
        route = '{} -> D0:1 -> G0'.format(host) if direction == 'htod' else 'G0 -> D0:1 -> {}'.format(host)
        rows = [
            'Executor: DMA 00, 999.000 GB/s, 1.000 ms, {} bytes, 999.000 GB/s (sum),'.format(size),
            'Transfer 0, 888.000 GB/s, 1.000 ms, {} bytes, {},'.format(size, route),
        ]
        for index, duration in reversed(list(enumerate(durations, 1))):
            rows.append('Iter {:03d}, {:.3f} GB/s, {:.3f} ms, , ,'.format(index, size / duration / 1e6, duration))
        rows.extend([
            'StandardDev, 0.000 GB/s, 0.000 ms, , ,',
            'Aggregate (CPU), 777.000 GB/s, 1.000 ms, {} bytes, Overhead 0.000 ms,'.format(size),
        ])
        return header + '\n'.join(rows) + '\n'

    @decorator.load_data('tests/data/rocm_memory_h2d_bw.log')
    @decorator.load_data('tests/data/rocm_memory_d2h_bw.log')
    def test_rocm_memory_bw_performance(self, raw_output_h2d, raw_output_d2h):
        """Test rocm mem-bw benchmark."""
        benchmark_name = 'mem-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.ROCM)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name)

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'mem-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list
        expected_command = ['hipBusBandwidth --h2d', 'hipBusBandwidth --d2h']
        for i in range(len(expected_command)):
            commnad = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (commnad == expected_command[i])

        # Check results and metrics.
        raw_output = [raw_output_h2d, raw_output_d2h]
        for i, metric in enumerate(['h2d_bw', 'd2h_bw']):
            assert (benchmark._process_raw_result(i, raw_output[i]))
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark.result['h2d_bw'][0] == 25.2351)
        assert (benchmark.result['d2h_bw'][0] == 27.9348)

    def test_legacy_default_and_arguments(self):
        """Retain the old executable, flags, normalization and ignored NUMA argument."""
        with mock.patch.dict(os.environ):
            os.environ.pop('SB_ROCM_MEM_BW_BACKEND', None)
            benchmark = RocmMemBwBenchmark('mem-bw', '--mem_type DTOH --memory unpinned --numa 3')
        self.assertTrue(benchmark._preprocess())
        self.assertEqual(benchmark._bin_name, 'hipBusBandwidth')
        self.assertEqual(len(benchmark._commands), 1)
        self.assertTrue(benchmark._commands[0].endswith('hipBusBandwidth --d2h --unpinned'))
        self.assertEqual(benchmark._args.numa, 3)
        self.assertNotIn('numactl', benchmark._commands[0])

    def test_transferbench_exact_legacy_policy(self):
        """Cover the complete HIP5.5 size/trial arrays and all controllable timing settings."""
        benchmark = self._transferbench('--numa 3')
        sizes = (
            64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
            524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
            134217728, 268435456, 536870912
        )
        iterations = (
            1000, 1000, 1000, 1000, 500, 500, 500, 500, 500, 200, 200, 200,
            200, 200, 100, 100, 100, 100, 50, 50, 50, 20, 20
        )
        self.assertEqual(benchmark._TRANSFERBENCH_SIZES, sizes)
        self.assertEqual(benchmark._TRANSFERBENCH_ITERATIONS, iterations)
        self.assertEqual(len(benchmark._commands), 2)
        for direction, command in enumerate(benchmark._commands):
            commands = command.split(' && ')
            self.assertEqual(len(commands), 23)
            for subcommand, size, count in zip(commands, sizes, iterations):
                tokens = shlex.split(subcommand)
                self.assertEqual(tokens[-3:], [
                    'cmdline', str(size), '1 1 P0 D0 G0' if direction == 0 else '1 1 G0 D0 P0'
                ])
                for setting in (
                    'NUM_ITERATIONS={}'.format(count), 'NUM_WARMUPS=0', 'NUM_SUBITERATIONS=1',
                    'SHOW_ITERATIONS=1', 'OUTPUT_TO_CSV=1', 'USE_HIP_EVENTS=1', 'USE_HSA_DMA=0',
                    'USE_INTERACTIVE=0', 'ALWAYS_VALIDATE=0'
                ):
                    self.assertIn(setting, tokens)
            self.assertNotIn('numactl', command)
            self.assertNotIn('G1', command)
        self.assertEqual(benchmark._args.numa, 3)

    def test_transferbench_pageable_subset(self):
        """Preserve memory and direction selection without introducing new CLI arguments."""
        benchmark = self._transferbench('--memory unpinned --mem_type DTOH')
        self.assertEqual(len(benchmark._commands), 1)
        self.assertIn("'1 1 G0 D0 H0'", benchmark._commands[0])
        self.assertNotIn('P0', benchmark._commands[0])

    def test_transferbench_numa_does_not_override_logical_gpu_locality(self):
        """P0 means the visible GPU's nearest node, not physical GPU zero or NUMA zero."""
        with mock.patch.dict(os.environ, {'HIP_VISIBLE_DEVICES': '5'}):
            for memory, host in (('pinned', 'P0'), ('unpinned', 'H0')):
                with self.subTest(memory=memory):
                    baseline = self._transferbench('--memory {}'.format(memory))
                    explicit = self._transferbench('--memory {} --numa 1'.format(memory))
                    self.assertEqual(baseline._commands, explicit._commands)
                    self.assertIn("'1 1 {} D0 G0'".format(host), explicit._commands[0])
                    self.assertIn("'1 1 G0 D0 {}'".format(host), explicit._commands[1])
                    self.assertNotIn('C0', explicit._commands[0])
                    self.assertNotIn('G5', explicit._commands[0])
                    self.assertEqual(explicit._args.numa, 1)

    def test_transferbench_pageable_worker_nodes(self):
        """Resolve node0/node1 from the worker CPU, not --numa or the GPU index."""
        for cpu, node in ((0, 0), (48, 1)):
            with self.subTest(cpu=cpu, node=node):
                self.libc.return_value.sched_getcpu.return_value = cpu
                self.cpu_nodes.return_value = [Path('/sys/devices/system/cpu/cpu{}/node{}'.format(cpu, node))]
                benchmark = self._transferbench('--memory unpinned --numa 3')
                self.cpu_nodes.assert_called_with(Path('/sys/devices/system/cpu/cpu{}'.format(cpu)))
                self.assertEqual(benchmark._worker_numa_node, node)
                self.assertIn("'1 1 H{} D0 G0'".format(node), benchmark._commands[0])
                self.assertIn("'1 1 G0 D0 H{}'".format(node), benchmark._commands[1])
                benchmark._TRANSFERBENCH_SIZES = (4194304,)
                benchmark._TRANSFERBENCH_ITERATIONS = (3,)
                raw = self._fixture('unpinned').replace('H0', 'H{}'.format(node))
                self.assertTrue(benchmark._process_raw_result(0, raw))
                wrong_node = self._fixture('unpinned').replace('H0', 'H{}'.format(1 - node))
                self.assertFalse(benchmark._process_raw_result(0, wrong_node))
                self.assertEqual(len(benchmark.result['h2d_bw']), 1)
                self.assertEqual(benchmark._args.numa, 3)

    def test_transferbench_pageable_detection_errors(self):
        """Fail explicitly on unavailable CPU detection or missing/ambiguous/inaccessible topology."""
        module = 'superbench.benchmarks.micro_benchmarks.rocm_memory_bw_performance'
        for error in ('library', 'symbol', 'cpu', 'missing', 'ambiguous', 'permission'):
            with self.subTest(error=error):
                self.libc.side_effect = None
                self.libc.return_value.sched_getcpu.side_effect = None
                self.libc.return_value.sched_getcpu.return_value = 0
                self.cpu_nodes.side_effect = None
                self.cpu_nodes.return_value = [Path('node0')]
                if error == 'library':
                    self.libc.side_effect = OSError('libc unavailable')
                elif error == 'symbol':
                    self.libc.return_value.sched_getcpu.side_effect = AttributeError('sched_getcpu unavailable')
                elif error == 'cpu':
                    self.libc.return_value.sched_getcpu.return_value = -1
                elif error == 'missing':
                    self.cpu_nodes.return_value = [Path('topology'), Path('node-invalid')]
                elif error == 'ambiguous':
                    self.cpu_nodes.return_value = [Path('node0'), Path('node1')]
                else:
                    self.cpu_nodes.side_effect = PermissionError('sysfs denied')
                with mock.patch.dict(os.environ, {'SB_ROCM_MEM_BW_BACKEND': 'transferbench'}):
                    benchmark = RocmMemBwBenchmark('mem-bw', '--memory unpinned')
                with mock.patch(module + '.logger.error') as log:
                    self.assertFalse(benchmark._preprocess())
                self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
                self.assertEqual(benchmark._commands, [])
                self.assertIsNone(benchmark._worker_numa_node)
                self.assertIn('Cannot determine worker-local NUMA placement', log.call_args[0][0])

    def test_pinned_and_legacy_do_not_detect_worker_numa(self):
        """Keep pinned P0 GPU-local and leave both legacy allocation modes unchanged."""
        self.libc.side_effect = OSError('CPU detection unavailable')
        benchmark = self._transferbench('--memory pinned --numa 1')
        self.assertIn("'1 1 P0 D0 G0'", benchmark._commands[0])
        for memory in ('pinned', 'unpinned'):
            with self.subTest(memory=memory):
                benchmark = RocmMemBwBenchmark('mem-bw', '--memory {} --numa 1'.format(memory))
                self.assertTrue(benchmark._preprocess())
                self.assertNotIn('numactl', benchmark._commands[0])
        self.libc.assert_not_called()

    def test_transferbench_measured_fixtures(self):
        """Use real pinned/pageable output, not flat CSV or aggregate executor bandwidth."""
        expected_times = {
            ('pinned', 'htod'): (0.100, 0.104, 23.315),
            ('pinned', 'dtoh'): (0.093, 0.094, 23.555),
            ('unpinned', 'htod'): (0.131, 0.139, 23.634),
            ('unpinned', 'dtoh'): (0.133, 0.135, 23.265),
        }
        for (memory, direction), times in expected_times.items():
            with self.subTest(memory=memory, direction=direction):
                benchmark = self._transferbench('--memory {} --mem_type {}'.format(memory, direction))
                benchmark._TRANSFERBENCH_SIZES = (4194304,)
                benchmark._TRANSFERBENCH_ITERATIONS = (3,)
                raw = self._fixture(memory, direction)
                self.assertTrue(benchmark._process_raw_result(0, raw))
                metric = 'h2d_bw' if direction == 'htod' else 'd2h_bw'
                self.assertEqual(len(benchmark.result[metric]), 1)
                self.assertAlmostEqual(benchmark.result[metric][0], sum(4.194 / time for time in times) / 3)
                self.assertEqual(benchmark.raw_data['raw_output_' + direction], [raw])

    def test_transferbench_arithmetic_mean_then_max(self):
        """Neither fastest sample, inverse mean duration nor aggregate rows are the metric."""
        benchmark = self._transferbench()
        benchmark._TRANSFERBENCH_SIZES = (1000000, 2000000)
        benchmark._TRANSFERBENCH_ITERATIONS = (2, 2)
        raw = self._report(1000000, (1, 0.125)) + self._report(2000000, (0.5, 0.5))
        self.assertTrue(benchmark._process_raw_result(0, raw))
        self.assertEqual(benchmark.result['h2d_bw'], [4.5])

    def test_transferbench_full_sweep_and_integer_conversion(self):
        """Include all trial IDs, sub-KB zero contributions and old integer-byte truncation."""
        benchmark = self._transferbench()
        raw = ''.join(
            self._report(size, [1] * count)
            for size, count in zip(benchmark._TRANSFERBENCH_SIZES, benchmark._TRANSFERBENCH_ITERATIONS)
        )
        self.assertTrue(benchmark._process_raw_result(0, raw))
        self.assertEqual(benchmark.result['h2d_bw'], [536.870])
        benchmark._TRANSFERBENCH_SIZES = (64, 1024)
        benchmark._TRANSFERBENCH_ITERATIONS = (2, 2)
        raw = self._report(64, (16, 1)) + self._report(1024, (1, 1))
        self.assertTrue(benchmark._process_raw_result(0, raw))
        self.assertEqual(benchmark.result['h2d_bw'][-1], 0.001)

    def test_transferbench_rejects_incomplete_and_wrong_reports(self):
        """Reject missing/extra sizes, samples, executors, settings and incorrect routes."""
        raw = self._fixture()
        corruptions = [
            '', raw + raw, raw[:raw.index('Aggregate (CPU)')],
            re.sub(r'^.*Iter 001.*\n', '', raw, flags=re.MULTILINE),
            raw.replace('Iter 001', 'Iter 002'), raw.replace('Iter 001', 'Iter 004'),
            raw.replace('DMA 00', 'DMA 01'), raw.replace('DMA 00', 'GFX 00'),
            raw.replace('Transfer 0', 'Transfer 1'), raw.replace('G0', 'G1'),
            raw.replace('D0:1', 'D1:1'), raw.replace('P0', 'H0'),
            raw.replace('P0 -> D0:1 -> G0', 'G0 -> D0:1 -> P0'),
            raw.replace('4194304 bytes', '4194305 bytes'),
            raw.replace('NUM_WARMUPS         ,           0', 'NUM_WARMUPS         ,           3'),
            raw.replace('NUM_ITERATIONS      ,           3', 'NUM_ITERATIONS      ,           2'),
            raw + '[ERROR] validation failed\n',
            raw.replace('Transfer 0', 'Transfer 0, bad'),
            raw.replace('StandardDev', 'missing'),
            raw.replace('      Iter 001', 'Transfer 1, 1 GB/s, 1 ms, 4194304 bytes, P0 -> D0:1 -> G0\nIter 001'),
        ]
        for index, corrupted in enumerate(corruptions):
            with self.subTest(case=index):
                benchmark = self._transferbench()
                benchmark._TRANSFERBENCH_SIZES = (4194304,)
                benchmark._TRANSFERBENCH_ITERATIONS = (3,)
                self.assertFalse(benchmark._process_raw_result(0, corrupted))
                self.assertNotIn('h2d_bw', benchmark.result)
                self.assertEqual(benchmark.raw_data['raw_output_htod'], [corrupted])

    def test_transferbench_rejects_invalid_numeric_samples(self):
        """Reject nonfinite/nonpositive timings and nonphysical bandwidths before reduction."""
        raw = self._fixture()
        for field in ('42.133 GB/s', '0.100 ms'):
            for invalid in ('nan', 'inf', '-inf', '-1', '0', 'bogus'):
                with self.subTest(field=field, invalid=invalid):
                    benchmark = self._transferbench()
                    benchmark._TRANSFERBENCH_SIZES = (4194304,)
                    benchmark._TRANSFERBENCH_ITERATIONS = (3,)
                    corrupted = raw.replace(field, invalid + ' ' + field.split()[1])
                    self.assertFalse(benchmark._process_raw_result(0, corrupted))
                    self.assertNotIn('h2d_bw', benchmark.result)

    def test_transferbench_rejects_partial_size_sweep(self):
        """Do not accept a valid single-size run as a full legacy-policy sweep."""
        benchmark = self._transferbench()
        self.assertFalse(benchmark._process_raw_result(0, self._fixture()))

    def test_transferbench_failure_does_not_fall_back(self):
        """Execution and parse errors use framework return codes without invoking hipBusBandwidth."""
        for code, raw, expected in (
            (1, '[ERROR] allocation failed', ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE),
            (0, 'malformed', ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE),
        ):
            with self.subTest(code=code):
                benchmark = self._transferbench()
                with mock.patch(
                    'superbench.benchmarks.micro_benchmarks.micro_base.run_command',
                    return_value=subprocess.CompletedProcess([], code, raw)
                ) as invoke:
                    self.assertFalse(benchmark._benchmark())
                self.assertEqual(benchmark.return_code, expected)
                self.assertEqual(invoke.call_count, 1)
                self.assertNotIn('hipBusBandwidth', invoke.call_args[0][0])

    def test_transferbench_missing_binary_and_unknown_backend(self):
        """Explicit selection is never replaced by an available legacy binary."""
        with mock.patch.dict(os.environ, {'SB_ROCM_MEM_BW_BACKEND': 'transferbench'}):
            benchmark = RocmMemBwBenchmark('mem-bw')
        with mock.patch('shutil.which', return_value=None):
            self.assertFalse(benchmark._preprocess())
        self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_BINARY_NOT_EXIST)
        with mock.patch.dict(os.environ, {'SB_ROCM_MEM_BW_BACKEND': 'invalid'}):
            benchmark = RocmMemBwBenchmark('mem-bw')
        with mock.patch('shutil.which', return_value=None) as lookup:
            self.assertFalse(benchmark._preprocess())
        lookup.assert_not_called()
        self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)

    def test_transferbench_tolerant_failure_keeps_direction_mapping(self):
        """An invalid H2D result must not mislabel a subsequent successful D2H result."""
        benchmark = self._transferbench('--tolerant_fail')
        benchmark._TRANSFERBENCH_SIZES = (4194304,)
        benchmark._TRANSFERBENCH_ITERATIONS = (3,)
        results = [
            subprocess.CompletedProcess([], 0, 'invalid'),
            subprocess.CompletedProcess([], 0, self._fixture(direction='dtoh')),
        ]
        with mock.patch(
            'superbench.benchmarks.micro_benchmarks.micro_base.run_command', side_effect=results
        ) as invoke:
            self.assertFalse(benchmark._benchmark())
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
        self.assertNotIn('h2d_bw', benchmark.result)
        self.assertEqual(len(benchmark.result['d2h_bw']), 1)

    def test_transferbench_shell_quoting_and_failure_chain(self):
        """Quote explicit binary paths and stop a failed sweep before subsequent sizes."""
        binary_dir = Path(self._tmp_dir) / 'bin with spaces'
        binary_dir.mkdir(exist_ok=True)
        binary = binary_dir / 'TransferBench'
        binary.write_text('#!/bin/sh\necho "[ERROR] synthetic failure"\nexit 3\n')
        binary.chmod(0o755)
        benchmark = self._transferbench('--bin_dir {}'.format(shlex.quote(str(binary_dir))))
        result = subprocess.run(benchmark._commands[0], shell=True, text=True, stdout=subprocess.PIPE, timeout=5)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout.count('[ERROR] synthetic failure'), 1)

    def test_transferbench_invalid_memory_and_direction(self):
        """Retain existing validation for unsupported YAML/CLI options."""
        for parameters in ('--memory registered', '--mem_type dtod', '--memory PINNED'):
            with self.subTest(parameters=parameters):
                with mock.patch.dict(os.environ, {'SB_ROCM_MEM_BW_BACKEND': 'transferbench'}):
                    benchmark = RocmMemBwBenchmark('mem-bw', parameters)
                self.assertFalse(benchmark._preprocess())
                self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)
