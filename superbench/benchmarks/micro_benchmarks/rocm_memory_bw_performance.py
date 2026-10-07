# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the ROCm memory performance benchmarks."""

import csv
import ctypes
import math
import os
from pathlib import Path
import re
import shlex
import statistics

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MemBwBenchmark


class RocmMemBwBenchmark(MemBwBenchmark):
    """The ROCm memory performance benchmark class."""

    # HIP rocm-5.5.0 samples/1_Utils/hipBusBandwidth/hipBusBandwidth.cpp.
    _TRANSFERBENCH_SIZES = (64, 256, 512) + tuple(2**power for power in range(10, 30))
    _TRANSFERBENCH_ITERATIONS = (
        1000, 1000, 1000, 1000, 500, 500, 500, 500, 500, 200, 200, 200,
        200, 200, 100, 100, 100, 100, 50, 50, 50, 20, 20
    )
    _TRANSFERBENCH_ENV = {
        'NUM_WARMUPS': 0,
        'NUM_SUBITERATIONS': 1,
        'SHOW_ITERATIONS': 1,
        'USE_HIP_EVENTS': 1,
        'USE_HSA_DMA': 0,
        'USE_SINGLE_STREAM': 1,
        'USE_INTERACTIVE': 0,
        'ALWAYS_VALIDATE': 0,
        'VALIDATE_DIRECT': 0,
        'VALIDATE_SOURCE': 0,
        'BYTE_OFFSET': 0,
    }

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._backend = os.getenv('SB_ROCM_MEM_BW_BACKEND', 'hipbusbandwidth').lower()
        self._bin_name = 'TransferBench' if self._backend == 'transferbench' else 'hipBusBandwidth'
        self._mem_types = ['htod', 'dtoh']
        self._parse_logline_map = {'htod': 'H2D_Bandwidth', 'dtoh': 'D2H_Bandwidth'}
        self._worker_numa_node = None

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

    def _set_binary_path(self):
        """Validate explicit backend selection before looking up an executable."""
        if self._backend not in ('hipbusbandwidth', 'transferbench'):
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Unsupported ROCm mem-bw backend: {}.'.format(self._backend))
            return False
        return super()._set_binary_path()

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if self._backend == 'transferbench' and self._args.memory == 'unpinned':
            if not self._set_worker_numa_node():
                return False

        # Check the arguments and generate the commands
        for mem_type in self._args.mem_type:
            if self._backend == 'transferbench':
                self._commands.append(self._transferbench_command(mem_type))
                continue
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' --' + mem_type.replace('to', '2')
            if self._args.memory == 'unpinned':
                command += ' --unpinned'
            self._commands.append(command)

        return True

    def _set_worker_numa_node(self):
        """Resolve the executing worker CPU's NUMA node once for this sweep."""
        try:
            getcpu = ctypes.CDLL(None, use_errno=True).sched_getcpu
            getcpu.argtypes = []
            getcpu.restype = ctypes.c_int
            cpu = getcpu()
            if cpu < 0:
                raise OSError(ctypes.get_errno(), 'sched_getcpu failed')
            cpu_path = Path('/sys/devices/system/cpu/cpu{}'.format(cpu))
            nodes = [entry for entry in cpu_path.iterdir() if re.fullmatch(r'node[0-9]+', entry.name)]
            if len(nodes) != 1:
                raise ValueError('Expected one NUMA node for CPU {}, found {}'.format(cpu, len(nodes)))
            self._worker_numa_node = int(nodes[0].name[4:])
        except (AttributeError, OSError, ValueError) as error:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error(
                'Cannot determine worker-local NUMA placement - benchmark: {}, error: {}.'.format(self._name, error)
            )
            return False
        logger.info(
            'TransferBench pageable placement - benchmark: {}, worker CPU: {}, NUMA node: {}, endpoint: H{}.'.format(
                self._name, cpu, self._worker_numa_node, self._worker_numa_node
            )
        )
        return True

    def _transferbench_route(self, mem_type):
        """Return the single-device DMA route, using HIP's logical device zero."""
        host = 'P0' if self._args.memory == 'pinned' else 'H{}'.format(self._worker_numa_node)
        return (host, 'G0') if mem_type == 'htod' else ('G0', host)

    def _transferbench_command(self, mem_type):
        """Build one fail-fast sweep per direction, retaining one raw log per direction."""
        source, destination = self._transferbench_route(mem_type)
        binary = shlex.quote(os.path.join(self._args.bin_dir, self._bin_name))
        environment = dict(self._TRANSFERBENCH_ENV, OUTPUT_TO_CSV=1, HIDE_ENV=0, SHOW_PERCENTILES='')
        options = ' '.join('{}={}'.format(key, value) for key, value in environment.items())
        commands = []
        for size, iterations in zip(self._TRANSFERBENCH_SIZES, self._TRANSFERBENCH_ITERATIONS):
            commands.append(
                "env {} NUM_ITERATIONS={} {} cmdline {} '1 1 {} D0 {}'".format(
                    options, iterations, binary, size, source, destination
                )
            )
        return ' && '.join(commands)

    @staticmethod
    def _transferbench_quantity(value, unit, allow_zero=False):
        """Parse a finite numeric field with its explicit unit."""
        fields = value.split()
        if len(fields) != 2 or fields[1] != unit:
            raise ValueError('Invalid TransferBench quantity: {}'.format(value))
        number = float(fields[0])
        if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
            raise ValueError('Invalid TransferBench quantity: {}'.format(value))
        return number

    def _transferbench_samples(self, rows, size, iterations):
        """Extract every timed iteration, not the reciprocal-mean-time summary."""
        samples = {}
        for row in rows:
            if not row[0].startswith('Iter '):
                continue
            index = int(row[0].split()[1])
            if index in samples or not 1 <= index <= iterations:
                raise ValueError('Duplicate or unexpected TransferBench iteration')
            bandwidth = self._transferbench_quantity(row[1], 'GB/s', allow_zero=True)
            duration = self._transferbench_quantity(row[2], 'ms')
            # Three-decimal CLI output can round a genuine small-copy bandwidth to zero.
            if bandwidth == 0 and size / (duration + 0.0005) / 1e6 >= 0.0005:
                raise ValueError('Nonpositive TransferBench bandwidth')
            # Preserve HIP5.5's integer nbytes/1000 division, including sub-KB zeros.
            samples[index] = (size // 1000) / 1000 / duration
        if len(samples) != iterations:
            raise ValueError('Incomplete TransferBench iterations')
        return statistics.mean(samples.values())

    def _transferbench_report(self, report, size, iterations, mem_type):
        """Validate one CSV-mode report, which also contains banners and typed table rows."""
        rows = [[field.strip() for field in row] for row in csv.reader(report.splitlines()) if row]
        settings = dict(self._TRANSFERBENCH_ENV, NUM_ITERATIONS=iterations)
        for key, value in settings.items():
            matches = [row for row in rows if row[0] == key]
            if len(matches) != 1 or int(matches[0][1]) != value:
                raise ValueError('Unexpected TransferBench setting: {}'.format(key))

        executors = [row for row in rows if row[0].startswith('Executor:')]
        transfers = [row for row in rows if row[0].startswith('Transfer ')]
        aggregates = [row for row in rows if row[0] == 'Aggregate (CPU)']
        deviations = [row for row in rows if row[0] == 'StandardDev']
        if len(executors) != 1 or len(transfers) != 1 or len(aggregates) != 1 or len(deviations) != 1:
            raise ValueError('Incomplete or multi-device TransferBench report')
        if executors[0][0] != 'Executor: DMA 00' or transfers[0][0] != 'Transfer 0':
            raise ValueError('Unexpected TransferBench executor')
        source, destination = self._transferbench_route(mem_type)
        if transfers[0][4].split() != [source, '->', 'D0:1', '->', destination]:
            raise ValueError('Unexpected TransferBench route')
        for row in (executors[0], transfers[0], aggregates[0]):
            if row[3].split() != [str(size), 'bytes']:
                raise ValueError('Unexpected TransferBench byte count')
        return self._transferbench_samples(rows, size, iterations)

    def _process_transferbench_result(self, cmd_idx, raw_output):
        """Reduce a complete size sweep to the maximum arithmetic mean bandwidth."""
        try:
            if '[ERROR]' in raw_output:
                raise ValueError('TransferBench reported an error')
            reports = re.split(r'(?m)^TransferBench v[^\n]*\n', raw_output)
            if len(reports) != len(self._TRANSFERBENCH_SIZES) + 1:
                raise ValueError('Incomplete TransferBench size sweep')
            means = [
                self._transferbench_report(report, size, iterations, self._args.mem_type[cmd_idx])
                for report, size, iterations in zip(
                    reports[1:], self._TRANSFERBENCH_SIZES, self._TRANSFERBENCH_ITERATIONS
                )
            ]
            bandwidth = max(means)
            if not math.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError('Nonpositive TransferBench result')
        except (ValueError, IndexError, OverflowError) as error:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, error: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, error, raw_output
                )
            )
            return False
        metric = self._metrics[self._mem_types.index(self._args.mem_type[cmd_idx])]
        self._result.add_result(metric, bandwidth)
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + self._args.mem_type[cmd_idx], raw_output, self._args.log_raw_data)

        if self._backend == 'transferbench':
            return self._process_transferbench_result(cmd_idx, raw_output)

        mem_bw = -1
        value_index = -1
        valid = True
        content = raw_output.splitlines()
        try:
            metric = self._metrics[self._mem_types.index(self._args.mem_type[cmd_idx])]
            parse_logline = self._parse_logline_map[self._args.mem_type[cmd_idx]]
            for line in content:
                if parse_logline in line and value_index != -1:
                    line = line.split()
                    mem_bw = max(mem_bw, float(line[value_index]))
                elif 'mean' in line:
                    line = line.split()
                    value_index = line.index('mean')
        except BaseException:
            valid = False
        finally:
            if valid is False or mem_bw == -1:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False
        self._result.add_result(metric, mem_bw)
        return True


BenchmarkRegistry.register_benchmark('mem-bw', RocmMemBwBenchmark, platform=Platform.ROCM)
