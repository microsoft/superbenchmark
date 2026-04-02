# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Kernel Launch overhead benchmarks."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class KernelLaunch(MicroBenchmarkWithInvoke):
    """The KernelLaunch overhead benchmark class."""
    _metric_names = ['e2e_latency_us', 'host_dispatch_us', 'launch_throughput_mkps', 'device_launch_us']

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'kernel_launch_overhead'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=100,
            required=False,
            help='The number of warmup step.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=2000000,
            required=False,
            help='The number of test step.',
        )
        self._parser.add_argument(
            '--interval',
            type=int,
            default=2000,
            required=False,
            help='The interval between different kernel launch tests, unit is millisecond.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)
        command += (' -w ' + str(self._args.num_warmup))
        command += (' -n ' + str(self._args.num_steps))
        command += (' -i ' + str(self._args.interval))
        self._commands.append(command)

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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        result = {}
        pattern = re.compile(r'^(e2e_latency_us|host_dispatch_us|launch_throughput_mkps|device_launch_us):\s*(-?\d+(?:\.\d+)?)$')
        for line in raw_output.splitlines():
            match = pattern.match(line.strip())
            if match:
                result[match.group(1)] = float(match.group(2))

        if set(result.keys()) != set(self._metric_names):
            logger.error(
                'Cannot extract kernel launch benchmark metrics - round: {}, benchmark: {}, raw data: {}.'
                .format(self._curr_run_index, self._name, raw_output)
            )
            return False

        for metric in self._metric_names:
            self._result.add_result(metric, result[metric])

        return True


BenchmarkRegistry.register_benchmark('kernel-launch', KernelLaunch)
