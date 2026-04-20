# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DTK memory performance benchmarks."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MemBwBenchmark


class DtkMemBwBenchmark(MemBwBenchmark):
    """The DTK memory performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'BandwidthTest'
        self._type_map = {'htod': 0, 'dtoh': 1, 'dtod': 2}
        self._mode_map = {'pinned': 0, 'unpinned': 1}

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # SuperBench runs one process per visible GPU. Select index 0 inside that visibility mask.
        for mem_type in self._args.mem_type:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' --type {} --index 0'.format(self._type_map[mem_type])
            if mem_type != 'dtod':
                command += ' --mode {}'.format(self._mode_map[self._args.memory])
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
        self._result.add_raw_data('raw_output_' + self._args.mem_type[cmd_idx], raw_output, self._args.log_raw_data)

        mem_bw = -1
        valid = True
        number = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
        row_pattern = re.compile(
            r'^\s*\d+(?:\.\d+)?\s*(?:B|KB|MB|GB)\s+'
            r'({number})\s+({number})\s+({number})\s+({number})\s+({number})\s+({number})\s*$'.format(number=number),
            re.IGNORECASE,
        )

        try:
            metric = self._metrics[self._mem_types.index(self._args.mem_type[cmd_idx])]
            for line in raw_output.splitlines():
                match = row_pattern.match(line)
                if match:
                    mem_bw = max(mem_bw, float(match.group(2)))
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


BenchmarkRegistry.register_benchmark('mem-bw', DtkMemBwBenchmark, platform=Platform.DTK)
