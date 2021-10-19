# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPCNet benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GPCNetBenchmark(MicroBenchmarkWithInvoke):
    """The TCP connectivity performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        if self._name == 'gpcnet-network-test':
            self._bin_name = 'network_test'
        if self._name == 'gpcnet-network-load':
            self._bin_name = 'network_load_test'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)
        self._commands.append(command)

        return True

    def _process_raw_result(self, idx, raw_output):    # noqa: C901
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            idx (int): the index corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + str(idx), raw_output)

        try:
            # Parse and add result
            if 'ERROR' not in raw_output:
                raw_output = raw_output.splitlines()
                labels = None
                label_len = -1
                test_name = ''
                for line in raw_output:
                    if not line.startswith('|'):
                        continue
                    items = line.split('|')
                    items = [item.strip() for item in items]
                    # Get table name
                    if len(items) == 3 and 'Tests' in items[1]:
                        test_name = items[1].replace(' ', '')
                    # Get the line of the table labels
                    elif 'Avg' in line:
                        label_len = len(items)
                        labels = items
                    # Get values related to the labels
                    else:
                        if len(items) == label_len:
                            name_prefix = items[1].replace(' ', '')
                            for i in range(2, len(items) - 1):
                                if labels[i] != 'Units':
                                    self._result.add_result(
                                        test_name + '_' + name_prefix + '_' + labels[i], float(items[i].strip('X'))
                                    )
            else:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False
        except Exception as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('gpcnet-network-test', GPCNetBenchmark)
BenchmarkRegistry.register_benchmark('gpcnet-network-load', GPCNetBenchmark)
