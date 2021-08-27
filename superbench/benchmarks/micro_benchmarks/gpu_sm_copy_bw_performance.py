# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU SM Copy Bandwidth Performance benchmark."""

from pathlib import Path
import json
import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuSmCopyBwBenchmark(MicroBenchmarkWithInvoke):
    """The GPU SM copy bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'gpu_sm_copy'

        self.__result_tags = []

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--numa_nodes',
            type=int,
            nargs='*',
            default=[],
            required=False,
            help='NUMA nodes to cover.',
        )

        self._parser.add_argument(
            '--gpu_ids',
            type=int,
            nargs='*',
            default=[],
            required=False,
            help='Device IDs of GPUs to cover.',
        )

        self._parser.add_argument(
            '--enable_dtoh',
            action='store_true',
            required=False,
            help='Enable device-to-host bandwidth test.',
        )

        self._parser.add_argument(
            '--enable_htod',
            action='store_true',
            required=False,
            help='Enable host-to-device bandwidth test.',
        )

        self._parser.add_argument(
            '--size',
            type=int,
            default=64 * 1024**2,
            required=False,
            help='Size of data buffer.',
        )

        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=100,
            required=False,
            help='Number of data buffer copies performed.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        gpu_sm_copy_path = os.path.join(self._args.bin_dir, self._bin_name)

        copy_directions = []
        if self._args.enable_dtoh:
            copy_directions.append("dtoh")
        if self._args.enable_htod:
            copy_directions.append("htod")

        for numa_node in self._args.numa_nodes:
            for gpu_id in self._args.gpu_ids:
                for copy_direction in copy_directions:
                    command = "numactl -N %d -m %d %s %d %s %d %d" % \
                        (numa_node, numa_node, gpu_sm_copy_path, gpu_id,
                         copy_direction, self._args.size, self._args.num_loops)
                    self.__result_tags.append(
                        "gpu_sm_copy_performance:numa%d:gpu%d:%s" % \
                        (numa_node, gpu_id, copy_direction))
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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output)

        try:
            output_prefix = "Bandwidth (GB/s): "
            assert (raw_output.startswith(output_prefix))
            self.__result_tags[cmd_idx] = float(raw_output[len(output_prefix):])
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('gpu-sm-copy-bw', GpuSmCopyBwBenchmark)
