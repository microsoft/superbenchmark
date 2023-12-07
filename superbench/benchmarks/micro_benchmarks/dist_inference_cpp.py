# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the distributed inference (cpp implementation) benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke
from superbench.benchmarks.reducer import ReduceType


class DistInferenceCpp(MicroBenchmarkWithInvoke):
    """The distributed inference (cpp implementation) benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'dist_inference'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--mnk_list',
            type=str,
            nargs='+',
            default=['128,80,128'],
            help='List of M(input),N(batch),K(hidden) tuples of sharded 2-layer-feed-forward block. E.g. {}.'.format(
                ' '.join(self._mnk_list)
            ),
        )
        self._parser.add_argument(
            '--alpha',
            type=float,
            default=1.0,
            required=False,
            help='Coefficient alpha in D = alpha*(A*B) + beta*(C).',
        )
        self._parser.add_argument(
            '--beta',
            type=float,
            default=1.0,
            required=False,
            help='Coefficient beta in D = alpha*(A*B) + beta*(C).',
        )
        self._parser.add_argument(
            '--num_layers',
            type=int,
            default=50,
            required=False,
            help='Number of layers in the model.',
        )
        self._parser.add_argument(
            '--num_warmups',
            type=int,
            default=20,
            required=False,
            help='Number of warmup runs.',
        )
        self._parser.add_argument(
            '--num_iters',
            type=int,
            default=100,
            required=False,
            help='Number of test runs.',
        )
        self._parser.add_argument(
            '--use_cuda_graph',
            action='store_true',
            help='Whether to launch kernels in CUDA graph mode.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        self._commands = []

        for mnk in self._args.mnk_list:
            m, n, k = [int(x) for x in mnk.split(',')]
            args = '-m %d -n %d -k %d' % (m, n, k)
            args += ' --alpha %g --beta %g' % (self._args.alpha, self._args.beta)
            args += ' --num_layers %d --num_warmups %d --num_iters %d' % \
                (self._args.num_layers, self._args.num_warmups, self._args.num_iters)
            if self._args.use_cuda_graph:
                args += ' --use_cuda_graph'
            self._commands.append('%s %s' % (self.__bin_path, args))

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

        try:
            output_lines = [x.strip() for x in raw_output.strip().splitlines()]
            for output_line in output_lines:
                if output_line.endswith(' us per layer'):
                    layer_latency = float(output_line.split(' us per layer')[0].split()[-1])
                    break
            return self._process_numeric_result(
                'layer_latency_%s' % self._args.mnk_list[cmd_idx], [layer_latency],
                reduce_type=ReduceType.MAX,
                cal_percentile=True
            )
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False


BenchmarkRegistry.register_benchmark('dist-inference-cpp', DistInferenceCpp, parameters='')
