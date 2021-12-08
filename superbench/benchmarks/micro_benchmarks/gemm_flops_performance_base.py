# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmark base class."""

from superbench.common.utils import logger
from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GemmFlopsBenchmark(MicroBenchmarkWithInvoke):
    """The GEMM FLOPs performance benchmark base class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._support_precisions = [
            'fp64', 'fp32', 'fp16', 'fp64_tc', 'tf32_tc', 'bf16_tc', 'fp16_tc', 'int8_tc', 'int4_tc'
        ]
        self._precision_need_to_run = list()
        self._metric_map = {
            'fp64': 'fp64_flops',
            'fp32': 'fp32_flops',
            'fp16': 'fp16_flops',
            'fp64_tc': 'fp64_tc_flops',
            'tf32_tc': 'tf32_tc_flops',
            'bf16_tc': 'bf16_tc_flops',
            'fp16_tc': 'fp16_tc_flops',
            'int8_tc': 'int8_tc_iops',
            'int4_tc': 'int4_tc_iops',
            'fp32_xdlops': 'fp32_xdlops_flops',
            'fp16_xdlops': 'fp16_xdlops_flops',
            'bf16_xdlops': 'bf16_xdlops_flops',
            'int8_xdlops': 'int8_xdlops_iops'
        }

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=5,
            required=False,
            help='The number of warmup step.',
        )
        self._parser.add_argument(
            '--n',
            type=int,
            default=16384,
            required=False,
            help='The N dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--k',
            type=int,
            default=16384,
            required=False,
            help='The K dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=16384,
            required=False,
            help='The M dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--precision',
            type=str,
            nargs='+',
            default=list(),
            help='Precision for benchmarking. E.g. {}.'.format(' '.join(self._support_precisions)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if len(self._args.precision) == 0:
            self._precision_need_to_run = self._support_precisions
        else:
            self._args.precision = [p.lower() for p in self._args.precision]
            for p in self._args.precision:
                if p not in self._support_precisions:
                    logger.warning(
                        'Unsupported precision - benchmark: {}, precision: {}, expected: {}.'.format(
                            self._name, p, self._support_precisions
                        )
                    )
                else:
                    self._precision_need_to_run.append(p)

        if len(self._precision_need_to_run) == 0:
            self._result.set_return_code(ReturnCode.NO_SUPPORTED_PRECISION)
            return False

        return True
