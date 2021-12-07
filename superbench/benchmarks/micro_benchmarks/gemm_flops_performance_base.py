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
            'FP64', 'FP32', 'FP16', 'FP64_TC', 'TF32_TC', 'BF16_TC', 'FP16_TC', 'INT8_TC', 'INT4_TC'
        ]
        self._precision_need_to_run = list()
        self._metric_map = {
            'FP64': 'fp64_flops',
            'FP32': 'fp32_flops',
            'FP16': 'fp16_flops',
            'FP64_TC': 'fp64_tc_flops',
            'TF32_TC': 'tp32_tc_flops',
            'BF16_TC': 'bf16_tc_flops',
            'FP16_TC': 'fp16_tc_flops',
            'INT8_TC': 'int8_tc_iops',
            'INT4_TC': 'int4_tc_iops',
            'FP32_xDLOPS': 'fp32_xdlops_flops',
            'FP16_xDLOPS': 'fp16_xdlops_flops',
            'BF16_xDLOPS': 'bf16_xdlops_flops',
            'INT8_xDLOPS': 'int8_xdlops_flops'
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
            self._args.precision = [p.upper() for p in self._args.precision]
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
