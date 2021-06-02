# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.common.utils import nv_helper
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GemmFlopsCuda(MicroBenchmarkWithInvoke):
    """The GEMM FLOPs performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'cutlass_profiler'

        self.__kernel_map = {
            'FP64': 'cutlass_simt_dgemm_128x128_8x2_*',
            'FP32': 'cutlass_simt_sgemm_128x128_8x2_*',
            'FP16': 'cutlass_simt_hgemm_256x128_8x2_*',
            'FP64_TC': 'cutlass_tensorop_d884gemm_128x128_16x3_*',
            'TF32_TC': 'cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_*',
            'BF16_TC': 'cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3_*',
            'FP16_TC': 'cutlass_tensorop_h16816gemm_256x128_32x3_*',
            'INT8_TC': 'cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_*',
            'INT4_TC': 'cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_*',
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
            default=list(self.__kernel_map.keys()),
            help='Precision for benchmarking. E.g. {}.'.format(' '.join(list(self.__kernel_map.keys()))),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self._args.precision = [p.upper() for p in self._args.precision]
        for p in self._args.precision:
            if p not in list(self.__kernel_map.keys()):
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported precision - benchmark: {}, precision: {}, expected: {}.'.format(
                        self._name, p, list(self.__kernel_map.keys())
                    )
                )
                return False
            else:
                command = os.path.join(self._args.bin_dir, self._bin_name)
                command += (' --warmup-iterations=' + str(self._args.num_warmup))
                command += (' --operation=gemm')
                command += (' --n=' + str(self._args.n))
                command += (' --k=' + str(self._args.k))
                command += (' --m=' + str(self._args.m))
                command += (' --kernels=' + self.__kernel_map[p])
                self._commands.append(command)

        # TODO - To support more architecutres, currently only support compute capability = 7.0 or 8.0
        capability = nv_helper.get_device_compute_capability()
        if capability == 7.0:
            self.__kernel_map['FP16_TC'] = 'cutlass_tensorop_h884gemm_256x128_32x2_*'

        if capability not in [7.0, 8.0]:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE)
            logger.error(
                'Unsupported architecture - benchmark: {}, compute capability: {}, expected: 7.0 or 8.0'.format(
                    self._name, capability
                )
            )
            return False

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
        precision = self._args.precision[cmd_idx]
        self._result.add_raw_data('raw_output_' + precision, raw_output)

        valid = True
        flops = list()
        content = raw_output.splitlines()
        try:
            for line in content:
                if 'gemm,cutlass_simt_dgemm_128x128_8x2' in line or \
                   'gemm,cutlass_simt_sgemm_128x128_8x2' in line or \
                   'gemm,cutlass_simt_hgemm_256x128_8x2' in line or \
                   'gemm,cutlass_tensorop_d884gemm_128x128_16x3' in line or \
                   'gemm,cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3' in line or \
                   'gemm,cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3' in line or \
                   'gemm,cutlass_tensorop_h16816gemm_256x128_32x3' in line or \
                   'gemm,cutlass_tensorop_h884gemm_256x128_32x2' in line or \
                   'gemm,cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3' in line or \
                   'gemm,cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3' in line:
                    flops.append(float(line.split(',')[-1]))
        except BaseException:
            valid = False
        finally:
            if valid is False or len(flops) == 0:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False

        self._result.add_result(precision, max(flops))

        return True


BenchmarkRegistry.register_benchmark('gemm-flops', GemmFlopsCuda, platform=Platform.CUDA)
