# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.common.utils import device_manager as dm
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import GemmFlopsBenchmark


class CudaGemmFlopsBenchmark(GemmFlopsBenchmark):
    """The GEMM FLOPs performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'cutlass_profiler'

        # TODO - To support more architecutres, currently only support compute capability = 7.0, 7.5, 8.0, 8.6
        self.__kernel_map = {
            7.0: {
                'fp64': 'cutlass_simt_dgemm_128x128_8x2_*',
                'fp32': 'cutlass_simt_sgemm_128x128_8x2_*',
                'fp16': 'cutlass_simt_hgemm_256x128_8x2_*',
                'fp16_tc': 'cutlass_tensorop_h884gemm_256x128_32x2_*',
            },
            8.0: {
                'fp64': 'cutlass_simt_dgemm_128x128_8x2_*',
                'fp32': 'cutlass_simt_sgemm_128x128_8x2_*',
                'fp16': 'cutlass_simt_hgemm_256x128_8x2_*',
                'fp64_tc': 'cutlass_tensorop_d884gemm_128x128_16x3_*',
                'tf32_tc': 'cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_*',
                'bf16_tc': 'cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3_*',
                'fp16_tc': 'cutlass_tensorop_h16816gemm_256x128_32x3_*',
                'int8_tc': 'cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_*',
                'int4_tc': 'cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_*',
            }
        }
        # Skip FP64 for RTX Turing/Ampere and Tesla T4/GA10x due to very limited FP64 TFLOP rate
        self.__kernel_map[7.5] = {k: self.__kernel_map[7.0][k] for k in self.__kernel_map[7.0] if 'fp64' not in k}
        self.__kernel_map[8.6] = {k: self.__kernel_map[8.0][k] for k in self.__kernel_map[8.0] if 'fp64' not in k}
        # Skip INT4 for Hopper due to no native CUDA/Tensor Cores
        self.__kernel_map[9.0] = {k: self.__kernel_map[8.0][k] for k in self.__kernel_map[8.0] if 'int4_tc' not in k}
        self.__parse_logline = [
            'gemm,cutlass_simt_dgemm_128x128_8x2', 'gemm,cutlass_simt_sgemm_128x128_8x2',
            'gemm,cutlass_simt_hgemm_256x128_8x2', 'gemm,cutlass_tensorop_d884gemm_128x128_16x3',
            'gemm,cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3',
            'gemm,cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3', 'gemm,cutlass_tensorop_h16816gemm_256x128_32x3',
            'gemm,cutlass_tensorop_h884gemm_256x128_32x2', 'gemm,cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3',
            'gemm,cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3'
        ]

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        # Reset kernels according to compute capability.
        capability = dm.device_manager.get_device_compute_capability()
        if capability not in self.__kernel_map:
            # After preprocess() self._result.return_code can be generated
            super()._preprocess()
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE)
            logger.error(
                'Unsupported architecture - benchmark: {}, compute capability: {}, supports {}'.format(
                    self._name, capability, ' '.join(sorted([str(k) for k in self.__kernel_map]))
                )
            )
            return False

        self._support_precisions = list(self.__kernel_map[capability].keys())
        # 'support_precisions' are precise only after getting capability,
        #  and then using super.preprocess() to check if the precision in arguments are supported to run
        if not super()._preprocess():
            return False

        for p in self._precision_need_to_run:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += (' --warmup-iterations=' + str(self._args.num_warmup))
            command += (' --operation=gemm')
            command += (' --n=' + str(self._args.n))
            command += (' --k=' + str(self._args.k))
            command += (' --m=' + str(self._args.m))
            command += (' --kernels=' + self.__kernel_map[capability][p])
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
        precision = self._precision_need_to_run[cmd_idx]
        self._result.add_raw_data('raw_output_' + precision, raw_output, self._args.log_raw_data)

        valid = True
        flops = list()
        content = raw_output.splitlines()
        try:
            for line in content:
                for item in self.__parse_logline:
                    if item in line:
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

        self._result.add_result(self._metric_map[precision], max(flops))

        return True


BenchmarkRegistry.register_benchmark('gemm-flops', CudaGemmFlopsBenchmark, platform=Platform.CUDA)
