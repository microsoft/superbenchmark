# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU-Burn Test."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuBurnBenchmark(MicroBenchmarkWithInvoke):
    """The GPU Burn Test benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'gpu_burn'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--doubles',
            action='store_true',
            default=False,
            help='Use doubles for the data type used in GPU-Burn',
        )
        self._parser.add_argument(
            '--tensor_core',
            action='store_true',
            default=False,
            help='Use tensor cores in GPU-Burn',
        )
        self._parser.add_argument(
            '--time',
            type=int,
            default=10,
            help='Length of time to run GPU-Burn for(in seconds)',
        )
        self._parser.add_argument(
            '--warmup_iters',
            type=int,
            default=0,
            help='Number of warmup iterations before performance measurement',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if not self._set_binary_path():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)

        if self._args.doubles:
            command += ' -d'

        if self._args.tensor_core:
            command += ' -tc'
        command += ' {} '.format(self._args.time)

        self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):    # noqa: C901
        """Function to parse raw results and save the summarized results.

           self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        content = raw_output.splitlines()
        gpu_res = []
        abort = False
        failure_msg = 'unknown failure'
        index = -1

        try:
            # detect fatal failure lines
            for idx, line in enumerate(content):
                if 'No clients are alive!' in line or "Couldn't init a GPU" \
                        in line or 'Failure during compute' in line or 'Low mem for result' in line:
                    abort = True
                    failure_msg = line
                    break
                if 'done' in line:
                    index = idx
                    break

            if not abort:
                if 'done' not in content[index]:
                    abort = True
                    failure_msg = 'The result format invalid'
                    raise failure_msg

                content = content[index + 2:len(content):]

                for line in content:
                    if 'Tested' in line:
                        continue
                    if 'GPU' in line:
                        gpu_res.append(line.strip('\n').strip('\t'))

                self._result.add_result('time', self._args.time)
                for res in gpu_res:
                    if 'OK' in res:
                        self._result.add_result(res.split(':')[0].replace(' ', '_').lower() + '_pass', 1)
                    else:
                        self._result.add_result(res.split(':')[0].replace(' ', '_').lower() + '_pass', 0)
                    self._result.add_raw_data('GPU-Burn_result', res, self._args.log_raw_data)
            else:
                self._result.add_raw_data('GPU Burn Failure: ', failure_msg, self._args.log_raw_data)
                self._result.add_result('abort', 1)
                return False

            # Parse and emit metrics for every perf snapshot
            # Find all performance snapshot lines containing Gflop/s
            perf_lines = [line for line in raw_output.splitlines() if 'Gflop/s' in line]
            per_gpu_flops, per_gpu_temps = {}, {}
            for snap_idx, perf_line in enumerate(perf_lines):
                # extract per-GPU Gflops values like '(581623 Gflop/s)'
                gflops = re.findall(r'\(([0-9]+(?:\.[0-9]+)?)\s*Gflop/s\)', perf_line)
                gflops = [float(x) for x in gflops]
                # extract temps: 'temps: 48 C - 49 C - 49 C - 49 C'
                temps = []
                m = re.search(r'temps:\s*(.+)$', perf_line)
                if m:
                    temps = []
                    for t in m.group(1).split(' - '):
                        match = re.search(r'(\d+)', t)
                        if match:
                            temps.append(int(match.group(1)))

                # Save snapshot raw line
                self._result.add_raw_data(f'GPU-Burn_perf_snapshot_{snap_idx}', perf_line, self._args.log_raw_data)

                # Emit per-GPU metrics for this snapshot
                num_gpus = max(len(gflops), len(temps), len(gpu_res))
                for i in range(num_gpus):
                    if i not in per_gpu_flops:
                        per_gpu_flops[i] = []
                    if i not in per_gpu_temps:
                        per_gpu_temps[i] = []
                    if i < len(gflops) and gflops[i] > 0:
                        self._result.add_result(f'gpu_{snap_idx}_gflops:{i}', gflops[i])
                        if snap_idx > self._args.warmup_iters:
                            per_gpu_flops[i].append(gflops[i])
                    else:
                        self._result.add_result(f'gpu_{snap_idx}_gflops:{i}', 0.0)
                    if i < len(temps):
                        self._result.add_result(f'gpu_{snap_idx}_temp:{i}', temps[i])
                        per_gpu_temps[i].append(temps[i])
                    else:
                        self._result.add_result(f'gpu_{snap_idx}_temp:{i}', -1)
            for i in per_gpu_flops:
                if len(per_gpu_flops[i]) > 0:
                    avg_flops = sum(per_gpu_flops[i]) / len(per_gpu_flops[i])
                    self._result.add_result(f'gpu_avg_gflops:{i}', avg_flops)
                    if avg_flops != 0:
                        self._result.add_result(
                            f'gpu_var_gflops:{i}', (max(per_gpu_flops[i]) - min(per_gpu_flops[i])) / avg_flops - 1
                        )
                    else:
                        self._result.add_result(f'gpu_var_gflops:{i}', 0.0)
            for i in per_gpu_temps:
                if len(per_gpu_temps[i]) > 0:
                    self._result.add_result(f'gpu_max_temp:{i}', max(per_gpu_temps[i]))

        except BaseException as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            self._result.add_result('abort', 1)
            return False
        self._result.add_result('abort', 0)
        return True


BenchmarkRegistry.register_benchmark('gpu-burn', GpuBurnBenchmark, platform=Platform.CUDA)
