# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU HPCG benchmark base class."""

import os
import re

from superbench.common.utils import logger

from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuHpcgBenchmark(MicroBenchmarkWithInvoke):
    """The GPU HPCG benchmark base class."""
    _operation_metric_map = {
        'DDOT': 'ddot',
        'WAXPBY': 'waxpby',
        'SpMV': 'spmv',
        'MG': 'mg',
        'Total': 'total',
        'Final': 'final',
    }
    _operation_pattern = re.compile(
        r'^(DDOT|WAXPBY|SpMV|MG|Total|Final)\s*=\s*'
        r'([0-9]+(?:\.[0-9]+)?)\s+GFlop/s\s+\(([0-9]+(?:\.[0-9]+)?)\s+GB/s\)\s+'
        r'([0-9]+(?:\.[0-9]+)?)\s+GFlop/s per process\s+\(\s*([0-9]+(?:\.[0-9]+)?)\s+GB/s per process\)$'
    )
    _time_pattern = re.compile(r'^(Total Time|Setup Time|Optimization Time):\s*([0-9]+(?:\.[0-9]+)?)\s+sec$')
    _domain_pattern = re.compile(r'^(Local|Global|Process) domain:\s*([0-9]+)\s+x\s+([0-9]+)\s+x\s+([0-9]+)$')
    _invalid_markers = ['*** WARNING *** INVALID RUN', '*** WARNING *** THIS IS NOT A VALID RUN ***']

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--npx',
            type=int,
            default=1,
            required=False,
            help='Number of MPI processes in x dimension.',
        )
        self._parser.add_argument(
            '--npy',
            type=int,
            default=1,
            required=False,
            help='Number of MPI processes in y dimension.',
        )
        self._parser.add_argument(
            '--npz',
            type=int,
            default=1,
            required=False,
            help='Number of MPI processes in z dimension.',
        )
        self._parser.add_argument(
            '--nx',
            type=int,
            default=560,
            required=False,
            help='Local problem size in x dimension.',
        )
        self._parser.add_argument(
            '--ny',
            type=int,
            default=280,
            required=False,
            help='Local problem size in y dimension.',
        )
        self._parser.add_argument(
            '--nz',
            type=int,
            default=280,
            required=False,
            help='Local problem size in z dimension.',
        )
        self._parser.add_argument(
            '--rt',
            type=int,
            default=60,
            required=False,
            help='Benchmark runtime in seconds.',
        )
        self._parser.add_argument(
            '--tol',
            type=float,
            default=1.0,
            required=False,
            help='Residual tolerance; reference verification is skipped if set.',
        )
        self._parser.add_argument(
            '--pz',
            type=int,
            default=0,
            required=False,
            help='Partition boundary in z process dimension.',
        )
        self._parser.add_argument(
            '--zl',
            type=int,
            required=False,
            help='Local nz value for processes with z rank < pz.',
        )
        self._parser.add_argument(
            '--zu',
            type=int,
            required=False,
            help='Local nz value for processes with z rank >= pz.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        bin_path = os.path.join(self._args.bin_dir, self._bin_name)
        zl = self._args.zl if self._args.zl is not None else self._args.nz
        zu = self._args.zu if self._args.zu is not None else self._args.nz

        command = (
            f'{bin_path}'
            f' --npx={self._args.npx}'
            f' --npy={self._args.npy}'
            f' --npz={self._args.npz}'
            f' --nx={self._args.nx}'
            f' --ny={self._args.ny}'
            f' --nz={self._args.nz}'
            f' --rt={self._args.rt}'
            f' --tol={self._args.tol}'
            f' --pz={self._args.pz}'
            f' --zl={zl}'
            f' --zu={zu}'
        )
        self._commands = [command]

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Parse rocHPCG stdout and save summarized results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            bool: True if rocHPCG summary metrics are extracted successfully.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        parsed_results = {}
        required_metrics = {
            'final_gflops',
            'final_gflops_per_process',
            'ddot_gflops',
            'ddot_bandwidth',
            'ddot_gflops_per_process',
            'ddot_bandwidth_per_process',
            'waxpby_gflops',
            'waxpby_bandwidth',
            'waxpby_gflops_per_process',
            'waxpby_bandwidth_per_process',
            'spmv_gflops',
            'spmv_bandwidth',
            'spmv_gflops_per_process',
            'spmv_bandwidth_per_process',
            'mg_gflops',
            'mg_bandwidth',
            'mg_gflops_per_process',
            'mg_bandwidth_per_process',
            'total_gflops',
            'total_bandwidth',
            'total_gflops_per_process',
            'total_bandwidth_per_process',
            'setup_time',
            'optimization_time',
            'total_time',
            'local_domain_x',
            'local_domain_y',
            'local_domain_z',
            'global_domain_x',
            'global_domain_y',
            'global_domain_z',
            'process_domain_x',
            'process_domain_y',
            'process_domain_z',
        }

        for raw_line in raw_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            operation_match = self._operation_pattern.match(line)
            if operation_match:
                prefix = self._operation_metric_map[operation_match.group(1)]
                total_gflops = float(operation_match.group(2))
                total_bandwidth = float(operation_match.group(3))
                per_process_gflops = float(operation_match.group(4))
                per_process_bandwidth = float(operation_match.group(5))

                parsed_results[f'{prefix}_gflops'] = total_gflops
                parsed_results[f'{prefix}_gflops_per_process'] = per_process_gflops
                if prefix != 'final':
                    parsed_results[f'{prefix}_bandwidth'] = total_bandwidth
                    parsed_results[f'{prefix}_bandwidth_per_process'] = per_process_bandwidth
                continue

            time_match = self._time_pattern.match(line)
            if time_match:
                metric_prefix = time_match.group(1).lower().replace(' ', '_')
                parsed_results[metric_prefix] = float(time_match.group(2))
                continue

            domain_match = self._domain_pattern.match(line)
            if domain_match:
                domain_prefix = domain_match.group(1).lower()
                parsed_results[f'{domain_prefix}_domain_x'] = int(domain_match.group(2))
                parsed_results[f'{domain_prefix}_domain_y'] = int(domain_match.group(3))
                parsed_results[f'{domain_prefix}_domain_z'] = int(domain_match.group(4))

        parsed_results['is_valid'] = 0 if any(marker in raw_output for marker in self._invalid_markers) else 1

        missing_metrics = sorted(metric for metric in required_metrics if metric not in parsed_results)
        if missing_metrics:
            logger.error(
                'The result format is invalid - round: %s, benchmark: %s, missing metrics: %s.',
                self._curr_run_index,
                self._name,
                ', '.join(missing_metrics),
            )
            return False

        for metric, value in parsed_results.items():
            self._result.add_result(metric, value)

        return True
