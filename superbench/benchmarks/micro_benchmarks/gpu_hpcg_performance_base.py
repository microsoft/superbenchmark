# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU HPCG benchmark base class."""

import os
import re

from superbench.common.utils import logger

from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuHpcgBenchmark(MicroBenchmarkWithInvoke):
    """The GPU HPCG benchmark base class."""
    _mpi_output_prefix_pattern = re.compile(r'^\[\d+,\d+\]<(?:stdout|stderr)>:\s*')
    _operation_metric_map = {'DDOT': 'ddot', 'WAXPBY': 'waxpby', 'SpMV': 'spmv', 'MG': 'mg', 'Total': 'total',
                             'Final': 'final'}
    _time_metric_map = {'Total Time': 'total_time', 'Setup Time': 'setup_time', 'Optimization Time': 'optimization_time'}
    _domain_metric_map = {'Local domain': 'local_domain', 'Global domain': 'global_domain',
                          'Process domain': 'process_domain'}
    _float_pattern = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s+(GFlop/s|GB/s)')
    _dimension_pattern = re.compile(r'([0-9]+)\s*x\s*([0-9]+)\s*x\s*([0-9]+)')
    _time_value_pattern = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s+sec')
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

        # Under MPI only rank 0 emits the complete rocHPCG summary.
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
        if rank > 0:
            return True

        parsed_results = {}
        required_metrics = {
            'final_gflops',
            'final_bandwidth',
            'final_gflops_per_process',
            'final_bandwidth_per_process',
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
            line = self._mpi_output_prefix_pattern.sub('', line)
            if not line:
                continue

            if self._parse_operation_line(line, parsed_results):
                continue

            if self._parse_time_line(line, parsed_results):
                continue

            self._parse_domain_line(line, parsed_results)

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

    def _parse_operation_line(self, line, parsed_results):
        """Parse one rocHPCG operation summary line."""
        operation_key = None
        for candidate in self._operation_metric_map:
            if line.startswith(candidate) and '=' in line:
                operation_key = candidate
                break

        if operation_key is None:
            return False

        matches = self._float_pattern.findall(line)
        if len(matches) < 4:
            return False

        prefix = self._operation_metric_map[operation_key]
        gflops_values = [float(value) for value, unit in matches if unit == 'GFlop/s']
        bandwidth_values = [float(value) for value, unit in matches if unit == 'GB/s']
        if len(gflops_values) < 2 or len(bandwidth_values) < 2:
            return False

        parsed_results[f'{prefix}_gflops'] = gflops_values[0]
        parsed_results[f'{prefix}_gflops_per_process'] = gflops_values[1]
        parsed_results[f'{prefix}_bandwidth'] = bandwidth_values[0]
        parsed_results[f'{prefix}_bandwidth_per_process'] = bandwidth_values[1]
        return True

    def _parse_time_line(self, line, parsed_results):
        """Parse one rocHPCG time summary line."""
        for label, metric in self._time_metric_map.items():
            if not line.startswith(label + ':'):
                continue

            match = self._time_value_pattern.search(line)
            if match:
                parsed_results[metric] = float(match.group(1))
                return True

        return False

    def _parse_domain_line(self, line, parsed_results):
        """Parse one rocHPCG domain summary line."""
        for label, metric_prefix in self._domain_metric_map.items():
            if not line.startswith(label + ':'):
                continue

            match = self._dimension_pattern.search(line)
            if not match:
                return False

            parsed_results[f'{metric_prefix}_x'] = int(match.group(1))
            parsed_results[f'{metric_prefix}_y'] = int(match.group(2))
            parsed_results[f'{metric_prefix}_z'] = int(match.group(3))
            return True

        return False
