# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU HPCG benchmark base class."""

import os

from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuHpcgBenchmark(MicroBenchmarkWithInvoke):
    """The GPU HPCG benchmark base class."""
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
        """Save raw output for later parser refinement.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            bool: Always True for now.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
        return True
