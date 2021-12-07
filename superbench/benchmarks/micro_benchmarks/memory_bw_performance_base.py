# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the memory performance benchmarks base class."""

from superbench.common.utils import logger
from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class MemBwBenchmark(MicroBenchmarkWithInvoke):
    """The Cuda memory performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._mem_types = ['htod', 'dtoh', 'dtod']
        self._metrics = ['h2d_bw', 'd2h_bw', 'd2d_bw']
        self._memory = ['pinned', 'unpinned']
        self._parse_logline_map = {'htod': 'H2D', 'dtoh': 'D2H', 'dtod': 'D2D'}

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--mem_type',
            type=str,
            nargs='+',
            default=self._mem_types,
            help='Memory types to benchmark. E.g. {}.'.format(' '.join(self._mem_types)),
        )
        self._parser.add_argument(
            '--memory',
            type=str,
            default='pinned',
            help='Memory argument for bandwidthtest. E.g. {}.'.format(' '.join(self._memory)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Format the arguments
        self._args.mem_type = [p.lower() for p in self._args.mem_type]

        # Check the arguments and generate the commands
        if self._args.memory not in self._memory:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error(
                'Unsupported mem_type of bandwidth test - benchmark: {}, memory: {}, expected: {}.'.format(
                    self._name, self._args.memory, ' '.join(self._memory)
                )
            )
            return False
        for mem_type in self._args.mem_type:
            if mem_type not in self._mem_types:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported mem_type of bandwidth test - benchmark: {}, mem_type: {}, expected: {}.'.format(
                        self._name, mem_type, ' '.join(self._mem_types)
                    )
                )
                return False

        return True
