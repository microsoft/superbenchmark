# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the BLASLt GEMM Base Class."""
import itertools

from superbench.common.utils import logger
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


def mrange(start, stop=-1, multiplication_factor=2, symbol='x'):
    """Range constructor with multiplication factor.

    Args:
        start (int): Start number.
        stop (int, optional): Stop number. Defaults to -1.
        multiplication_factor (int, optional): Multiplication factor. Defaults to 2.
        symbol (str, optional): Symbol. Defaults to 'x' (multiplication).

    Yields:
        int: number in the range.
    """
    if symbol == 'x':
        while True:
            yield start
            start *= multiplication_factor
            if start > stop or start == 0 or multiplication_factor < 2:
                break
    elif symbol == '+':
        while True:
            yield start
            start = start + multiplication_factor
            if start > stop or start == 0 or multiplication_factor < 1:
                break
    else:
        raise ValueError(f'Invalid symbol {symbol}.')


def validate_mrange(string):
    """Validate mrange string in format start[[:stop]:multiplication_factor].

    Args:
        string (str): mrange string.

    Returns:
        bool: whether the mrange is expected.
    """
    nums = string.split(':')
    if len(nums) > 3:
        return False

    if len(nums) < 3:
        return all(x.isdigit() for x in nums)
    return nums[0].isdigit() and nums[1].isdigit() and (nums[2].lstrip('+').isdigit() or nums[2].lstrip('x').isdigit())


class BlasLtBaseBenchmark(MicroBenchmarkWithInvoke):
    """The BLASLt GEMM Base class."""
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
            '--shapes',
            type=str,
            nargs='+',
            default=[f'{x},{x},{x}' for x in [2048, 4096, 8192]],
            help='Shapes in m,n,k format. Support format start:stop:multiplication_factor, e.g., 16:128:2.',
        )
        self._parser.add_argument(
            '--batch',
            type=str,
            default='0',
            required=False,
            help=(
                'Batch size for strided batch GEMM, set 0 to disable.'
                ' Support format start:stop:multiplication_factor, e.g., 16:128:2.'
            ),
        )
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=20,
            required=False,
            help='Number of warm up steps.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=50,
            required=False,
            help='Number of steps to measure.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if not validate_mrange(self._args.batch):
            logger.error(f'Invalid batch size {self._args.batch}.')
            return False

        for _in_type in self._args.in_types:
            if _in_type not in self._in_types:
                logger.error(f'Invalid input type {_in_type}.')
                return False

        self._shapes_to_run = []
        for _in_type in self._args.in_types:
            for _b in mrange(*map(int, self._args.batch.split(':'))):
                for shape in self._args.shapes:
                    shape_list = shape.replace(',', ' ').split()
                    if len(shape_list) != 3 or not all(validate_mrange(x) for x in shape_list):
                        logger.error(f'Invalid shape {shape}.')
                        return False
                    for _m, _n, _k in itertools.product(
                        *map(
                            lambda shape: mrange(
                                *map(lambda dim: int(dim.lstrip('+').lstrip('x')), shape.split(':')),
                                symbol=shape.split(':')[2][0]
                                if len(shape.split(':')) == 3 and any([i in shape for i in ['+', 'x']]) else 'x'
                            ), shape_list
                        )
                    ):
                        self._shapes_to_run.append((_m, _n, _k, _b, _in_type))

        return True
