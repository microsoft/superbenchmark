# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the TCP connectivity benchmarks."""

import tcping
from joblib import Parallel, delayed

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark


def run_tcping(host, port, count, timeout):
    """Run tcping for the given host address, port, count and timeout.

    Args:
        host (str): the address of the host
        port (int): listened tcp port of the target node
        count (int): try connections counts
        timeout (int): timeout of each connection try in seconds

    Returns:
        str: Table-like output of the tcping. Error message if error or exception happened.
    """
    ping_obj = tcping.Ping(host, port, timeout)
    output = None
    try:
        ping_obj.ping(count)
        output = ping_obj.result.table
    except Exception as e:
        return 'Socket connection failure, address: {}, port: {}, message: {}.'.format(host, port, str(e))
    return output


class TCPConnectivityBenchmark(MicroBenchmark):
    """The TCP connectivity performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self.__hosts = []

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--hostfile',
            type=str,
            default='/root/hostfile',
            required=False,
            help='The path of the hostfile including addresses to test',
        )
        self._parser.add_argument(
            '--port',
            type=int,
            default=22,
            required=False,
            help='Listened tcp port of the target node',
        )
        self._parser.add_argument(
            '--count',
            type=int,
            default=10,
            required=False,
            help='Try connections counts',
        )
        self._parser.add_argument(
            '--timeout',
            type=int,
            default=1,
            required=False,
            help='Timeout of each connection try in seconds',
        )
        self._parser.add_argument(
            '--parallel',
            type=int,
            default=-1,
            required=False,
            help='The maximum number of concurrently running jobs, if -1 all CPUs are used',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Check if the content of hostfile is valid and not empty
        valid = True
        try:
            with open(self._args.hostfile, 'r') as f:
                self.__hosts = f.readlines()
            for i in range(0, len(self.__hosts)):
                self.__hosts[i] = self.__hosts[i].rstrip('\n')
        except Exception:
            valid = False
        if not valid or len(self.__hosts) == 0:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Invalid hostfile - benchmark: {}, hostfile: {}.'.format(self._name, self._args.hostfile))
            return False

        return True

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        logger.info('TCP validation - round: {0}, name: {1}'.format(self._curr_run_index, self._name))

        # Run TCPing on host in the hostfile in parallel
        try:
            outputs = Parallel(n_jobs=min(len(self.__hosts), self._args.parallel))(
                delayed(run_tcping)(self.__hosts[i], self._args.port, self._args.count, self._args.timeout)
                for i in (range(len(self.__hosts)))
            )
        except Exception as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            logger.error(
                'Microbenchmark execution failed - round: {}, benchmark: {}, error message: {}.'.format(
                    self._curr_run_index, self._name, str(e)
                )
            )
            return False

        # Parse the output and get the results
        for host_index, out in enumerate(outputs):
            if not self._process_raw_result(host_index, out):
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
                return False

        return True

    def _process_raw_result(self, idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            idx (int): the index corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        host = self.__hosts[idx]
        self._result.add_raw_data('raw_output_' + host, raw_output, self._args.log_raw_data)

        try:
            # If socket error or exception happens on TCPing, add result values as failed
            suc = 0
            fail = self._args.count
            mininum = 0.00
            maximum = 0.00
            average = 0.00
            rate = 0
            # Parse and add result from table-like output of TCPing
            if 'failure' not in raw_output:
                raw_output = raw_output.splitlines()
                labels = None
                for line in raw_output:
                    # Get the line of the table labels
                    if 'Host' in line:
                        labels = line.split('|')
                        labels = [label.strip() for label in labels]
                    if host in line:
                        res = line.split('|')
                        res = [result.strip() for result in res]
                        suc = int(res[labels.index('Successed')])
                        fail = int(res[labels.index('Failed')])
                        rate = float(res[labels.index('Success Rate')].strip('%'))
                        mininum = float(res[labels.index('Minimum')].strip('ms'))
                        maximum = float(res[labels.index('Maximum')].strip('ms'))
                        average = float(res[labels.index('Average')].strip('ms'))
            self._result.add_result(host + '_successed_count', suc)
            self._result.add_result(host + '_failed_count', fail)
            self._result.add_result(host + '_success_rate', rate)
            self._result.add_result(host + '_time_min', mininum)
            self._result.add_result(host + '_time_max', maximum)
            self._result.add_result(host + '_time_avg', average)
        except Exception as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, address: {}, raw output: {}, message: {}.'.
                format(self._curr_run_index, self._name, host, raw_output, str(e))
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('tcp-connectivity', TCPConnectivityBenchmark)
