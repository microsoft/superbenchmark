# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FAMBench benchmarks.

Including:
  DLRM
  XLMR
"""

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.docker_benchmarks.docker_base import CudaDockerBenchmark


class FAMBenchBenchmark(CudaDockerBenchmark):
    """The FAMBench E2E model benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        # Image uri of the current docker-benchmark.
        self._image_uri = 'superbench/benchmark:cuda11.1.1-fambench'

        # Image digest of the current docker-benchmark.
        self._digest = 'b7b0d07270055287129e8b4b32be0863cbc3cc061610fcfaccf3a7450906e36f'

        # Container name of the current docker-benchmark.
        self._container_name = 'fambench-benchmarks'

        # Entrypoint option of the current docker-benchmark.
        self._entrypoint = '/workspace/FAMBench/benchmarks/run_all_benchmarks.sh'

        # CMD option of the current docker-benchmark.
        self._cmd = None

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output', raw_output, self._args.log_raw_data)

        content = raw_output.splitlines(False)
        try:
            result_header = 'benchmark implementation mode config score'
            found = False
            for line in content:
                if result_header in line:
                    found = True
                elif found:
                    items = line.split(' ')
                    if len(items) == 7:
                        name = '_'.join(items[0:4] + [items[5]])
                        for char in ['-', ' ', '=', '/']:
                            name = name.replace(char, '_')
                        score = float(items[4])
                        self._result.add_result(name.lower(), score)
        except BaseException as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, message: {}.'.format(
                    self._curr_run_index, self._name, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('fambench', FAMBenchBenchmark, platform=Platform.CUDA)
