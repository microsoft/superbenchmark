# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the onnxruntime E2E model benchmarks.

Including:
  bert-large-uncased ngpu=1
  bert-large-uncased ngpu=8
  distilbert-base-uncased ngpu=1
  distilbert-base-uncased ngpu=8
  gpt2 ngpu=1
  gpt2 ngpu=8
  facebook/bart-large ngpu=1
  facebook/bart-large ngpu=8
  roberta-large ngpu=1
  roberta-large ngpu=8
"""

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.docker_benchmarks.docker_base import RocmDockerBenchmark


class RocmOnnxRuntimeModelBenchmark(RocmDockerBenchmark):
    """The onnxruntime E2E model benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        # Image uri of the current docker-benchmark.
        self._image_uri = 'superbench/benchmark:rocm4.3.1-onnxruntime1.9.0'

        # Image digest of the current docker-benchmark.
        self._digest = 'f5e6c832e3cdcbba9820c619bb30fa47ca7117aa7f2c15944d17e6983d37ab9a'

        # Container name of the current docker-benchmark.
        self._container_name = 'rocm-onnxruntime-model-benchmarks'

        # Entrypoint option of the current docker-benchmark.
        self._entrypoint = '/stage/onnxruntime-training-examples/huggingface/azureml/run_benchmark.sh'

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
            name_prefix = '__superbench__ begin '
            value_prefix = '    "samples_per_second": '
            model_name = None
            for line in content:
                if name_prefix in line:
                    model_name = line[len(name_prefix):]
                    for char in ['-', ' ', '=', '/']:
                        model_name = model_name.replace(char, '_')
                elif value_prefix in line and model_name is not None:
                    throughput = float(line[len(value_prefix):])
                    self._result.add_result(model_name + '_throughput', throughput)
                    model_name = None
        except BaseException as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, message: {}.'.format(
                    self._curr_run_index, self._name, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('onnxruntime-ort-models', RocmOnnxRuntimeModelBenchmark, platform=Platform.ROCM)
