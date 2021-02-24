# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

from superbench.benchmarks import Platform, Framework, Precision, \
    BenchmarkContext, BenchmarkRegistry, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks import ModelBenchmark


class FakeModelBenchmark(ModelBenchmark):
    """Fake benchmark inherit from ModelBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)
        self._supported_precision = [Precision.FLOAT32, Precision.FLOAT16]

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--hidden_size',
            type=int,
            default=1024,
            required=False,
            help='Hidden size',
        )

        self._parser.add_argument(
            '--seq_len',
            type=int,
            default=512,
            required=False,
            help='Sequence length',
        )

    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU."""
        pass

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info."""
        pass

    def _init_dataloader(self):
        """Initialize the distributed dataloader."""
        pass

    def _create_optimizer(self):
        """Create the optimzier instance used for training."""
        pass

    def _create_model(self, precision):
        """Construct the model for benchmarking."""
        pass

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (str): precision of model and input data,
              such as float, half.

        Return:
            The step-time list of every training step.
        """
        duration = []
        for i in range(self._args.num_steps):
            duration.append(2)
        return duration

    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (str): precision of model and input data,
              such as float, half.

        Return:
            The latency list of every inference operation.
        """
        duration = []
        for i in range(self._args.num_steps):
            duration.append(4)
        return duration

    def _cal_params_size(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        return 200


def create_benchmark(params='--num_steps=8'):
    """Register and create benchmark."""
    # Register the FakeModelBenchmark benchmark.
    BenchmarkRegistry.register_benchmark(
        'pytorch-fake-model',
        FakeModelBenchmark,
        parameters='--hidden_size=2',
        platform=Platform.CUDA,
    )
    context = BenchmarkContext('fake-model', Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name)
    (benchmark_class, predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(name, context.platform)
    assert (benchmark_class)
    BenchmarkRegistry.clean_benchmarks()
    return benchmark_class(name, predefine_params + ' ' + context.parameters)


def test_arguments_related_interfaces():
    """Test arguments related interfaces.

    Benchmark.add_parser_arguments(),
    Benchmark.parse_args(),
    Benchmark.get_configurable_settings()
    """
    # Positive case for parse_args().
    benchmark = create_benchmark('--num_steps=9')
    benchmark.add_parser_arguments()
    (ret, args, unknown) = benchmark.parse_args()
    assert (ret and args.num_steps == 9)

    # Negative case for parse_args() - invalid precision.
    benchmark = create_benchmark('--num_steps=8 --precision=fp32')
    benchmark.add_parser_arguments()
    (ret, args, unknown) = benchmark.parse_args()
    assert (ret is False)

    # Test get_configurable_settings().
    settings = benchmark.get_configurable_settings()
    expected_settings = (
        """optional arguments:
  --run_count int       The run count of benchmark.
  --duration int        The elapsed time of benchmark in seconds.
  --num_warmup int      The number of warmup step
  --num_steps int       The number of test step
  --batch_size int      The number of batch size
  --precision Precision [Precision ...]
                        Model precision. E.g. float16 float32 float64 bfloat16
                        uint8 int8 int16 int32 int64.
  --model_action ModelAction [ModelAction ...]
                        Benchmark model process. E.g. train inference.
  --distributed_impl DistributedImpl
                        Distributed implementations. E.g. ddp mirrored
                        multiworkermirrored parameterserver horovod
  --distributed_backend DistributedBackend
                        Distributed backends. E.g. nccl mpi gloo
  --hidden_size int     Hidden size
  --seq_len int         Sequence length"""
    )
    assert (settings == expected_settings)


def test_preprocess():
    """Test interface Benchmark._preprocess()."""
    # Positive case for _preprocess().
    benchmark = create_benchmark('--num_steps=8')
    assert (benchmark._preprocess())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    settings = benchmark.get_configurable_settings()
    expected_settings = (
        """optional arguments:
  --run_count int       The run count of benchmark.
  --duration int        The elapsed time of benchmark in seconds.
  --num_warmup int      The number of warmup step
  --num_steps int       The number of test step
  --batch_size int      The number of batch size
  --precision Precision [Precision ...]
                        Model precision. E.g. float16 float32 float64 bfloat16
                        uint8 int8 int16 int32 int64.
  --model_action ModelAction [ModelAction ...]
                        Benchmark model process. E.g. train inference.
  --distributed_impl DistributedImpl
                        Distributed implementations. E.g. ddp mirrored
                        multiworkermirrored parameterserver horovod
  --distributed_backend DistributedBackend
                        Distributed backends. E.g. nccl mpi gloo
  --hidden_size int     Hidden size
  --seq_len int         Sequence length"""
    )
    print(settings)
    assert (settings == expected_settings)

    # Negative case for _preprocess() - invalid precision.
    benchmark = create_benchmark('--num_steps=8 --precision=fp32')
    assert (benchmark._preprocess() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

    # Negative case for _preprocess() - invalid benchmark type.
    benchmark = create_benchmark('--num_steps=8 --precision=float32')
    benchmark._benchmark_type = Platform.CUDA
    assert (benchmark._preprocess() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_BENCHMARK_TYPE)


def test_train():
    """Test interface Benchmark.__train()."""
    benchmark = create_benchmark()
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {"steptime_train_float32": [[2, 2, 2, 2, 2, 2, 2, 2]], '
        '"throughput_train_float32": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]}, '
        '"result": {"steptime_train_float32": [2.0], "throughput_train_float32": [16000.0]}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__train(Precision.FLOAT32))
    assert (benchmark.serialized_result == expected_result)

    # Step time list is empty (simulate training failure).
    benchmark = create_benchmark('--num_steps=0')
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {}, "result": {}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__train(Precision.FLOAT32) is False)
    assert (benchmark.serialized_result == expected_result)


def test_inference():
    """Test interface Benchmark.__inference()."""
    benchmark = create_benchmark()
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {"steptime_inference_float16": [[4, 4, 4, 4, 4, 4, 4, 4]], '
        '"throughput_inference_float16": [[8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0]]}, '
        '"result": {"steptime_inference_float16": [4.0], "throughput_inference_float16": [8000.0]}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__inference(Precision.FLOAT16))
    assert (benchmark.serialized_result == expected_result)

    # Step time list is empty (simulate inference failure).
    benchmark = create_benchmark('--num_steps=0')
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {}, "result": {}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__inference(Precision.FLOAT16) is False)
    assert (benchmark.serialized_result == expected_result)


def test_benchmark():
    """Test interface Benchmark._benchmark()."""
    # Positive case for _benchmark().
    benchmark = create_benchmark()
    benchmark._preprocess()
    assert (benchmark._benchmark())
    assert (benchmark.name == 'pytorch-fake-model')
    assert (benchmark.type == BenchmarkType.MODEL)
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    expected_raw_data = {
        'steptime_train_float32': [[2, 2, 2, 2, 2, 2, 2, 2]],
        'throughput_train_float32': [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]],
        'steptime_train_float16': [[2, 2, 2, 2, 2, 2, 2, 2]],
        'throughput_train_float16': [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]
    }
    assert (benchmark.raw_data == expected_raw_data)
    expected_result = {
        'steptime_train_float32': [2.0],
        'throughput_train_float32': [16000.0],
        'steptime_train_float16': [2.0],
        'throughput_train_float16': [16000.0]
    }
    assert (benchmark.result == expected_result)

    expected_serialized_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, "start_time": null, '
        '"end_time": null, "raw_data": {"steptime_train_float32": [[2, 2, 2, 2, 2, 2, 2, 2]], '
        '"throughput_train_float32": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]], '
        '"steptime_train_float16": [[2, 2, 2, 2, 2, 2, 2, 2]], '
        '"throughput_train_float16": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]}, '
        '"result": {"steptime_train_float32": [2.0], "throughput_train_float32": [16000.0], '
        '"steptime_train_float16": [2.0], "throughput_train_float16": [16000.0]}}'
    )
    assert (benchmark.serialized_result == expected_serialized_result)

    # Negative case for _benchmark() - no supported precision found.
    benchmark = create_benchmark('--precision=int16')
    assert (benchmark._preprocess())
    assert (benchmark._benchmark() is False)
    assert (benchmark.return_code == ReturnCode.NO_SUPPORTED_PRECISION)

    # Negative case for _benchmark() - model train failure, step time list is empty.
    benchmark = create_benchmark('--num_steps=0')
    assert (benchmark._preprocess())
    assert (benchmark._benchmark() is False)
    assert (benchmark.return_code == ReturnCode.MODEL_TRAIN_FAILURE)

    # Negative case for _benchmark() - model inference failure, step time list is empty.
    benchmark = create_benchmark('--model_action=inference --num_steps=0')
    assert (benchmark._preprocess())
    assert (benchmark._benchmark() is False)
    assert (benchmark.return_code == ReturnCode.MODEL_INFERENCE_FAILURE)


def test_check_result_format():
    """Test interface Benchmark.__check_result_format()."""
    # Positive case for __check_result_format().
    benchmark = create_benchmark()
    benchmark._preprocess()
    assert (benchmark._benchmark())
    assert (benchmark._Benchmark__check_result_type())
    assert (benchmark._Benchmark__check_summarized_result())
    assert (benchmark._Benchmark__check_raw_data())

    # Negative case for __check_result_format() - change List[int] to List[str].
    benchmark._result._BenchmarkResult__result = {'metric1': ['2.0']}
    assert (benchmark._Benchmark__check_summarized_result() is False)

    # Negative case for __check_raw_data() - change List[List[int]] to List[List[str]].
    benchmark._result._BenchmarkResult__raw_data = {'metric1': [['2.0']]}
    assert (benchmark._Benchmark__check_raw_data() is False)

    # Negative case for __check_raw_data() - invalid benchmark result.
    assert (benchmark._Benchmark__check_result_format() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_BENCHMARK_RESULT)
