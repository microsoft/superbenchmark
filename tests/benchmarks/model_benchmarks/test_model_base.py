# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

from superbench.benchmarks import Platform, Framework, Precision, BenchmarkContext, BenchmarkRegistry, BenchmarkType
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
        self._supported_precision = [Precision.FLOAT32.value, Precision.FLOAT16.value]

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


def create_benchmark():
    """Register and create benchmark."""
    # Register the FakeModelBenchmark benchmark.
    BenchmarkRegistry.register_benchmark(
        'pytorch-fake-model',
        FakeModelBenchmark,
        parameters='--hidden_size=2',
        platform=Platform.CUDA,
    )
    context = BenchmarkContext('fake-model', Platform.CUDA, parameters='--num_steps=8', framework=Framework.PYTORCH)
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name)
    (benchmark_class, predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(name, context.platform)
    assert (benchmark_class)
    BenchmarkRegistry.clean_benchmarks()
    return (benchmark_class, name, predefine_params + ' ' + context.parameters)


def test_arguments_related_interfaces():
    """Test arguments related interfaces.

    Benchmark.add_parser_arguments(),
    Benchmark.parse_args(),
    Benchmark.get_configurable_settings()
    """
    (benchmark_class, name, parameters) = create_benchmark()

    benchmark = benchmark_class(name, parameters)
    benchmark.add_parser_arguments()
    benchmark.parse_args()

    settings = benchmark.get_configurable_settings()
    expected_settings = (
        """optional arguments:
  --run_count int       The run count of benchmark.
  --duration int        The elapsed time of benchmark in seconds.
  --num_warmup int      The number of warmup step
  --num_steps int       The number of test step
  --batch_size int      The number of batch size
  --precision {float16,float32,float64,bfloat16,uint8,int8,int16,int32,int64} """
        """[{float16,float32,float64,bfloat16,uint8,int8,int16,int32,int64} ...]
                        Model precision. E.g. float16 float32 float64 bfloat16
                        uint8 int8 int16 int32 int64.
  --model_action {train,inference} [{train,inference} ...]
                        Benchmark type. E.g. train inference.
  --distributed_mode {pytorch-ddp-nccl,tf-mirrored,tf-multiworkermirrored,tf-parameterserver,horovod}
                        Distributed mode. E.g. pytorch-ddp-nccl tf-mirrored
                        tf-multiworkermirrored tf-parameterserver horovod
  --hidden_size int     Hidden size
  --seq_len int         Sequence length"""
    )
    assert (settings == expected_settings)


def test_preprocess():
    """Test interface Benchmark._preprocess()."""
    (benchmark_class, name, parameters) = create_benchmark()

    benchmark = benchmark_class(name, parameters)
    benchmark._preprocess()
    assert (benchmark._result._BenchmarkResult__type == BenchmarkType.MODEL.value)

    settings = benchmark.get_configurable_settings()
    expected_settings = (
        """optional arguments:
  --run_count int       The run count of benchmark.
  --duration int        The elapsed time of benchmark in seconds.
  --num_warmup int      The number of warmup step
  --num_steps int       The number of test step
  --batch_size int      The number of batch size
  --precision {float16,float32,float64,bfloat16,uint8,int8,int16,int32,int64} """
        """[{float16,float32,float64,bfloat16,uint8,int8,int16,int32,int64} ...]
                        Model precision. E.g. float16 float32 float64 bfloat16
                        uint8 int8 int16 int32 int64.
  --model_action {train,inference} [{train,inference} ...]
                        Benchmark type. E.g. train inference.
  --distributed_mode {pytorch-ddp-nccl,tf-mirrored,tf-multiworkermirrored,tf-parameterserver,horovod}
                        Distributed mode. E.g. pytorch-ddp-nccl tf-mirrored
                        tf-multiworkermirrored tf-parameterserver horovod
  --hidden_size int     Hidden size
  --seq_len int         Sequence length"""
    )
    assert (settings == expected_settings)


def test_train():
    """Test interface Benchmark.__train()."""
    (benchmark_class, name, parameters) = create_benchmark()

    benchmark = benchmark_class(name, parameters)
    benchmark._preprocess()
    benchmark._ModelBenchmark__train(Precision.FLOAT32.value)
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {"steptime_train_float32": [[2, 2, 2, 2, 2, 2, 2, 2]], '
        '"throughput_train_float32": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]}, '
        '"result": {"steptime_train_float32": [2.0], "throughput_train_float32": [16000.0]}}'
    )
    assert (benchmark._result.to_string() == expected_result)


def test_inference():
    """Test interface Benchmark.__inference()."""
    (benchmark_class, name, parameters) = create_benchmark()

    benchmark = benchmark_class(name, parameters)
    benchmark._preprocess()
    benchmark._ModelBenchmark__inference(Precision.FLOAT16.value)
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {"steptime_inference_float16": [[4, 4, 4, 4, 4, 4, 4, 4]], '
        '"throughput_inference_float16": [[8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0]]}, '
        '"result": {"steptime_inference_float16": [4.0], "throughput_inference_float16": [8000.0]}}'
    )
    assert (benchmark._result.to_string() == expected_result)


def test_benchmark():
    """Test interface Benchmark._benchmark()."""
    (benchmark_class, name, parameters) = create_benchmark()

    benchmark = benchmark_class(name, parameters)
    benchmark._preprocess()
    benchmark._benchmark()
    assert (benchmark.name == 'pytorch-fake-model')
    assert (benchmark.type == BenchmarkType.MODEL.value)
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == 0)
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
