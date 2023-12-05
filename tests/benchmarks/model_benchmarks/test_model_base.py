# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

import json

from superbench.benchmarks import Platform, Framework, Precision, BenchmarkRegistry, BenchmarkType, ReturnCode
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
        self._sub_benchmark_start_time = 0

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--hidden_size',
            type=int,
            default=1024,
            required=False,
            help='Hidden size.',
        )

        self._parser.add_argument(
            '--seq_len',
            type=int,
            default=512,
            required=False,
            help='Sequence length.',
        )

    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        self._gpu_available = False

    def _set_force_fp32(self):
        """Set the config that controls whether full float32 precision will be used."""
        pass

    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU."""
        return True

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info."""
        return True

    def _init_dataloader(self):
        """Initialize the distributed dataloader."""
        return True

    def _create_optimizer(self):
        """Create the optimzier instance used for training."""
        return True

    def _create_model(self, precision):
        """Construct the model for benchmarking."""
        return True

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
            duration.append(2.0)
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
            duration.append(4.0)
        return duration

    def _cal_params_count(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        return 200


def create_benchmark(params='--num_steps 8'):
    """Register and create benchmark."""
    # Register the FakeModelBenchmark benchmark.
    BenchmarkRegistry.register_benchmark(
        'pytorch-fake-model',
        FakeModelBenchmark,
        parameters='--hidden_size 2',
        platform=Platform.CUDA,
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'fake-model', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH
    )
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name)
    (benchmark_class, predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(name, context.platform)
    assert (benchmark_class)
    return benchmark_class(name, predefine_params + ' ' + context.parameters)


def test_arguments_related_interfaces():
    """Test arguments related interfaces.

    Benchmark.add_parser_arguments(),
    Benchmark.parse_args(),
    Benchmark.get_configurable_settings()
    """
    # Positive case for parse_args().
    benchmark = create_benchmark('--num_steps 9')
    benchmark.add_parser_arguments()
    (ret, args, unknown) = benchmark.parse_args()
    assert (ret and args.num_steps == 9)

    # Negative case for parse_args() - invalid precision.
    benchmark = create_benchmark('--num_steps 8 --precision fp32')
    benchmark.add_parser_arguments()
    (ret, args, unknown) = benchmark.parse_args()
    assert (ret is False)

    # Test get_configurable_settings().
    settings = benchmark.get_configurable_settings()
    expected_settings = (
        """optional arguments:
  --batch_size int      The number of batch size.
  --distributed_backend DistributedBackend
                        Distributed backends. E.g. nccl mpi gloo.
  --distributed_impl DistributedImpl
                        Distributed implementations. E.g. ddp mirrored
                        multiworkermirrored parameterserver horovod.
  --duration int        The elapsed time of benchmark in seconds.
  --force_fp32          Enable option to use full float32 precision.
  --hidden_size int     Hidden size.
  --log_flushing        Real-time log flushing.
  --log_n_steps int     Real-time log every n steps.
  --log_raw_data        Log raw data into file instead of saving it into
                        result object.
  --model_action ModelAction [ModelAction ...]
                        Benchmark model process. E.g. train inference.
  --no_gpu              Disable GPU training.
  --num_steps int       The number of test step.
  --num_warmup int      The number of warmup step.
  --num_workers int     Number of subprocesses to use for data loading.
  --pin_memory          Enable option to pin memory in data loader.
  --precision Precision [Precision ...]
                        Model precision. E.g. fp8_hybrid fp8_e4m3 fp8_e5m2
                        float16 float32 float64 bfloat16 uint8 int8 int16
                        int32 int64.
  --run_count int       The run count of benchmark.
  --sample_count int    The number of data samples in dataset.
  --seq_len int         Sequence length."""
    )
    assert (settings == expected_settings)


def test_preprocess():
    """Test interface Benchmark._preprocess()."""
    # Positive case for _preprocess().
    benchmark = create_benchmark('--num_steps 8')
    assert (benchmark._preprocess())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    settings = benchmark.get_configurable_settings()
    expected_settings = (
        """optional arguments:
  --batch_size int      The number of batch size.
  --distributed_backend DistributedBackend
                        Distributed backends. E.g. nccl mpi gloo.
  --distributed_impl DistributedImpl
                        Distributed implementations. E.g. ddp mirrored
                        multiworkermirrored parameterserver horovod.
  --duration int        The elapsed time of benchmark in seconds.
  --force_fp32          Enable option to use full float32 precision.
  --hidden_size int     Hidden size.
  --log_flushing        Real-time log flushing.
  --log_n_steps int     Real-time log every n steps.
  --log_raw_data        Log raw data into file instead of saving it into
                        result object.
  --model_action ModelAction [ModelAction ...]
                        Benchmark model process. E.g. train inference.
  --no_gpu              Disable GPU training.
  --num_steps int       The number of test step.
  --num_warmup int      The number of warmup step.
  --num_workers int     Number of subprocesses to use for data loading.
  --pin_memory          Enable option to pin memory in data loader.
  --precision Precision [Precision ...]
                        Model precision. E.g. fp8_hybrid fp8_e4m3 fp8_e5m2
                        float16 float32 float64 bfloat16 uint8 int8 int16
                        int32 int64.
  --run_count int       The run count of benchmark.
  --sample_count int    The number of data samples in dataset.
  --seq_len int         Sequence length."""
    )
    assert (settings == expected_settings)

    # Negative case for _preprocess() - invalid precision.
    benchmark = create_benchmark('--num_steps 8 --precision fp32')
    assert (benchmark._preprocess() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

    # Negative case for _preprocess() - invalid benchmark type.
    benchmark = create_benchmark('--num_steps 8 --precision float32')
    benchmark._benchmark_type = Platform.CUDA
    assert (benchmark._preprocess() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_BENCHMARK_TYPE)


def test_train():
    """Test interface Benchmark.__train()."""
    benchmark = create_benchmark()
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {'
        '"fp32_train_step_time": [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]], '
        '"fp32_train_throughput": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]}, '
        '"result": {"return_code": [0], "fp32_train_step_time": [2.0], "fp32_train_throughput": [16000.0]}, '
        '"reduce_op": {"return_code": null, "fp32_train_step_time": null, "fp32_train_throughput": null}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__train(Precision.FLOAT32))
    assert (json.loads(benchmark.serialized_result) == json.loads(expected_result))

    # Step time list is empty (simulate training failure).
    benchmark = create_benchmark('--num_steps 0')
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 3, '
        '"start_time": null, "end_time": null, "raw_data": {}, '
        '"result": {"return_code": [3]}, "reduce_op": {"return_code": null}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__train(Precision.FLOAT32) is False)
    assert (json.loads(benchmark.serialized_result) == json.loads(expected_result))


def test_inference():
    """Test interface Benchmark.__inference()."""
    benchmark = create_benchmark()
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, '
        '"start_time": null, "end_time": null, "raw_data": {'
        '"fp16_inference_step_time": [[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]], '
        '"fp16_inference_throughput": [[8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0, 8000.0]]}, '
        '"result": {"return_code": [0], "fp16_inference_step_time": [4.0], '
        '"fp16_inference_step_time_50": [4.0], "fp16_inference_step_time_90": [4.0], '
        '"fp16_inference_step_time_95": [4.0], "fp16_inference_step_time_99": [4.0], '
        '"fp16_inference_step_time_99.9": [4.0], '
        '"fp16_inference_throughput": [8000.0], '
        '"fp16_inference_throughput_50": [8000.0], "fp16_inference_throughput_90": [8000.0], '
        '"fp16_inference_throughput_95": [8000.0], "fp16_inference_throughput_99": [8000.0], '
        '"fp16_inference_throughput_99.9": [8000.0]}, '
        '"reduce_op": {"return_code": null, "fp16_inference_step_time": null, '
        '"fp16_inference_step_time_50": null, "fp16_inference_step_time_90": null, '
        '"fp16_inference_step_time_95": null, "fp16_inference_step_time_99": null, '
        '"fp16_inference_step_time_99.9": null, "fp16_inference_throughput": null, '
        '"fp16_inference_throughput_50": null, "fp16_inference_throughput_90": null, '
        '"fp16_inference_throughput_95": null, "fp16_inference_throughput_99": null, '
        '"fp16_inference_throughput_99.9": null}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__inference(Precision.FLOAT16))
    assert (json.loads(benchmark.serialized_result) == json.loads(expected_result))

    # Step time list is empty (simulate inference failure).
    benchmark = create_benchmark('--num_steps 0')
    expected_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 3, '
        '"start_time": null, "end_time": null, "raw_data": {}, '
        '"result": {"return_code": [3]}, "reduce_op": {"return_code": null}}'
    )
    assert (benchmark._preprocess())
    assert (benchmark._ModelBenchmark__inference(Precision.FLOAT16) is False)
    assert (json.loads(benchmark.serialized_result) == json.loads(expected_result))


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
        'fp32_train_step_time': [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
        'fp32_train_throughput': [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]],
        'fp16_train_step_time': [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
        'fp16_train_throughput': [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]
    }
    assert (benchmark.raw_data == expected_raw_data)
    expected_result = {
        'return_code': [0],
        'fp32_train_step_time': [2.0],
        'fp32_train_throughput': [16000.0],
        'fp16_train_step_time': [2.0],
        'fp16_train_throughput': [16000.0]
    }
    assert (benchmark.result == expected_result)

    expected_serialized_result = (
        '{"name": "pytorch-fake-model", "type": "model", "run_count": 1, "return_code": 0, "start_time": null, '
        '"end_time": null, "raw_data": {"fp32_train_step_time": [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]], '
        '"fp32_train_throughput": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]], '
        '"fp16_train_step_time": [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]], '
        '"fp16_train_throughput": [[16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0, 16000.0]]}, '
        '"result": {"return_code": [0], "fp32_train_step_time": [2.0], "fp32_train_throughput": [16000.0], '
        '"fp16_train_step_time": [2.0], "fp16_train_throughput": [16000.0]}, '
        '"reduce_op": {"return_code": null, "fp32_train_step_time": null, "fp32_train_throughput": null, '
        '"fp16_train_step_time": null, "fp16_train_throughput": null}}'
    )
    assert (json.loads(benchmark.serialized_result) == json.loads(expected_serialized_result))

    # Negative case for _benchmark() - no supported precision found.
    benchmark = create_benchmark('--precision int16')
    assert (benchmark._preprocess())
    assert (benchmark._benchmark() is False)
    assert (benchmark.return_code == ReturnCode.NO_SUPPORTED_PRECISION)

    # Negative case for _benchmark() - model train failure, step time list is empty.
    benchmark = create_benchmark('--num_steps 0')
    assert (benchmark._preprocess())
    assert (benchmark._benchmark() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_BENCHMARK_RESULT)

    # Negative case for _benchmark() - model inference failure, step time list is empty.
    benchmark = create_benchmark('--model_action inference --num_steps 0')
    assert (benchmark._preprocess())
    assert (benchmark._benchmark() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_BENCHMARK_RESULT)


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
    benchmark._result._BenchmarkResult__result = {'return_code': [0], 'metric1': ['2.0']}
    assert (benchmark._Benchmark__check_summarized_result() is False)

    # Negative case for __check_raw_data() - change List[List[int]] to List[List[str]].
    benchmark._result._BenchmarkResult__raw_data = {'metric1': [['2.0']]}
    assert (benchmark._Benchmark__check_raw_data() is False)

    # Negative case for __check_raw_data() - invalid benchmark result.
    assert (benchmark._Benchmark__check_result_format() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_BENCHMARK_RESULT)


def test_is_finished():
    """Test interface Benchmark._is_finished()."""
    # Only step takes effect, benchmarking finish due to step.
    benchmark = create_benchmark('--num_warmup 32 --num_steps 128 --duration 0')
    benchmark._preprocess()
    end_time = 2
    curr_step = 50
    assert (benchmark._is_finished(curr_step, end_time) is False)
    curr_step = 160
    assert (benchmark._is_finished(curr_step, end_time))

    # Only duration takes effect, benchmarking finish due to duration.
    benchmark = create_benchmark('--num_warmup 32 --num_steps 0 --duration 10')
    benchmark._preprocess()
    benchmark._sub_benchmark_start_time = 0
    curr_step = 50
    end_time = 1
    assert (benchmark._is_finished(curr_step, end_time) is False)
    end_time = 10
    assert (benchmark._is_finished(curr_step, end_time))

    # Both step and duration take effect.
    benchmark = create_benchmark('--num_warmup 32 --num_steps 128 --duration 10')
    benchmark._preprocess()
    # Benchmarking finish due to step.
    curr_step = 160
    end_time = 2
    assert (benchmark._is_finished(curr_step, end_time))
    # Benchmarking finish due to duration.
    curr_step = 50
    end_time = 10
    assert (benchmark._is_finished(curr_step, end_time))
