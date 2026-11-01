# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

import builtins
import json
import numbers
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import transformers

from tests.helper import decorator
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision, ReturnCode
from superbench.benchmarks.model_benchmarks import pytorch_base
from superbench.benchmarks.model_benchmarks.model_base import (
    Optimizer, DistributedImpl, DistributedBackend, ModelBenchmark
)
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class MNISTModel(torch.nn.Module):
    """The MNIST model for benchmarking."""
    def __init__(self):
        """Constructor."""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        """Forward propagation function.

        Args:
            x (torch.Tensor): Image tensor.

        Return:
            output (torch.Tensor): Tensor of the log_softmax result.
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output


class PytorchMNIST(PytorchBase):
    """The MNIST benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._supported_precision = [Precision.FLOAT32]
        self._optimizer_type = Optimizer.ADAMW
        self._loss_fn = torch.nn.functional.nll_loss

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        samples_count = (self._args.batch_size * (self._args.num_warmup + self._args.num_steps))
        self._dataset = TorchRandomDataset([samples_count, 1, 28, 28], self._world_size, dtype=torch.float32)
        if len(self._dataset) == 0:
            logger.error('Generate random dataset failed - model: {}'.format(self._name))
            return False

        return True

    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        try:
            self._model = MNISTModel()
            self._model = self._model.to(dtype=getattr(torch, precision.value))
            if self._gpu_available:
                self._model = self._model.cuda()
        except BaseException as e:
            logger.error(
                'Create model with specified precision failed - model: {}, precision: {}, message: {}.'.format(
                    self._name, precision, str(e)
                )
            )
            return False

        self._target = torch.LongTensor(self._args.batch_size).random_(10)
        if self._gpu_available:
            self._target = self._target.cuda()

        return True

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            The step-time list of every training step.
        """
        duration = []
        for idx, sample in enumerate(self._dataloader):
            sample = sample.to(dtype=getattr(torch, precision.value))
            start = self._timer()
            if self._gpu_available:
                sample = sample.cuda()
            self._optimizer.zero_grad()
            output = self._model(sample)
            loss = self._loss_fn(output, self._target)
            loss.backward()
            self._optimizer.step()
            end = self._timer()
            if idx % 10 == 0:
                logger.info(
                    'Train step [{}/{} ({:.0f}%)]'.format(
                        idx, len(self._dataloader), 100. * idx / len(self._dataloader)
                    )
                )
            if idx >= self._args.num_warmup:
                duration.append((end - start) * 1000)

        return duration

    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (Precision): precision of model and input data,
              such as float32, float16.

        Return:
            The latency list of every inference operation.
        """
        duration = []
        with torch.no_grad():
            self._model.eval()
            for idx, sample in enumerate(self._dataloader):
                sample = sample.to(dtype=getattr(torch, precision.value))
                start = self._timer()
                if self._gpu_available:
                    sample = sample.cuda()
                self._model(sample)
                if self._gpu_available:
                    torch.cuda.synchronize()
                end = self._timer()
                if idx % 10 == 0:
                    logger.info(
                        'Inference step [{}/{} ({:.0f}%)]'.format(
                            idx, len(self._dataloader), 100. * idx / len(self._dataloader)
                        )
                    )
                if idx >= self._args.num_warmup:
                    duration.append((end - start) * 1000)
        return duration


@decorator.pytorch_test
def test_pytorch_base():
    """Test PytorchBase class."""
    # Register mnist benchmark.
    BenchmarkRegistry.register_benchmark('pytorch-mnist', PytorchMNIST)

    # Launch benchmark with --no_gpu for testing.
    parameters = '--batch_size 32 --num_warmup 8 --num_steps 64 --model_action train inference --no_gpu --force_fp32'
    benchmark = PytorchMNIST('pytorch-mnist', parameters=parameters)
    assert (benchmark)
    assert (benchmark._preprocess())
    assert (benchmark._benchmark())
    assert (benchmark.name == 'pytorch-mnist')
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    # Test results.
    for metric in [
        'fp32_train_step_time', 'fp32_inference_step_time', 'fp32_train_throughput', 'fp32_inference_throughput'
    ]:
        assert (len(benchmark.raw_data[metric]) == 1)
        assert (len(benchmark.raw_data[metric][0]) == 64)
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))

    # Test _cal_params_count().
    assert (benchmark._cal_params_count() == 1199882)

    # Test _judge_gpu_availability().
    assert (benchmark._gpu_available is False)

    # Test _set_force_fp32().
    assert (benchmark._args.force_fp32 is True)

    # Test _init_distributed_setting().
    assert (benchmark._args.distributed_impl is None)
    assert (benchmark._args.distributed_backend is None)
    assert (benchmark._init_distributed_setting() is True)
    benchmark._args.distributed_impl = DistributedImpl.DDP
    benchmark._args.distributed_backend = DistributedBackend.NCCL
    assert (benchmark._init_distributed_setting() is False)
    benchmark._args.distributed_impl = DistributedImpl.MIRRORED
    assert (benchmark._init_distributed_setting() is False)

    # Test _init_dataloader().
    benchmark._args.distributed_impl = None
    assert (benchmark._init_dataloader() is True)
    benchmark._args.distributed_impl = DistributedImpl.DDP
    assert (benchmark._init_dataloader() is False)
    benchmark._args.distributed_impl = DistributedImpl.MIRRORED
    assert (benchmark._init_dataloader() is False)

    # Test _create_optimizer().
    if hasattr(torch.optim, 'AdamW'):
        assert (isinstance(benchmark._optimizer, torch.optim.AdamW))
    else:
        assert (isinstance(benchmark._optimizer, transformers.AdamW))
    benchmark._optimizer_type = Optimizer.ADAM
    assert (benchmark._create_optimizer() is True)
    assert (isinstance(benchmark._optimizer, torch.optim.Adam))
    benchmark._optimizer_type = Optimizer.SGD
    assert (benchmark._create_optimizer() is True)
    assert (isinstance(benchmark._optimizer, torch.optim.SGD))
    benchmark._optimizer_type = None
    assert (benchmark._create_optimizer() is False)

    # Test _sync_result().
    step_time = [2.0, 2.0]
    benchmark._args.distributed_impl = DistributedImpl.DDP
    step_time = benchmark._sync_result(step_time)
    assert (not step_time)
    benchmark._args.distributed_impl = None

    # Test _postprocess().
    assert (benchmark._postprocess())


@pytest.fixture(params=['cuda', 'hip'])
def mocked_profiler(request, monkeypatch, tmp_path):
    """Provide CPU-safe profiler events for the real trace exporter."""
    event = SimpleNamespace(
        device_type=torch.autograd.DeviceType.CPU,
        name='capture',
        input_shapes=[],
        concrete_inputs=[],
        cpu_time=2.0,
        cuda_time=1.0,
        time_range=SimpleNamespace(start=0.0)
    )
    profiler = MagicMock()
    profiler.events.return_value = [event, SimpleNamespace(device_type=torch.autograd.DeviceType.CUDA)]
    monkeypatch.setenv('SB_ENABLE_PYTORCH_PROFILER', '1')
    monkeypatch.setenv('SB_TORCH_PROFILER_TRACE_DIR', str(tmp_path))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.version, 'cuda', '12.0' if request.param == 'cuda' else None)
    monkeypatch.setattr(torch.version, 'hip', '6.4' if request.param == 'hip' else None, raising=False)
    monkeypatch.setattr(torch.profiler, 'profile', lambda **kwargs: profiler)
    monkeypatch.setattr(ModelBenchmark, '_benchmark', lambda self: True)
    return PytorchMNIST('pytorch-mnist'), event


@decorator.pytorch_test
@pytest.mark.parametrize('distributed', [False, True])
def test_pytorch_profiler_config_traces(mocked_profiler, monkeypatch, tmp_path, distributed):
    """Keep configuration/rank traces distinct while preserving legacy writes."""
    benchmark, event = mocked_profiler
    captures = {}
    for config_name in ['model-benchmarks:mnist@fp16', 'model-benchmarks:mnist@fp32']:
        for rank in range(2):
            work_dir = tmp_path / 'benchmarks' / config_name / f'rank{rank}'
            work_dir.mkdir(parents=True)
            monkeypatch.chdir(work_dir)
            benchmark._local_rank = rank if distributed else None
            event.name = f'{config_name}-rank{rank}'
            assert (benchmark._benchmark())
            qualified = tmp_path / f'torch-profiler-sb-pytorch-mnist__cfg={config_name}-{rank}.json'
            legacy = tmp_path / f'torch-profiler-sb-pytorch-mnist-{rank if distributed else 0}.json'
            assert (qualified.read_bytes() == legacy.read_bytes())
            assert (qualified.stat().st_ino != legacy.stat().st_ino)
            captures[qualified] = event.name
    assert (len(list(tmp_path.glob('*__cfg=*.json'))) == 4)
    for path, marker in captures.items():
        events = json.loads(path.read_text(encoding='utf-8'))
        assert (len(events) == 1)
        assert (events[0]['name'] == marker)


@decorator.pytorch_test
@pytest.mark.parametrize('replacement', ['', '[{"name": "other-worker"}]'])
def test_pytorch_profiler_shared_legacy(mocked_profiler, monkeypatch, tmp_path, replacement):
    """Do not source qualified events from a concurrently changed legacy file."""
    benchmark, event = mocked_profiler
    work_dir = tmp_path / 'benchmarks' / 'model-benchmarks:mnist' / 'rank1'
    work_dir.mkdir(parents=True)
    monkeypatch.chdir(work_dir)
    legacy = tmp_path / 'torch-profiler-sb-pytorch-mnist-0.json'

    @contextmanager
    def racing_open(path, *args, **kwargs):
        with builtins.open(path, *args, **kwargs) as stream:
            yield stream
        if str(path) == str(legacy):
            legacy.write_text(replacement, encoding='utf-8')

    monkeypatch.setattr(pytorch_base, 'open', racing_open, raising=False)
    assert (benchmark._benchmark())
    qualified = tmp_path / 'torch-profiler-sb-pytorch-mnist__cfg=model-benchmarks:mnist-1.json'
    assert (json.loads(qualified.read_text(encoding='utf-8'))[0]['name'] == event.name)
    assert (legacy.read_text(encoding='utf-8') == replacement)


@decorator.pytorch_test
@pytest.mark.parametrize('directory', ['other', 'benchmarks/config/worker0', 'benchmarks/config/rankx'])
def test_pytorch_profiler_legacy_fallback(mocked_profiler, monkeypatch, tmp_path, directory):
    """Keep legacy-only output outside the recognized executor layout."""
    benchmark, event = mocked_profiler
    work_dir = tmp_path / directory
    work_dir.mkdir(parents=True)
    monkeypatch.chdir(work_dir)
    assert (benchmark._benchmark())
    legacy = tmp_path / 'torch-profiler-sb-pytorch-mnist-0.json'
    assert ([path.name for path in tmp_path.glob('*.json')] == [legacy.name])
    assert (json.loads(legacy.read_text(encoding='utf-8'))[0]['name'] == event.name)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_empty_cache():
    """Test PytorchBase class."""
    # Register mnist benchmark.
    BenchmarkRegistry.register_benchmark('pytorch-mnist', PytorchMNIST)

    # Get initial memory reserved
    init_res_memory = torch.cuda.memory_reserved()

    # Test cache empty by manually calling torch.cuda.empty_cache().
    parameters = '--batch_size 32 --num_warmup 8 --num_steps 64 --model_action train'
    benchmark = PytorchMNIST('pytorch-mnist', parameters=parameters)

    assert (benchmark)
    assert (benchmark._preprocess())
    assert (benchmark._benchmark())
    del benchmark

    # Get current reserved memory after benchmark
    post_bm_res_memory = torch.cuda.memory_reserved()

    # Assert that memory is increased after benchmark
    assert (post_bm_res_memory >= init_res_memory)

    # Manually empty cache and get reserved memory
    # Calling empty_cache() releases all unused cached memory from PyTorch so that those can be used by
    # other GPU applications. However, the occupied GPU memory by tensors will not be freed so it can not
    # increase the amount of GPU memory available for PyTorch.
    # https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
    torch.cuda.empty_cache()
    post_empty_cache_res_memory = torch.cuda.memory_reserved()

    # Assert that some memory is released after manually empty cache. The cache is not guaranteed to be reset
    # back to the init_res_memory due to some tensors not being released.
    assert (post_empty_cache_res_memory <= post_bm_res_memory)

    # Test automatic cache empty.
    context = BenchmarkRegistry.create_benchmark_context(
        'pytorch-mnist', parameters='--batch_size 32 --num_warmup 8 --num_steps 64 --model_action train'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark)
    assert (torch.cuda.memory_reserved() == post_empty_cache_res_memory)
