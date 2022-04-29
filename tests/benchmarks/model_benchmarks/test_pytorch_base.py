# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

import numbers

import torch
import transformers

from tests.helper import decorator
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision, ReturnCode
from superbench.benchmarks.model_benchmarks.model_base import Optimizer, DistributedImpl, DistributedBackend
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


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_empty_cache():
    """Test PytorchBase class."""
    # Register mnist benchmark.
    BenchmarkRegistry.register_benchmark('pytorch-mnist', PytorchMNIST)

    # Test cache empty by manually calling torch.cuda.empty_cache().
    parameters = '--batch_size 32 --num_warmup 8 --num_steps 64 --model_action train'
    benchmark = PytorchMNIST('pytorch-mnist', parameters=parameters)
    assert (benchmark)
    assert (benchmark._preprocess())
    assert (benchmark._benchmark())
    del benchmark
    assert (torch.cuda.memory_stats()['reserved_bytes.all.current'] > 0)
    torch.cuda.empty_cache()
    assert (torch.cuda.memory_stats()['reserved_bytes.all.current'] == 0)

    # Test automatic cache empty.
    context = BenchmarkRegistry.create_benchmark_context(
        'pytorch-mnist', parameters='--batch_size 32 --num_warmup 8 --num_steps 64 --model_action train'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark)
    assert (torch.cuda.memory_stats()['reserved_bytes.all.current'] == 0)
