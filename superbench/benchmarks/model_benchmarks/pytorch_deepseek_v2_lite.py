# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench adapter for isolated official-checkpoint V2-Lite FSDP training."""

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, Platform, ReturnCode
from superbench.benchmarks.base import Benchmark
from superbench.common.utils import logger
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite_utils import (
    numeric_series, read_json, run_worker, validate_identity
)


class DeepSeekV2LiteTraining(Benchmark):
    """Run one official BF16 checkpoint through a separate FSDP worker runtime."""
    def __init__(self, name, parameters=''):
        """Initialize the fixed model benchmark without importing its new runtime."""
        super().__init__(name, parameters)
        self._benchmark_type = BenchmarkType.MODEL

    def add_parser_arguments(self):
        """Configure exactly one download-inclusive training profile."""
        super().add_parser_arguments()
        self._parser.set_defaults(duration=3300)
        self._parser.add_argument('--runtime_python', type=str, default=sys.executable)
        self._parser.add_argument('--cache_root', type=str, default='model-cache')
        self._parser.add_argument('--num_gpus', type=int, default=4)
        self._parser.add_argument('--batch_size', type=int, default=2, help='Microbatch per GPU.')
        self._parser.add_argument('--seq_len', type=int, default=512)
        self._parser.add_argument('--num_warmup', type=int, default=2)
        self._parser.add_argument('--num_steps', type=int, default=8)
        self._parser.add_argument('--learning_rate', type=float, default=1e-5)

    def _preprocess(self):
        """Reject unsupported execution modes and invalid shapes explicitly."""
        if not super()._preprocess():
            return False
        args = self._args
        invalid = (
            args.run_count != 1 or not 30 < args.duration < 3600 or args.num_gpus < 2 or args.batch_size < 1
            or args.seq_len < 32 or args.num_warmup < 0 or args.num_steps < 1 or not 0 < args.learning_rate < 1
        )
        try:
            world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', '1')))
        except ValueError:
            logger.error('V2-Lite training launcher world size must be an integer.')
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False
        if invalid or world_size != 1:
            logger.error('V2-Lite training requires one local launcher, valid positive dimensions and duration <3600s.')
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False
        return True

    def _worker_command(self, job):
        """Launch all local GPUs once, retaining argv boundaries."""
        worker = Path(__file__).with_name('deepseek_v2_lite_training_worker.py')
        args = self._args
        return [
            args.runtime_python,
            '-m',
            'torch.distributed.run',
            '--standalone',
            '--nproc_per_node={}'.format(args.num_gpus),
            str(worker),
            '--cache_dir',
            str(job / 'cache'),
            '--output',
            str(job / 'report.json'),
            '--num_gpus',
            str(args.num_gpus),
            '--batch_size',
            str(args.batch_size),
            '--seq_len',
            str(args.seq_len),
            '--num_warmup',
            str(args.num_warmup),
            '--num_steps',
            str(args.num_steps),
            '--learning_rate',
            str(args.learning_rate),
            '--duration',
            str(args.duration),
        ]

    def _benchmark(self):
        """Accept only complete worker metrics and verified full-model training."""
        start = time.monotonic()
        try:
            root = Path(self._args.cache_root).resolve()
            root.mkdir(parents=True, exist_ok=True)
            job = Path(tempfile.mkdtemp(prefix='v2lite-training-', dir=str(root)))
            logger.info('V2-Lite training evidence and isolated cache: %s', job)
            run_worker(self._worker_command(job), job / 'worker.log', self._args.duration - 20)
            report = read_json(job / 'report.json')
            validate_identity(report)
            steps = numeric_series(report, 'step_ms', self._args.num_steps)
            losses = numeric_series(report, 'loss', self._args.num_steps)
            norms = numeric_series(report, 'gradient_norm', self._args.num_steps)
            expected_tokens = (
                self._args.num_steps * self._args.batch_size * self._args.num_gpus * (self._args.seq_len - 1)
            )
            updates = report.get('updated_parameter_elements')
            if report.get('loss_token_count') != expected_tokens or type(updates) is not int or updates <= 0:
                raise ValueError('Training token count or verified optimizer update is invalid.')
            if report.get('world_size') != self._args.num_gpus:
                raise ValueError('Worker used a different number of GPUs.')
            total = time.monotonic() - start
            if total >= self._args.duration:
                raise TimeoutError('Complete worker job exceeded the profile duration.')
            for name, values in [('step_time', steps), ('loss', losses), ('gradient_norm', norms)]:
                metric = 'bf16_train_' + name
                self._result.add_raw_data(metric, values, self._args.log_raw_data)
                self._result.add_result(metric, statistics.mean(values))
            self._result.add_result('bf16_train_tokens_per_second', expected_tokens * 1000 / sum(steps))
            self._result.add_result('end_to_end_seconds', total)
            self._result.add_result('checkpoint_bytes', report['weight_bytes'])
            return True
        except TimeoutError as error:
            logger.error('V2-Lite training timeout: %s', error)
            self._result.set_return_code(ReturnCode.KILLED_BY_TIMEOUT)
            return False
        except (OSError, ValueError, RuntimeError) as error:
            logger.error('V2-Lite training failed: %s', error)
            self._result.set_return_code(ReturnCode.RUNTIME_EXCEPTION_ERROR)
            return False


BenchmarkRegistry.register_benchmark('deepseek-v2-lite-training', DeepSeekV2LiteTraining, platform=Platform.CUDA)
BenchmarkRegistry.register_benchmark('deepseek-v2-lite-training', DeepSeekV2LiteTraining, platform=Platform.ROCM)
