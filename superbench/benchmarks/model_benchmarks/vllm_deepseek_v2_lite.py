# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench adapter for isolated official-checkpoint V2-Lite generation."""

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


class DeepSeekV2LiteInference(Benchmark):
    """Benchmark one BF16 inference replica, not a training or forward-only loop."""

    def __init__(self, name, parameters=''):
        """Initialize without importing vLLM into the legacy SuperBench process."""
        super().__init__(name, parameters)
        self._benchmark_type = BenchmarkType.MODEL

    def add_parser_arguments(self):
        """Configure one separately timed checkpoint and generation profile."""
        super().add_parser_arguments()
        self._parser.set_defaults(duration=3300)
        self._parser.add_argument('--runtime_python', type=str, default=sys.executable)
        self._parser.add_argument('--reference_python', type=str, default='/opt/deepseek-reference/bin/python')
        self._parser.add_argument('--cache_root', type=str, default='model-cache')
        self._parser.add_argument('--batch_size', type=int, default=8)
        self._parser.add_argument('--seq_len', type=int, default=512)
        self._parser.add_argument('--max_new_tokens', type=int, default=64)
        self._parser.add_argument('--num_warmup', type=int, default=2)
        self._parser.add_argument('--num_steps', type=int, default=8)
        self._parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)

    def _preprocess(self):
        """Reject duplicate launchers and unsupported profile settings explicitly."""
        if not super()._preprocess():
            return False
        args = self._args
        try:
            world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', '1')))
        except ValueError:
            world_size = 0
        invalid = (
            args.run_count != 1 or not 30 < args.duration < 3600 or args.batch_size < 1 or args.seq_len < 32
            or args.max_new_tokens < 4 or args.num_warmup < 0 or args.num_steps < 1
            or not 0 < args.gpu_memory_utilization <= 0.95 or world_size != 1
        )
        if invalid:
            logger.error('V2-Lite inference requires one local replica and a valid BF16 generation/deadline profile.')
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False
        return True

    def _worker_command(self, job):
        """Keep executable paths and every worker argument separate."""
        args = self._args
        return [
            args.runtime_python, str(Path(__file__).with_name('deepseek_v2_lite_inference_worker.py')),
            '--cache_dir', str(job / 'cache'), '--output', str(job / 'report.json'),
            '--reference_python', args.reference_python, '--batch_size', str(args.batch_size),
            '--seq_len', str(args.seq_len), '--max_new_tokens', str(args.max_new_tokens),
            '--num_warmup', str(args.num_warmup), '--num_steps', str(args.num_steps),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization), '--duration', str(args.duration),
        ]

    def _benchmark(self):
        """Accept actual request/token counters and complete numeric validation."""
        start = time.monotonic()
        try:
            root = Path(self._args.cache_root).resolve()
            root.mkdir(parents=True, exist_ok=True)
            job = Path(tempfile.mkdtemp(prefix='v2lite-inference-', dir=str(root)))
            logger.info('V2-Lite inference evidence and isolated cache: %s', job)
            run_worker(self._worker_command(job), job / 'worker.log', self._args.duration - 20)
            report = read_json(job / 'report.json')
            validate_identity(report)
            requests = self._args.num_steps * self._args.batch_size
            steps = numeric_series(report, 'step_ms', self._args.num_steps)
            inputs = numeric_series(report, 'input_token_counts', requests)
            outputs = numeric_series(report, 'output_token_counts', requests)
            prefills = numeric_series(report, 'request_prefill_ms', requests)
            ttfts = numeric_series(report, 'ttft_ms', requests)
            decode_count = sum(value > 1 for value in outputs)
            tpots = numeric_series(report, 'decode_tpot_ms', decode_count)
            if decode_count == 0 or report.get('request_count') != requests or report.get('num_gpus') != 1:
                raise ValueError('Generation did not complete the requested replica/cache workload.')
            if any(value != self._args.seq_len for value in inputs):
                raise ValueError('Input token counts differ from the frozen prompt shape.')
            if any(not value.is_integer() or value > self._args.max_new_tokens for value in outputs):
                raise ValueError('Actual generated token counts are invalid.')
            total = time.monotonic() - start
            if total >= self._args.duration:
                raise TimeoutError('Complete inference job exceeded its duration.')
            for name, values in [
                ('step_time', steps), ('request_prefill_time', prefills), ('ttft', ttfts), ('decode_tpot', tpots),
                ('input_token_count', inputs), ('output_token_count', outputs)
            ]:
                metric = 'bf16_inference_' + name
                self._result.add_raw_data(metric, values, self._args.log_raw_data)
                self._result.add_result(metric, statistics.mean(values))
            self._result.add_result('bf16_inference_output_tokens_per_second', sum(outputs) * 1000 / sum(steps))
            self._result.add_result('successful_requests', requests)
            self._result.add_result('actual_output_tokens', sum(outputs))
            self._result.add_result('end_to_end_seconds', total)
            self._result.add_result('checkpoint_bytes', report['weight_bytes'])
            return True
        except TimeoutError as error:
            logger.error('V2-Lite inference timeout: %s', error)
            self._result.set_return_code(ReturnCode.KILLED_BY_TIMEOUT)
            return False
        except (OSError, ValueError, RuntimeError) as error:
            logger.error('V2-Lite inference failed: %s', error)
            self._result.set_return_code(ReturnCode.RUNTIME_EXCEPTION_ERROR)
            return False


BenchmarkRegistry.register_benchmark('deepseek-v2-lite-inference', DeepSeekV2LiteInference, platform=Platform.CUDA)
BenchmarkRegistry.register_benchmark('deepseek-v2-lite-inference', DeepSeekV2LiteInference, platform=Platform.ROCM)
