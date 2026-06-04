# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ort-inference benchmark."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchvision.models

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Precision, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.ort_inference_performance import ORTInferenceBenchmark
from superbench.benchmarks.result import BenchmarkResult


@decorator.cuda_test
@decorator.pytorch_test
@mock.patch('torch.hub.get_dir')
@mock.patch('onnxruntime.InferenceSession.run')
def test_ort_inference_performance(mock_ort_session_run, mock_get_dir):
    """Test ort-inference benchmark."""
    benchmark_name = 'ort-inference'
    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
    assert (benchmark_class)

    mock_get_dir.return_value = '/tmp/superbench/'
    benchmark = benchmark_class(
        benchmark_name,
        parameters='--pytorch_models resnet50 --graph_opt_level 1 --precision float16'
        ' --batch_size 16 --num_warmup 128 --num_steps 512'
    )

    assert (isinstance(benchmark, ORTInferenceBenchmark))
    assert (benchmark._preprocess())

    # Check basic information.
    assert (benchmark.name == 'ort-inference')
    assert (benchmark.type == BenchmarkType.MICRO)
    assert (benchmark._ORTInferenceBenchmark__model_cache_path == Path(torch.hub.get_dir()) / 'checkpoints')
    for model in benchmark._args.pytorch_models:
        assert (hasattr(torchvision.models, model))
        file_name = '{model}.{precision}.onnx'.format(model=model, precision=benchmark._args.precision)
        assert ((benchmark._ORTInferenceBenchmark__model_cache_path / file_name).is_file())

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.pytorch_models == ['resnet50'])
    assert (benchmark._args.graph_opt_level == 1)
    assert (benchmark._args.precision == Precision.FLOAT16)
    assert (benchmark._args.batch_size == 16)
    assert (benchmark._args.num_warmup == 128)
    assert (benchmark._args.num_steps == 512)

    # Check results and metrics.
    assert (benchmark._benchmark())
    shutil.rmtree(benchmark._ORTInferenceBenchmark__model_cache_path)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'int8': 'int8'}
    for model in benchmark._args.pytorch_models:
        if benchmark._args.precision.value in precision_metric:
            precision = precision_metric[benchmark._args.precision.value]
        else:
            precision = benchmark._args.precision.value
        metric = '{}_{}_time'.format(precision, model)
        assert (metric in benchmark.result)
        assert (metric in benchmark.raw_data)


# ---------------------------------------------------------------------------
# HuggingFace-path coverage for _preprocess_huggingface_models and
# _export_hf_model_to_onnx. These tests are pure unit tests with no CUDA / no
# HF network access; the model loader, ModelSourceConfig, and torch2onnxExporter
# are all mocked to keep the suite fast and deterministic.
# ---------------------------------------------------------------------------

_ORT_MODULE = 'superbench.benchmarks.micro_benchmarks.ort_inference_performance'


def _make_ort_benchmark(**arg_overrides):
    """Build an ORTInferenceBenchmark and minimally initialise its mutable state.

    Returns the benchmark with ``_args``, ``_result``, and the name-mangled
    cache-path attribute populated so HF-path methods can be exercised in
    isolation without going through the full ``_preprocess`` pipeline.
    """
    benchmark = ORTInferenceBenchmark('ort-inference', parameters='')
    benchmark._result = BenchmarkResult('ort-inference', BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)
    defaults = dict(
        model_source='huggingface',
        model_identifier='prajjwal1/bert-tiny',
        allow_remote_code=False,
        precision=Precision.FLOAT16,
        batch_size=8,
        seq_length=128,
        graph_opt_level=3,
        num_warmup=1,
        num_steps=1,
        pytorch_models=[],
        require_cuda=False,
        log_raw_data=False,
    )
    defaults.update(arg_overrides)
    benchmark._args = SimpleNamespace(**defaults)
    # The HF helpers reference the name-mangled cache path; set it explicitly so
    # we don't depend on torch.hub.get_dir() in unit tests.
    benchmark._ORTInferenceBenchmark__model_cache_path = Path('/tmp/sb-ort-test-cache')
    return benchmark


# ---------------------------------------------------------------------------
# _preprocess_huggingface_models
# ---------------------------------------------------------------------------


def test_preprocess_hf_missing_model_identifier():
    """Missing --model_identifier is rejected before any HF I/O."""
    benchmark = _make_ort_benchmark(model_identifier=None)

    assert benchmark._preprocess_huggingface_models() is False
    assert benchmark.return_code == ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE


def test_preprocess_hf_invalid_identifier():
    """Path-like / unsafe identifier is rejected by validate_model_identifier."""
    benchmark = _make_ort_benchmark(model_identifier='../etc/passwd')

    assert benchmark._preprocess_huggingface_models() is False
    assert benchmark.return_code == ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE


def test_preprocess_hf_memory_check_fails():
    """check_memory_fits=False short-circuits with EXECUTION_FAILURE."""
    benchmark = _make_ort_benchmark()

    with patch('transformers.AutoConfig') as mock_auto_config, \
            patch(f'{_ORT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls:
        mock_auto_config.from_pretrained.return_value = MagicMock(name='hf_config')
        mock_loader_cls.check_memory_fits.return_value = (False, 1000.0, 30.0, 16.0)

        assert benchmark._preprocess_huggingface_models() is False

    assert benchmark.return_code == ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE
    mock_auto_config.from_pretrained.assert_called_once()


def test_preprocess_hf_auto_config_exception():
    """An exception while downloading the config is converted to failure."""
    benchmark = _make_ort_benchmark()

    with patch('transformers.AutoConfig') as mock_auto_config:
        mock_auto_config.from_pretrained.side_effect = RuntimeError('boom')

        assert benchmark._preprocess_huggingface_models() is False

    assert benchmark.return_code == ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE


def test_preprocess_hf_happy_path_delegates_to_export():
    """Happy path: config + memory check pass and the export helper runs."""
    benchmark = _make_ort_benchmark()

    fake_hf_config = MagicMock(name='hf_config')
    with patch('transformers.AutoConfig') as mock_auto_config, \
            patch(f'{_ORT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls, \
            patch.object(benchmark, '_export_hf_model_to_onnx', return_value=True) as mock_export:
        mock_auto_config.from_pretrained.return_value = fake_hf_config
        mock_loader_cls.check_memory_fits.return_value = (True, 4.0, 0.02, 16.0)

        assert benchmark._preprocess_huggingface_models() is True

    # AutoConfig is called with trust_remote_code matching --allow_remote_code (False).
    config_kwargs = mock_auto_config.from_pretrained.call_args.kwargs
    assert config_kwargs['trust_remote_code'] is False
    # _hf_config is stashed for __inference() to read vocab_size later.
    assert benchmark._hf_config is fake_hf_config
    # Memory check uses the runtime precision (float16 here).
    mem_args, mem_kwargs = mock_loader_cls.check_memory_fits.call_args
    assert mem_args[2] == 'float16'
    assert mem_kwargs.get('mode') == 'inference'
    # Export helper receives the pre-downloaded config to avoid a redundant fetch.
    export_args, _ = mock_export.call_args
    assert export_args[2] is fake_hf_config


def test_preprocess_hf_int8_uses_float32_for_memory_check():
    """INT8 precision still does the memory check against float32 weights."""
    benchmark = _make_ort_benchmark(precision=Precision.INT8)

    with patch('transformers.AutoConfig') as mock_auto_config, \
            patch(f'{_ORT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls, \
            patch.object(benchmark, '_export_hf_model_to_onnx', return_value=True):
        mock_auto_config.from_pretrained.return_value = MagicMock()
        mock_loader_cls.check_memory_fits.return_value = (True, 1.0, 0.01, 16.0)

        assert benchmark._preprocess_huggingface_models() is True

    mem_args, _ = mock_loader_cls.check_memory_fits.call_args
    assert mem_args[2] == 'float32'


def test_preprocess_hf_allow_remote_code_propagates():
    """--allow_remote_code is forwarded as trust_remote_code=True to AutoConfig."""
    benchmark = _make_ort_benchmark(allow_remote_code=True)

    with patch('transformers.AutoConfig') as mock_auto_config, \
            patch(f'{_ORT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls, \
            patch.object(benchmark, '_export_hf_model_to_onnx', return_value=True):
        mock_auto_config.from_pretrained.return_value = MagicMock()
        mock_loader_cls.check_memory_fits.return_value = (True, 1.0, 0.01, 16.0)

        benchmark._preprocess_huggingface_models()

    assert mock_auto_config.from_pretrained.call_args.kwargs['trust_remote_code'] is True


# ---------------------------------------------------------------------------
# _export_hf_model_to_onnx
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_export_dependencies(tmp_path):
    """Patch the loader, ModelSourceConfig, exporter, and torch.cuda for export tests.

    Yields a SimpleNamespace bundle of mock handles plus the exporter's resolved
    ONNX output path, so each test can assert on whichever it needs.
    """
    rank_dir = tmp_path / 'checkpoints'

    with patch(f'{_ORT_MODULE}.HuggingFaceModelLoader') as loader_cls, \
            patch(f'{_ORT_MODULE}.ModelSourceConfig') as msc, \
            patch(f'{_ORT_MODULE}.torch.cuda') as torch_cuda:
        loader = MagicMock()
        loader_cls.return_value = loader
        loader.load_model_from_config.return_value = (MagicMock(name='hf_model'), MagicMock(), None)
        torch_cuda.is_available.return_value = False

        # Patch the exporter where it is imported (inside _export_hf_model_to_onnx).
        with patch('superbench.benchmarks.micro_benchmarks._export_torch_to_onnx.torch2onnxExporter') as exporter_cls:
            exporter = MagicMock()
            exporter_cls.return_value = exporter

            def _fake_export(model, model_name, batch_size, seq_length, output_dir):
                """Simulate a successful ONNX export by writing the file the exporter would produce."""
                out = Path(output_dir) / f'{model_name}.onnx'
                out.parent.mkdir(parents=True, exist_ok=True)
                out.touch()
                return str(out)

            exporter.export_huggingface_model.side_effect = _fake_export

            yield SimpleNamespace(
                loader_cls=loader_cls,
                loader=loader,
                msc=msc,
                exporter_cls=exporter_cls,
                exporter=exporter,
                rank_dir=rank_dir,
            )


def test_export_hf_model_to_onnx_fp16_success(mock_export_dependencies, tmp_path):
    """fp16 path: ModelSourceConfig dtype=float16, exporter writes ONNX, no quantization."""
    benchmark = _make_ort_benchmark(precision=Precision.FLOAT16)
    benchmark._ORTInferenceBenchmark__model_cache_path = tmp_path / 'checkpoints'

    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
        ok = benchmark._export_hf_model_to_onnx(hf_token='abc', allow_remote_code=False, hf_config=MagicMock())

    assert ok is True
    # ModelSourceConfig built with float16 (precision dtype) and device_map=None.
    msc_kwargs = mock_export_dependencies.msc.call_args.kwargs
    assert msc_kwargs['torch_dtype'] == 'float16'
    assert msc_kwargs['device_map'] is None
    assert msc_kwargs['hf_token'] == 'abc'
    # load_model_from_config is invoked with the pre-downloaded config to skip a redundant fetch.
    load_kwargs = mock_export_dependencies.loader.load_model_from_config.call_args.kwargs
    assert load_kwargs['device'] == 'cpu'
    assert load_kwargs['config_pretrained'] is not None
    # Exporter receives precision-tagged model name and the rank-scoped output dir.
    export_kwargs = mock_export_dependencies.exporter.export_huggingface_model.call_args.kwargs
    assert export_kwargs['model_name'] == 'prajjwal1_bert-tiny.float16'
    assert export_kwargs['output_dir'].endswith('rank_0')
    assert export_kwargs['batch_size'] == 8
    assert export_kwargs['seq_length'] == 128
    # pytorch_models is rewritten to the bare HF id (no precision suffix).
    assert benchmark._args.pytorch_models == ['prajjwal1_bert-tiny']
    # Cache path now points at the rank subdirectory.
    assert str(benchmark._ORTInferenceBenchmark__model_cache_path).endswith('rank_0')


def test_export_hf_model_to_onnx_int8_invokes_quantize(mock_export_dependencies, tmp_path):
    """INT8 path: ONNX is exported as float32 first, then quantize_dynamic is called."""
    benchmark = _make_ort_benchmark(precision=Precision.INT8)
    benchmark._ORTInferenceBenchmark__model_cache_path = tmp_path / 'checkpoints'

    fake_quantize_module = MagicMock()
    with patch.dict('sys.modules', {'onnxruntime.quantization': fake_quantize_module}), \
            patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
        ok = benchmark._export_hf_model_to_onnx(hf_token=None, allow_remote_code=False, hf_config=MagicMock())

    assert ok is True
    # ModelSourceConfig dtype is float32 because INT8 is generated post-export.
    msc_kwargs = mock_export_dependencies.msc.call_args.kwargs
    assert msc_kwargs['torch_dtype'] == 'float32'
    # Exporter wrote the float32 ONNX, then quantize_dynamic was called with that file.
    export_kwargs = mock_export_dependencies.exporter.export_huggingface_model.call_args.kwargs
    assert export_kwargs['model_name'] == 'prajjwal1_bert-tiny.float32'
    fake_quantize_module.quantize_dynamic.assert_called_once()
    quantize_args = fake_quantize_module.quantize_dynamic.call_args.args
    assert quantize_args[0].endswith('prajjwal1_bert-tiny.float32.onnx')
    assert quantize_args[1].endswith('prajjwal1_bert-tiny.int8.onnx')


def test_export_hf_model_to_onnx_export_failure(mock_export_dependencies, tmp_path):
    """If exporter returns falsy, the helper fails without touching pytorch_models."""
    benchmark = _make_ort_benchmark()
    benchmark._ORTInferenceBenchmark__model_cache_path = tmp_path / 'checkpoints'
    mock_export_dependencies.exporter.export_huggingface_model.side_effect = None
    mock_export_dependencies.exporter.export_huggingface_model.return_value = None

    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
        ok = benchmark._export_hf_model_to_onnx(hf_token=None, allow_remote_code=False, hf_config=MagicMock())

    assert ok is False
    assert benchmark.return_code == ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE
    assert benchmark._args.pytorch_models == []


def test_export_hf_model_to_onnx_uses_proc_rank_env(mock_export_dependencies, tmp_path):
    """PROC_RANK env var (or CUDA_VISIBLE_DEVICES) controls the rank subdirectory."""
    benchmark = _make_ort_benchmark()
    benchmark._ORTInferenceBenchmark__model_cache_path = tmp_path / 'checkpoints'

    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0', 'PROC_RANK': '7'}, clear=False):
        ok = benchmark._export_hf_model_to_onnx(hf_token=None, allow_remote_code=False, hf_config=MagicMock())

    assert ok is True
    export_kwargs = mock_export_dependencies.exporter.export_huggingface_model.call_args.kwargs
    assert export_kwargs['output_dir'].endswith('rank_7')
    assert str(benchmark._ORTInferenceBenchmark__model_cache_path).endswith('rank_7')


def test_export_hf_model_to_onnx_passes_allow_remote_code_to_loader(mock_export_dependencies, tmp_path):
    """allow_remote_code is forwarded to the HuggingFaceModelLoader constructor."""
    benchmark = _make_ort_benchmark()
    benchmark._ORTInferenceBenchmark__model_cache_path = tmp_path / 'checkpoints'

    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
        benchmark._export_hf_model_to_onnx(hf_token=None, allow_remote_code=True, hf_config=MagicMock())

    loader_kwargs = mock_export_dependencies.loader_cls.call_args.kwargs
    assert loader_kwargs['allow_remote_code'] is True


def test_export_hf_model_to_onnx_releases_cuda_cache(mock_export_dependencies, tmp_path):
    """When CUDA is available, torch.cuda.empty_cache() is invoked after export."""
    benchmark = _make_ort_benchmark()
    benchmark._ORTInferenceBenchmark__model_cache_path = tmp_path / 'checkpoints'

    with patch(f'{_ORT_MODULE}.torch.cuda') as torch_cuda, \
            patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
        torch_cuda.is_available.return_value = True

        ok = benchmark._export_hf_model_to_onnx(hf_token=None, allow_remote_code=False, hf_config=MagicMock())

    assert ok is True
    torch_cuda.empty_cache.assert_called_once()
