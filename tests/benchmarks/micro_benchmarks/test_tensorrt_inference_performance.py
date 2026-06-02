# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for tensorrt-inference benchmark."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.result import BenchmarkResult


def _make_onnx_dim(value):
    """Build an ONNX-graph-input-style dim mock that exposes ``dim_value``."""
    return SimpleNamespace(dim_value=value)


def _make_onnx_input(name, dims):
    """Build an ONNX-graph-input mock with the given name and dim values.

    A ``dim_value`` of ``0`` mimics a dynamic/unknown dimension, matching how
    ``onnx`` represents symbolic dims (``dim_param`` set, ``dim_value`` == 0).
    """
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=[_make_onnx_dim(d) for d in dims]))
        ),
    )


def _make_onnx_model(inputs, initializer_names=()):
    """Build an ONNX model mock with the given graph inputs and initializers."""
    initializers = [SimpleNamespace(name=n) for n in initializer_names]
    return SimpleNamespace(graph=SimpleNamespace(input=list(inputs), initializer=initializers))


class TensorRTInferenceBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for tensorrt-inferencee benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'tensorrt-inference'
        cls._model_path = Path(cls._tmp_dir) / 'hub' / 'onnx'
        cls.createMockEnvs(cls, {
            'TORCH_HOME': cls._tmp_dir,
            'SB_MICRO_PATH': cls._tmp_dir,
        })
        cls.createMockFiles(cls, ['bin/trtexec'])

    def test_tensorrt_inference_cls(self):
        """Test tensorrt-inference benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.CUDA:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    @decorator.cuda_test
    @decorator.pytorch_test
    def test_tensorrt_inference_params(self):
        """Test tensorrt-inference benchmark preprocess with different parameters."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)

        test_cases = [
            {
                'precision': 'fp32',
            },
            {
                'pytorch_models': ['resnet50', 'mnasnet0_5'],
                'precision': 'fp16',
            },
            {
                'pytorch_models': ['resnet50'],
                'batch_size': 4,
            },
            {
                'pytorch_models': ['lstm', 'bert-base', 'gpt2-small'],
                'batch_size': 4,
                'seq_length': 128,
                'iterations': 256,
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                parameter_list = []
                if 'pytorch_models' in test_case:
                    parameter_list.append(f'--pytorch_models {" ".join(test_case["pytorch_models"])}')
                if 'precision' in test_case:
                    parameter_list.append(f'--precision {test_case["precision"]}')
                if 'batch_size' in test_case:
                    parameter_list.append(f'--batch_size {test_case["batch_size"]}')
                if 'seq_length' in test_case:
                    parameter_list.append(f'--seq_length {test_case["seq_length"]}')
                if 'iterations' in test_case:
                    parameter_list.append(f'--iterations {test_case["iterations"]}')

                # Check basic information
                benchmark = benchmark_cls(self.benchmark_name, parameters=' '.join(parameter_list))
                self.assertTrue(benchmark)

                # Limit model number
                benchmark._pytorch_models = benchmark._pytorch_models[:1]

                # Preprocess
                ret = benchmark._preprocess()
                self.assertTrue(ret)
                self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)
                self.assertEqual(BenchmarkType.MICRO, benchmark.type)
                self.assertEqual(self.benchmark_name, benchmark.name)

                # Check parameters
                self.assertEqual(
                    test_case.get('pytorch_models', benchmark._pytorch_models),
                    benchmark._args.pytorch_models,
                )
                self.assertEqual(
                    test_case.get('precision', 'int8'),
                    benchmark._args.precision,
                )
                self.assertEqual(
                    test_case.get('batch_size', 32),
                    benchmark._args.batch_size,
                )
                self.assertEqual(
                    test_case.get('iterations', 2048),
                    benchmark._args.iterations,
                )

                # Check models
                for model in benchmark._args.pytorch_models:
                    self.assertTrue((self._model_path / f'{model}.onnx').is_file())

                # Command list should equal to default model number
                self.assertEqual(
                    len(test_case.get('pytorch_models', benchmark._pytorch_models)), len(benchmark._commands)
                )

    @decorator.load_data('tests/data/tensorrt_inference.1.log')
    @decorator.load_data('tests/data/tensorrt_inference.2.log')
    def test_tensorrt_inference_result_parsing(self, test_raw_log_1, test_raw_log_2):
        """Test tensorrt-inference benchmark result parsing."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        benchmark = benchmark_cls(self.benchmark_name, parameters='')
        benchmark._args = SimpleNamespace(pytorch_models=['model_0', 'model_1'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # Positive case 1 - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, test_raw_log_1))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertEqual(6 + benchmark.default_metric_count, len(benchmark.result))
        for tag in ['mean', '99']:
            self.assertEqual(0.5, benchmark.result[f'model_0_gpu_time_{tag}'][0])
            self.assertEqual(0.6, benchmark.result[f'model_0_host_time_{tag}'][0])
            self.assertEqual(1.0, benchmark.result[f'model_0_end_to_end_time_{tag}'][0])

        # Positive case 2 - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, test_raw_log_2))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)
        for tag in ['mean', '99']:
            self.assertEqual(1.5, benchmark.result[f'model_0_gpu_time_{tag}'][1])
            self.assertEqual(2.0, benchmark.result[f'model_0_host_time_{tag}'][1])

        # Negative case - invalid raw output
        self.assertFalse(benchmark._process_raw_result(1, 'Invalid raw output'))


_TENSORRT_MODULE = 'superbench.benchmarks.micro_benchmarks.tensorrt_inference_performance'


class TensorRTInferenceHuggingFaceTestCase(unittest.TestCase):
    """Unit tests for the HuggingFace-specific helpers on TensorRTInferenceBenchmark.

    These tests exercise the methods that previously had zero coverage:
    ``_preprocess_huggingface_models``, ``_build_trtexec_command_for_hf``,
    ``_derive_trt_input_shapes``, ``_derive_vision_input_shape``, and
    ``_derive_nlp_input_shapes``. They are pure unit tests (no CUDA / no HF
    network) and rely on mocking the model loader, ONNX exporter, and the
    ``onnx`` loader to keep them fast and deterministic.
    """

    benchmark_name = 'tensorrt-inference'

    def _make_benchmark(self, **arg_overrides):
        """Build a benchmark instance with mock args and bin/workspace state.

        Mimics the post-``_preprocess`` state of the object (bin path and
        workspace flag already resolved) without actually invoking trtexec or
        touching the filesystem.
        """
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(
            self.benchmark_name, Platform.CUDA
        )
        benchmark = benchmark_cls(self.benchmark_name, parameters='')
        benchmark._result = BenchmarkResult(
            self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1
        )
        defaults = dict(
            model_source='huggingface',
            model_identifier='prajjwal1/bert-tiny',
            allow_remote_code=False,
            precision='fp16',
            batch_size=8,
            seq_length=128,
            iterations=128,
            pytorch_models=[],
            log_raw_data=False,
        )
        defaults.update(arg_overrides)
        benchmark._args = SimpleNamespace(**defaults)
        # Set name-mangled private attributes that _preprocess() normally fills in.
        benchmark._TensorRTInferenceBenchmark__bin_path = '/fake/bin/trtexec'
        benchmark._TensorRTInferenceBenchmark__workspace_flag = '--memPoolSize=workspace:8192M'
        benchmark._commands = []
        return benchmark

    # ------------------------------------------------------------------
    # _preprocess_huggingface_models
    # ------------------------------------------------------------------

    def test_preprocess_hf_missing_model_identifier(self):
        """Missing --model_identifier is rejected before any HF I/O."""
        benchmark = self._make_benchmark(model_identifier=None)

        self.assertFalse(benchmark._preprocess_huggingface_models())
        self.assertEqual(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE, benchmark.return_code)
        self.assertEqual([], benchmark._commands)

    def test_preprocess_hf_invalid_identifier(self):
        """Path-like / unsafe identifier is rejected by validate_model_identifier."""
        benchmark = self._make_benchmark(model_identifier='../etc/passwd')

        self.assertFalse(benchmark._preprocess_huggingface_models())
        self.assertEqual(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE, benchmark.return_code)

    def test_preprocess_hf_int8_rejected(self):
        """INT8 on the HF path is rejected (no calibration data emitted)."""
        benchmark = self._make_benchmark(precision='int8')

        self.assertFalse(benchmark._preprocess_huggingface_models())
        self.assertEqual(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE, benchmark.return_code)

    def test_preprocess_hf_memory_check_fails(self):
        """When check_memory_fits reports fits=False, preprocess fails."""
        benchmark = self._make_benchmark()

        fake_config = MagicMock(name='AutoConfigInstance')
        with patch('transformers.AutoConfig') as mock_auto_config, \
                patch(f'{_TENSORRT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls:
            mock_auto_config.from_pretrained.return_value = fake_config
            mock_loader_cls.check_memory_fits.return_value = (False, 1000.0, 30.0, 16.0)

            self.assertFalse(benchmark._preprocess_huggingface_models())

        self.assertEqual(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE, benchmark.return_code)
        mock_auto_config.from_pretrained.assert_called_once()

    def test_preprocess_hf_auto_config_exception(self):
        """An exception while downloading the config is caught and converted to failure."""
        benchmark = self._make_benchmark()

        with patch('transformers.AutoConfig') as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = RuntimeError('boom')

            self.assertFalse(benchmark._preprocess_huggingface_models())

        self.assertEqual(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE, benchmark.return_code)

    def test_preprocess_hf_happy_path_delegates_to_build_command(self):
        """Happy path: config + memory check pass and the build helper is invoked."""
        benchmark = self._make_benchmark()

        fake_config = MagicMock(name='AutoConfigInstance')
        with patch('transformers.AutoConfig') as mock_auto_config, \
                patch(f'{_TENSORRT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls, \
                patch.object(
                    benchmark, '_build_trtexec_command_for_hf', return_value=True
                ) as mock_build:
            mock_auto_config.from_pretrained.return_value = fake_config
            mock_loader_cls.check_memory_fits.return_value = (True, 4.0, 0.02, 16.0)

            self.assertTrue(benchmark._preprocess_huggingface_models())

        # AutoConfig must be called with trust_remote_code matching --allow_remote_code (False here).
        config_kwargs = mock_auto_config.from_pretrained.call_args.kwargs
        self.assertFalse(config_kwargs['trust_remote_code'])
        # Memory check must run for fp32 (ONNX export dtype) regardless of --precision.
        mem_args, mem_kwargs = mock_loader_cls.check_memory_fits.call_args
        self.assertEqual('fp32', mem_args[2])
        self.assertEqual('inference', mem_kwargs.get('mode'))
        mock_build.assert_called_once()

    def test_preprocess_hf_allow_remote_code_propagates(self):
        """--allow_remote_code is forwarded as trust_remote_code=True."""
        benchmark = self._make_benchmark(allow_remote_code=True)

        with patch('transformers.AutoConfig') as mock_auto_config, \
                patch(f'{_TENSORRT_MODULE}.HuggingFaceModelLoader') as mock_loader_cls, \
                patch.object(benchmark, '_build_trtexec_command_for_hf', return_value=True):
            mock_auto_config.from_pretrained.return_value = MagicMock()
            mock_loader_cls.check_memory_fits.return_value = (True, 1.0, 0.01, 16.0)

            benchmark._preprocess_huggingface_models()

        self.assertTrue(mock_auto_config.from_pretrained.call_args.kwargs['trust_remote_code'])

    # ------------------------------------------------------------------
    # _build_trtexec_command_for_hf
    # ------------------------------------------------------------------

    def _patch_build_dependencies(self, onnx_path='/tmp/fake.onnx', input_shapes='input_ids:8x128'):
        """Common patch context for _build_trtexec_command_for_hf tests."""
        loader_patch = patch(f'{_TENSORRT_MODULE}.HuggingFaceModelLoader')
        msc_patch = patch(f'{_TENSORRT_MODULE}.ModelSourceConfig')
        exporter_patch = patch(f'{_TENSORRT_MODULE}.torch2onnxExporter')
        makedirs_patch = patch(f'{_TENSORRT_MODULE}.os.makedirs')
        torch_patch = patch(f'{_TENSORRT_MODULE}.torch')
        return loader_patch, msc_patch, exporter_patch, makedirs_patch, torch_patch

    def test_build_trtexec_command_for_hf_success(self):
        """Happy path: command is appended and shape/precision flags are correct."""
        benchmark = self._make_benchmark(precision='fp16')

        loader_p, msc_p, exporter_p, makedirs_p, torch_p = self._patch_build_dependencies()
        derived_shapes = 'input_ids:8x128,attention_mask:8x128'
        with loader_p as mock_loader_cls, msc_p as mock_msc, exporter_p as mock_exporter_cls, \
                makedirs_p as mock_makedirs, torch_p as mock_torch, \
                patch.object(benchmark, '_derive_trt_input_shapes', return_value=derived_shapes) as mock_derive:
            mock_torch.hub.get_dir.return_value = '/tmp/torchhub'
            mock_torch.cuda.is_available.return_value = False

            mock_loader = MagicMock()
            mock_loader_cls.return_value = mock_loader
            mock_hf_model = MagicMock(name='HFModel')
            mock_hf_config = MagicMock(name='HFConfig')
            mock_loader.load_model_from_config.return_value = (mock_hf_model, mock_hf_config, None)

            mock_exporter = MagicMock()
            mock_exporter_cls.return_value = mock_exporter
            mock_exporter.export_huggingface_model.return_value = '/tmp/torchhub/checkpoints/trt_rank_0/m.onnx'

            ok = benchmark._build_trtexec_command_for_hf(hf_token=None, allow_remote_code=False)

        self.assertTrue(ok)
        self.assertIs(benchmark._hf_config, mock_hf_config)
        # makedirs called once with the rank-scoped output dir.
        mock_makedirs.assert_called_once()
        self.assertTrue(mock_makedirs.call_args.args[0].endswith('trt_rank_0'))
        # ModelSourceConfig is constructed with float32 + device_map=None (CPU load).
        msc_kwargs = mock_msc.call_args.kwargs
        self.assertEqual('float32', msc_kwargs['torch_dtype'])
        self.assertIsNone(msc_kwargs['device_map'])
        self.assertEqual('huggingface', msc_kwargs['source'])
        # Exporter called with the configured batch_size / seq_length.
        export_kwargs = mock_exporter.export_huggingface_model.call_args.kwargs
        self.assertEqual(8, export_kwargs['batch_size'])
        self.assertEqual(128, export_kwargs['seq_length'])
        # _derive_trt_input_shapes was invoked with the exported ONNX path.
        mock_derive.assert_called_once_with('/tmp/torchhub/checkpoints/trt_rank_0/m.onnx')
        # Exactly one command appended, containing the expected flags.
        self.assertEqual(1, len(benchmark._commands))
        cmd = benchmark._commands[0]
        self.assertIn('/fake/bin/trtexec', cmd)
        self.assertIn('--onnx=/tmp/torchhub/checkpoints/trt_rank_0/m.onnx', cmd)
        self.assertIn(f'--optShapes={derived_shapes}', cmd)
        self.assertIn('--memPoolSize=workspace:8192M', cmd)
        self.assertIn('--fp16', cmd)
        self.assertIn('--iterations=128', cmd)
        self.assertIn('--percentile=99', cmd)
        # pytorch_models is rewritten so _process_raw_result can key off the HF id.
        self.assertEqual(['prajjwal1_bert-tiny'], benchmark._args.pytorch_models)

    def test_build_trtexec_command_for_hf_fp32_omits_precision_flag(self):
        """fp32 precision must not emit a ``--fp32`` or ``--int8`` flag."""
        benchmark = self._make_benchmark(precision='fp32')

        loader_p, msc_p, exporter_p, makedirs_p, torch_p = self._patch_build_dependencies()
        with loader_p as mock_loader_cls, msc_p, exporter_p as mock_exporter_cls, \
                makedirs_p, torch_p as mock_torch, \
                patch.object(benchmark, '_derive_trt_input_shapes', return_value='input_ids:8x128'):
            mock_torch.hub.get_dir.return_value = '/tmp/torchhub'
            mock_torch.cuda.is_available.return_value = False
            mock_loader_cls.return_value.load_model_from_config.return_value = (MagicMock(), MagicMock(), None)
            mock_exporter_cls.return_value.export_huggingface_model.return_value = (
                '/tmp/torchhub/checkpoints/trt_rank_0/m.onnx'
            )

            self.assertTrue(benchmark._build_trtexec_command_for_hf(None, False))

        cmd = benchmark._commands[0]
        self.assertNotIn('--fp16', cmd)
        self.assertNotIn('--fp32', cmd)
        self.assertNotIn('--int8', cmd)

    def test_build_trtexec_command_for_hf_export_failure(self):
        """If ONNX export returns falsy, the build fails and no command is queued."""
        benchmark = self._make_benchmark()

        loader_p, msc_p, exporter_p, makedirs_p, torch_p = self._patch_build_dependencies()
        with loader_p as mock_loader_cls, msc_p, exporter_p as mock_exporter_cls, \
                makedirs_p, torch_p as mock_torch:
            mock_torch.hub.get_dir.return_value = '/tmp/torchhub'
            mock_torch.cuda.is_available.return_value = False
            mock_loader_cls.return_value.load_model_from_config.return_value = (MagicMock(), MagicMock(), None)
            mock_exporter_cls.return_value.export_huggingface_model.return_value = None

            self.assertFalse(benchmark._build_trtexec_command_for_hf(None, False))

        self.assertEqual([], benchmark._commands)
        self.assertEqual(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE, benchmark.return_code)

    def test_build_trtexec_command_for_hf_uses_proc_rank_env(self):
        """PROC_RANK env var (or CUDA_VISIBLE_DEVICES) controls the rank subdir."""
        benchmark = self._make_benchmark()

        loader_p, msc_p, exporter_p, makedirs_p, torch_p = self._patch_build_dependencies()
        with loader_p as mock_loader_cls, msc_p, exporter_p as mock_exporter_cls, \
                makedirs_p as mock_makedirs, torch_p as mock_torch, \
                patch.object(benchmark, '_derive_trt_input_shapes', return_value='input_ids:8x128'), \
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0', 'PROC_RANK': '3'}, clear=False):
            mock_torch.hub.get_dir.return_value = '/tmp/torchhub'
            mock_torch.cuda.is_available.return_value = False
            mock_loader_cls.return_value.load_model_from_config.return_value = (MagicMock(), MagicMock(), None)
            mock_exporter_cls.return_value.export_huggingface_model.return_value = (
                '/tmp/torchhub/checkpoints/trt_rank_3/m.onnx'
            )

            self.assertTrue(benchmark._build_trtexec_command_for_hf(None, False))

        self.assertTrue(mock_makedirs.call_args.args[0].endswith('trt_rank_3'))

    # ------------------------------------------------------------------
    # _derive_trt_input_shapes
    # ------------------------------------------------------------------

    def test_derive_trt_input_shapes_vision_by_pixel_values_name(self):
        """Inputs named ``pixel_values`` are routed to the vision helper."""
        benchmark = self._make_benchmark(batch_size=4)
        # 3D so we rely on the name heuristic, not the dim-count heuristic.
        vision_input = _make_onnx_input('pixel_values', [0, 3, 224])
        # Pad the input to 4D so vision helper can index dims[1..3] safely.
        vision_input_4d = _make_onnx_input('pixel_values', [0, 3, 224, 224])
        model = _make_onnx_model([vision_input_4d])

        with patch('onnx.load', return_value=model):
            shapes = benchmark._derive_trt_input_shapes('/tmp/fake.onnx')

        self.assertEqual('pixel_values:4x3x224x224', shapes)
        _ = vision_input    # silence unused-warning in case of future refactor

    def test_derive_trt_input_shapes_vision_by_4d_shape(self):
        """A 4D non-``pixel_values`` input is still treated as vision."""
        benchmark = self._make_benchmark(batch_size=2)
        vision_input = _make_onnx_input('image', [0, 3, 256, 256])
        model = _make_onnx_model([vision_input])

        with patch('onnx.load', return_value=model):
            shapes = benchmark._derive_trt_input_shapes('/tmp/fake.onnx')

        self.assertEqual('image:2x3x256x256', shapes)

    def test_derive_trt_input_shapes_nlp_multi_input(self):
        """NLP routing: 2D inputs are emitted as ``name:BxS`` and comma-joined."""
        benchmark = self._make_benchmark(batch_size=4, seq_length=64)
        inputs = [
            _make_onnx_input('input_ids', [0, 0]),
            _make_onnx_input('attention_mask', [0, 0]),
        ]
        model = _make_onnx_model(inputs)

        with patch('onnx.load', return_value=model):
            shapes = benchmark._derive_trt_input_shapes('/tmp/fake.onnx')

        self.assertEqual('input_ids:4x64,attention_mask:4x64', shapes)

    def test_derive_trt_input_shapes_filters_initializers(self):
        """Initializer-named graph inputs are excluded from runtime inputs."""
        benchmark = self._make_benchmark(batch_size=1, seq_length=16)
        runtime = _make_onnx_input('input_ids', [0, 0])
        weight = _make_onnx_input('weight', [768, 768])
        model = _make_onnx_model([weight, runtime], initializer_names=['weight'])

        with patch('onnx.load', return_value=model):
            shapes = benchmark._derive_trt_input_shapes('/tmp/fake.onnx')

        self.assertEqual('input_ids:1x16', shapes)

    def test_derive_trt_input_shapes_no_runtime_inputs_raises(self):
        """A graph with only initializer-shadowed inputs raises ValueError."""
        benchmark = self._make_benchmark()
        weight = _make_onnx_input('weight', [768, 768])
        model = _make_onnx_model([weight], initializer_names=['weight'])

        with patch('onnx.load', return_value=model):
            with self.assertRaises(ValueError):
                benchmark._derive_trt_input_shapes('/tmp/fake.onnx')

    # ------------------------------------------------------------------
    # _derive_vision_input_shape
    # ------------------------------------------------------------------

    def test_derive_vision_input_shape_static_dims(self):
        """Static ONNX dims are used verbatim (apart from the batch dim)."""
        benchmark = self._make_benchmark(batch_size=16)
        vision_input = _make_onnx_input('pixel_values', [0, 3, 384, 384])

        result = benchmark._derive_vision_input_shape(vision_input, 'pixel_values')

        self.assertEqual('pixel_values:16x3x384x384', result)

    def test_derive_vision_input_shape_dynamic_with_hf_config_scalar(self):
        """Dynamic dims fall back to ``_hf_config`` (scalar ``image_size``)."""
        benchmark = self._make_benchmark(batch_size=4)
        benchmark._hf_config = SimpleNamespace(num_channels=1, image_size=160)
        vision_input = _make_onnx_input('pixel_values', [0, 0, 0, 0])

        result = benchmark._derive_vision_input_shape(vision_input, 'pixel_values')

        self.assertEqual('pixel_values:4x1x160x160', result)

    def test_derive_vision_input_shape_dynamic_with_hf_config_tuple(self):
        """Dynamic dims fall back to ``_hf_config`` (tuple/list ``image_size``)."""
        benchmark = self._make_benchmark(batch_size=2)
        benchmark._hf_config = SimpleNamespace(num_channels=3, image_size=(192, 384))
        vision_input = _make_onnx_input('pixel_values', [0, 0, 0, 0])

        result = benchmark._derive_vision_input_shape(vision_input, 'pixel_values')

        self.assertEqual('pixel_values:2x3x192x384', result)

    def test_derive_vision_input_shape_dynamic_without_hf_config_uses_defaults(self):
        """No ``_hf_config`` + dynamic dims → default (3, 224, 224)."""
        benchmark = self._make_benchmark(batch_size=1)
        # Ensure no _hf_config is set.
        if hasattr(benchmark, '_hf_config'):
            del benchmark._hf_config
        vision_input = _make_onnx_input('pixel_values', [0, 0, 0, 0])

        result = benchmark._derive_vision_input_shape(vision_input, 'pixel_values')

        self.assertEqual('pixel_values:1x3x224x224', result)

    # ------------------------------------------------------------------
    # _derive_nlp_input_shapes
    # ------------------------------------------------------------------

    def test_derive_nlp_input_shapes_single_2d(self):
        """A single 2D input emits a single ``name:BxS`` entry."""
        benchmark = self._make_benchmark(batch_size=8, seq_length=256)
        inputs = [_make_onnx_input('input_ids', [0, 0])]

        result = benchmark._derive_nlp_input_shapes(inputs)

        self.assertEqual('input_ids:8x256', result)

    def test_derive_nlp_input_shapes_multiple_inputs(self):
        """Multiple inputs are joined with commas in declaration order."""
        benchmark = self._make_benchmark(batch_size=4, seq_length=64)
        inputs = [
            _make_onnx_input('input_ids', [0, 0]),
            _make_onnx_input('attention_mask', [0, 0]),
            _make_onnx_input('token_type_ids', [0, 0]),
        ]

        result = benchmark._derive_nlp_input_shapes(inputs)

        self.assertEqual(
            'input_ids:4x64,attention_mask:4x64,token_type_ids:4x64',
            result,
        )

    def test_derive_nlp_input_shapes_4d_input_uses_bx1xsxs(self):
        """A 4D input (rare for NLP) gets the ``Bx1xSxS`` shape."""
        benchmark = self._make_benchmark(batch_size=2, seq_length=32)
        inputs = [_make_onnx_input('attention_bias', [0, 0, 0, 0])]

        result = benchmark._derive_nlp_input_shapes(inputs)

        self.assertEqual('attention_bias:2x1x32x32', result)

    def test_derive_nlp_input_shapes_default_seq_length_when_missing(self):
        """When ``_args.seq_length`` is absent, the helper defaults to 512."""
        benchmark = self._make_benchmark()
        # Strip seq_length to trigger the getattr-default branch.
        del benchmark._args.seq_length
        inputs = [_make_onnx_input('input_ids', [0, 0])]

        result = benchmark._derive_nlp_input_shapes(inputs)

        self.assertEqual('input_ids:8x512', result)
