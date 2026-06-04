# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for ``torch2onnxExporter`` HuggingFace export helpers.

Covers:
- ``export_huggingface_model`` (orchestration, error path, vision/NLP routing).
- ``_build_vision_export_inputs`` (config-driven C/H/W, VisionModelWrapper).
- ``_build_nlp_export_inputs`` (input_ids + attention_mask, NLPModelWrapper).
- ``_build_onnx_export_kwargs`` (opset/dynamic_axes; external-data branch).

Tests are pure-CPU and pure-unit: ``torch.onnx.export`` is patched out so we
never touch the ONNX runtime, and dummy ``torch.nn.Module`` instances stand in
for HuggingFace models.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from superbench.benchmarks.micro_benchmarks._export_torch_to_onnx import torch2onnxExporter

_EXPORTER_MODULE = 'superbench.benchmarks.micro_benchmarks._export_torch_to_onnx'

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    """Build a torch2onnxExporter rooted at a tmp dir to avoid touching the real torch hub."""
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
    return torch2onnxExporter()


class _TinyVisionModel(torch.nn.Module):
    """Minimal stand-in for a HuggingFace vision model.

    Mimics enough of the HF API for the export helpers: ``main_input_name``,
    a ``config`` namespace, and a ``forward`` that accepts ``pixel_values`` and
    returns an object with a ``logits`` attribute.
    """

    main_input_name = 'pixel_values'

    def __init__(self, num_channels=3, image_size=224, num_classes=4):
        super().__init__()
        self.config = SimpleNamespace(
            num_channels=num_channels,
            image_size=image_size,
            use_cache=True,
        )
        # A trivial trainable parameter so .parameters() / .element_size() are exercised.
        self.linear = torch.nn.Linear(num_channels, num_classes)

    def forward(self, pixel_values):
        # Reduce H/W and project channel dim, mimicking a classifier head.
        flat = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(logits=self.linear(flat))


class _TinyNLPModel(torch.nn.Module):
    """Minimal stand-in for a HuggingFace NLP model with input_ids + attention_mask."""

    main_input_name = 'input_ids'

    def __init__(self, vocab_size=128, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embed = torch.nn.Embedding(vocab_size, hidden)

    def forward(self, input_ids, attention_mask):
        h = self.embed(input_ids)
        # last_hidden_state path is exercised here.
        return SimpleNamespace(last_hidden_state=h * attention_mask.unsqueeze(-1).to(h.dtype))


# ---------------------------------------------------------------------------
# _build_vision_export_inputs
# ---------------------------------------------------------------------------


def test_build_vision_export_inputs_default_shape(exporter):
    """Default config (3 channels, 224 image_size) → (B, 3, 224, 224) tensor."""
    model = _TinyVisionModel(num_channels=3, image_size=224)

    wrapped, args, names, axes = exporter._build_vision_export_inputs(
        model, batch_size=2, model_dtype=torch.float32, device='cpu'
    )

    assert names == ['pixel_values']
    assert axes == {'pixel_values': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    assert len(args) == 1
    pixel_values = args[0]
    assert tuple(pixel_values.shape) == (2, 3, 224, 224)
    assert pixel_values.dtype == torch.float32
    assert pixel_values.device.type == 'cpu'
    # Wrapper is callable and returns the inner model's logits tensor (not a SimpleNamespace).
    out = wrapped(pixel_values)
    assert isinstance(out, torch.Tensor)


def test_build_vision_export_inputs_custom_channels_and_size(exporter):
    """Non-default num_channels / scalar image_size are honored."""
    model = _TinyVisionModel(num_channels=1, image_size=384)

    _, args, _, _ = exporter._build_vision_export_inputs(model, batch_size=4, model_dtype=torch.float32, device='cpu')
    pixel_values = args[0]
    assert tuple(pixel_values.shape) == (4, 1, 384, 384)


def test_build_vision_export_inputs_tuple_image_size(exporter):
    """Tuple/list ``image_size`` is unpacked as (H, W)."""
    model = _TinyVisionModel(num_channels=3)
    model.config.image_size = (192, 384)

    _, args, _, _ = exporter._build_vision_export_inputs(model, batch_size=1, model_dtype=torch.float32, device='cpu')
    assert tuple(args[0].shape) == (1, 3, 192, 384)


def test_build_vision_export_inputs_wrapper_handles_last_hidden_state(exporter):
    """The wrapper falls back to ``last_hidden_state`` when ``logits`` is absent."""
    model = _TinyVisionModel()
    # Override forward to return only last_hidden_state.
    hidden = torch.zeros(2, 4)

    class _ModelOnlyHidden(torch.nn.Module):
        main_input_name = 'pixel_values'

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_channels=3, image_size=8)
            self.dummy = torch.nn.Linear(1, 1)

        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=hidden)

    custom = _ModelOnlyHidden()
    wrapped, args, _, _ = exporter._build_vision_export_inputs(
        custom, batch_size=2, model_dtype=torch.float32, device='cpu'
    )
    out = wrapped(args[0])
    assert torch.equal(out, hidden)
    _ = model    # keep fixture-ish ref


def test_build_vision_export_inputs_wrapper_handles_tuple_output(exporter):
    """The wrapper returns ``outputs[0]`` when the model emits a tuple."""

    class _TupleModel(torch.nn.Module):
        main_input_name = 'pixel_values'

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_channels=3, image_size=8)
            self.dummy = torch.nn.Linear(1, 1)

        def forward(self, pixel_values):
            return (torch.ones(pixel_values.size(0), 2), torch.zeros(1))

    wrapped, args, _, _ = exporter._build_vision_export_inputs(
        _TupleModel(), batch_size=3, model_dtype=torch.float32, device='cpu'
    )
    out = wrapped(args[0])
    assert tuple(out.shape) == (3, 2)


# ---------------------------------------------------------------------------
# _build_nlp_export_inputs
# ---------------------------------------------------------------------------


def test_build_nlp_export_inputs_basic(exporter):
    """NLP path emits int64 ``input_ids`` + ``attention_mask`` of shape (B, S)."""
    model = _TinyNLPModel()

    wrapped, args, names, axes = exporter._build_nlp_export_inputs(model, batch_size=2, seq_length=16, device='cpu')

    assert names == ['input_ids', 'attention_mask']
    # Dynamic axes: batch_size + seq_length on both inputs and the output.
    assert axes['input_ids'] == {0: 'batch_size', 1: 'seq_length'}
    assert axes['attention_mask'] == {0: 'batch_size', 1: 'seq_length'}
    assert axes['output'] == {0: 'batch_size', 1: 'seq_length'}
    assert len(args) == 2
    input_ids, attention_mask = args
    assert tuple(input_ids.shape) == (2, 16)
    assert tuple(attention_mask.shape) == (2, 16)
    assert input_ids.dtype == torch.int64
    assert attention_mask.dtype == torch.int64
    # All ones ⇒ token id 1 is within the embedding's vocab.
    assert torch.all(input_ids == 1)
    # Wrapper runs the inner model and unwraps last_hidden_state.
    out = wrapped(input_ids, attention_mask)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (2, 16, 8)


def test_build_nlp_export_inputs_wrapper_handles_logits(exporter):
    """When the inner model exposes ``logits``, the wrapper returns those."""

    class _LogitsModel(torch.nn.Module):
        main_input_name = 'input_ids'

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.embed = torch.nn.Embedding(8, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.embed(input_ids))

    wrapped, args, _, _ = exporter._build_nlp_export_inputs(_LogitsModel(), batch_size=1, seq_length=4, device='cpu')
    out = wrapped(*args)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (1, 4, 4)


def test_build_nlp_export_inputs_wrapper_handles_tuple(exporter):
    """The wrapper returns ``outputs[0]`` when the model emits a tuple."""

    class _TupleNLP(torch.nn.Module):
        main_input_name = 'input_ids'

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.embed = torch.nn.Embedding(8, 4)

        def forward(self, input_ids, attention_mask):
            return (self.embed(input_ids), torch.zeros(1))

    wrapped, args, _, _ = exporter._build_nlp_export_inputs(_TupleNLP(), batch_size=1, seq_length=2, device='cpu')
    out = wrapped(*args)
    assert tuple(out.shape) == (1, 2, 4)


# ---------------------------------------------------------------------------
# _build_onnx_export_kwargs
# ---------------------------------------------------------------------------


def test_build_onnx_export_kwargs_small_model_no_external_data(exporter):
    """Small models (< 2GB) do not request external-data format."""
    model = _TinyNLPModel()
    input_names = ['input_ids', 'attention_mask']
    dynamic_axes = {'input_ids': {0: 'b'}}

    kwargs = exporter._build_onnx_export_kwargs(model, input_names, dynamic_axes)

    assert kwargs['opset_version'] == 14
    assert kwargs['do_constant_folding'] is True
    assert kwargs['input_names'] == input_names
    assert kwargs['output_names'] == ['output']
    assert kwargs['dynamic_axes'] is dynamic_axes
    assert 'external_data' not in kwargs
    assert 'use_external_data_format' not in kwargs


def test_build_onnx_export_kwargs_large_model_uses_external_data_modern(exporter):
    """For >2GB models on PyTorch with ``external_data`` param, that key is used."""
    fake_model = MagicMock()
    # 3 GB worth of fp32 params = 3 * (1024**3) / 4 numel.
    big_param = SimpleNamespace(
        numel=lambda: int(3 * (1024**3) / 4),
        element_size=lambda: 4,
    )
    fake_model.parameters.return_value = [big_param]

    fake_sig = inspect.Signature(
        parameters=[inspect.Parameter('external_data', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    with patch(f'{_EXPORTER_MODULE}.inspect.signature', return_value=fake_sig):
        kwargs = exporter._build_onnx_export_kwargs(fake_model, ['input_ids'], {})

    assert kwargs['external_data'] is True
    assert 'use_external_data_format' not in kwargs


def test_build_onnx_export_kwargs_large_model_uses_external_data_legacy(exporter):
    """For >2GB models on older PyTorch, ``use_external_data_format`` is used instead."""
    fake_model = MagicMock()
    big_param = SimpleNamespace(
        numel=lambda: int(3 * (1024**3) / 4),
        element_size=lambda: 4,
    )
    fake_model.parameters.return_value = [big_param]

    fake_sig = inspect.Signature(
        parameters=[inspect.Parameter('use_external_data_format', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    with patch(f'{_EXPORTER_MODULE}.inspect.signature', return_value=fake_sig):
        kwargs = exporter._build_onnx_export_kwargs(fake_model, ['input_ids'], {})

    assert kwargs['use_external_data_format'] is True
    assert 'external_data' not in kwargs


# ---------------------------------------------------------------------------
# export_huggingface_model
# ---------------------------------------------------------------------------


def test_export_huggingface_model_vision_routes_to_vision_helper(exporter, tmp_path):
    """Vision model (main_input_name='pixel_values') uses the vision helper."""
    model = _TinyVisionModel(num_channels=3, image_size=32)

    captured = {}

    def fake_export(wrapped_model, args, file_name, **kwargs):
        captured['wrapped_model'] = wrapped_model
        captured['args'] = args
        captured['file_name'] = file_name
        captured['kwargs'] = kwargs
        Path(file_name).touch()

    with patch(f'{_EXPORTER_MODULE}.torch.onnx.export', side_effect=fake_export):
        result = exporter.export_huggingface_model(
            model=model,
            model_name='vit-tiny',
            batch_size=2,
            seq_length=16,
            output_dir=str(tmp_path),
        )

    assert result == str(tmp_path / 'vit-tiny.onnx')
    # Vision shape: (B, C, H, W) tuple of length 1.
    assert len(captured['args']) == 1
    assert tuple(captured['args'][0].shape) == (2, 3, 32, 32)
    assert captured['kwargs']['input_names'] == ['pixel_values']
    assert captured['kwargs']['opset_version'] == 14
    # use_cache disabled to avoid DynamicCache issues.
    assert model.config.use_cache is False


def test_export_huggingface_model_nlp_routes_to_nlp_helper(exporter, tmp_path):
    """NLP model (main_input_name='input_ids') uses the NLP helper."""
    model = _TinyNLPModel()

    captured = {}

    def fake_export(wrapped_model, args, file_name, **kwargs):
        captured['args'] = args
        captured['file_name'] = file_name
        captured['kwargs'] = kwargs
        Path(file_name).touch()

    with patch(f'{_EXPORTER_MODULE}.torch.onnx.export', side_effect=fake_export):
        result = exporter.export_huggingface_model(
            model=model,
            model_name='bert-tiny',
            batch_size=2,
            seq_length=8,
            output_dir=str(tmp_path),
        )

    assert result == str(tmp_path / 'bert-tiny.onnx')
    assert len(captured['args']) == 2
    input_ids, attention_mask = captured['args']
    assert tuple(input_ids.shape) == (2, 8)
    assert tuple(attention_mask.shape) == (2, 8)
    assert captured['kwargs']['input_names'] == ['input_ids', 'attention_mask']


def test_export_huggingface_model_default_output_dir(exporter):
    """When ``output_dir`` is None, the exporter writes under self._onnx_model_path."""
    model = _TinyNLPModel()

    written = {}

    def fake_export(wrapped_model, args, file_name, **kwargs):
        written['file_name'] = file_name
        Path(file_name).touch()

    with patch(f'{_EXPORTER_MODULE}.torch.onnx.export', side_effect=fake_export):
        result = exporter.export_huggingface_model(model=model, model_name='bert-default')

    expected = str(exporter._onnx_model_path / 'bert-default.onnx')
    assert result == expected
    assert written['file_name'] == expected


def test_export_huggingface_model_handles_export_failure(exporter, tmp_path):
    """If ``torch.onnx.export`` raises, the helper returns '' and logs the error."""
    model = _TinyNLPModel()

    with patch(f'{_EXPORTER_MODULE}.torch.onnx.export', side_effect=RuntimeError('boom')):
        result = exporter.export_huggingface_model(
            model=model,
            model_name='bert-fail',
            batch_size=1,
            seq_length=4,
            output_dir=str(tmp_path),
        )

    assert result == ''


def test_export_huggingface_model_disables_use_cache(exporter, tmp_path):
    """``model.config.use_cache`` is forced to False before export."""
    model = _TinyNLPModel()
    model.config.use_cache = True

    with patch(f'{_EXPORTER_MODULE}.torch.onnx.export') as mock_export:
        mock_export.side_effect = lambda *a, **kw: Path(a[2]).touch()
        exporter.export_huggingface_model(
            model=model,
            model_name='bert-cache',
            batch_size=1,
            seq_length=4,
            output_dir=str(tmp_path),
        )

    assert model.config.use_cache is False


def test_export_huggingface_model_default_main_input_name_is_nlp(exporter, tmp_path):
    """Models without ``main_input_name`` default to the NLP path."""

    class _NoMainInput(torch.nn.Module):
        # Intentionally no main_input_name attribute.
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=False)
            self.embed = torch.nn.Embedding(8, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    captured = {}

    def fake_export(wrapped_model, args, file_name, **kwargs):
        captured['kwargs'] = kwargs
        Path(file_name).touch()

    with patch(f'{_EXPORTER_MODULE}.torch.onnx.export', side_effect=fake_export):
        result = exporter.export_huggingface_model(
            model=_NoMainInput(),
            model_name='no-main',
            batch_size=1,
            seq_length=4,
            output_dir=str(tmp_path),
        )

    assert result == str(tmp_path / 'no-main.onnx')
    assert captured['kwargs']['input_names'] == ['input_ids', 'attention_mask']
