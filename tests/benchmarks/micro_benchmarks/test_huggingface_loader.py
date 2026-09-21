# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for HuggingFaceModelLoader."""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import (
    HuggingFaceModelLoader,
    ModelNotFoundError,
)
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class TestHuggingFaceModelLoader:
    """Test cases for HuggingFaceModelLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create a loader instance for testing."""
        return HuggingFaceModelLoader(cache_dir=str(tmp_path / 'test_cache'), token=None)

    def test_initialization(self, loader, tmp_path):
        """Test loader initialization."""
        assert loader.cache_dir == str(tmp_path / 'test_cache')
        assert loader.token is None

    def test_initialization_with_env_token(self, monkeypatch, tmp_path):
        """Test loader picks up token from environment."""
        monkeypatch.setenv('HF_TOKEN', 'env_token')
        monkeypatch.setenv('HF_HOME', str(tmp_path / 'hf_cache'))
        loader = HuggingFaceModelLoader()
        assert loader.token == 'env_token'

    def test_get_torch_dtype_valid(self, loader):
        """Test torch dtype conversion."""
        assert loader._get_torch_dtype('float32') == torch.float32
        assert loader._get_torch_dtype('float16') == torch.float16
        assert loader._get_torch_dtype('fp16') == torch.float16
        assert loader._get_torch_dtype('bfloat16') == torch.bfloat16

    def test_get_torch_dtype_invalid(self, loader):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError, match='Invalid dtype'):
            loader._get_torch_dtype('invalid_dtype')

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModel')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoTokenizer')
    def test_load_model_success(self, mock_tokenizer, mock_config, mock_model, loader):
        """Test successful model loading."""
        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.model_type = 'bert'
        mock_config.from_pretrained.return_value = mock_cfg

        # Mock model
        mock_mdl = MagicMock()
        mock_mdl.parameters.return_value = [torch.randn(100, 100)]
        mock_mdl.to.return_value = mock_mdl
        mock_model.from_pretrained.return_value = mock_mdl

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        model, config, tokenizer = loader.load_model('test/model', device='cpu')

        assert model == mock_mdl
        assert config == mock_cfg
        assert tokenizer == mock_tok

        # Verify mocks were called with correct arguments. trust_remote_code must
        # default to False (matches loader.allow_remote_code=False) so that arbitrary
        # repo Python is not executed unless the caller explicitly opts in.
        mock_config.from_pretrained.assert_called_once()
        call_kwargs = mock_config.from_pretrained.call_args
        assert call_kwargs[0][0] == 'test/model'
        assert call_kwargs[1]['trust_remote_code'] is False
        assert call_kwargs[1]['cache_dir'] == loader.cache_dir

        mock_model.from_pretrained.assert_called_once()
        model_call_kwargs = mock_model.from_pretrained.call_args
        assert model_call_kwargs[1]['trust_remote_code'] is False
        assert model_call_kwargs[1]['cache_dir'] == loader.cache_dir

        mock_tokenizer.from_pretrained.assert_called_once()

        # Verify model was moved to the requested device
        mock_mdl.to.assert_called_once_with('cpu')

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModelForCausalLM')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModel')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoTokenizer')
    def test_load_model_uses_causal_lm_architecture(self, mock_tokenizer, mock_model, mock_causal_model, loader):
        """Causal-LM configs load the model with its language-model head."""
        config = MagicMock(architectures=['Qwen2ForCausalLM'])
        model = MagicMock()
        model.parameters.return_value = []
        model.to.return_value = model
        mock_causal_model.from_pretrained.return_value = model

        loaded_model, _, _ = loader.load_model('test/model', device='cpu', config=config)

        assert loaded_model is model
        mock_causal_model.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_not_called()

    def test_estimate_param_count_requires_attention_heads(self):
        """Configs without usable attention-head metadata are not estimated."""
        config = MagicMock(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=0,
            num_key_value_heads=0,
        )

        assert HuggingFaceModelLoader.estimate_param_count_from_config(config) is None

    def test_estimate_param_count_dense_and_moe_models(self):
        """Parameter estimation accounts for dense, rotary, gated, and MoE layers."""
        dense_config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=128,
            position_embedding_type='absolute',
            hidden_act='gelu',
            num_local_experts=1,
        )
        moe_config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=128,
            position_embedding_type='rotary',
            hidden_act='silu',
            num_local_experts=4,
        )

        dense_count = HuggingFaceModelLoader.estimate_param_count_from_config(dense_config)
        moe_count = HuggingFaceModelLoader.estimate_param_count_from_config(moe_config)

        assert dense_count is not None
        assert moe_count is not None
        assert moe_count > dense_count

    def test_estimate_memory_for_cpu_and_gpu(self):
        """Memory estimates apply precision/mode multipliers and device capacity."""
        with patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.torch.cuda') as cuda:
            cuda.is_available.return_value = False
            estimated, available, fits = HuggingFaceModelLoader.estimate_memory(1_000_000, 'fp16', 'inference')
            assert estimated == 2_400_000
            assert available > 0
            assert fits is True

            cuda.is_available.return_value = True
            cuda.get_device_properties.return_value = SimpleNamespace(total_memory=1_000_000)
            estimated, available, fits = HuggingFaceModelLoader.estimate_memory(1_000_000, 'int8', 'training')
            assert estimated == 4_000_000
            assert available == 1_000_000
            assert fits is False

    def test_check_memory_fits_reports_fit_failure_and_unknown(self):
        """Preflight reports fit status and skips only genuinely unestimable configs."""
        with patch.object(HuggingFaceModelLoader, 'estimate_param_count_from_config', return_value=None):
            assert HuggingFaceModelLoader.check_memory_fits('test/model', MagicMock(), 'fp16') == (True, 0, 0, 0)

        with patch.object(HuggingFaceModelLoader, 'estimate_param_count_from_config', return_value=2_000_000), \
                patch.object(
                    HuggingFaceModelLoader, 'estimate_memory', return_value=(4_000_000, 8_000_000, True)
                ):
            result = HuggingFaceModelLoader.check_memory_fits('test/model', MagicMock(), 'fp16', mode='inference')
            assert result == (True, 2.0, 0.004, 0.008)

        with patch.object(HuggingFaceModelLoader, 'estimate_param_count_from_config', return_value=2_000_000), \
                patch.object(
                    HuggingFaceModelLoader, 'estimate_memory', return_value=(8_000_000, 4_000_000, False)
                ), patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.torch.cuda') as cuda:
            cuda.is_available.return_value = True
            result = HuggingFaceModelLoader.check_memory_fits('test/model', MagicMock(), 'fp32')
            assert result == (False, 2.0, 0.008, 0.004)

    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoTokenizer')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoModel')
    @patch('superbench.benchmarks.micro_benchmarks.huggingface_model_loader.AutoConfig')
    def test_load_model_not_found(self, mock_config, mock_model, mock_tokenizer, loader):
        """Test loading non-existent model."""
        mock_config.from_pretrained.side_effect = OSError('404 Client Error')

        with pytest.raises(ModelNotFoundError, match='not found'):
            loader.load_model('nonexistent/model')

    def test_load_model_from_config_invalid_source(self, loader):
        """Test loading with invalid source in config."""
        config = ModelSourceConfig(source='in-house', identifier='bert-base')

        with pytest.raises(ValueError, match='Cannot load model'):
            loader.load_model_from_config(config)

    def test_get_model_size(self, loader):
        """Test model size calculation."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            torch.randn(1000, 1000),    # 1M params
            torch.randn(500, 500),    # 0.25M params
        ]

        size = loader._get_model_size(mock_model)
        assert abs(size - 1.25) < 0.01    # Should be ~1.25M
