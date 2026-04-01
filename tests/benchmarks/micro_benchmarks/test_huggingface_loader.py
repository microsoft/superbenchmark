# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for HuggingFaceModelLoader."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import (
    HuggingFaceModelLoader,
    ModelNotFoundError,
)
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class TestHuggingFaceModelLoader:
    """Test cases for HuggingFaceModelLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        return HuggingFaceModelLoader(cache_dir='/tmp/test_cache', token=None)

    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.cache_dir == '/tmp/test_cache'
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

        # Verify mocks were called with correct arguments
        mock_config.from_pretrained.assert_called_once()
        call_kwargs = mock_config.from_pretrained.call_args
        assert call_kwargs[0][0] == 'test/model'
        assert call_kwargs[1]['trust_remote_code'] is True
        assert call_kwargs[1]['cache_dir'] == '/tmp/test_cache'

        mock_model.from_pretrained.assert_called_once()
        model_call_kwargs = mock_model.from_pretrained.call_args
        assert model_call_kwargs[1]['trust_remote_code'] is True
        assert model_call_kwargs[1]['cache_dir'] == '/tmp/test_cache'

        mock_tokenizer.from_pretrained.assert_called_once()

        # Verify model was moved to the requested device
        mock_mdl.to.assert_called_once_with('cpu')

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
            torch.randn(1000, 1000),  # 1M params
            torch.randn(500, 500),     # 0.25M params
        ]

        size = loader._get_model_size(mock_model)
        assert abs(size - 1.25) < 0.01  # Should be ~1.25M
