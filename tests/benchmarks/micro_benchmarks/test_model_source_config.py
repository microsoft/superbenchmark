# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for ModelSourceConfig."""

import pytest
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class TestModelSourceConfig:
    """Test cases for ModelSourceConfig class."""
    def test_default_config(self):
        """Test default configuration."""
        config = ModelSourceConfig(identifier='bert-base')
        assert config.source == 'in-house'
        assert config.identifier == 'bert-base'
        assert config.torch_dtype == 'float32'
        assert config.hf_token is None

    def test_huggingface_config(self):
        """Test HuggingFace configuration."""
        config = ModelSourceConfig(source='huggingface', identifier='meta-llama/Llama-2-7b-hf', torch_dtype='float16')
        assert config.source == 'huggingface'
        assert config.identifier == 'meta-llama/Llama-2-7b-hf'
        assert config.torch_dtype == 'float16'

    def test_invalid_source(self):
        """Test invalid source raises error."""
        with pytest.raises(ValueError, match='Invalid model source'):
            ModelSourceConfig(source='invalid', identifier='test')

    def test_invalid_dtype(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError, match='Invalid torch_dtype'):
            ModelSourceConfig(identifier='test', torch_dtype='invalid')

    def test_missing_identifier(self):
        """Test missing identifier raises error."""
        with pytest.raises(ValueError, match='identifier must be provided'):
            ModelSourceConfig(identifier='')

    def test_validate_huggingface_empty(self):
        """Test validation of empty HuggingFace model identifier."""
        config = ModelSourceConfig(source='huggingface', identifier='   ')
        is_valid, message = config.validate()
        assert not is_valid
        assert 'cannot be empty' in message

    def test_validate_valid_huggingface(self):
        """Test validation of valid HuggingFace model."""
        config = ModelSourceConfig(source='huggingface', identifier='meta-llama/Llama-2-7b-hf')
        is_valid, message = config.validate()
        assert is_valid
        assert message == ''

    def test_validate_valid_huggingface_short_name(self):
        """Test validation of valid HuggingFace model with short name (no org)."""
        config = ModelSourceConfig(source='huggingface', identifier='bert-base-uncased')
        is_valid, message = config.validate()
        assert is_valid
        assert message == ''

    def test_is_huggingface(self):
        """Test is_huggingface method."""
        hf_config = ModelSourceConfig(source='huggingface', identifier='test/model')
        inhouse_config = ModelSourceConfig(source='in-house', identifier='bert-base')
        assert hf_config.is_huggingface() is True
        assert inhouse_config.is_huggingface() is False

    def test_deprecated_use_auth_token(self):
        """Test deprecated use_auth_token parameter."""
        config = ModelSourceConfig(identifier='test', use_auth_token='old_token')
        assert config.hf_token == 'old_token'
