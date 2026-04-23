# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end integration tests for HuggingFace model loading.

These tests actually download and load models from HuggingFace Hub.
The test class is skipped according to ``@decorator.cuda_test``, and
``test_load_model_to_gpu`` is additionally skipped when
``torch.cuda.is_available()`` is false.
"""

import pytest
import torch

transformers = pytest.importorskip('transformers')

from tests.helper import decorator
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


@decorator.cuda_test
class TestHuggingFaceE2E:
    """End-to-end tests for HuggingFace model loading."""
    @pytest.fixture
    def loader(self):
        """Create a loader instance."""
        return HuggingFaceModelLoader(cache_dir='/tmp/hf_test_cache')

    def test_load_tiny_bert_model(self, loader):
        """Test loading a tiny BERT model from HuggingFace Hub.

        Uses prajjwal1/bert-tiny which is a small public BERT model (~17MB).
        """
        model, config, tokenizer = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        assert model is not None
        assert config is not None
        assert config.model_type == 'bert'

        # Verify model can do a forward pass
        dummy_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(dummy_input)
        assert output is not None

    def test_load_distilgpt2_model(self, loader):
        """Test loading DistilGPT2 model from HuggingFace Hub.

        Uses distilbert/distilgpt2 which is a small public GPT-2 model (~82MB).
        """
        model, config, tokenizer = loader.load_model('distilbert/distilgpt2', device='cpu')

        assert model is not None
        assert config is not None
        assert config.model_type == 'gpt2'

        # Verify model can do a forward pass
        dummy_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(dummy_input)
        assert output is not None

    def test_load_model_from_config(self, loader):
        """Test loading model using ModelSourceConfig via load_model_from_config."""
        config = ModelSourceConfig(source='huggingface', identifier='prajjwal1/bert-tiny', torch_dtype='float32')

        model, hf_config, tokenizer = loader.load_model_from_config(config, device='cpu')

        assert model is not None
        assert hf_config.model_type == 'bert'

    def test_load_model_with_dtype(self, loader):
        """Test loading model and converting dtype after load."""
        model, config, tokenizer = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        # Convert to float32 after loading
        model = model.float()

        # Check model parameters are float32
        param = next(model.parameters())
        assert param.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires GPU')
    def test_load_model_to_gpu(self, loader):
        """Test loading model and moving to GPU."""
        model, config, tokenizer = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        # Move to GPU manually
        model = model.cuda()

        # Check model is on GPU
        param = next(model.parameters())
        assert param.device.type == 'cuda'

    def test_architecture_detection(self, loader):
        """Test that architecture is correctly detected from loaded model."""
        model, config, tokenizer = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        # Architecture should be detected from config
        assert config.model_type is not None
        assert 'bert' in config.model_type.lower()
