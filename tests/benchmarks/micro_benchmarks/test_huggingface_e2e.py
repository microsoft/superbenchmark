# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end integration tests for HuggingFace model loading.

These tests actually download and load models from HuggingFace Hub.
The test class is skipped unless ``SB_TEST_HF_E2E=1`` is set, and
``test_load_model_to_gpu`` is additionally skipped when
``torch.cuda.is_available()`` is false.
"""

import os

import pytest
import torch

pytest.importorskip('transformers')

# Imports below this point depend on `transformers` being available, so they
# must be deferred until after the `importorskip` call above.
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import (    # noqa: E402
    HuggingFaceModelLoader,
)
from superbench.benchmarks.micro_benchmarks.model_source_config import (    # noqa: E402
    ModelSourceConfig,
)


@pytest.mark.skipif(
    os.environ.get('SB_TEST_HF_E2E', '0') != '1',
    reason='Skip HF E2E tests. Set SB_TEST_HF_E2E=1 to enable.',
)
class TestHuggingFaceE2E:
    """End-to-end tests for HuggingFace model loading."""
    @pytest.fixture
    def loader(self, tmp_path):
        """Create a loader instance with an isolated per-test cache dir."""
        return HuggingFaceModelLoader(cache_dir=str(tmp_path / 'hf_cache'))

    def test_load_tiny_bert_model(self, loader):
        """Test loading a tiny BERT model from HuggingFace Hub.

        Uses prajjwal1/bert-tiny which is a small public BERT model (~17MB).
        """
        model, config, _ = loader.load_model('prajjwal1/bert-tiny', device='cpu')

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
        model, config, _ = loader.load_model('distilbert/distilgpt2', device='cpu')

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

        model, hf_config, _ = loader.load_model_from_config(config, device='cpu')

        assert model is not None
        assert hf_config.model_type == 'bert'

    def test_load_model_with_dtype(self, loader):
        """Test loading model and converting dtype after load."""
        model, _, _ = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        # Convert to float32 after loading
        model = model.float()

        # Check model parameters are float32
        param = next(model.parameters())
        assert param.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires GPU')
    def test_load_model_to_gpu(self, loader):
        """Test loading model and moving to GPU."""
        model, _, _ = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        # Move to GPU manually
        model = model.cuda()

        # Check model is on GPU
        param = next(model.parameters())
        assert param.device.type == 'cuda'

    def test_architecture_detection(self, loader):
        """Test that architecture is correctly detected from loaded model."""
        _, config, _ = loader.load_model('prajjwal1/bert-tiny', device='cpu')

        # Architecture should be detected from config
        assert config.model_type is not None
        assert 'bert' in config.model_type.lower()
