# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugging Face model loader for benchmarking."""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

from superbench.common.utils import logger
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelNotFoundError(ModelLoadError):
    """Exception raised when model is not found."""
    pass


class ModelIncompatibleError(ModelLoadError):
    """Exception raised when model is incompatible with ONNX export."""
    pass


class HuggingFaceModelLoader:
    """Loads models from Hugging Face Hub for benchmarking.

    This class handles downloading, caching, and loading models from
    Hugging Face Hub with support for authentication, device mapping,
    and compatibility validation.

    Attributes:
        cache_dir: Directory to cache downloaded models.
        token: HuggingFace authentication token for private/gated models.
    """
    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """Initialize the HuggingFace model loader.

        Args:
            cache_dir: Directory to cache downloaded models. If None, uses HF default.
            token: HuggingFace authentication token for private/gated models.
        """
        self.cache_dir = cache_dir or os.getenv('HF_HOME') or os.path.expanduser('~/.cache/huggingface')
        self.token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f'HuggingFaceModelLoader initialized with cache_dir: {self.cache_dir}')
        if self.token:
            logger.info('Authentication token provided for private/gated models (token not logged)')

    def load_model(
        self,
        model_identifier: str,
        torch_dtype: Optional[str] = None,
        device: str = 'cuda',
        revision: Optional[str] = None,
        device_map: Optional[str] = None,
        config: Optional[PretrainedConfig] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PretrainedConfig, AutoTokenizer]:
        """Load a model from Hugging Face Hub.

        Args:
            model_identifier: HF model ID (e.g., 'meta-llama/Llama-2-7b-hf').
            torch_dtype: Data type for model weights ('float32', 'float16', 'bfloat16').
            device: Device to load model on ('cuda', 'cpu').
            revision: Specific model version/commit/tag to use.
            device_map: Device mapping strategy for large models.
            config: Pre-downloaded model config. If None, downloads from Hub.
            **kwargs: Additional arguments passed to from_pretrained().

        Returns:
            Tuple of (model, config, tokenizer).

        Raises:
            ModelNotFoundError: If model doesn't exist on HF Hub.
            ModelLoadError: If model loading fails for any reason.
        """
        logger.info(f'Loading model: {model_identifier}')

        try:
            # Convert torch_dtype string to torch dtype
            dtype = self._get_torch_dtype(torch_dtype) if torch_dtype else None

            # Prepare loading kwargs
            load_kwargs = {'cache_dir': self.cache_dir, 'revision': revision, **kwargs}

            # Add token if available
            if self.token:
                load_kwargs['token'] = self.token

            # Add dtype if specified
            if dtype:
                load_kwargs['torch_dtype'] = dtype

            # Load config (use pre-downloaded config if provided)
            if config is None:
                logger.info('Loading model configuration...')
                config = AutoConfig.from_pretrained(model_identifier, trust_remote_code=True, **load_kwargs)
            else:
                logger.info('Using pre-downloaded model configuration.')

            # Load tokenizer (may fail for some models, that's ok)
            tokenizer = None
            try:
                logger.info('Loading tokenizer...')
                tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True, **load_kwargs)
            except Exception as e:
                logger.warning(f'Could not load tokenizer: {e}. Continuing without tokenizer.')

            # Load model
            logger.info(f'Loading model weights (dtype={torch_dtype}, device={device})...')
            model_kwargs = load_kwargs.copy()
            model_kwargs['trust_remote_code'] = True

            # Handle device mapping for large models
            if device_map:
                model_kwargs['device_map'] = device_map
            elif device == 'cuda' and torch.cuda.is_available():
                # Don't set device_map if device is explicitly cuda
                pass
            elif device != 'cpu':
                model_kwargs['device_map'] = device

            # Pass pre-downloaded config to from_pretrained so any overrides take effect
            if config is not None:
                model_kwargs['config'] = config

            try:
                model = AutoModel.from_pretrained(model_identifier, **model_kwargs)
            except ValueError:
                logger.info('AutoModel failed, trying AutoModelForCausalLM...')
                model = AutoModelForCausalLM.from_pretrained(model_identifier, **model_kwargs)

            # Move to device if not using device_map
            if not device_map and device != 'auto':
                model = model.to(device)

            logger.info(
                f'Successfully loaded model: {model_identifier} '
                f'({self._get_model_size(model):.2f}M parameters)'
            )

            return model, config, tokenizer

        except OSError as e:
            if 'not found' in str(e).lower() or '404' in str(e):
                raise ModelNotFoundError(
                    f"Model '{model_identifier}' not found on Hugging Face Hub. "
                    f'Please check the model ID at https://huggingface.co/models'
                ) from e
            raise ModelLoadError(f"Failed to load model '{model_identifier}': {e}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading model '{model_identifier}': {e}") from e

    def load_model_from_config(
        self,
        config: ModelSourceConfig,
        device: Optional[str] = None,
        config_pretrained: Optional[PretrainedConfig] = None,
    ) -> Tuple[PreTrainedModel, PretrainedConfig, AutoTokenizer]:
        """Load a model using ModelSourceConfig.

        Args:
            config: ModelSourceConfig instance with loading parameters.
            device: Device to load model on. If None, uses CUDA when available, else CPU.
            config_pretrained: Pre-downloaded HF model config. If provided, skips redundant download.

        Returns:
            Tuple of (model, config, tokenizer).

        Raises:
            ValueError: If config source is not 'huggingface'.
            ModelLoadError: If model loading fails.
        """
        if not config.is_huggingface():
            raise ValueError(f"Cannot load model with source '{config.source}'. Use 'huggingface' source.")

        # Validate config
        is_valid, error = config.validate()
        if not is_valid:
            raise ValueError(f'Invalid configuration: {error}')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Extract loading parameters
        return self.load_model(
            model_identifier=config.identifier,
            torch_dtype=config.torch_dtype,
            device=device,
            revision=config.revision,
            device_map=config.device_map,
            config=config_pretrained,
            **config.additional_kwargs
        )

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch.dtype.

        Args:
            dtype_str: String representation of dtype ('float32', 'float16', etc.).

        Returns:
            Corresponding torch.dtype.

        Raises:
            ValueError: If dtype string is invalid.
        """
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int8': torch.int8,
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }

        if dtype_str.lower() not in dtype_map:
            raise ValueError(f"Invalid dtype '{dtype_str}'.Must be one of {list(dtype_map.keys())}")

        return dtype_map[dtype_str.lower()]

    def _get_model_size(self, model: PreTrainedModel) -> float:
        """Calculate model size in millions of parameters.

        Args:
            model: The model to measure.

        Returns:
            Number of parameters in millions.
        """
        return float(sum(p.numel() for p in model.parameters())) / 1_000_000

    @staticmethod
    def estimate_param_count_from_config(hf_config) -> Optional[int]:
        """Estimate parameter count from a HuggingFace config without instantiating the model.

        This avoids allocating tens/hundreds of GB of CPU RAM for large models (e.g. 70B).
        The estimate covers embedding + transformer layers + LM head for common architectures.

        Args:
            hf_config: A HuggingFace PretrainedConfig object.

        Returns:
            int: Estimated number of parameters, or None if estimation is not possible.
        """
        try:
            vocab = getattr(hf_config, 'vocab_size', 0)
            hidden = getattr(hf_config, 'hidden_size', 0)
            layers = getattr(hf_config, 'num_hidden_layers', 0)
            intermediate = getattr(hf_config, 'intermediate_size', hidden * 4)
            num_heads = getattr(hf_config, 'num_attention_heads', 0)
            num_kv_heads = getattr(hf_config, 'num_key_value_heads', num_heads)
            head_dim = hidden // num_heads if num_heads > 0 else 0

            if vocab == 0 or hidden == 0 or layers == 0:
                return None

            # Embeddings: token + (optional) position
            max_pos = getattr(hf_config, 'max_position_embeddings', 0)
            has_pos_embed = getattr(hf_config, 'position_embedding_type', None) not in ('rotary', None)
            embed_params = vocab * hidden
            if has_pos_embed and max_pos > 0:
                embed_params += max_pos * hidden

            # Per transformer layer:
            #   Self-attention: Q, K, V projections + output projection
            #   MLP: gate_proj + up_proj + down_proj (LLaMA-style) or fc1 + fc2
            #   Layer norms: 2 * hidden
            qkv_params = (num_heads * head_dim + 2 * num_kv_heads * head_dim) * hidden
            attn_out = hidden * hidden
            # For gated MLPs (LLaMA/Mistral), there are 3 matrices; otherwise 2
            has_gate = getattr(hf_config, 'hidden_act', 'gelu') in ('silu', 'swiglu')
            mlp_params = (3 if has_gate else 2) * hidden * intermediate
            norm_params = 2 * hidden
            layer_params = qkv_params + attn_out + mlp_params + norm_params

            # MoE: if num_local_experts > 1, MLP is replicated per expert
            num_experts = getattr(hf_config, 'num_local_experts', 1)
            if num_experts > 1:
                # Router + replicated MLP experts (attention is shared)
                router_params = hidden * num_experts
                layer_params = qkv_params + attn_out + norm_params + \
                    num_experts * mlp_params + router_params

            total_params = embed_params + layers * layer_params
            # LM head (often tied to embedding, but count it for safety)
            total_params += vocab * hidden
            # Final layer norm
            total_params += hidden

            return total_params
        except Exception as e:
            logger.warning(f'Could not estimate param count from config: {e}')
            return None

    @staticmethod
    def estimate_memory(param_count, precision_str, mode='training'):
        """Estimate GPU memory required for a model.

        For training: weights + gradients + optimizer states (Adam uses 2x) = 4x multiplier.
        For inference: weights only + overhead for runtime buffers = ~1.2x multiplier.

        Args:
            param_count (int): Number of model parameters.
            precision_str (str): Precision string ('float32', 'float16', 'bfloat16', 'fp16', 'fp32', 'int8').
            mode (str): 'training' or 'inference'.

        Returns:
            tuple: (estimated_bytes, gpu_total_bytes, fits) where fits is True if
                   the model is estimated to fit in available memory.
        """
        precision_lower = precision_str.lower()
        if precision_lower in ('float16', 'fp16', 'bfloat16', 'bf16'):
            bytes_per_param = 2
        elif precision_lower in ('int8', ):
            bytes_per_param = 1
        else:
            bytes_per_param = 4

        if mode == 'training':
            # weights + gradients + 2x Adam optimizer states = 4x
            multiplier = 4
        else:
            # inference: weights + runtime overhead (~20%)
            multiplier = 1.2

        estimated_bytes = int(param_count * bytes_per_param * multiplier)

        gpu_available = torch.cuda.is_available()
        if not gpu_available:
            try:
                import psutil
                sys_mem = psutil.virtual_memory().total
            except ImportError:
                logger.warning('psutil not installed — cannot check system memory. Skipping memory check.')
                return 0, 0, True
            max_gpu_mem = 80 * (1024**3)    # 80GB — largest common single-GPU memory
            effective_mem = min(sys_mem, max_gpu_mem)
            fits = (estimated_bytes / effective_mem) < 0.85
            return estimated_bytes, effective_mem, fits

        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        # Use 85% threshold to leave headroom for activations, framework overhead, etc.
        fits = (estimated_bytes / gpu_mem) < 0.85
        return estimated_bytes, gpu_mem, fits

    @staticmethod
    def check_memory_fits(model_identifier, hf_config, precision_str, mode='training', token=None):
        """Check if a model fits in GPU memory before downloading weights.

        Downloads only the config (few KB) via hf_config, estimates memory, and returns
        whether the model fits. Use this before calling load_model() to avoid wasting
        time downloading large models that won't fit.

        Args:
            model_identifier (str): HF model ID (for logging).
            hf_config: A HuggingFace PretrainedConfig object.
            precision_str (str): Precision string ('float32', 'float16', etc.).
            mode (str): 'training' or 'inference'.
            token (str, optional): HF token (unused, kept for API consistency).

        Returns:
            tuple: (fits, param_count_millions, estimated_gb, available_gb)
                   fits is True if model is estimated to fit.
        """
        param_count = HuggingFaceModelLoader.estimate_param_count_from_config(hf_config)
        if param_count is None:
            logger.warning(
                f'Could not estimate param count from config for {model_identifier}. '
                f'Proceeding with download — memory check skipped.'
            )
            return True, 0, 0, 0

        estimated_bytes, available_bytes, fits = HuggingFaceModelLoader.estimate_memory(
            param_count, precision_str, mode=mode
        )

        param_millions = param_count / 1e6
        estimated_gb = estimated_bytes / 1e9
        available_gb = available_bytes / 1e9

        if fits:
            logger.info(
                f'Model {model_identifier} ({param_millions:.1f}M params) estimated to need '
                f'~{estimated_gb:.1f}GB for {mode}, fits in available memory ({available_gb:.1f}GB).'
            )
        else:
            mem_type = 'GPU memory' if torch.cuda.is_available() else 'system RAM'
            logger.error(
                f'Model {model_identifier} ({param_millions:.1f}M params) estimated to need '
                f'~{estimated_gb:.1f}GB for {mode} (weights'
                f'{" + gradients + optimizer states" if mode == "training" else " + runtime overhead"}), '
                f'which exceeds available {mem_type} ({available_gb:.1f}GB). '
                f'Skipping benchmark. Use a smaller model variant or a machine with more memory.'
            )

        return fits, param_millions, estimated_gb, available_gb

    def __repr__(self) -> str:
        """String representation of the loader."""
        token_status = 'authenticated' if self.token else 'no authentication'
        return f"HuggingFaceModelLoader(cache_dir='{self.cache_dir}', {token_status})"
