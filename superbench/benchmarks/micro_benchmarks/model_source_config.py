# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration classes for model source and loading."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


@dataclass
class ModelSourceConfig:
    """Configuration for model source and loading parameters.

    This class encapsulates all configuration needed to load a model
    from either in-house definitions or Hugging Face Hub.

    Attributes:
        source: Source of the model ('in-house' or 'huggingface').
        identifier: Model name (in-house) or model ID (HuggingFace).
        hf_token: Optional HuggingFace authentication token for private/gated models.
        torch_dtype: Data type for model weights ('float32', 'float16', 'bfloat16').
        revision: Specific model version/commit/tag to use.
        cache_dir: Directory to cache downloaded models.
        device_map: Device mapping strategy for model loading.
        use_auth_token: Deprecated, use hf_token instead.
        additional_kwargs: Additional keyword arguments for model loading.
    """

    source: str = 'in-house'
    identifier: str = ''
    hf_token: Optional[str] = None
    torch_dtype: str = 'float32'
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    device_map: Optional[str] = None
    use_auth_token: Optional[str] = None    # Deprecated
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and normalization."""
        # Handle deprecated use_auth_token
        if self.use_auth_token is not None and self.hf_token is None:
            self.hf_token = self.use_auth_token

        # Normalize and validate source
        self.source = self.source.lower()
        if self.source not in ['in-house', 'huggingface']:
            raise ValueError(f"Invalid model source '{self.source}'. "
                             f"Must be 'in-house' or 'huggingface'.")

        # Validate torch_dtype
        valid_dtypes = ['float32', 'float16', 'bfloat16', 'int8']
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"Invalid torch_dtype '{self.torch_dtype}'. "
                             f'Must be one of {valid_dtypes}.')

        # Validate identifier is provided
        if not self.identifier:
            raise ValueError('Model identifier must be provided.')

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration parameters.

        Returns:
            Tuple of (is_valid, error_message).
            If is_valid is True, error_message is empty.
        """
        # Check identifier is not empty for HuggingFace models
        if self.source == 'huggingface':
            if not self.identifier or not self.identifier.strip():
                return (False, 'HuggingFace model identifier cannot be empty')

        return (True, '')

    def is_huggingface(self) -> bool:
        """Check if this configuration is for a HuggingFace model.

        Returns:
            True if source is 'huggingface', False otherwise.
        """
        return self.source == 'huggingface'

    def __repr__(self) -> str:
        """String representation of the configuration."""
        token_status = 'set' if self.hf_token else 'not set'
        return (
            f"ModelSourceConfig(source='{self.source}', "
            f"identifier='{self.identifier}', "
            f"torch_dtype='{self.torch_dtype}', "
            f'hf_token={token_status})'
        )
