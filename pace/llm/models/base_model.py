# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from abc import ABC, abstractmethod
from typing import Dict, List

from torch import nn
from transformers import PretrainedConfig

from pace.ops.base import BackendBase
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_DEBUG


class BaseModelForCausalLM(ABC, nn.Module):
    """
    Abstract base class for all models.

    All models should subclass this class and should implement
    the forward and load_weights methods.
    """

    rename_layers: Dict[str, str] = None
    target_map: Dict[str, List[str]] = None

    def __init__(self, config: PretrainedConfig):

        super().__init__()
        PACE_LLM_ASSERT(
            isinstance(config, PretrainedConfig),
            f"Config should be an instance of PretrainedConfig., got {type(config)}",
        )
        self.config = config

    def get_config(
        self,
    ) -> PretrainedConfig:
        """
        Returns the model configuration.

        Returns:
            PretrainedConfig: The model configuration.
        """
        return self.config

    @abstractmethod
    def forward(self, input_ids, positions, kv_cache):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.LongTensor): Only the new (unprocessed) token IDs.
                The caller is responsible for sending only tokens not yet in the
                KV cache. Shape: [batch_size, num_new_tokens].
            positions (torch.LongTensor): Absolute position indices for each token
                in input_ids. Computed by the caller from a num_computed_tokens
                counter. Shape: [batch_size, num_new_tokens].
            kv_cache: The key-value cache manager (KVCacheManager or list of them).

        Returns:
            ModelOutput: The model outputs containing logits.
        """
        pass

    @abstractmethod
    def load_weights(self, *args, **kwargs):
        pass

    def rename_fused_params(self, param_name: str) -> str:
        if not self.rename_layers:
            return param_name

        for old_name, new_name in self.rename_layers.items():
            if old_name in param_name:
                param_name = param_name.replace(old_name, new_name)

        return param_name

    def preprocess_weights(
        self,
    ):

        for _, module in self.named_modules():
            backend = getattr(module, "backend", None)
            if isinstance(backend, BackendBase):
                PACE_LLM_DEBUG(
                    f"Preprocessing module: {module.__class__.__name__} "
                    f"with backend: {backend.__class__.__name__}"
                )
                backend.preprocess(module)
