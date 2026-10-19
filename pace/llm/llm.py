# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
from typing import Union, Optional

import torch
from transformers import (
    PreTrainedTokenizer,
    PretrainedConfig,
    BatchEncoding,
    TextStreamer,
)

from pace.llm.attention import KVCacheType
from pace.llm.attention.eviction import EvictionConfig
from pace.llm.generation import GenerationBackendConfig, NativeGenerationBackendConfig
from pace.llm.outputs import GeneratorOutput
from pace.llm.configs import SamplingConfig, OperatorConfig, SpecDecodeConfig


class LLMModel(object):
    """
    LLMModel is a class that wraps a language model and provides a generate method to generate text from a prompt.
    The class is designed to be used in inference mode, which means that the model is not trainable and the forward pass
    is optimized for inference.

    This is the front-facing class that users should use to generate text from a prompt.
    The model expects a prompt which is tokenized by the tokenizer and then passed to the model to generate text.

    Args:
        model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
            The model identifier or the path to the directory where the pre-trained
            model is stored.
        tokenizer_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
            The tokenizer identifier or the path to the directory where the pre-trained
            tokenizer is stored. If not provided, the model will use the default tokenizer in the model.
        dtype (:obj:`torch.dtype`, `optional`, defaults to :obj:`torch.bfloat16`):
            The data type to use for the model. The model will be loaded in this dtype.
            If not provided, the model will be loaded in torch.bfloat16.
        eviction_config (:obj:`EvictionConfig`, `optional`):
            Enables StreamingLLM / EQUIP KV eviction on contiguous caches
            (``KVCacheType.DYNAMIC`` / ``BMC``). Decode keeps at most
            ``sink_size + recent_size`` slots.

    Examples:
        >>> model_name = "meta-llama/Llama-3.1-8B"
        >>> prompt = "Once upon a time"
        >>> prompt_encoded = tokenizer([prompt], return_tensors="pt")
        >>> from pace.llm import LLMModel, SamplingConfig
        >>> model = LLMModel(model_name)
        >>> sampling_config = SamplingConfig(max_length=50, do_sample=True, temperature=0.7)
        >>> output = model.generate(prompt_encoded, sampling_config)
        >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_type: Optional[KVCacheType] = None,
        spec_config: Optional[SpecDecodeConfig] = None,
        opconfig: Optional[OperatorConfig] = None,
        generation_backend_config: Optional[GenerationBackendConfig] = None,
        disable_tqdm: bool = False,
        cache_kwargs: Optional[dict] = None,
        eviction_config: Optional[EvictionConfig] = None,
    ):
        config = generation_backend_config
        if config is None:
            config = NativeGenerationBackendConfig()
        backend_kwargs = {
            "serving": False,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "dtype": dtype,
            "disable_tqdm": disable_tqdm,
            "kv_cache_type": kv_cache_type
            or (KVCacheType.PAGED if dtype == torch.float32 else KVCacheType.SLAB_POOL),
            "spec_config": spec_config,
            "opconfig": opconfig,
            "cache_kwargs": cache_kwargs,
            "eviction_config": eviction_config,
        }
        self._backend = config.initialize_backend(model_name_or_path, **backend_kwargs)

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[torch.Tensor, BatchEncoding],
        sampling_config: Optional[SamplingConfig] = None,
        text_streamer: Optional[TextStreamer] = None,
    ) -> GeneratorOutput:
        """
        Generate text from a prompt through the configured backend.
        """

        return self._backend.generate(
            prompt,
            sampling_config,
            text_streamer=text_streamer,
        )

    def __repr__(
        self,
    ):
        """Return the underlying model's string representation."""
        return self._backend.model.__repr__()

    @property
    def model(self):
        return self._backend.model

    def get_tokenizer(
        self,
    ) -> PreTrainedTokenizer:
        """
        Returns the tokenizer

        Returns:
            PreTrainedTokenizer: The tokenizer
        """
        return self._backend.tokenizer

    def get_config(self) -> PretrainedConfig:
        """
        Returns the model config

        Returns:
            PretrainedConfig: The model config
        """
        return self._backend.model_config
