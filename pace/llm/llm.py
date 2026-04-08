# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
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
from pace.llm.generator import Generator
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

    Examples:
        >>> model_name = "meta-llama/Llama-3.1-8B"
        >>> prompt = "Once upon a time"
        >>> prompt_encoded = tokenizer.batch_encode_plus([prompt], return_tensors="pt")
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
        kv_cache_type: Optional[KVCacheType] = KVCacheType.DYNAMIC,
        spec_config: Optional[SpecDecodeConfig] = None,
        opconfig: Optional[OperatorConfig] = None,
        disable_tqdm: bool = False,
    ):

        self.generator = Generator(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dtype=dtype,
            kv_cache_type=kv_cache_type,
            spec_config=spec_config,
            opconfig=opconfig,
            disable_tqdm=disable_tqdm,
        )

    @classmethod
    def for_serving(
        cls,
        model_name_or_path: Union[str, os.PathLike],
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_type: Optional[KVCacheType] = KVCacheType.DYNAMIC,
        opconfig: Optional[OperatorConfig] = None,
    ):
        """
        A class method to create an LLMModel instance for serving.
        This is a convenience method that sets some default parameters for serving.
        """
        return cls(
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            opconfig=opconfig,
            disable_tqdm=True,
            kv_cache_type=kv_cache_type,
        )

    @torch.inference_mode()
    def step(self, *args, **kwargs):
        """
        Perform a single generation step. This is a wrapper around the generator's step method.
        This is an advanced feature for users who want to control the generation process step-by-step.
        """
        return self.generator.step(*args, **kwargs)

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[torch.Tensor, BatchEncoding],
        sampling_config: Optional[SamplingConfig] = None,
        text_streamer: Optional[TextStreamer] = None,
    ) -> GeneratorOutput:
        """
        Generate text from a prompt. generate makes use of the generator to generate text from a prompt.
        """

        inputs = self.generator.prepare_for_generate(
            prompt, sampling_config, text_streamer
        )
        outputs = self.generator.generate(inputs)
        return outputs

    def __repr__(
        self,
    ):
        return self.generator.model.__repr__()

    def get_tokenizer(
        self,
    ) -> PreTrainedTokenizer:
        """
        Returns the tokenizer

        Returns:
            PreTrainedTokenizer: The tokenizer
        """
        return self.generator.get_tokenizer()

    def get_config(self) -> PretrainedConfig:
        """
        Returns the model config

        Returns:
            PretrainedConfig: The model config
        """
        return self.generator.get_config()
