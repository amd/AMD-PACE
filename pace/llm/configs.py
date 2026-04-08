# *******************************************************************************
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
import math
import json
import re
import functools
import inspect
from enum import Enum
from collections import UserDict
from dataclasses import dataclass
from typing import Union, Optional, List

import torch
from transformers import PreTrainedTokenizerBase

from pace.llm.ops import LLMOperatorType, BackendType
from pace.llm.attention import AttentionBackendType, KVCacheType
from pace.utils.logging import PACE_LLM_ASSERT, PACE_LLM_WARNING, PACE_LLM_DEBUG

CACHE_COMPATIBLE_ATTENTION = {
    KVCacheType.DYNAMIC: {AttentionBackendType.JIT, AttentionBackendType.NATIVE},
    KVCacheType.BMC: {AttentionBackendType.JIT, AttentionBackendType.NATIVE},
    KVCacheType.SLAB_POOL: {AttentionBackendType.SLAB},
    KVCacheType.PAGED: {AttentionBackendType.PAGED},
}

CACHE_DEFAULT_ATTENTION = {
    KVCacheType.DYNAMIC: AttentionBackendType.JIT,
    KVCacheType.BMC: AttentionBackendType.JIT,
    KVCacheType.SLAB_POOL: AttentionBackendType.SLAB,
    KVCacheType.PAGED: AttentionBackendType.PAGED,
}

# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4/vllm/model_executor/sampling_metadata.py#L13
# to check for a small value (instead of directly checking for 0)
_SAMPLING_EPS = 1e-5


class SamplingMode(Enum):
    """
    SamplingMode is an enumeration class that contains the different sampling methods used.
    """

    GREEDY_SEARCH = 1
    RANDOM_SAMPLING = 2


# Adapted from GenerationConfig in
# https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/generation/configuration_utils.py#L71
class SamplingConfig(object):
    """
    SamplingConfig is a class that contains all the configuration parameters for the generation process.
    It is used to control the generation process.
    The user needs to create a SamplingConfig object and pass it to the genertae method. If not
    provided, the default values defined in the __init__ method will be used.

    Args:
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to None. But will be set to 20 if not provided.
        min_new_tokens (int, optional): The minimum number of tokens to generate. Defaults to 0.
        repetition_penalty(float, optional): Multiplicative penalty on tokens present in the prompt+output. >1 discourages repetition, <1 encourages it. (Matches HuggingFace/vLLM convention.)
        frequency_penalty(float, optional): Additive penalty proportional to how often a token has appeared in the generated output so far. >0 discourages repetition, <0 encourages it. (Matches OpenAI/vLLM convention.)
        do_sample (bool, optional): Whether to use sampling. Defaults to False.
        temperature (float, optional): The temperature to use for sampling. Defaults to 1.0.
        top_k (int, optional): The number of tokens to sample from. Defaults to 0.
        top_p (float, optional): The cumulative probability for sampling from the top_k tokens. Defaults to 1.0.
        min_p (float, optional): The minimum cumulative probability for sampling from the top_k tokens. Defaults to 0.0.
        seed (int, optional): The seed to use for sampling. Defaults to None.
        pad_token_id (Union[int, List[int], torch.Tensor], optional): The id of the pad token. Defaults to None.
        eos_token_id (Union[int, List[int], torch.Tensor], optional): The id of the eos token. Defaults to None.
        stop_strings (Union[str, List[str]], optional): The strings to stop generation at. Defaults to None.
        return_probs (bool, optional): Whether to return the probabilities of the generated tokens. Defaults to False.
        return_logprobs (bool, optional): Whether to return the log probabilities of the generated tokens. Defaults to False.
        return_input_logprobs (bool, optional): Whether to return the log probabilities of the input tokens. Defaults to False.
        return_text (bool, optional): Whether to return the generated text. Defaults to False.
    """

    @classmethod
    def from_pretrained(
        cls, generation_config_from_model: Union[str, os.PathLike]
    ) -> "SamplingConfig":
        """
        Create a SamplingConfig object from the generation_config.json file saved in the model directory.

        Tolerates trailing commas in JSON, which are common in HuggingFace
        model repositories but rejected by Python's strict json parser.

        Args:
            generation_config_from_model: Path to the generation_config.json file saved in the model directory.

        Returns:
            SamplingConfig: A SamplingConfig object created from the generation_config.json file.
        """
        with open(generation_config_from_model, "r", encoding="utf-8") as reader:
            text = reader.read()
        try:
            config_dict = json.loads(text)
        except json.JSONDecodeError:
            text = re.sub(r",\s*([}\]])", r"\1", text)
            config_dict = json.loads(text)
        return cls.from_dict(**config_dict)

    @classmethod
    def from_dict(cls, **kwargs) -> "SamplingConfig":
        """
        Create a SamplingConfig object from a dictionary.

        Args:
            **kwargs: A dictionary containing the configuration parameters.

        Returns:
            SamplingConfig: A SamplingConfig object created from the dictionary.
        """
        return cls(**kwargs)

    @staticmethod
    def _track_explicit_args(init_fn):
        """Decorator that records which ``__init__`` args the caller explicitly passed.

        After construction, ``self._explicit_keys`` is a frozenset of the
        parameter names that were supplied by the caller (as opposed to falling
        back to their signature defaults).
        """
        sig = inspect.signature(init_fn)

        @functools.wraps(init_fn)
        def wrapper(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            explicit = set()
            for k, v in bound.arguments.items():
                if v is None:
                    continue
                if k == "self":
                    continue
                if k == "kwargs":
                    explicit.update(v.keys())
                else:
                    explicit.add(k)
            self._explicit_keys = frozenset(explicit)
            return init_fn(self, *args, **kwargs)

        return wrapper

    @_track_explicit_args.__func__
    def __init__(
        self,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = 0,
        ignore_eos: Optional[bool] = False,
        repetition_penalty: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = 1.0,
        min_p: Optional[float] = 0.0,
        seed: Optional[int] = None,
        pad_token_id: Optional[Union[int, torch.Tensor]] = None,
        eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        stop_strings: Optional[Union[str, List[str]]] = None,
        return_probs: Optional[bool] = False,
        return_logprobs: Optional[bool] = False,
        return_input_logprobs: Optional[bool] = False,
        return_text: Optional[bool] = False,
        **kwargs,
    ):
        # Warn about removed beam search parameters that may still be
        # present in generation_config.json files or user code
        _removed_params = {"num_beams", "early_stopping", "use_beam_search"}
        _found = _removed_params & kwargs.keys()
        if _found:
            PACE_LLM_WARNING(
                f"Beam search has been removed. The following parameters "
                f"will be ignored: {_found}"
            )

        # Parameters that control the length of the output
        self.max_new_tokens: int = max_new_tokens
        self.min_new_tokens: int = min_new_tokens
        self.ignore_eos: bool = bool(ignore_eos)
        self.repetition_penalty: float = repetition_penalty
        self.frequency_penalty: float = frequency_penalty
        # Parameters that control the generation strategy used
        self.do_sample: bool = do_sample

        # Parameters for manipulation of the model output logits
        self.temperature: float = temperature
        self.top_k: int = top_k
        self.top_p: float = top_p
        self.min_p: float = min_p
        self.sampling_seed: Optional[int] = seed

        # Pad and eos token ids
        self.pad_token_id: Union[int, torch.Tensor] = pad_token_id
        self.eos_token_id: Union[int, List[int], torch.Tensor] = eos_token_id
        self.stop_strings: Union[str, List[str], None] = stop_strings

        # Return parameters
        self.return_probs: bool = return_probs
        self.return_logprobs: bool = return_logprobs
        self.return_input_logprobs: bool = return_input_logprobs
        self.return_text: bool = return_text

        # There could be multiple stop eos tokens, stop tokens, or stop texts
        # They will be stored in a list, and converted to tensors
        if self.eos_token_id is not None and not isinstance(
            self.eos_token_id, (list, torch.Tensor)
        ):
            self.eos_token_id = [self.eos_token_id]
        if self.stop_strings is not None and not isinstance(self.stop_strings, list):
            self.stop_strings = [self.stop_strings]

    def _merge_lists(self, other: Optional["SamplingConfig"], key: str):
        """
        Merge the lists of the same key from the current and other SamplingConfig objects.

        Args:
            other: Another SamplingConfig object.
            key: The key of the list to be merged.
        """
        other_attr_id = getattr(other, key, None)
        self_attr_id = getattr(self, key, None)

        if other_attr_id is not None and (self_attr_id != other_attr_id):

            if isinstance(self_attr_id, int):
                self_attr_id = [self_attr_id]
            elif isinstance(self_attr_id, torch.Tensor):
                self_attr_id = self_attr_id.tolist()
            elif self_attr_id is None:
                self_attr_id = []

            if isinstance(other_attr_id, int):
                other_attr_id = [other_attr_id]
            elif isinstance(other_attr_id, torch.Tensor):
                other_attr_id = other_attr_id.tolist()
            elif other_attr_id is None:
                other_attr_id = []

            combined = sorted(list(set(other_attr_id + self_attr_id)))
            setattr(self, key, combined)

    def merge_from(
        self,
        other: Optional["SamplingConfig"] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Merge the configuration parameters from another SamplingConfig object or a tokenizer object.

        Args:
            other: Another SamplingConfig object.
            tokenizer: A tokenizer object.
        """

        if tokenizer:
            for attr in ["pad_token_id", "eos_token_id"]:
                self._merge_lists(tokenizer, attr)

                # We will not sample top_k if top_k is 0 or it's the same as vocab size
                self.top_k = tokenizer.vocab_size if self.top_k == 0 else self.top_k
                self.vocab_size = tokenizer.vocab_size

        # The other SamplingConfig object will override the current object (takes precedence)
        if other is not None:
            explicit = getattr(other, "_explicit_keys", frozenset())
            for key, value in other.__dict__.items():
                if key.startswith("_"):
                    continue
                # For pad_token_id and eos_token_id, we will merge the values
                if key in ["pad_token_id", "eos_token_id"]:
                    self._merge_lists(other, key)
                    continue

                # Only apply values that the caller explicitly provided.
                # This lets us distinguish "user wrote do_sample=False" from
                # "do_sample just defaulted to False", so defaults from one
                # config never silently overwrite explicit values in another.
                if key in explicit:
                    setattr(self, key, value)

    def _set_sampling_method(
        self,
    ):
        """
        Set the sampling method based on the configuration parameters.

        The sampling method can be one of the following:
        - GREEDY_SEARCH
        - RANDOM_SAMPLING (default)
        """
        if (self.do_sample is False) or self.temperature == 0:
            self.sampling_mode = SamplingMode.GREEDY_SEARCH
        else:
            self.sampling_mode = SamplingMode.RANDOM_SAMPLING

    def _verify_sampling_config(
        self,
    ):
        """
        Verify the configuration parameters.

        The configuration parameters must satisfy the following constraints:
            * max_new_tokens should be an integer >= 1
            * min_new_tokens should be an integer >= 0

            * temperature should be a float >= 0
            * top_k should be an integer >= 0
            * top_p should be a float between 0 and 1
            * min_p should be a float between 0 and 1

            * if temperature is 0, then top_k should be 0, top_p should be 1.0 and min_p should be 0.0

            * eos_token_id should not be None, and should be an integer or a tensor
            * pad_token_id should be an integer or a tensor, and if None, should be set to eos_token_id
            * stop_strings should be a string or a list of strings

        Raises:
            AssertionError: If any of the constraints are not satisfied.
        """

        PACE_LLM_ASSERT(
            isinstance(self.max_new_tokens, int),
            f"max_new_tokens should be of type int, got: {type(self.max_new_tokens)}",
        )
        PACE_LLM_ASSERT(
            self.max_new_tokens >= 1,
            f"max_new_tokens should be >=1, got: {self.max_new_tokens}",
        )

        PACE_LLM_ASSERT(
            isinstance(self.min_new_tokens, int),
            f"min_new_tokens should be of type int, got: {type(self.min_new_tokens)}",
        )
        PACE_LLM_ASSERT(
            self.min_new_tokens >= 0,
            f"min_new_tokens should be >=0, got {self.min_new_tokens}",
        )

        # Temperature should be float and >= 0
        self.temperature = float(self.temperature)
        PACE_LLM_ASSERT(
            self.temperature >= 0,
            f"temperature should be >= 0, Got temperature: {self.temperature}",
        )
        # top_k should be int and >= 0
        self.top_k = int(self.top_k)
        PACE_LLM_ASSERT(
            self.top_k >= 0, f"top_k should be >= 0, Got top_k: {self.top_k}"
        )
        # top_p should be float and  between 0 and 1
        self.top_p = float(self.top_p)
        PACE_LLM_ASSERT(
            0 <= self.top_p <= 1.0,
            f"top_p should be between 0 and 1, Got top_p: {self.top_p}",
        )
        # min_p should be float and  between 0 and 1
        self.min_p = float(self.min_p)
        PACE_LLM_ASSERT(
            0 <= self.min_p <= 1.0,
            f"min_p should be between 0 and 1, Got min_p: {self.min_p}",
        )

        self.repetition_penalty = float(self.repetition_penalty)
        PACE_LLM_ASSERT(
            self.repetition_penalty > 0,
            f"repetition_penalty must be > 0 (division by zero), got: {self.repetition_penalty}",
        )
        self.frequency_penalty = float(self.frequency_penalty)
        PACE_LLM_ASSERT(
            math.isfinite(self.frequency_penalty),
            f"frequency_penalty must be finite, got: {self.frequency_penalty}",
        )

        # https://github.com/vllm-project/vllm/blob/v0.6.4/vllm/model_executor/sampling_metadata.py#L413
        # NOTE: Zero temperature means deterministic sampling
        # (i.e., greedy sampling).
        # Set the temperature to 1 to avoid division by zero, also temperature,
        # as 0 means it's deterministic sampling, so override the values
        # of top_k, top_p and min_p to 0, 1.0 and 0.0 respectively (so that, they are not done)
        if self.temperature < _SAMPLING_EPS:
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.min_p = 0.0
            self.do_sample = False

        PACE_LLM_ASSERT(self.eos_token_id is not None, "eos_token_id cannot be None")
        PACE_LLM_ASSERT(
            isinstance(self.eos_token_id, (torch.Tensor, int))
            or all(isinstance(item, int) for item in self.eos_token_id),
            f"eos_token_id should be an integer, list of integers or a tensor, got: {type(self.eos_token_id)}",
        )
        self.eos_token_id = torch.tensor(self.eos_token_id)

        if self.pad_token_id is not None:
            if isinstance(self.pad_token_id, list):
                PACE_LLM_WARNING(
                    f"pad_token_id is a list. Using the first element as pad_token_id: {self.pad_token_id[0]}."
                )
                self.pad_token_id = self.pad_token_id[0]  # use the first element

            PACE_LLM_ASSERT(
                isinstance(self.pad_token_id, (torch.Tensor, int)),
                f"pad_token_id should be an integer or a tensor, got: {type(self.pad_token_id)}",
            )
            self.pad_token_id = torch.tensor(self.pad_token_id)
        else:  # if pad_token_id is None, set eos_token_id as pad_token_id
            PACE_LLM_WARNING(
                f"pad_token_id is None. Setting pad_token_id to eos_token_id: {self.eos_token_id[0]}."
            )
            self.pad_token_id = self.eos_token_id[0]
        if self.stop_strings is not None:
            PACE_LLM_ASSERT(
                all(isinstance(item, str) for item in self.stop_strings),
                f"stop_strings should be a string or a list of strings, got: {type(self.stop_strings)}",
            )

    def __str__(self):
        return (
            "SamplingConfig("
            + ", ".join([f"{key}: {value}" for key, value in self.__dict__.items()])
            + ")"
        )

    def verify_max_new_tokens(
        self,
        initial_decoder_input_length: int = 0,
        model_max_position_embeddings: int = 2048,
    ):
        """
        Truncate the max_new_tokens parameter based on the initial decoder input length and model max position embeddings.
        max_new_tokens should be less than or equal to model_max_position_embeddings - initial_decoder_input_length.

        Args:
            initial_decoder_input_length (int): The initial decoder input length.
            model_max_position_embeddings (int): The model max position embeddings.

        Returns:
            int: The truncated max_new_tokens value.
        """
        if self.max_new_tokens is None:
            PACE_LLM_WARNING("max_new_tokens is not set. Setting max_new_tokens to 20.")
            self.max_new_tokens = 20
        max_tokens = initial_decoder_input_length + self.max_new_tokens
        if max_tokens > model_max_position_embeddings:
            PACE_LLM_WARNING(
                f"Max tokens {max_tokens} is greater than the model's max position "
                f"embeddings {model_max_position_embeddings}, "
                f"truncating to {model_max_position_embeddings}"
            )
            self.max_new_tokens = max(
                1, model_max_position_embeddings - initial_decoder_input_length
            )  # Ensure at least 1 token is generated

    def repr(self):
        return str(self)

    def finalize(
        self,
    ):
        """
        Finalize the SamplingConfig object. Once finalized, the configuration parameters cannot be modified.

        Raises:
            ValueError: If the SamplingConfig object is already finalized.
        """

        self._verify_sampling_config()
        self._set_sampling_method()
        self._finalized = True

    def __setattr__(self, name, value):
        if hasattr(self, "_finalized") and self._finalized and name != "_finalized":
            PACE_LLM_ASSERT(
                False,
                "This SamplingConfig instance is finalized and cannot be modified.",
            )
        super().__setattr__(name, value)


# Holds a config for operator to backend mapping like a dictionary
# e.g. {LLMOperatorType: BackendType}
class OperatorConfig(UserDict):
    """
    OperatorConfig is a class that holds the configuration for LLMOperatorType or str to BackendType mapping.
    It is used to control the operator to backend mapping for the LLM.
    The user needs to create an OperatorConfig object and pass it to the LLM. If not provided, the default values
    will be used.

    Args:
        **kwargs: A dictionary containing the operator to backend mapping.
    """

    def __init__(self, **kwargs):
        """
        Initialize the OperatorConfig object.

        Args:
            **kwargs: A dictionary containing the operator to backend mapping.
        """
        super().__init__(**kwargs)
        self._finalized = False

    def finalize(self, cache_type=None):
        """
        Finalize the OperatorConfig object. Once finalized, the configuration parameters cannot be modified.

        Args:
            cache_type: Optional KVCacheType. If provided, validates that the
                attention backend is compatible with the cache type and overrides
                with a warning if not.
        """

        if hasattr(self, "_finalized") and self._finalized:
            PACE_LLM_ASSERT(
                False,
                "This OperatorConfig instance is finalized and cannot be modified.",
            )

        for key in self.keys():
            if not isinstance(key, (LLMOperatorType, str)):
                PACE_LLM_ASSERT(
                    False,
                    f"{key} is not a valid LLMOperatorType. "
                    f"Valid keys are: {list(LLMOperatorType)} or str",
                )
            if key == LLMOperatorType.Attention:
                if not isinstance(self[key], AttentionBackendType):
                    PACE_LLM_ASSERT(
                        False,
                        f"{self[key]} is not a valid AttentionBackendType. "
                        f"Valid values are: {list(AttentionBackendType)}",
                    )
            elif not isinstance(self[key], BackendType):
                PACE_LLM_ASSERT(
                    False,
                    f"{self[key]} is not a valid BackendType. "
                    f"Valid values are: {list(BackendType)}",
                )

        for key in LLMOperatorType:
            if key not in self.keys():
                if key == LLMOperatorType.Attention:
                    self[key] = AttentionBackendType.JIT
                else:
                    self[key] = BackendType.NATIVE
                PACE_LLM_DEBUG(
                    f"{key} is not set in the OperatorConfig. "
                    f"Defaulting to {self[key]}."
                )

        if cache_type is not None:
            user_attn = self.get(LLMOperatorType.Attention)
            compatible = CACHE_COMPATIBLE_ATTENTION.get(cache_type, set())
            if compatible and user_attn not in compatible:
                forced = CACHE_DEFAULT_ATTENTION[cache_type]
                PACE_LLM_WARNING(
                    f"Attention backend '{user_attn}' is incompatible with "
                    f"cache type '{cache_type.value}'. Overriding to '{forced}'."
                )
                self[LLMOperatorType.Attention] = forced

        self.qkv_projection = self.pop(
            LLMOperatorType.QKVProjection, BackendType.NATIVE
        )
        self.rope = self.pop(LLMOperatorType.RoPE, BackendType.NATIVE)
        self.attention = self.pop(LLMOperatorType.Attention, AttentionBackendType.JIT)
        self.out_projection = self.pop(
            LLMOperatorType.OutProjection, BackendType.NATIVE
        )
        self.mlp = self.pop(LLMOperatorType.MLP, BackendType.NATIVE)
        self.norm = self.pop(LLMOperatorType.Norm, BackendType.NATIVE)
        self.lm_head = self.pop(LLMOperatorType.LMHead, BackendType.NATIVE)

        # These are the extra keys that might not be in the LLMOperatorType
        # but are still valid keys, so make sure they are valid
        for key in self.keys():
            PACE_LLM_ASSERT(
                self[key] is not None,
                f"Key {key} is not a valid BackendType. "
                f"Valid values are: {list(BackendType)}",
            )

        self._finalized = True
        return self

    def __setitem__(self, key: LLMOperatorType, value: BackendType):
        if hasattr(self, "_finalized") and self._finalized:
            PACE_LLM_ASSERT(
                False,
                "This OperatorConfig instance is finalized and cannot be modified.",
            )
        super().__setitem__(key, value)


@dataclass
class SpecDecodeConfig:
    """Base configuration class for speculative decoding algorithms.

    All speculative decoding configs should inherit from this class so that
    ``Generator`` and ``LLMModel`` can accept any algorithm-specific config
    through a single ``spec_config`` parameter.
    """

    pass


@dataclass
class PardSpecDecodeConfig(SpecDecodeConfig):

    model_name_or_path: str
    pard_token: Optional[torch.Tensor] = None
    num_speculative_tokens: int = 12
    draft_kv_cache_memory_gb: Optional[float] = None

    def merge_and_verify(
        self,
        model_pard_token: Optional[Union[int, torch.Tensor]] = None,
    ):
        self.pard_token = (
            model_pard_token if model_pard_token is not None else self.pard_token
        )  # Override with config pard token
        PACE_LLM_ASSERT(
            self.pard_token is not None,
            "PARD token is not set. Please set the pard_token in the PARD config.",
        )
        self.pard_token = torch.tensor(self.pard_token)

        PACE_LLM_ASSERT(
            isinstance(self.num_speculative_tokens, int)
            and self.num_speculative_tokens > 0,
            f"num_speculative_tokens should be a positive integer, got: {self.num_speculative_tokens} "
            f"of type {type(self.num_speculative_tokens)}",
        )

    def __str__(self):
        return (
            "PardSpecDecodeConfig("
            + ", ".join(
                [
                    f"{key}: {value}"
                    for key, value in self.__dict__.items()
                    if value is not None
                ]
            )
            + ")"
        )
