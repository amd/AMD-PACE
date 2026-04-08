# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

from functools import partial
from typing import Optional
import torch
from transformers import PreTrainedTokenizerBase, StopStringCriteria

from pace.llm.configs import SamplingConfig
from pace.utils.logging import PACE_LLM_ASSERT


# Stopping criteria for sampling
class StoppingCriteria(object):
    """
    Class to define stopping criteria for sampling. StoppingCriteria
    uses the SamplingConfig provided to define the stopping conditions.

    Args:
        sampling_config (SamplingConfig): The sampling configuration object.
    """

    def __init__(
        self,
        sampling_config: SamplingConfig,
        input_prompts: torch.Tensor,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):

        self.initial_decoder_input_length = input_prompts.shape[-1]
        self._min_length = (
            getattr(sampling_config, "min_new_tokens", 0) or 0
        ) + self.initial_decoder_input_length

        self._max_length = None
        self.stop_conditions = []  # To be used with conditions implemented in PACE
        self.hf_stop_conditions = []  # To be used with conditions implemented in HF
        if sampling_config.max_new_tokens is not None:
            self._max_length = (
                sampling_config.max_new_tokens + self.initial_decoder_input_length
            )
            self.stop_conditions.append(
                partial(self._stop_if_max_len, max_length=self._max_length)
            )

        if not getattr(sampling_config, "ignore_eos", False):
            if sampling_config.eos_token_id is not None:
                self.stop_conditions.append(
                    partial(
                        self._stop_if_eos_token,
                        eos_token_id=torch.as_tensor(sampling_config.eos_token_id),
                    )
                )

        if (
            sampling_config.stop_strings is not None
            and len(sampling_config.stop_strings) > 0
        ):
            if tokenizer is None:
                PACE_LLM_ASSERT(False, "Tokenizer is required for stop strings.")
            self.hf_stop_conditions.append(
                StopStringCriteria(
                    tokenizer=tokenizer, stop_strings=sampling_config.stop_strings
                )
            )

    def __str__(self):
        return (
            f"StoppingCriteria("
            f"stop_conditions={[(condition.func.__name__, condition.keywords) for condition in self.stop_conditions]}, "
            f"hf_stop_conditions={self.hf_stop_conditions})"
        )

    def __repr__(self):
        return str(self)

    def _stop_if_max_len(
        self, logits: torch.Tensor, max_length: int, **kwargs
    ) -> torch.Tensor:
        """
        Check if the number of new tokens generated is greater than max_new_tokens.

        Args:
            logits (torch.Tensor): The logits tensor.
            max_length (int): The maximum total sequence length allowed.

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the length
                of the generated sequence is greater than max_new_tokens.
        """
        cur_len = logits.shape[-1]
        is_done = cur_len >= max_length
        return torch.full((logits.shape[0],), is_done, dtype=torch.bool)

    def _stop_if_eos_token(
        self,
        logits: torch.Tensor,
        eos_token_id: torch.Tensor,
        num_new_tokens: int = 1,
    ) -> torch.Tensor:
        """
        Check if any of the newly appended tokens is an EOS token.

        ``eos_token_id`` can be a scalar or a tensor of multiple IDs;
        ``torch.isin`` handles both transparently.  With speculative
        decoding multiple tokens can be appended in a single step, so
        ``num_new_tokens`` tells the method how many trailing tokens to
        inspect.

        Args:
            logits (torch.Tensor): The full token sequence so far.
            eos_token_id (torch.Tensor): One or more EOS token ids.
            num_new_tokens (int): Number of tokens appended this step.

        Returns:
            torch.Tensor: A boolean tensor indicating whether any of the
                new tokens is an EOS token.
        """
        return torch.isin(logits[:, -num_new_tokens:], eos_token_id).any(dim=-1)

    def _check_for_conditions(
        self, logits: torch.Tensor, num_new_tokens: int = 1
    ) -> torch.Tensor:
        """
        Check if any of the stop conditions are met.

        Args:
            logits (torch.Tensor): The logits tensor.
            num_new_tokens (int): Number of tokens appended this step.

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the sampling
                should be stopped.
        """

        # Initialize is_done to False, with one value for each sample
        # in the batch. If any of the stop conditions are met, the
        # corresponding value in is_done is set to True.
        is_done = torch.full((logits.shape[0],), False, dtype=torch.bool)
        for condition in self.stop_conditions:
            is_done = is_done | condition(logits=logits, num_new_tokens=num_new_tokens)

        if self.hf_stop_conditions:
            for condition in self.hf_stop_conditions:
                is_done = is_done | condition(logits, scores=None)
        return is_done

    def stop_now(self, logits: torch.Tensor, num_new_tokens: int = 1) -> torch.Tensor:
        """
        Check if the sampling should be stopped. If any of the stop conditions
        are met, the corresponding value in the output tensor is set to True.

        Args:
            logits (torch.Tensor): The logits tensor.
            num_new_tokens (int): Number of tokens appended this step
                (defaults to 1).

        Returns:
            torch.Tensor: A tensor of boolean values indicating if the sampling
                should be stopped.
        """

        if logits.shape[-1] < self._min_length:
            if self._max_length is not None:
                return self._stop_if_max_len(logits, self._max_length)
            return torch.full((logits.shape[0],), False, dtype=torch.bool)
        return self._check_for_conditions(logits, num_new_tokens)
