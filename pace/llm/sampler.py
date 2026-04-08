# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

from functools import partial
from typing import Optional

import torch
import math

from pace.llm.configs import SamplingConfig, SamplingMode
from pace.llm.outputs import SamplerOutput
from pace.utils.logging import PACE_LLM_ASSERT


class Sampler(object):
    """
    Sampler class to sample from the model's output logits. The sampler can be configured to use
    different sampling strategies like Greedy Search, Random Sampling, Top-k, Top-p, Min-p sampling.

    Args:
        sampling_config: SamplingConfig object with sampling configuration.
    """

    def __init__(
        self,
        sampling_config: SamplingConfig,
        input_encodings: torch.Tensor,
    ) -> None:
        self.sampling_config = sampling_config
        self._set_sampling_seed(sampling_config.sampling_seed)
        self.repetition_penalty = self.sampling_config.repetition_penalty
        self.frequency_penalty = self.sampling_config.frequency_penalty
        self.initial_input_length = input_encodings.shape[-1]
        self.sampler_preprocessors = []
        if sampling_config.top_k != 0:
            self.sampler_preprocessors.append(
                partial(Sampler._apply_top_k, top_k=sampling_config.top_k)
            )
        if sampling_config.top_p < 1.0:
            self.sampler_preprocessors.append(
                partial(Sampler._apply_tok_p, top_p=sampling_config.top_p)
            )
        if sampling_config.min_p > 0:
            self.sampler_preprocessors.append(
                partial(Sampler._apply_min_p, min_p=sampling_config.min_p)
            )

        self.do_sampler_preprocessors = False
        if len(self.sampler_preprocessors) > 0:
            self.do_sampler_preprocessors = True

        # Pre-decide greedy fast path (direct argmax, no preprocessing).
        self._greedy_fast_path = (
            sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH
            and self.repetition_penalty == 1.0
            and self.frequency_penalty == 0.0
            and getattr(sampling_config, "min_new_tokens", 0) == 0
            and not getattr(sampling_config, "return_probs", False)
            and not getattr(sampling_config, "return_logprobs", False)
        )

    def _set_sampling_seed(self, sampling_seed: Optional[int] = None) -> None:
        """
        Set the seed for sampling operations. This is useful for reproducibility.

        Args:
            sampling_seed: Seed for sampling operations
        """
        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)

    # Top-k, Top-p, Min-p samplers adapted from Huggingface's transformers library
    # https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/generation/logits_process.py
    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits_processed = logits.masked_fill(
            indices_to_remove, torch.finfo(logits.dtype).min
        )
        return logits_processed

    @staticmethod
    def _apply_tok_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -1:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits_processed = logits.masked_fill(
            indices_to_remove, torch.finfo(logits.dtype).min
        )
        return logits_processed

    @staticmethod
    def _apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(logits, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(
            tokens_to_remove, dim=-1, index=sorted_indices
        )
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :1] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits_processed = logits.masked_fill(
            indices_to_remove, torch.finfo(logits.dtype).min
        )
        return logits_processed

    def apply_repetition_penalty(
        self, logits: torch.Tensor, input_encodings: torch.Tensor
    ) -> torch.Tensor:
        """Multiplicative penalty on tokens already in prompt+output (vLLM/HF convention).

        penalty > 1.0 discourages repeated tokens; penalty < 1.0 encourages them.
        Positive logits are divided by the penalty; negative logits are multiplied.
        """
        logit = torch.gather(logits, 1, input_encodings)
        logit = torch.where(
            logit > 0, logit / self.repetition_penalty, logit * self.repetition_penalty
        )
        logits_processed = logits.scatter(1, input_encodings, logit)
        return logits_processed

    def apply_frequency_penalty(
        self, logits: torch.Tensor, input_encodings: torch.Tensor
    ) -> torch.Tensor:
        """Additive penalty proportional to token frequency in generated output only (OpenAI/vLLM convention).

        penalty > 0 discourages frequent tokens; penalty < 0 encourages them.
        Formula: logits[token] -= frequency_penalty * count(token in output)
        """
        output_tokens = input_encodings[:, self.initial_input_length :]
        if output_tokens.shape[-1] == 0:
            return logits
        bin_counts = torch.zeros_like(logits)
        bin_counts.scatter_add_(
            1, output_tokens, torch.ones_like(output_tokens, dtype=logits.dtype)
        )
        logits = logits - self.frequency_penalty * bin_counts
        return logits

    def set_min_new_tokens(self, logits: torch.Tensor, input_encodings) -> torch.Tensor:
        """
        enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
        Args:
            logits: Model's output logits
            input_encodings: Encoded input representations.
        Returns:
            logits: Processed logits with EOS token probability set to 0
        """
        eos_token_id = self.sampling_config.eos_token_id
        vocab_tensor = torch.arange(logits.shape[-1], device=logits.device)
        eos_token_mask = torch.isin(vocab_tensor, eos_token_id)
        scores_processed = logits.clone()
        if input_encodings.shape[-1] < (
            self.initial_input_length + self.sampling_config.min_new_tokens
        ):
            scores_processed = torch.where(eos_token_mask, -math.inf, logits)
        return scores_processed

    def sample(
        self,
        input_encodings: torch.Tensor,
        logits: torch.Tensor,
    ) -> SamplerOutput:
        """
        Sample the next token from the model's output logits using the configured sampling strategy.

        Args:
            input_encodings: Encoded input representations.
            logits: Model's output logits

        Returns:
            SamplerOutput object with the sampled token, probabilities, and log probabilities
        """

        PACE_LLM_ASSERT(
            logits is not None and not torch.isnan(logits).any(),
            "Invalid logits provided for sampling, something went wrong!",
        )

        # Greedy fast path: direct argmax, skip softmax/topk/temperature.
        # Eligibility pre-decided at __init__ (no per-call overhead).
        if self._greedy_fast_path:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            return SamplerOutput(next_tokens, None, None)

        if self.sampling_config.min_new_tokens > 0:
            logits = self.set_min_new_tokens(logits, input_encodings)

        # Penalties expect 2D [batch, vocab]. For speculative decode logits are
        # 3D [batch, num_candidates, vocab] — flatten, apply, then restore.
        needs_reshape = logits.dim() == 3
        if needs_reshape:
            batch, num_cand, vocab = logits.shape
            logits = logits.reshape(batch * num_cand, vocab)
            input_encodings = input_encodings.repeat_interleave(num_cand, dim=0)

        if self.repetition_penalty != 1.0:
            logits = self.apply_repetition_penalty(logits, input_encodings)
        if self.frequency_penalty != 0.0:
            logits = self.apply_frequency_penalty(logits, input_encodings)

        if needs_reshape:
            logits = logits.reshape(batch, num_cand, vocab)

        # Apply temperature scaling.
        if self.sampling_config.temperature != 1.0:
            logits.div_(self.sampling_config.temperature)

        # Apply top_k, top_p, min_p
        preprocessed_logits = logits
        if self.do_sampler_preprocessors:
            for sampler_preprocessor in self.sampler_preprocessors:
                preprocessed_logits = sampler_preprocessor(logits=preprocessed_logits)

        # Compute the probabilities.
        probs = torch.softmax(preprocessed_logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(preprocessed_logits, dim=-1, dtype=torch.float)

        next_tokens = None
        if self.sampling_config.sampling_mode == SamplingMode.GREEDY_SEARCH:
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)
        else:  # By default, use random sampling
            next_tokens = torch.multinomial(probs, num_samples=1)

        # Make sure the next token is valid.
        PACE_LLM_ASSERT(
            next_tokens is not None or torch.isnan(next_tokens).any(),
            "Invalid next token sampled, something went wrong!",
        )

        return SamplerOutput(next_tokens, probs, logprobs)
