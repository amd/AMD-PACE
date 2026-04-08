# *******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
# reserved. Notified per clause 4(b) of the license.
# Portions of this file consist of AI-generated content
# *******************************************************************************

import os
from typing import Tuple, Union, Optional

import torch
from transformers import (
    PreTrainedTokenizer,
    PretrainedConfig,
    BatchEncoding,
    TextStreamer,
)
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME

from pace.llm.sampler import Sampler
from pace.llm.configs import (
    SamplingConfig,
    OperatorConfig,
    SpecDecodeConfig,
)
from pace.llm.attention import KVCacheType, KVCacheManager, create_cache
from pace.llm.attention.paged.utils import (
    create_paged_kv_cache_manager,
    build_paged_attention_metadata,
)
from pace.llm.stopping_criteria import StoppingCriteria
from pace.llm.models.hf_utils import resolve_model_path
from pace.llm.models.model_utils import init_model, get_tokenizer
from pace.llm.outputs import (
    ModelOutput,
    SamplerOutput,
    GeneratorOutput,
)
from pace.llm.speculative import create_speculative_decoder
from pace.utils.logging import PACE_LLM_DEBUG, PACE_LLM_INFO, PACE_LLM_ASSERT


def validate_generator_inputs(
    model_path: Union[str, os.PathLike],
    tokenizer_path: Optional[Union[str, os.PathLike]] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
) -> None:
    """
    Validates the inputs for the Generator class

    Args:
        model_path (Union[str, os.PathLike]): Path to the model
        tokenizer_path (Optional[Union[str, os.PathLike]]): Path to the tokenizer
        dtype (Optional[torch.dtype]): Data type for the model

    Raises:
        FileNotFoundError: If the model or tokenizer path does not exist
        TypeError: If the dtype is not a torch.dtype
    """

    def validate_path(path: os.PathLike, error_message: str):
        if not os.path.exists(path):
            raise FileNotFoundError(error_message)

    validate_path(
        model_path,
        f"The model path provided does not exist. Please receck path: {model_path}",
    )

    config_path = os.path.join(model_path, CONFIG_NAME)
    validate_path(
        config_path,
        f"The model path provided does not have {CONFIG_NAME}. Please recheck path: , {model_path}",
    )

    if tokenizer_path is not None:
        validate_path(
            tokenizer_path,
            f"The tokenizer path provided does not exist. Please recheck path: {tokenizer_path}",
        )

    if not (isinstance(dtype, torch.dtype)):
        raise TypeError(
            f"Generator input dtype should be a torch.dtype, got {type(dtype)}"
        )


class Generator(object):
    """
    A class to generate text from a given model. This is the backend to
    the LLMModel class and is to be used internally. The class is initialized
    with a model, an optional tokenizer path, and an optional data type.

    Tokenizer is requried for the model to work. If the tokenizer is not provided,
    the model will try to load the tokenizer from the model path. If the tokenizer
    is not present in the model path, an error will be raised.

    Args:
        model_name_or_path (Union[str, os.PathLike]): Path to the model or the model name
        tokenizer_name_or_path (Optional[Union[str, os.PathLike]]):Path to the tokenizer or tokenizer name, if any
        dtype (Optional[torch.dtype]): Data type for the model, defaults to torch.bfloat16

    Raises:
        FileNotFoundError: If the model or tokenizer path does not exist
        TypeError: If the dtype is not a torch.dtype
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        tokenizer_name_or_path: Optional[Union[str, os.PathLike]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_type: Optional[KVCacheType] = KVCacheType.DYNAMIC,
        spec_config: Optional[SpecDecodeConfig] = None,
        opconfig: Optional[OperatorConfig] = None,
        disable_tqdm: Optional[bool] = False,
    ):

        self.model_path = resolve_model_path(model_name_or_path)
        self.tokenizer_path = None
        if tokenizer_name_or_path is not None:
            self.tokenizer_path = resolve_model_path(tokenizer_name_or_path)
        validate_generator_inputs(self.model_path, self.tokenizer_path, dtype)

        opconfig = (
            opconfig.finalize(cache_type=kv_cache_type)
            if opconfig is not None
            else OperatorConfig().finalize(cache_type=kv_cache_type)
        )
        self.model = init_model(
            self.model_path, dtype=dtype, opconfig=opconfig, disable_tqdm=disable_tqdm
        )
        self.tokenizer = get_tokenizer(self.model_path, self.tokenizer_path)

        self.spec_decoder = None
        if spec_config is not None:
            self.spec_decoder = create_speculative_decoder(
                spec_config, dtype=dtype, opconfig=opconfig, cache_type=kv_cache_type
            )

        self.kv_cache_type = kv_cache_type
        if kv_cache_type != KVCacheType.PAGED:
            self._cache = create_cache(kv_cache_type, model_config=self.model.config)
        else:
            self._cache = None

    def _prepare_inputs(
        self, prompts: Union[torch.Tensor, BatchEncoding]
    ) -> torch.Tensor:
        """
        Prepares the inputs for the model. If the inputs are a BatchEncoding, extracts the input_ids
        and returns it. If the inputs are a tensor, returns the tensor as is.

        Args:
            prompts (Union[torch.Tensor, BatchEncoding]): The input prompts

        Returns:
            torch.Tensor: The input prompts
        """

        if isinstance(prompts, BatchEncoding):
            input_prompts = prompts.input_ids
        else:
            input_prompts = prompts

        return input_prompts

    def _prepare_sampling_config(
        self,  # type: ignore
        user_sampling_config: Optional[SamplingConfig] = None,
        initial_decoder_input_length: Optional[int] = 0,
        model_max_new_tokens: Optional[int] = 2048,
    ) -> SamplingConfig:
        """
        Prepares the sampler for the model. If the user sampling config is present, merges it with the model's
        sampling config. If the user sampling config is not present, uses the model's sampling config.

        Args:
            user_sampling_config (Optional[SamplingConfig]): The user sampling config
            initial_decoder_input_length: Original length of inputs.
            model_max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            SamplingConfig: The sampling config
        """

        # Create an empty sampling config
        sampling_config = SamplingConfig()

        # If the generation config is present in the model, load it
        generation_config_from_model = os.path.join(
            self.model_path, GENERATION_CONFIG_NAME
        )
        if os.path.exists(generation_config_from_model):
            sampling_config = SamplingConfig.from_pretrained(
                generation_config_from_model
            )

        # Merge the user sampling config with the model's
        sampling_config.merge_from(user_sampling_config, self.tokenizer)
        sampling_config.verify_max_new_tokens(
            initial_decoder_input_length, model_max_new_tokens
        )
        sampling_config.finalize()

        return sampling_config

    def _prepare_streamer(
        self, text_streamer: TextStreamer, input_prompts: torch.Tensor
    ) -> Optional[TextStreamer]:

        if text_streamer:
            if isinstance(text_streamer, TextStreamer):
                PACE_LLM_ASSERT(
                    input_prompts.shape[0] == 1,
                    "Text streamer is only supported for batch size of 1",
                )
            return text_streamer
        return None

    def _update_probs_logprobs(
        self,
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampler_output: SamplerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the probs and logprobs based on the sampler output.

        Args:
            probs (torch.Tensor): The probabilities
            logprobs (torch.Tensor): The log probabilities
            sampler_output (SamplerOutput): The sampler output

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated probabilities and log probabilities
        """

        if self.sampling_config.return_probs:
            if probs is None:
                probs = sampler_output.probs
            else:
                probs = torch.cat([probs, sampler_output.probs], dim=-1)

        if self.sampling_config.return_logprobs:
            if logprobs is None:
                logprobs = sampler_output.logprobs
            else:
                logprobs = torch.cat([logprobs, sampler_output.logprobs], dim=-1)

        return probs, logprobs

    def _prepare_output_for_generate(
        self,
        output_token_ids: torch.Tensor,
        input_logprobs: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logprobs: Optional[torch.Tensor] = None,
        decoded_text: Optional[str] = None,
    ) -> GeneratorOutput:
        """
        Prepares the output for the generate method in the GeneratorOutput format

        Args:
            output_token_ids (torch.Tensor): The output token ids
            logprobs (torch.Tensor): The log probabilities
            input_logprobs (torch.Tensor): The input log probabilities
            decoded_text (Optional[str]): The decoded text

        Returns:
            GeneratorOutput: The generated outputs
        """

        decoded_text = None
        if self.sampling_config.return_text:
            decoded_text = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in output_token_ids
            ]

        speculative_stats = (
            self.spec_decoder.get_stats() if self.spec_decoder is not None else None
        )

        output = GeneratorOutput(
            output_token_ids=output_token_ids,
            probs=probs,
            logprobs=logprobs,
            input_logprobs=input_logprobs,
            decoded_text=decoded_text,
            speculative_stats=speculative_stats,
        )
        return output

    def _prepare_kv_cache(
        self, kv_cache_type: KVCacheType, input_prompts: torch.Tensor
    ):
        max_seq_length = self.sampling_config.max_new_tokens + input_prompts.size(-1)

        self.use_paged_attention = kv_cache_type == KVCacheType.PAGED
        self.block_size = getattr(self.model.config, "block_size", 16)

        if self.use_paged_attention:
            self.dtype = next(self.model.parameters()).dtype
            batch_size = input_prompts.size(0)
            cache_length = max_seq_length
            if self.spec_decoder is not None:
                cache_length += self.spec_decoder.num_speculative_tokens
            self.kv_cache_manager = create_paged_kv_cache_manager(
                self.model.config,
                cache_length,
                self.block_size,
                self.dtype,
                batch_size=batch_size,
            )
        else:
            batch_size = input_prompts.shape[0]
            spec_headroom = 0
            if self.spec_decoder is not None:
                spec_headroom = self.spec_decoder.num_speculative_tokens + 1
            self.kv_cache_manager = self._cache.create_context(
                self.model.config,
                max_seq_length,
                batch_size=batch_size,
                spec_headroom=spec_headroom,
            )

    def _compute_positions(
        self, num_tokens: int, num_computed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute positions from num_computed_tokens counter.
        Returns tensor of shape [batch_size, num_tokens].
        """
        if num_computed is None:
            num_computed = self.num_computed_tokens
        offsets = torch.arange(num_tokens, dtype=torch.long)
        return num_computed.unsqueeze(1) + offsets.unsqueeze(0)

    def prepare_for_generate(
        self,
        prompts: Union[torch.Tensor, BatchEncoding],
        sampling_config: Optional[SamplingConfig] = None,
        text_streamer: Optional[TextStreamer] = None,
    ) -> torch.Tensor:
        """
        Prepares the inputs for the model, prepares the sampler.

        Args:
            prompts (Union[torch.Tensor, BatchEncoding]): The input prompts
            sampling_config (Optional[SamplingConfig]): The sampling config

        Returns:
            torch.Tensor: The input prompts

        Raises:
            NotImplementedError: If the input prompts or sampling config is not a torch.Tensor or BatchEncoding
            NotImplementedError: If the sampling config is not a SamplingConfig
        """

        if not isinstance(prompts, (torch.Tensor, BatchEncoding)):
            raise NotImplementedError(
                f"Only input types of torch.Tensor or BatchEncoding is allowed for now, got {type(prompts)}"
            )

        if sampling_config is not None and not isinstance(
            sampling_config, SamplingConfig
        ):
            raise NotImplementedError(
                f"Only input types of SamplingConfig is allowed for now, got {type(sampling_config)}"
            )

        # Converts everything into tensors
        input_prompts = self._prepare_inputs(prompts)
        initial_decoder_input_length = input_prompts.shape[-1]

        # Prepare configs and instantiate the sampler
        self.sampling_config = self._prepare_sampling_config(
            sampling_config,
            initial_decoder_input_length,
            self.model.config.max_position_embeddings,
        )
        self.sampler = Sampler(self.sampling_config, input_prompts)

        # Initialize position tracking
        batch_size = input_prompts.shape[0]
        seq_len = input_prompts.shape[-1]
        self.num_computed_tokens = torch.zeros(batch_size, dtype=torch.long)

        # Detect padding and compute actual sequence lengths.
        # Priority: (1) BatchEncoding.attention_mask from the tokenizer
        # (always correct, even when pad_token == eos_token),
        # (2) leading-pad detection via cumprod (handles pad == eos by
        # counting only left-padding, not trailing EOS in content),
        # (3) assume no padding.
        pad_token_id = self.sampling_config.pad_token_id
        if isinstance(prompts, BatchEncoding) and hasattr(prompts, "attention_mask"):
            attn_mask = prompts.attention_mask
            PACE_LLM_ASSERT(
                not (attn_mask[:, -1] == 0).any(),
                "Right-padded inputs are not supported. "
                "Set tokenizer.padding_side = 'left'.",
            )
            actual_lengths = attn_mask.sum(-1).long()
        elif pad_token_id is not None and input_prompts[:, 0].eq(pad_token_id).any():
            leading_pads = (
                input_prompts.eq(pad_token_id).long().cumprod(dim=-1).sum(dim=-1)
            )
            actual_lengths = seq_len - leading_pads
        else:
            actual_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        self.actual_lengths = actual_lengths

        # Build initial positions for prefill (handles left-padding)
        pad_lens = seq_len - actual_lengths
        base_positions = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        self.initial_positions = torch.clamp(
            base_positions - pad_lens.unsqueeze(1), min=0
        )

        self._prepare_kv_cache(self.kv_cache_type, input_prompts)

        # Creates stopping config
        self.stopping_criteria = StoppingCriteria(
            self.sampling_config, input_prompts, self.tokenizer
        )

        self.text_streamer = self._prepare_streamer(text_streamer, input_prompts)

        # If a speculative decoder is configured, prepare it
        if self.spec_decoder is not None:
            self.spec_decoder.prepare(
                input_prompts, self.sampling_config, self.kv_cache_type
            )

        PACE_LLM_INFO(str(self.sampling_config))
        PACE_LLM_INFO(str(self.stopping_criteria))

        return input_prompts

    def step(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        kv_cache_mgr: Union[KVCacheManager, list[KVCacheManager]],
        paged_attn_metadata=None,
    ) -> ModelOutput:
        """Performs a single step of inference with the model."""
        pa_kwargs = (
            {"paged_attn_metadata": paged_attn_metadata}
            if paged_attn_metadata is not None
            else {}
        )
        model_out: ModelOutput = self.model(
            inputs,
            positions,
            kv_cache=kv_cache_mgr,
            **pa_kwargs,
        )
        return model_out

    def _pack_ragged(
        self,
        inputs: torch.Tensor,
        per_seq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length sequences into a single contiguous tensor.

        Removes left-padding by extracting only the real tokens from each
        sequence, concatenating them into a flat [1, total_tokens] tensor.
        """
        packed_tokens = []
        packed_positions = []
        for i in range(inputs.shape[0]):
            seq_len_i = int(per_seq_lengths[i].item())
            if seq_len_i <= 0:
                continue
            packed_tokens.append(inputs[i, -seq_len_i:])
            packed_positions.append(
                torch.arange(seq_len_i, dtype=torch.long, device=inputs.device)
            )
        if not packed_tokens:
            raise ValueError("Cannot pack: all sequences have zero length")
        return (
            torch.cat(packed_tokens).unsqueeze(0),
            torch.cat(packed_positions).unsqueeze(0),
        )

    def _prefill_forward(
        self, inputs: torch.Tensor
    ) -> tuple[ModelOutput, torch.Tensor]:
        """Runs the prefill step."""
        actual_lengths = self.actual_lengths

        if self.spec_decoder is not None:
            spec_output = self.spec_decoder.speculate(inputs)
            speculated_inputs = spec_output.extended_input
        else:
            speculated_inputs = inputs

        positions = self.initial_positions
        if speculated_inputs.shape[-1] > inputs.shape[-1]:
            spec_count = speculated_inputs.shape[-1] - inputs.shape[-1]
            spec_pos = actual_lengths.unsqueeze(1) + torch.arange(
                spec_count, dtype=torch.long
            ).unsqueeze(0)
            positions = torch.cat([self.initial_positions, spec_pos], dim=-1)

        # TODO: Unify BMC and PAGED speculative prefill paths into a single
        # forward, similar to how _decode_forward handles both uniformly.
        spec_count = speculated_inputs.shape[-1] - inputs.shape[-1]
        paged_meta = None
        if self.use_paged_attention:
            per_seq_lengths = actual_lengths + spec_count
            packed_input, packed_pos = self._pack_ragged(
                speculated_inputs, per_seq_lengths
            )
            paged_meta = build_paged_attention_metadata(
                self.kv_cache_manager,
                self.model.config,
                per_seq_lengths,
                self.dtype,
                self.block_size,
            )
            model_out: ModelOutput = self.model(
                packed_input,
                packed_pos,
                kv_cache=self.kv_cache_manager,
                paged_attn_metadata=paged_meta,
            )
            self._prefill_full_logits = model_out.logits
            if self.spec_decoder is None:
                end_positions = torch.cumsum(per_seq_lengths, dim=0) - 1
                per_seq_logits = model_out.logits[0, end_positions, :].unsqueeze(1)
                model_out = ModelOutput(logits=per_seq_logits)
            self.num_computed_tokens += per_seq_lengths
        else:
            model_out = self.model(
                speculated_inputs,
                positions,
                kv_cache=self.kv_cache_manager,
            )
            self.num_computed_tokens += actual_lengths + spec_count
        return model_out, speculated_inputs

    def _decode_forward(
        self, model_input: torch.Tensor
    ) -> tuple[ModelOutput, torch.Tensor]:
        """Runs a single decode step: speculation (if configured), counter-based
        position computation, and model forward.

        When PARD is active, ``model_input`` may contain multiple
        accepted tokens from the previous verification step.  All of
        them are forwarded through the draft model so it rebuilds the
        correct KV-cache context, but only the tokens that are *not*
        yet in the target KV cache are sent to the target model.

        Args:
            model_input (torch.Tensor): The new token(s) to decode

        Returns:
            Tuple of (ModelOutput, speculated_inputs tensor)
        """
        if self.spec_decoder is not None:
            spec_output = self.spec_decoder.speculate(model_input)
            speculated_inputs = spec_output.extended_input

            n_target = 1 + self.spec_decoder.num_speculative_tokens
            target_input = speculated_inputs[:, -n_target:]
        else:
            speculated_inputs = model_input
            target_input = model_input

        positions = self._compute_positions(target_input.shape[-1])

        paged_meta = None
        if self.use_paged_attention:
            batch_size = target_input.shape[0]
            query_len = target_input.shape[-1]
            query_lengths = torch.full((batch_size,), query_len, dtype=torch.long)
            paged_meta = build_paged_attention_metadata(
                self.kv_cache_manager,
                self.model.config,
                query_lengths,
                self.dtype,
                self.block_size,
                past_lengths=self.num_computed_tokens,
            )

        pa_kwargs = (
            {"paged_attn_metadata": paged_meta} if paged_meta is not None else {}
        )
        model_out: ModelOutput = self.model(
            target_input,
            positions,
            kv_cache=self.kv_cache_manager,
            **pa_kwargs,
        )
        self.num_computed_tokens += target_input.shape[-1]
        return model_out, speculated_inputs

    @staticmethod
    def _trim_sampler_output_to_accepted(
        sampler_output: SamplerOutput, n_accepted: int
    ) -> SamplerOutput:
        """Trim speculative sampler output to only the accepted positions.

        During speculative decoding the sampler produces probs/logprobs for
        all ``draft_size + 1`` candidate positions, but only ``n_accepted``
        tokens survive verification.  This method slices the 3-D
        ``[batch, positions, vocab]`` tensors down to ``n_accepted``
        positions and flattens them to ``[batch, n_accepted * vocab]`` so
        they can be concatenated by ``_update_probs_logprobs``.

        If the tensors are already 2-D (non-speculative path) or ``None``,
        they are returned unchanged.
        """

        def _trim(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None or tensor.dim() != 3:
                return tensor
            trimmed = tensor[:, :n_accepted, :]
            return trimmed.reshape(trimmed.shape[0], -1)

        new_probs = _trim(sampler_output.probs)
        new_logprobs = _trim(sampler_output.logprobs)

        if (
            new_probs is sampler_output.probs
            and new_logprobs is sampler_output.logprobs
        ):
            return sampler_output

        return SamplerOutput(
            next_tokens=sampler_output.next_tokens,
            probs=new_probs,
            logprobs=new_logprobs,
        )

    def _sample_and_postprocess(
        self,
        model_out: ModelOutput,
        speculated_inputs: torch.Tensor,
        all_tokens: torch.Tensor,
        unfinished_sequences: torch.Tensor,
        probs: Optional[torch.Tensor],
        logprobs: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        SamplerOutput,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Samples tokens from model output, postprocesses speculated tokens,
        and updates accumulated state (all_tokens, stopping, mask, probs/logprobs).

        Returns:
            Tuple of (all_tokens, next_tokens, unfinished_sequences,
                       sampler_output, probs, logprobs)
        """
        logits = model_out.logits

        # Sample the next token using the sampler
        draft_size = (
            self.spec_decoder.num_speculative_tokens
            if self.spec_decoder is not None
            else 0
        )
        next_token_logits = logits[:, -draft_size - 1 :, :].clone().float()
        if next_token_logits.size(1) == 1 and draft_size == 0:
            next_token_logits = next_token_logits.squeeze(dim=1)

        sampler_output: SamplerOutput = self.sampler.sample(
            all_tokens, next_token_logits
        )
        next_tokens = sampler_output.next_tokens
        if next_tokens.dim() == 3:
            next_tokens = next_tokens.squeeze(-1)

        # Verify speculated tokens (adjusts target KV cache if needed)
        if self.spec_decoder is not None:
            result = self.spec_decoder.verify(next_tokens, speculated_inputs)
            next_tokens = result.accepted_tokens
            n_accepted = next_tokens.shape[-1]

            sampler_output = self._trim_sampler_output_to_accepted(
                sampler_output, n_accepted
            )

            if result.target_kv_cache_trim > 0:
                self.kv_cache_manager.remove_cache(result.target_kv_cache_trim)
                self.num_computed_tokens -= result.target_kv_cache_trim

        # Pad finished sequences with pad_token_id
        next_tokens = next_tokens * unfinished_sequences.reshape(
            -1, 1
        ) + self.sampling_config.pad_token_id * (
            1 - unfinished_sequences.reshape(-1, 1)
        )

        # Stream tokens
        if self.text_streamer:
            self.text_streamer.put(next_tokens)

        # Accumulate tokens for stopping criteria / output (NOT sent to model)
        all_tokens = torch.cat([all_tokens, next_tokens], dim=-1)

        # Update probs and logprobs
        probs, logprobs = self._update_probs_logprobs(probs, logprobs, sampler_output)

        # Check stopping criteria
        stopping_criteria_output = self.stopping_criteria.stop_now(
            all_tokens, num_new_tokens=next_tokens.shape[-1]
        )
        unfinished_sequences = unfinished_sequences & ~stopping_criteria_output

        return (
            all_tokens,
            next_tokens,
            unfinished_sequences,
            sampler_output,
            probs,
            logprobs,
        )

    def generate(self, inputs: torch.Tensor) -> GeneratorOutput:
        """
        Generates text from the model given the input prompts.
        NOTE: prepare_for_generate should be called before calling this method.

        Args:
            inputs (torch.Tensor): The input prompts

        Returns:
            GeneratorOutput: The generated outputs
        """

        all_tokens = inputs
        input_logprobs = None
        probs = None
        logprobs = None

        try:
            if self.text_streamer:
                self.text_streamer.put(inputs)

            unfinished_sequences = torch.ones(inputs.shape[0], dtype=torch.long)

            # === PREFILL ===
            model_out, speculated_inputs = self._prefill_forward(inputs)

            # One-time calculation of input logprobs.
            # For paged attention, _prefill_forward truncates logits to
            # last-token-per-sequence; _prefill_full_logits stores the full
            # logits before truncation. For non-paged it is None.
            if self.sampling_config.return_input_logprobs:
                full_logits = getattr(self, "_prefill_full_logits", None)
                if full_logits is None:
                    full_logits = model_out.logits
                input_logprobs = torch.log_softmax(full_logits, dim=-1)
                self._prefill_full_logits = None

            all_tokens, next_tokens, unfinished_sequences, _, probs, logprobs = (
                self._sample_and_postprocess(
                    model_out,
                    speculated_inputs,
                    all_tokens,
                    unfinished_sequences,
                    probs,
                    logprobs,
                )
            )
            model_input = next_tokens

            # === DECODE LOOP ===
            while unfinished_sequences.max() != 0:
                model_out, speculated_inputs = self._decode_forward(model_input)

                all_tokens, next_tokens, unfinished_sequences, _, probs, logprobs = (
                    self._sample_and_postprocess(
                        model_out,
                        speculated_inputs,
                        all_tokens,
                        unfinished_sequences,
                        probs,
                        logprobs,
                    )
                )
                model_input = next_tokens

            if self.text_streamer:
                self.text_streamer.end()

            if self.spec_decoder is not None:
                stats = self.spec_decoder.get_stats()
                if stats:
                    PACE_LLM_DEBUG(
                        f"Mean accepted tokens: {stats.mean_accepted_tokens}"
                    )
                    PACE_LLM_DEBUG(
                        f"Total accepted speculated tokens: "
                        f"{stats.total_speculated_tokens}"
                    )
            output = self._prepare_output_for_generate(
                output_token_ids=all_tokens,
                input_logprobs=input_logprobs,
                logprobs=logprobs,
                probs=probs,
            )

            return output
        finally:
            if hasattr(self, "kv_cache_manager") and self.kv_cache_manager is not None:
                if self._cache is not None:
                    self._cache.remove_context(self.kv_cache_manager)
                self.kv_cache_manager = None
            if self.spec_decoder is not None:
                self.spec_decoder.cleanup()

    def __repr__(self):
        return f"Generator(model_path={self.model_path}, tokenizer_path={self.tokenizer_path})"

    def get_tokenizer(
        self,
    ) -> PreTrainedTokenizer:
        """
        Returns the tokenizer

        Returns:
            PreTrainedTokenizer: The tokenizer
        """
        return self.tokenizer

    def get_config(self) -> PretrainedConfig:
        """
        Returns the model config

        Returns:
            PretrainedConfig: The model config
        """
        return self.model.get_config()
