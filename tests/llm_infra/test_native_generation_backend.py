# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.testing._internal.common_utils import TestCase
from transformers import BatchEncoding

from pace.llm.attention import KVCacheType
from pace.llm.configs import PardSpecDecodeConfig, SamplingConfig
from pace.llm.generation import (
    GenerationRequest,
    GenerationStatus,
    NativeGenerationBackend,
    TerminationReason,
)
from pace.llm.speculative import SpeculationOutput, VerificationOutput


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    vocab_size = 8

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in token_ids)


class FakeCacheContext:
    def __init__(self):
        self.removed = []

    def remove_cache(self, remove_len):
        self.removed.append(remove_len)


class FakeCacheBackend:
    prefix_cache_enabled = False
    supports_packing = False

    def __init__(self):
        self.released = []
        self.create_kwargs = []
        self.prepared_batches = []

    def prepare_batch(self, max_seq_length, batch_size, spec_headroom=0):
        self.prepared_batches.append((max_seq_length, batch_size, spec_headroom))

    def create_context(self, config, max_seq_length, **kwargs):
        self.create_kwargs.append(kwargs)
        return FakeCacheContext()

    @staticmethod
    def build_prefill_metadata(context, seq_len, past_len=0):
        return None

    @staticmethod
    def merge_contexts(contexts, query_len=1):
        return contexts[0] if len(contexts) == 1 else contexts

    def remove_context(self, context):
        self.released.append(context)


class FakePackedCacheBackend(FakeCacheBackend):
    supports_packing = True

    def merge_contexts_mixed(self, contexts_with_query_lens):
        return SimpleNamespace(paged_attn_metadata=None)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.calls = []

    def __call__(self, input_ids, positions, kv_cache, **kwargs):
        self.calls.append((input_ids.clone(), positions.clone()))
        logits = torch.zeros(*input_ids.shape, 8)
        next_by_token = {1: 3, 3: 2, 4: 5, 5: 6, 6: 2}
        for batch in range(input_ids.shape[0]):
            for position in range(input_ids.shape[1]):
                token = int(input_ids[batch, position])
                logits[batch, position, next_by_token.get(token, 2)] = 10
        return SimpleNamespace(logits=logits)


class FakeSpeculativeDecoder:
    num_speculative_tokens = 1
    supports_mixed_batch = False

    def __init__(self):
        self.draft_context = FakeCacheContext()
        self.released = []

    def prepare_draft_batch(self, max_seq_length, kv_cache_type, batch_size):
        pass

    def create_draft_kv_cache(
        self,
        max_seq_length,
        kv_cache_type,
    ):
        return self.draft_context

    def speculate(
        self,
        model_input,
        draft_kv_cache,
        draft_num_computed,
        initial_positions=None,
    ):
        draft_num_computed += model_input.shape[-1]
        proposed = torch.full(
            (1, 1),
            3,
            dtype=model_input.dtype,
            device=model_input.device,
        )
        return SpeculationOutput(
            extended_input=torch.cat([model_input, proposed], dim=-1)
        )

    @staticmethod
    def verify(
        sampled_tokens,
        speculated_input,
        draft_kv_cache,
        draft_num_computed,
    ):
        return VerificationOutput(
            accepted_tokens=sampled_tokens[:, :1],
            target_kv_cache_trim=1,
        )

    def remove_draft_context(self, context):
        self.released.append(context)


class TestNativeGenerationBackend(TestCase):
    @staticmethod
    def _backend(
        *,
        cache_backend=None,
        model=None,
        kv_cache_type=KVCacheType.DYNAMIC,
        spec_decoder=None,
    ):
        return NativeGenerationBackend(
            model=model or FakeModel(),
            tokenizer=FakeTokenizer(),
            cache_backend=cache_backend or FakeCacheBackend(),
            kv_cache_type=kv_cache_type,
            spec_decoder=spec_decoder,
        )

    @patch(
        "pace.llm.generation.native.num_recurrent_layers",
        return_value=1,
    )
    def test_speculation_rejects_recurrent_models(self, mock_num_recurrent):
        with self.assertRaisesRegex(
            AssertionError,
            "Speculative decoding is not supported for Gated DeltaNet",
        ):
            NativeGenerationBackend._create_speculative_decoder(
                spec_config=object(),
                model_config=SimpleNamespace(),
                dtype=torch.bfloat16,
                opconfig=None,
                kv_cache_type=KVCacheType.DYNAMIC,
            )

    def test_offline_generate_uses_request_local_native_state(self):
        cache_backend = FakeCacheBackend()
        backend = self._backend(
            cache_backend=cache_backend,
        )

        output = backend.generate(
            torch.tensor([[1]]),
            SamplingConfig(max_new_tokens=4),
        )

        self.assertEqual(output.output_token_ids, torch.tensor([[1, 3, 2]]))
        self.assertIsNone(output.probs)
        self.assertIsNone(output.logprobs)
        self.assertEqual(len(cache_backend.released), 1)

    def test_offline_generate_accumulates_requested_distributions(self):
        backend = self._backend()

        output = backend.generate(
            torch.tensor([[1]]),
            SamplingConfig(
                max_new_tokens=4,
                return_probs=True,
                return_logprobs=True,
                return_input_logprobs=True,
            ),
        )

        self.assertIsNotNone(output.probs)
        self.assertIsNotNone(output.logprobs)
        self.assertIsNotNone(output.input_logprobs)
        self.assertEqual(output.probs.shape, output.logprobs.shape)

    def test_chunked_prefill_accumulates_prompt_logprobs(self):
        backend = self._backend()
        state = backend.create_request(
            GenerationRequest(
                request_id="chunked",
                input_ids=torch.tensor([[1, 3]]),
                sampling_config=SamplingConfig(
                    max_new_tokens=2,
                    return_input_logprobs=True,
                ),
            )
        )

        first = backend.prefill(
            [state],
            chunk_info={state.request_id: (0, 1)},
        )
        second = backend.prefill(
            [state],
            chunk_info={state.request_id: (1, 1)},
        )

        self.assertEqual(
            first.deltas[state.request_id].status,
            GenerationStatus.PREFILL_IN_PROGRESS,
        )
        self.assertEqual(
            second.deltas[state.request_id].prompt_logprobs.shape,
            (2, 8),
        )

    def test_chunked_prefill_without_next_chunk_resumes_progress(self):
        model = FakeModel()
        backend = self._backend(model=model)
        state = backend.create_request(
            GenerationRequest(
                request_id="chunked",
                input_ids=torch.tensor([[1, 3, 4]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )

        backend.prefill([state], chunk_info={state.request_id: (0, 1)})
        backend.prefill([state])

        self.assertEqual(model.calls[0][0], torch.tensor([[1]]))
        self.assertEqual(model.calls[1][0], torch.tensor([[3, 4]]))

    def test_finished_rows_follow_cache_batching_strategy(self):
        cases = (
            (FakeCacheBackend, KVCacheType.DYNAMIC, False),
            (FakePackedCacheBackend, KVCacheType.SLAB_POOL, True),
        )
        for cache_type, kv_cache_type, packed in cases:
            with self.subTest(kv_cache_type=kv_cache_type):
                model = FakeModel()
                cache_backend = cache_type()
                backend = self._backend(
                    cache_backend=cache_backend,
                    model=model,
                    kv_cache_type=kv_cache_type,
                )

                output = backend.generate(
                    torch.tensor([[1], [4]]),
                    SamplingConfig(max_new_tokens=4),
                )

                self.assertEqual(
                    output.output_token_ids,
                    torch.tensor([[1, 3, 2, 0], [4, 5, 6, 2]]),
                )
                if packed:
                    self.assertEqual(model.calls[-1][0].numel(), 1)
                else:
                    self.assertEqual(
                        [
                            kwargs["batch_size"]
                            for kwargs in cache_backend.create_kwargs
                        ],
                        [2],
                    )
                    self.assertTrue(all(call[0].shape[0] == 2 for call in model.calls))
                    self.assertEqual(model.calls[-1][0][0].item(), 0)
                self.assertEqual(len(cache_backend.released), 2 if packed else 1)

    def test_shared_contiguous_cache_releases_after_last_request(self):
        cache_backend = FakeCacheBackend()
        backend = self._backend(cache_backend=cache_backend)
        states = backend.create_requests(
            [
                GenerationRequest(
                    request_id=str(index),
                    input_ids=torch.tensor([[1]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                )
                for index in range(2)
            ]
        )

        backend.release(states[0], TerminationReason.ERROR)
        self.assertEqual(cache_backend.released, [])

        backend.release(states[1], TerminationReason.CANCELLED)
        self.assertEqual(len(cache_backend.released), 1)

    def test_variable_length_contiguous_prefill_is_batched_with_padding(self):
        model = FakeModel()
        backend = self._backend(model=model)
        states = backend.create_requests(
            [
                GenerationRequest(
                    request_id="short",
                    input_ids=torch.tensor([[1]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                ),
                GenerationRequest(
                    request_id="long",
                    input_ids=torch.tensor([[4, 5]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                ),
            ]
        )

        result = backend.prefill(states)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], torch.tensor([[0, 1], [4, 5]]))
        self.assertEqual(result.deltas["short"].token_ids, [3])
        self.assertEqual(result.deltas["long"].token_ids, [6])

    def test_packed_batch_prepares_capacity_from_longest_request(self):
        cache_backend = FakePackedCacheBackend()
        backend = self._backend(
            cache_backend=cache_backend,
            kv_cache_type=KVCacheType.SLAB_POOL,
        )

        backend.create_requests(
            [
                GenerationRequest(
                    request_id="short",
                    input_ids=torch.tensor([[1]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                ),
                GenerationRequest(
                    request_id="long",
                    input_ids=torch.tensor([[1, 2, 3]]),
                    sampling_config=SamplingConfig(max_new_tokens=4),
                ),
            ]
        )

        self.assertEqual(cache_backend.prepared_batches, [(7, 2, 0)])

    def test_ragged_probabilities_are_padded(self):
        states = [
            SimpleNamespace(accumulated_probs=[torch.ones(1, 4)]),
            SimpleNamespace(accumulated_probs=[torch.ones(1, 4), torch.ones(1, 4)]),
        ]

        result = NativeGenerationBackend._stack_accumulated(states, "accumulated_probs")

        self.assertEqual(result.shape, (2, 8))
        self.assertEqual(result[0, 4:], torch.zeros(4))

    def test_batched_sampling_isolates_invalid_request(self):
        backend = self._backend()
        states = backend.create_requests(
            [
                GenerationRequest(
                    request_id=str(index),
                    input_ids=torch.tensor([[1]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                )
                for index in range(2)
            ]
        )

        proposals = backend._sample_batch(
            states,
            [torch.ones(1, 8), torch.full((1, 8), torch.nan)],
            [GenerationStatus.DECODING_IN_PROGRESS] * 2,
            [1, 1],
        )

        self.assertEqual(
            proposals["0"].status,
            GenerationStatus.DECODING_IN_PROGRESS,
        )
        self.assertEqual(proposals["1"].status, GenerationStatus.ERROR)

    def test_left_padded_pooled_prefill_is_packed_once(self):
        cache_backend = FakePackedCacheBackend()
        model = FakeModel()
        backend = self._backend(
            model=model,
            cache_backend=cache_backend,
            kv_cache_type=KVCacheType.SLAB_POOL,
        )
        prompts = BatchEncoding(
            {
                "input_ids": torch.tensor([[0, 0, 1], [0, 3, 1]]),
                "attention_mask": torch.tensor([[0, 0, 1], [0, 1, 1]]),
            }
        )

        backend.generate(prompts, SamplingConfig(max_new_tokens=1))

        prefill_ids, prefill_positions = model.calls[0]
        self.assertEqual(prefill_ids, torch.tensor([[1, 3, 1]]))
        self.assertEqual(prefill_positions, torch.tensor([[0, 0, 1]]))

    def test_padding_inference_can_be_disabled_for_server_tokens(self):
        state = SimpleNamespace(
            prompt_length=2,
            attention_mask=None,
            infer_padding=False,
            prompt_token_ids=torch.tensor([[0, 1]]),
            sampling_config=SimpleNamespace(pad_token_id=0),
        )

        prompt_start, actual_length = NativeGenerationBackend._prompt_bounds(
            Mock(), state
        )

        self.assertEqual(prompt_start, 0)
        self.assertEqual(actual_length, 2)

    def test_prefix_cache_evicts_for_batch_headroom(self):
        cache_backend = FakeCacheBackend()
        cache_backend.prefix_cache_enabled = True
        cache_backend._manager = Mock(block_size=4)
        cache_backend._manager.get_free_blocks.return_value = 1
        cache_backend.evict_prefix_blocks = Mock()
        backend = self._backend(
            cache_backend=cache_backend,
        )

        backend._ensure_pool_headroom(
            SimpleNamespace(total_tokens=8),
            prefill_count=1,
            decode_count=1,
        )

        cache_backend.evict_prefix_blocks.assert_called_once_with(3)

    def test_prefix_hit_advances_chunk_cursor(self):
        cache_backend = FakeCacheBackend()
        cache_backend.prefix_cache_enabled = True
        cache_backend._prefix_block_size = 2
        cache_backend.try_prefix_match = Mock(return_value=([7], 2))
        cache_backend.claim_prefix_blocks = Mock()
        backend = self._backend(
            cache_backend=cache_backend,
            kv_cache_type=KVCacheType.SLAB_POOL,
        )
        state = backend.create_request(
            GenerationRequest(
                request_id="request",
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )

        backend._claim_prefix(state)
        start, size = backend._effective_chunk(state, chunk_info=(0, 2))

        self.assertEqual(state.num_computed_tokens, 2)
        self.assertEqual((start, size), (2, 2))

    def test_new_prefill_reuses_later_in_progress_prefix(self):
        cache_backend = FakePackedCacheBackend()
        cache_backend.prefix_cache_enabled = True
        cache_backend._prefix_block_size = 2
        cached = {}
        claimed = []

        def try_prefix_match(block_hashes, query_token_count):
            if block_hashes and block_hashes[0] in cached:
                return [cached[block_hashes[0]]], 2
            return [], 0

        def preinsert(context, block_hashes, num_full_blocks):
            if block_hashes and num_full_blocks:
                cached[block_hashes[0]] = 7
            return num_full_blocks

        cache_backend.try_prefix_match = try_prefix_match
        cache_backend.preinsert_computed_blocks = preinsert
        cache_backend.claim_prefix_blocks = (
            lambda context, hashes, blocks, tokens: claimed.append(tokens)
        )
        cache_backend.insert_prefix = Mock(return_value=0)
        model = FakeModel()
        backend = self._backend(
            cache_backend=cache_backend,
            model=model,
            kv_cache_type=KVCacheType.SLAB_POOL,
        )

        states = backend.create_requests(
            [
                GenerationRequest(
                    request_id=str(index),
                    input_ids=torch.tensor([[1, 2, 3]]),
                    sampling_config=SamplingConfig(max_new_tokens=1),
                )
                for index in range(2)
            ]
        )
        states[1].num_computed_tokens = 2

        backend.prefill(
            states,
            chunk_info={states[1].request_id: (2, 1)},
        )

        self.assertEqual(model.calls[0][0].shape[-1], 2)
        self.assertEqual(claimed, [2])

    def test_only_serving_constructor_reserves_pool(self):
        for constructor, should_reserve in (
            (NativeGenerationBackend.for_offline, False),
            (NativeGenerationBackend.for_serving, True),
        ):
            with self.subTest(constructor=constructor.__name__):
                backend = Mock()
                with patch.object(
                    NativeGenerationBackend,
                    "_from_pretrained",
                    return_value=backend,
                ) as load:
                    constructor("model", kv_cache_type=KVCacheType.PAGED)

                if should_reserve:
                    backend.cache_backend.reserve_pool.assert_called_once_with()
                else:
                    backend.cache_backend.reserve_pool.assert_not_called()
                    self.assertFalse(
                        load.call_args.kwargs["cache_kwargs"]["enable_prefix_cache"]
                    )

    def test_pretrained_does_not_mutate_spec_config(self):
        spec_config = PardSpecDecodeConfig(model_name_or_path="draft")
        target_config = SimpleNamespace(
            max_position_embeddings=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=4,
            hidden_size=16,
        )
        draft_config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=1,
            num_hidden_layers=2,
            hidden_size=16,
        )
        model = Mock(config=target_config)
        decoder = Mock(model=Mock(config=draft_config))
        captured = {}

        def create_decoder(config, *args, **kwargs):
            captured["config"] = config
            return decoder

        with (
            patch(
                "pace.llm.generation.native.resolve_model_path",
                side_effect=lambda path: path,
            ),
            patch("pace.llm.generation.native.validate_model_inputs"),
            patch("pace.llm.generation.native.init_model", return_value=model),
            patch(
                "pace.llm.generation.native.get_tokenizer",
                return_value=FakeTokenizer(),
            ),
            patch("pace.llm.generation.native.create_cache", return_value=Mock()),
            patch.object(
                NativeGenerationBackend,
                "_create_speculative_decoder",
                side_effect=create_decoder,
            ),
        ):
            NativeGenerationBackend._from_pretrained(
                "model",
                spec_config=spec_config,
                cache_kwargs={"kv_cache_memory_gb": 1.0},
            )

        self.assertIsNone(spec_config.draft_kv_cache_memory_gb)
        self.assertIsNot(captured["config"], spec_config)
        self.assertIsNotNone(captured["config"].draft_kv_cache_memory_gb)

    def test_speculative_decoder_controls_mixed_batch_support(self):
        cache_backend = FakePackedCacheBackend()
        decoder = Mock(supports_mixed_batch=False)
        backend = self._backend(
            cache_backend=cache_backend,
            kv_cache_type=KVCacheType.SLAB_POOL,
            spec_decoder=decoder,
        )

        self.assertFalse(backend.supports_mixed_batch)
        decoder.supports_mixed_batch = True
        self.assertTrue(backend.supports_mixed_batch)

    def test_partial_draft_allocation_is_released(self):
        cache_backend = FakeCacheBackend()
        draft_context = object()
        decoder = Mock(num_speculative_tokens=2)
        decoder.create_draft_kv_cache.return_value = draft_context
        backend = self._backend(
            cache_backend=cache_backend,
            spec_decoder=decoder,
        )

        with (
            patch(
                "pace.llm.generation.native.torch.zeros",
                side_effect=RuntimeError("counter allocation failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "counter allocation failed"),
        ):
            backend.create_request(
                GenerationRequest(
                    request_id="request",
                    input_ids=torch.tensor([[1]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                )
            )

        decoder.remove_draft_context.assert_called_once_with(draft_context)
        self.assertEqual(len(cache_backend.released), 1)

    def test_batched_speculation_allocates_one_draft_sequence_per_request(self):
        decoder = Mock(num_speculative_tokens=2)
        decoder.create_draft_kv_cache.side_effect = [object(), object()]
        backend = self._backend(
            cache_backend=FakePackedCacheBackend(),
            kv_cache_type=KVCacheType.SLAB_POOL,
            spec_decoder=decoder,
        )

        backend.create_requests(
            [
                GenerationRequest(
                    request_id=str(index),
                    input_ids=torch.tensor([[1]]),
                    sampling_config=SamplingConfig(max_new_tokens=2),
                )
                for index in range(2)
            ]
        )

        self.assertEqual(decoder.create_draft_kv_cache.call_count, 2)
        decoder.prepare_draft_batch.assert_called_once_with(
            3,
            KVCacheType.SLAB_POOL,
            2,
        )

    def test_speculative_lifecycle_releases_target_and_draft(self):
        cache_backend = FakeCacheBackend()
        decoder = FakeSpeculativeDecoder()
        backend = self._backend(
            cache_backend=cache_backend,
            spec_decoder=decoder,
        )

        output = backend.generate(
            torch.tensor([[1]]),
            SamplingConfig(max_new_tokens=4),
        )

        self.assertEqual(output.output_token_ids, torch.tensor([[1, 3, 2]]))
        self.assertEqual(len(cache_backend.released), 1)
        self.assertEqual(decoder.released, [decoder.draft_context])
        self.assertIsNotNone(output.speculative_stats)

    def test_batched_speculation_rejects_non_packing_cache(self):
        cache_backend = FakeCacheBackend()
        decoder = FakeSpeculativeDecoder()
        backend = self._backend(
            cache_backend=cache_backend,
            spec_decoder=decoder,
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires a cache with packing support",
        ):
            backend.generate(
                torch.tensor([[1], [1]]),
                SamplingConfig(max_new_tokens=4),
            )

    def test_speculative_top_p_with_logprobs(self):
        backend = self._backend(
            spec_decoder=FakeSpeculativeDecoder(),
        )

        output = backend.generate(
            torch.tensor([[1]]),
            SamplingConfig(
                max_new_tokens=4,
                do_sample=False,
                top_p=0.9,
                return_logprobs=True,
            ),
        )

        self.assertEqual(output.output_token_ids, torch.tensor([[1, 3, 2]]))
        self.assertIsNotNone(output.logprobs)

    def test_one_sided_mixed_batch_delegates_on_contiguous_cache(self):
        backend = self._backend()
        prefill = Mock()
        decode = Mock()
        prefill_result = {"prefill": Mock()}
        decode_result = {"decode": Mock()}
        backend._prefill = Mock(return_value=prefill_result)
        backend._decode = Mock(return_value=decode_result)

        self.assertIs(
            backend._mixed([prefill], [], {"prefill": (0, 1)}),
            prefill_result,
        )
        self.assertIs(backend._mixed([], [decode], None), decode_result)
        backend._prefill.assert_called_once_with(
            [prefill],
            {"prefill": (0, 1)},
        )
        backend._decode.assert_called_once_with([decode])

    def test_unsupported_mixed_batch_fails_without_sequential_fallback(self):
        cache_backend = FakeCacheBackend()
        backend = self._backend(
            cache_backend=cache_backend,
        )
        prefill = backend.create_request(
            GenerationRequest(
                request_id="prefill",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )
        decode = backend.create_request(
            GenerationRequest(
                request_id="decode",
                input_ids=torch.tensor([[1]]),
                sampling_config=SamplingConfig(max_new_tokens=2),
            )
        )

        result = backend.mixed([prefill], [decode])

        self.assertEqual(
            result.deltas["prefill"].status,
            GenerationStatus.ERROR,
        )
        self.assertEqual(
            result.deltas["decode"].status,
            GenerationStatus.ERROR,
        )
        self.assertEqual(len(cache_backend.released), 2)

    def test_chunk_start_at_or_beyond_prompt_is_rejected(self):
        state = SimpleNamespace(
            prompt_length=4,
            num_computed_tokens=0,
            backend_state=SimpleNamespace(
                prefix_length=0,
                actual_prompt_length=4,
            ),
        )

        for chunk_start in (4, 5):
            with self.subTest(chunk_start=chunk_start):
                with self.assertRaisesRegex(ValueError, "must be below prompt length"):
                    NativeGenerationBackend._effective_chunk(
                        Mock(), state, chunk_info=(chunk_start, 1)
                    )

    def test_stale_chunk_start_is_rejected(self):
        state = SimpleNamespace(
            prompt_length=4,
            num_computed_tokens=2,
            backend_state=SimpleNamespace(
                prefix_length=0,
                actual_prompt_length=4,
            ),
        )

        with self.assertRaisesRegex(ValueError, "does not match computed tokens"):
            NativeGenerationBackend._effective_chunk(Mock(), state, chunk_info=(1, 1))
