# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from types import SimpleNamespace

import torch
from torch.testing._internal.common_utils import TestCase
from transformers import (
    DynamicCache,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    LlamaConfig,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.masking_utils import sliding_window_causal_mask_function

from pace.llm.configs import SamplingConfig
from pace.llm.outputs import SamplerOutput
from pace.llm.sampler import Sampler
from pace.llm.step.tree import (
    AttentionWindows,
    DraftTree,
    TreeDrafter,
    TreeVerifier,
    additive_mask,
    compact_cache,
    level_widths,
    draft_keep_indices,
    gather_cache,
    windowed,
)
from pace.utils.logging import suppress_logging_cls

VOCAB = 32
SLIDING_WINDOW = 4
SMALL = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
    max_position_embeddings=256,
)
ALTERNATING = ["sliding_attention", "full_attention"] * 2


def chain_tree(tokens):
    """A single-branch tree: each token is the only child of the previous one."""
    tree = DraftTree()
    parent = -1
    for depth, token in enumerate(tokens):
        parent = tree.add(token, parent, depth, 0.0)
    return tree


def greedy_branch(tree):
    """The tokens along the path that takes the most probable child at every level."""
    children = tree.children()
    branch, candidates = [], children.get(-1, [])
    while candidates:
        best = max(candidates, key=lambda node: tree.log_probs[node])
        branch.append(tree.tokens[best])
        candidates = children.get(best, [])
    return branch


def gemma4_text_config():
    """Shaped like Gemma 4 31B: wider heads, with keys reused as values, on the full-attention layers."""
    return Gemma4TextConfig(
        **SMALL,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        sliding_window=SLIDING_WINDOW,
        layer_types=ALTERNATING,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        use_bidirectional_attention="vision",
    )


def sliding_models():
    """Random models with 4-token windows, taking either a mask per layer type or one mask for every layer."""
    return {
        "gemma3": lambda: Gemma3ForCausalLM(
            Gemma3TextConfig(
                **SMALL,
                head_dim=16,
                sliding_window=SLIDING_WINDOW,
                layer_types=ALTERNATING,
            )
        ),
        "multimodal gemma4": lambda: Gemma4ForConditionalGeneration(
            Gemma4Config(text_config=gemma4_text_config())
        ),
        "qwen3": lambda: Qwen3ForCausalLM(
            Qwen3Config(
                **SMALL,
                head_dim=16,
                use_sliding_window=True,
                sliding_window=SLIDING_WINDOW,
                layer_types=ALTERNATING,
            )
        ),
        "mistral": lambda: MistralForCausalLM(
            MistralConfig(**SMALL, sliding_window=SLIDING_WINDOW)
        ),
    }


class FakeLayer:
    def __init__(self, length, heads=2, head_dim=4):
        self.is_initialized = True
        self.keys = torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1)
        self.keys = self.keys.expand(1, heads, length, head_dim).clone()
        self.values = self.keys.clone() * 10


class FakeCache:
    def __init__(self, length, layers=2):
        self.layers = [FakeLayer(length) for _ in range(layers)]

    def crop(self, length):
        for layer in self.layers:
            layer.keys = layer.keys[:, :, :length]
            layer.values = layer.values[:, :, :length]


class ScriptedModel(torch.nn.Module):
    """
    A stand-in that returns preset next-token predictions.

    `predictions` maps a token id to the token the "full model" would produce
    after it, which is all the acceptance walk depends on.
    """

    def __init__(self, predictions, dtype=torch.float32):
        super().__init__()
        self.predictions = predictions
        self.dtype = dtype
        self.register_parameter(
            "anchor", torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        )
        self.seen = []

    def forward(self, input_ids, **kwargs):
        self.seen.append(kwargs)
        rows = input_ids.shape[1]
        logits = torch.zeros(1, rows, VOCAB, dtype=self.dtype)
        for row in range(rows):
            token = int(input_ids[0, row])
            logits[0, row, self.predictions.get(token, 0)] = 10.0
        return type("Output", (), {"logits": logits})()


HISTORY = torch.tensor([[9]])


def greedy_sampler():
    config = SamplingConfig(temperature=0.0, max_new_tokens=8, eos_token_id=[0])
    config.finalize()
    return Sampler(config, HISTORY)


class ScriptedSampler:
    """Draws preset tokens, and tags each row's distribution with its row index."""

    def __init__(self, draws):
        self.draws = draws

    def sample(self, history, logits):
        rows = logits.shape[0]
        tokens = torch.tensor(self.draws[:rows]).reshape(rows, 1)
        probs = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
        probs = probs.expand(rows, VOCAB).clone()
        return SamplerOutput(tokens, probs, probs.log())


@suppress_logging_cls()
class TestDraftTree(TestCase):

    def test_ancestors_walk_to_the_root(self):
        tree = chain_tree([5, 6, 7])
        self.assertEqual(tree.ancestors(0), [])
        self.assertEqual(tree.ancestors(2), [1, 0])

    def test_children_groups_roots_under_minus_one(self):
        tree = DraftTree()
        first = tree.add(1, -1, 0, 0.0)
        second = tree.add(2, -1, 0, 0.0)
        child = tree.add(3, first, 1, 0.0)
        mapping = tree.children()
        self.assertEqual(mapping[-1], [first, second])
        self.assertEqual(mapping[first], [child])
        self.assertNotIn(second, mapping)


@suppress_logging_cls()
class TestLevelWidths(TestCase):

    def test_narrow_tapers_but_never_below_one(self):
        self.assertEqual(level_widths(3, 3, "narrow"), [3, 2, 1])
        self.assertEqual(level_widths(2, 4, "narrow"), [2, 1, 1, 1])

    def test_full_holds_the_width(self):
        self.assertEqual(level_widths(3, 3, "full"), [3, 3, 3])


@suppress_logging_cls()
class TestAdditiveMask(TestCase):

    def test_context_is_always_visible(self):
        mask = additive_mask(4, [[0], [1]], 2, torch.float32, torch.device("cpu"))
        self.assertEqual(mask.shape, (1, 1, 2, 6))
        self.assertTrue(torch.all(mask[0, 0, :, :4] == 0.0))

    def test_only_listed_slots_are_visible(self):
        mask = additive_mask(2, [[0], [0, 1]], 2, torch.float32, torch.device("cpu"))
        self.assertEqual(mask[0, 0, 0, 2].item(), 0.0)
        self.assertLess(mask[0, 0, 0, 3].item(), -1e30)
        self.assertEqual(mask[0, 0, 1, 2].item(), 0.0)
        self.assertEqual(mask[0, 0, 1, 3].item(), 0.0)

    def test_fill_value_is_representable_in_the_dtype(self):
        mask = additive_mask(1, [[0]], 1, torch.bfloat16, torch.device("cpu"))
        self.assertFalse(torch.isinf(mask).any())
        self.assertFalse(torch.isnan(mask).any())


@suppress_logging_cls()
class TestAttentionWindows(TestCase):

    def test_models_without_sliding_layers_have_no_windows(self):
        for config in (
            LlamaConfig(**SMALL),
            Qwen2Config(**SMALL),
            MistralConfig(**SMALL, sliding_window=None),
        ):
            with self.subTest(model=config.model_type):
                windows = AttentionWindows.of(config)
                self.assertEqual(windows, AttentionWindows())
                self.assertFalse(windows.sliding)

    def test_attention_a_mask_cannot_express_is_refused(self):
        for config in (
            SimpleNamespace(layer_types=["linear_attention", "full_attention"]),
            SimpleNamespace(attention_chunk_size=8192),
        ):
            with self.subTest(config=config), self.assertRaises(AssertionError):
                AttentionWindows.of(config)

    def test_the_window_reaches_as_far_back_as_transformers(self):
        context, rows = 6, 5
        causal = [list(range(row + 1)) for row in range(rows)]
        mask = additive_mask(context, causal, rows, torch.float32, torch.device("cpu"))
        queries = torch.arange(context, context + rows)
        keys = torch.arange(context + rows)
        expected = sliding_window_causal_mask_function(SLIDING_WINDOW)(
            0, 0, queries.reshape(-1, 1), keys.reshape(1, -1)
        )
        self.assertEqual(
            windowed(mask, SLIDING_WINDOW, queries, keys)[0, 0] == 0, expected
        )


@suppress_logging_cls()
class TestSlidingWindowSpeculation(TestCase):
    """Drafting and verifying three windows into the context reproduce HuggingFace's greedy decoding."""

    def setUp(self):
        generator = torch.Generator().manual_seed(1)
        self.prompt = torch.randint(
            3, SMALL["vocab_size"], (1, 3 * SLIDING_WINDOW), generator=generator
        )

    def _greedy(self, model, tokens):
        generated = model.generate(
            self.prompt, max_new_tokens=tokens, do_sample=False, pad_token_id=0
        )
        return generated[0, self.prompt.shape[1] :].tolist()

    def _cache(self, model, tokens):
        cache = DynamicCache()
        model(input_ids=self.prompt[:, :tokens], past_key_values=cache, use_cache=True)
        return cache

    @torch.no_grad()
    def test_verification_commits_what_huggingface_generates(self):
        for name, build in sliding_models().items():
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = build().eval()
                expected = self._greedy(model, 6)
                # Three prompt tokens are left pending, so pending rows are windowed too.
                context = self.prompt.shape[1] - 3
                result = TreeVerifier(model, AttentionWindows.of(model.config)).verify(
                    chain_tree(expected[:5]),
                    self._cache(model, context),
                    context,
                    self.prompt[0, context:].tolist(),
                    greedy_sampler(),
                    self.prompt,
                )
                self.assertEqual(result.committed, expected)

    @torch.no_grad()
    def test_the_draft_follows_huggingface(self):
        for name, build in sliding_models().items():
            with self.subTest(model=name):
                torch.manual_seed(0)
                model = build().eval()
                expected = self._greedy(model, 5)
                context = self.prompt.shape[1] - 1
                # Two roots, so the greedy branch grows beside a sibling it must not attend to.
                drafter = TreeDrafter(
                    model,
                    width=2,
                    depth=5,
                    attention_windows=AttentionWindows.of(model.config),
                )
                tree, _ = drafter.grow(
                    self.prompt[0, context:].tolist(),
                    self._cache(model, context),
                    context,
                    context,
                )
                self.assertEqual(greedy_branch(tree), expected)


@suppress_logging_cls()
class TestCompactCache(TestCase):

    def test_accepted_slots_slide_onto_the_context(self):
        cache = FakeCache(length=10)
        compact_cache(cache, context_length=6, keep_slots=[1, 3])
        for layer in cache.layers:
            self.assertEqual(layer.keys.shape[2], 8)
            # Context is untouched, then the kept slots in commit order.
            self.assertEqual(layer.keys[0, 0, :6, 0].tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual(layer.keys[0, 0, 6:, 0].tolist(), [7, 9])

    def test_rejecting_everything_truncates_to_the_context(self):
        cache = FakeCache(length=10)
        compact_cache(cache, context_length=6, keep_slots=[])
        for layer in cache.layers:
            self.assertEqual(layer.keys.shape[2], 6)

    def test_values_move_with_keys(self):
        cache = FakeCache(length=8)
        compact_cache(cache, context_length=4, keep_slots=[2])
        layer = cache.layers[0]
        self.assertEqual(layer.values[0, 0, 4, 0].item(), 60.0)


@suppress_logging_cls()
class TestTreeVerifier(TestCase):

    def _verify(self, tree, predictions, pending=(9,), context_length=4, sampler=None):
        model = ScriptedModel(predictions)
        cache = FakeCache(length=context_length + len(pending) + len(tree))
        verifier = TreeVerifier(model)
        result = verifier.verify(
            tree,
            cache,
            context_length,
            list(pending),
            sampler or greedy_sampler(),
            HISTORY,
        )
        return result, cache

    def test_a_matching_chain_is_fully_accepted(self):
        tree = chain_tree([1, 2, 3])
        predictions = {9: 1, 1: 2, 2: 3, 3: 4}
        result, _ = self._verify(tree, predictions)
        self.assertEqual(result.accepted, 3)
        self.assertEqual(result.committed, [1, 2, 3, 4])
        self.assertEqual(result.proposed, 3)

    def test_a_wrong_root_accepts_nothing_but_still_commits_a_bonus(self):
        tree = chain_tree([1, 2, 3])
        predictions = {9: 7}
        result, _ = self._verify(tree, predictions)
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.committed, [7])

    def test_acceptance_stops_at_the_first_mismatch(self):
        tree = chain_tree([1, 2, 3])
        predictions = {9: 1, 1: 2, 2: 20}
        result, _ = self._verify(tree, predictions)
        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.committed, [1, 2, 20])

    def test_the_matching_branch_is_chosen_among_siblings(self):
        tree = DraftTree()
        wrong = tree.add(5, -1, 0, 0.0)
        right = tree.add(1, -1, 0, 0.0)
        tree.add(50, wrong, 1, 0.0)
        tree.add(2, right, 1, 0.0)
        predictions = {9: 1, 1: 2, 2: 3}
        result, _ = self._verify(tree, predictions)
        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.committed, [1, 2, 3])

    def test_context_length_advances_by_pending_plus_accepted(self):
        tree = chain_tree([1, 2, 3])
        predictions = {9: 1, 1: 2, 2: 3, 3: 4}
        result, cache = self._verify(tree, predictions, pending=(9,), context_length=4)
        self.assertEqual(result.context_length, 4 + 1 + 3)
        self.assertEqual(cache.layers[0].keys.shape[2], result.context_length)

    def test_cache_is_compacted_to_the_accepted_path(self):
        tree = DraftTree()
        wrong = tree.add(5, -1, 0, 0.0)
        right = tree.add(1, -1, 0, 0.0)
        tree.add(50, wrong, 1, 0.0)
        tree.add(2, right, 1, 0.0)
        predictions = {9: 1, 1: 2, 2: 3}
        result, cache = self._verify(tree, predictions, context_length=4)
        self.assertEqual(cache.layers[0].keys.shape[2], result.context_length)

    def test_an_empty_tree_is_rejected(self):
        model = ScriptedModel({})
        verifier = TreeVerifier(model)
        with self.assertRaises(AssertionError):
            verifier.verify(
                DraftTree(), FakeCache(4), 4, [9], greedy_sampler(), HISTORY
            )

    def test_without_windows_the_model_gets_one_plain_mask(self):
        model = ScriptedModel({9: 1, 1: 2, 2: 3})
        TreeVerifier(model).verify(
            chain_tree([1, 2]), FakeCache(7), 4, [9], greedy_sampler(), HISTORY
        )
        self.assertEqual(
            model.seen[0]["attention_mask"],
            additive_mask(
                4, [[0], [0, 1], [0, 2, 1]], 3, torch.float32, torch.device("cpu")
            ),
        )

    def test_tree_positions_share_a_slot_per_depth(self):
        tree = chain_tree([1, 2, 3])
        predictions = {9: 1, 1: 2, 2: 3, 3: 4}
        model = ScriptedModel(predictions)
        verifier = TreeVerifier(model)
        verifier.verify(tree, FakeCache(8), 4, [9], greedy_sampler(), HISTORY)
        positions = model.seen[0]["position_ids"][0].tolist()
        self.assertEqual(positions, [4, 5, 6, 7])

    def test_acceptance_follows_the_sampler_rather_than_the_argmax(self):
        tree = chain_tree([1, 2, 3])
        # The model's argmax agrees with the whole chain, but the sampler drew 7 after token 1.
        predictions = {9: 1, 1: 2, 2: 3, 3: 4}
        result, _ = self._verify(
            tree, predictions, sampler=ScriptedSampler([1, 7, 3, 4])
        )
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.committed, [1, 7])

    def test_each_committed_token_carries_its_parents_distribution(self):
        tree = DraftTree()
        wrong = tree.add(5, -1, 0, 0.0)
        right = tree.add(1, -1, 0, 0.0)
        tree.add(50, wrong, 1, 0.0)
        tree.add(2, right, 1, 0.0)
        # Rows: 0 is the root, then 1 + node for nodes 0 to 3.
        result, _ = self._verify(tree, {}, sampler=ScriptedSampler([1, 0, 2, 0, 3]))
        self.assertEqual(result.committed, [1, 2, 3])
        self.assertEqual(result.probs[:, 0].tolist(), [0.0, 2.0, 4.0])

    def test_greedy_sampling_reports_no_distributions(self):
        result, _ = self._verify(chain_tree([1]), {9: 1, 1: 2})
        self.assertIsNone(result.probs)
        self.assertIsNone(result.logprobs)


@suppress_logging_cls()
class TestDraftKeepIndices(TestCase):
    """The selection policy, which a token pruner would replace wholesale."""

    def test_nothing_to_do_when_within_budget(self):
        self.assertIsNone(draft_keep_indices(10, budget=16, sink=4))
        self.assertIsNone(draft_keep_indices(10, budget=10, sink=4))

    def test_no_budget_never_trims(self):
        self.assertIsNone(draft_keep_indices(10**6, budget=None, sink=4))

    def test_keeps_a_head_and_the_most_recent_tail(self):
        self.assertEqual(draft_keep_indices(10, budget=6, sink=2), [0, 1, 6, 7, 8, 9])

    def test_without_a_sink_only_the_tail_survives(self):
        self.assertEqual(draft_keep_indices(10, budget=4, sink=0), [6, 7, 8, 9])

    def test_the_result_always_fits_the_budget(self):
        for budget in (1, 2, 5, 17):
            for sink in (0, 1, 4, 100):
                keep = draft_keep_indices(1000, budget=budget, sink=sink)
                self.assertEqual(len(keep), budget)
                self.assertEqual(len(set(keep)), budget)
                self.assertEqual(keep, sorted(keep))

    def test_a_sink_wider_than_the_budget_still_leaves_room_for_recency(self):
        keep = draft_keep_indices(1000, budget=8, sink=50)
        self.assertEqual(len(keep), 8)
        self.assertIn(999, keep)


@suppress_logging_cls()
class TestGatherCache(TestCase):
    """The mechanism, which must express arbitrary selections, not just head and tail."""

    def test_scattered_indices_are_honoured(self):
        cache = FakeCache(length=10)
        gather_cache(cache, [0, 3, 7, 9])
        for layer in cache.layers:
            self.assertEqual(layer.keys.shape[2], 4)
            self.assertEqual(layer.keys[0, 0, :, 0].tolist(), [0, 3, 7, 9])

    def test_values_move_with_keys(self):
        cache = FakeCache(length=8)
        gather_cache(cache, [1, 5])
        self.assertEqual(cache.layers[0].values[0, 0, :, 0].tolist(), [10, 50])

    def test_head_and_tail_selection_matches_the_policy(self):
        cache = FakeCache(length=10)
        gather_cache(cache, draft_keep_indices(10, budget=6, sink=2))
        self.assertEqual(cache.layers[0].keys[0, 0, :, 0].tolist(), [0, 1, 6, 7, 8, 9])


@suppress_logging_cls()
class TestTreeDrafter(TestCase):

    def _drafter(self, window=None):
        model = ScriptedModel({token: token + 1 for token in range(VOCAB)})
        return TreeDrafter(model, width=2, depth=2, mode="narrow", window=window), model

    def test_grow_reports_the_committed_cache_length(self):
        drafter, _ = self._drafter()
        cache = FakeCache(length=6)
        tree, cached = drafter.grow([9], cache, 6, absolute_position=6)
        self.assertEqual(cached, 7)
        self.assertGreater(len(tree), 0)

    def test_grow_needs_a_pending_token(self):
        drafter, _ = self._drafter()
        with self.assertRaises(AssertionError):
            drafter.grow([], FakeCache(4), 4, absolute_position=4)

    def test_window_trims_the_cache_but_not_the_positions(self):
        drafter, model = self._drafter(window=4)
        cache = FakeCache(length=10)
        _, cached = drafter.grow([9], cache, 10, absolute_position=10)
        # Ten cached entries trimmed to the window, then one replayed token.
        self.assertEqual(cached, 5)
        # RoPE still sees the true sequence position, not the shrunken slot.
        self.assertEqual(model.seen[0]["position_ids"][0].tolist(), [10])
        self.assertEqual(model.seen[0]["cache_position"].tolist(), [4])

    def test_without_a_window_the_cache_is_untrimmed(self):
        drafter, model = self._drafter()
        _, cached = drafter.grow([9], FakeCache(10), 10, absolute_position=10)
        self.assertEqual(cached, 11)
        self.assertEqual(model.seen[0]["cache_position"].tolist(), [10])

    def test_a_window_wider_than_the_context_changes_nothing(self):
        """A window that trims nothing must be indistinguishable from no window."""
        plain, plain_model = self._drafter()
        wide, wide_model = self._drafter(window=1000)

        plain_tree, plain_cached = plain.grow([9], FakeCache(10), 10, 10)
        wide_tree, wide_cached = wide.grow([9], FakeCache(10), 10, 10)

        self.assertEqual(plain_cached, wide_cached)
        self.assertEqual(plain_tree.tokens, wide_tree.tokens)
        self.assertEqual(plain_tree.parents, wide_tree.parents)
        for plain_call, wide_call in zip(plain_model.seen, wide_model.seen):
            self.assertEqual(
                plain_call["cache_position"].tolist(),
                wide_call["cache_position"].tolist(),
            )
            self.assertEqual(
                plain_call["position_ids"].tolist(), wide_call["position_ids"].tolist()
            )
