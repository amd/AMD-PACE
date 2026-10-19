# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Numerical parity tests for PACE encoder models vs HuggingFace.

Covers BERT (bge family) and ModernBERT (mmBERT family) embedders and
sequence classifiers, plus the bidirectional (non-causal) attention path and
pooling. The fp32 checks run against the NATIVE attention backend so they are
pure-PyTorch and do not require AVX512/JIT kernels; a separate bf16 class
covers the dtype ``pace-vllm`` actually serves, which is also the only dtype
that reaches the fused ``pace::fused_rope_apply`` kernel.

Every parity class draws its weights through :func:`_rescale_weight_matrices`
rather than trusting HuggingFace's initialiser, and carries a mutation guard
asserting that the attention path it claims to check is load-bearing. See
:func:`_rescale_weight_matrices` for why.
"""

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from pace.llm.attention import AttentionBackendType
from pace.llm.configs import OperatorConfig
from pace.llm.ops import LLMOperatorType
from pace.llm.models.model_utils import no_init_weights, set_default_torch_dtype
from pace.llm.pooling import pool


def _native_opconfig() -> OperatorConfig:
    op = OperatorConfig()
    op[LLMOperatorType.Attention] = AttentionBackendType.NATIVE
    return op.finalize()


def _build_pace(model_cls, config, dtype=torch.float32):
    with set_default_torch_dtype(dtype):
        with no_init_weights():
            model = model_cls(config, _native_opconfig())
    return model.eval()


def _copy_weights(pace_model, hf_model):
    pace_model.load_weights((k, v) for k, v in hf_model.state_dict().items())
    # What model_utils.load_weights does once a checkpoint is in: operators
    # with a backend lay their weights out for it here. Skipping it would leave
    # the fused kernels unreachable, so the bf16 classes would be testing a
    # path production never runs.
    pace_model.preprocess_weights()


# Std of the redrawn weight matrices. Large enough that attention dominates
# the residual stream, small enough that 2-4 layers of pre/post-norm blocks
# stay numerically tame in bf16.
_WEIGHT_STD = 0.2


def _rescale_weight_matrices(model, seed, std=_WEIGHT_STD):
    """Redraw every weight *matrix* at a realistic scale, in place.

    HuggingFace's ``_init_weights`` deliberately initialises attention and MLP
    out-projections tiny -- ModernBERT's std is
    ``initializer_range / sqrt(2 * num_layers)`` -- so a randomly initialised
    encoder's output is dominated by the embeddings and the final LayerNorm and
    is nearly independent of what attention computes. Measured on the
    4-layer/64-hidden config used here, ablating attention outright (zeroing
    every ``Wo``) moves ``last_hidden_state`` by max-abs 2.1e-2 at a per-token
    cosine of 0.999986.

    A parity assertion built on those weights therefore has no usable margin
    against attention-path faults, and against some it is blind outright:
    giving the sliding layers the global RoPE base -- a 16x error, and the real
    regression that motivated this -- shifts the output by only 4.3e-5, three
    orders of magnitude inside the 1e-3 tolerance, and passes. Faults that were
    caught were caught narrowly: forcing the local window off failed by 3.7e-3
    against a 1e-3 tolerance, and swapping BERT's Q and K by 1.0e-3 against
    1e-4, both close enough to the bound to turn on the weight seed.

    Redrawing at std ``0.2`` puts the attention/MLP contribution on the same
    footing as the residual stream, which is the regime a *trained* checkpoint
    is in -- the tiny values are an artefact of the initialiser, not of real
    weights. The same ablation then moves the output by 4.3 at a cosine of
    0.11, and the RoPE error by 3.0.

    Only tensors with ``dim() >= 2`` are touched, so LayerNorm gains and biases
    keep their meaningful 1.0 / 0.0 initialisation. Callers pass the *same*
    seed to both the HF and PACE side so parity stays a fair comparison.
    """
    generator = torch.Generator().manual_seed(seed)
    for param in model.parameters():
        if param.dim() >= 2:
            param.data = (
                torch.randn(param.shape, generator=generator, dtype=param.dtype) * std
            )
    return model


def _paired_models(pace_cls, hf_cls, cfg, seed=0, dtype=torch.float32, **hf_kwargs):
    """An HF and a PACE model holding bit-identical, rescaled weights.

    The rescale happens in fp32 and the cast to ``dtype`` afterwards, so both
    sides round identically and a bf16 comparison measures the encoder rather
    than a difference in weights.
    """
    torch.manual_seed(seed)
    hf = hf_cls(cfg, **hf_kwargs).eval()
    _rescale_weight_matrices(hf, seed=seed + 1)
    if dtype != torch.float32:
        hf = hf.to(dtype)
    pace = _build_pace(pace_cls, cfg, dtype=dtype)
    _copy_weights(pace, hf)
    return hf, pace


# bf16 parity budgets.
#
# bfloat16 keeps an 8-bit mantissa (eps ~= 7.8e-3), so on encoder activations
# of magnitude ~3 the *representable* granularity alone is ~2.3e-2. An
# absolute ``atol=1e-3`` of the kind the fp32 tests use is not merely tight
# here, it is unsatisfiable by any implementation -- it would measure rounding
# luck rather than correctness. Worse, ``last_hidden_state`` leaves a
# LayerNorm, so individual coordinates sit near zero and their *relative*
# error is unbounded even when the vector as a whole is right.
#
# So the criterion is distributional and scale-relative instead:
#   - mean per-token cosine: the aggregate "is this the same representation"
#   - min per-token cosine: catches a single badly-wrong token that the mean
#     would average away (exactly the failure mode that let a sliding-window
#     fault read 1.0000 on mean cosine)
#   - max-abs-diff relative to the reference's own scale, rather than absolute
#
# Values fixed by measuring PACE-vs-HF over 8 weight seeds at seq 12 and 64;
# worst case observed is quoted against each. Real attention bugs land far
# outside these (mean cosine <= 0.89, relative diff >= 0.87), so the gap
# between "bf16 noise" and "bug" is ~500x in (1 - cosine).
_BF16_MEAN_COS = 0.999  # worst observed 0.99983
_BF16_MIN_COS = 0.99  # worst observed 0.99930
_BF16_REL_DIFF = 0.15  # worst observed 0.040


def _parity_stats(got, ref, attention_mask=None):
    """``(mean cosine, min cosine, relative max-abs-diff)`` over real tokens.

    Padded positions are dropped when ``attention_mask`` is given: they are
    still computed as queries but carry no meaning, so letting them into the
    statistics would add noise unrelated to what is being tested. Everything
    is compared in fp32 so the metric itself is not quantised.
    """
    got = got.float().reshape(-1, got.shape[-1])
    ref = ref.float().reshape(-1, ref.shape[-1])
    if attention_mask is not None:
        keep = attention_mask.reshape(-1).bool()
        got, ref = got[keep], ref[keep]
    cos = F.cosine_similarity(got, ref, dim=-1)
    scale = ref.abs().max().clamp(min=1e-6)
    rel = ((got - ref).abs().max() / scale).item()
    return cos.mean().item(), cos.min().item(), rel


class _ParityAssertions:
    """Shared assertions for the parity classes."""

    def _encode(self, model, ids, am):
        with torch.inference_mode():
            return model(input_ids=ids, attention_mask=am).last_hidden_state

    def assert_bf16_parity(self, got, ref, attention_mask=None, label="output"):
        mean_cos, min_cos, rel = _parity_stats(got, ref, attention_mask)
        self.assertGreater(
            mean_cos,
            _BF16_MEAN_COS,
            f"{label}: mean per-token cosine {mean_cos:.6f} below "
            f"{_BF16_MEAN_COS} -- worse than bf16 rounding explains",
        )
        self.assertGreater(
            min_cos,
            _BF16_MIN_COS,
            f"{label}: worst per-token cosine {min_cos:.6f} below "
            f"{_BF16_MIN_COS} -- some token is substantially wrong",
        )
        self.assertLess(
            rel,
            _BF16_REL_DIFF,
            f"{label}: max-abs-diff is {rel:.4f} of the reference scale, "
            f"above {_BF16_REL_DIFF}",
        )
        return mean_cos, min_cos, rel

    def assert_attention_is_load_bearing(self, intact, ablated, ids, am):
        """Assert the encoder's output actually depends on attention.

        Ablating every attention out-projection must visibly move the output.
        This is the guard that keeps the parity assertions in the same class
        from passing vacuously: under HuggingFace's own initialiser the same
        ablation leaves a per-token cosine of 0.9997+, i.e. the parity check
        would be comparing two encoders whose attention could be arbitrarily
        broken without either noticing.
        """
        base = self._encode(intact, ids, am)
        gone = self._encode(ablated, ids, am)
        max_diff = (base - gone).abs().max().item()
        mean_cos, _, _ = _parity_stats(gone, base, am)
        self.assertGreater(
            max_diff,
            0.5,
            f"ablating attention moved the output by only {max_diff:.3e}; "
            "the parity tests in this class cannot see attention bugs",
        )
        self.assertLess(
            mean_cos,
            0.9,
            f"ablating attention left a per-token cosine of {mean_cos:.6f}; "
            "attention contributes almost nothing to this model's output",
        )
        return max_diff, mean_cos


class TestBertParity(unittest.TestCase, _ParityAssertions):
    def _config(self, num_labels=None):
        from transformers import BertConfig

        kwargs = dict(
            vocab_size=200,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=64,
            type_vocab_size=2,
            hidden_act="gelu",
        )
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
        return BertConfig(**kwargs)

    def _models(
        self,
        pace_cls,
        hf_cls,
        num_labels=None,
        seed=0,
        dtype=torch.float32,
        **hf_kwargs,
    ):
        """HF + PACE BERT sharing one set of realistically-scaled weights."""
        cfg = self._config(num_labels=num_labels)
        hf, pace = _paired_models(
            pace_cls, hf_cls, cfg, seed=seed, dtype=dtype, **hf_kwargs
        )
        return cfg, hf, pace

    def _inputs(self, seq_len=10, pad_from=7):
        torch.manual_seed(0)
        ids = torch.randint(1, 200, (2, seq_len))
        am = torch.ones(2, seq_len, dtype=torch.long)
        am[0, pad_from:] = 0
        return ids, am

    def test_bert_model_matches_hf(self):
        from transformers.models.bert.modeling_bert import BertModel as HFBert

        from pace.llm.models.bert import BertModel

        _, hf, pace = self._models(BertModel, HFBert, add_pooling_layer=False)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
            got = pace(input_ids=ids, attention_mask=am).last_hidden_state
        # Observed max-abs-diff is 0.0: PACE reproduces HF bit-exactly here.
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_bert_classifier_matches_hf(self):
        from transformers.models.bert.modeling_bert import (
            BertForSequenceClassification as HF,
        )

        from pace.llm.models.bert import BertForSequenceClassification

        _, hf, pace = self._models(BertForSequenceClassification, HF, num_labels=3)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_bert_token_classifier_matches_hf(self):
        from transformers.models.bert.modeling_bert import (
            BertForTokenClassification as HF,
        )

        from pace.llm.models.bert import BertForTokenClassification

        _, hf, pace = self._models(BertForTokenClassification, HF, num_labels=5)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        # Per-token logits: [batch, seq_len, num_labels].
        self.assertEqual(tuple(got.shape), (2, 10, 5))
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    # Mutation guards: the parity checks above are only meaningful if the
    # attention path they run through is actually observable in the output.

    def test_attention_is_load_bearing(self):
        from transformers.models.bert.modeling_bert import BertModel as HFBert

        from pace.llm.models.bert import BertModel

        _, _, intact = self._models(BertModel, HFBert, add_pooling_layer=False)
        _, _, ablated = self._models(BertModel, HFBert, add_pooling_layer=False)
        with torch.no_grad():
            for layer in ablated.encoder.layer:
                layer.attention.output.dense.weight.zero_()

        ids, am = self._inputs()
        self.assert_attention_is_load_bearing(intact, ablated, ids, am)

    def test_output_depends_on_bidirectional_context(self):
        """A token's representation must depend on tokens *after* it.

        This is the defining property of the non-causal encoder path, and it
        can only hold if attention both runs and mixes across positions. It
        would also catch the backend silently falling back to a causal mask.
        """
        from transformers.models.bert.modeling_bert import BertModel as HFBert

        from pace.llm.models.bert import BertModel

        _, _, pace = self._models(BertModel, HFBert, add_pooling_layer=False)
        ids, am = self._inputs()
        tail_swapped = ids.clone()
        torch.manual_seed(5)
        tail_swapped[:, 5:] = torch.randint(1, 200, (2, ids.shape[1] - 5))

        base = self._encode(pace, ids, am)
        moved = self._encode(pace, tail_swapped, am)
        # Only position 0 is inspected; its own input token is unchanged, so
        # any movement has to have arrived through attention.
        diff = (base[:, 0] - moved[:, 0]).abs().max().item()
        self.assertGreater(
            diff,
            1e-1,
            f"replacing the sequence tail moved token 0 by only {diff:.3e}; "
            "attention is not mixing across positions",
        )

    def test_padding_mask_is_applied(self):
        """The ``[B, S]`` padding mask must be honoured, and must matter.

        Two halves, because either alone is passable by a broken
        implementation: real tokens must be *invariant* to what sits in the
        padded slots (the mask is applied), and dropping the mask must
        *change* them (the mask is not a no-op being silently ignored).
        """
        from transformers.models.bert.modeling_bert import BertModel as HFBert

        from pace.llm.models.bert import BertModel

        _, _, pace = self._models(BertModel, HFBert, add_pooling_layer=False)
        ids, am = self._inputs()
        real = am[0].sum().item()

        masked = self._encode(pace, ids, am)
        garbage = ids.clone()
        garbage[0, real:] = torch.tensor([199, 198, 197])[: ids.shape[1] - real]
        invariant = self._encode(pace, garbage, am)
        torch.testing.assert_close(
            invariant[0, :real], masked[0, :real], rtol=0, atol=1e-6
        )

        unmasked = self._encode(pace, ids, torch.ones_like(am))
        diff = (masked[0, :real] - unmasked[0, :real]).abs().max().item()
        self.assertGreater(
            diff,
            1e-1,
            f"dropping the padding mask moved real tokens by only {diff:.3e}; "
            "padded keys are not being excluded",
        )


def _modernbert_available() -> bool:
    try:
        from transformers import ModernBertConfig  # noqa: F401
        from transformers.models.modernbert.modeling_modernbert import (  # noqa: F401
            ModernBertModel,
        )

        return True
    except Exception:
        return False


@unittest.skipUnless(
    _modernbert_available(), "transformers ModernBert not available in this env"
)
class TestModernBertParity(unittest.TestCase, _ParityAssertions):
    # Long-sequence settings for the tests that need the local window to
    # actually bite: at seq 12 with local_attention 8 the window clips only the
    # sequence corners, which is a weak exercise of the sliding-window path.
    LONG_SEQ = 64
    LONG_WINDOW = 16

    def _config(self, num_labels=None, **overrides):
        from transformers import ModernBertConfig

        kwargs = dict(
            vocab_size=200,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=64,
            global_attn_every_n_layers=2,
            local_attention=8,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            hidden_activation="gelu",
            norm_bias=False,
            attention_bias=False,
            mlp_bias=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
        )
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
            kwargs["classifier_pooling"] = "cls"
        kwargs.update(overrides)
        return ModernBertConfig(**kwargs)

    def _models(
        self,
        pace_cls,
        hf_cls,
        num_labels=None,
        seed=0,
        dtype=torch.float32,
        cfg=None,
        **overrides,
    ):
        """HF + PACE ModernBERT sharing one set of realistically-scaled weights.

        Mirrors :meth:`TestModernBertRopeScaling._models`: without the rescale
        a wrong RoPE base or a mis-sized local window is invisible here (see
        :func:`_rescale_weight_matrices`).
        """
        if cfg is None:
            cfg = self._config(num_labels=num_labels, **overrides)
        hf, pace = _paired_models(pace_cls, hf_cls, cfg, seed=seed, dtype=dtype)
        return cfg, hf, pace

    def _inputs(self, seq_len=12, pad_from=9):
        torch.manual_seed(0)
        ids = torch.randint(3, 200, (2, seq_len))
        am = torch.ones(2, seq_len, dtype=torch.long)
        am[0, pad_from:] = 0
        return ids, am

    def _long_config(self, **overrides):
        kwargs = dict(
            max_position_embeddings=128,
            local_attention=self.LONG_WINDOW,
        )
        kwargs.update(overrides)
        return self._config(**kwargs)

    def test_modernbert_model_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        _, hf, pace = self._models(ModernBertModel, HF)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
            got = pace(input_ids=ids, attention_mask=am).last_hidden_state
        # Tightened from 1e-3 now that the weights make attention observable:
        # the measured max-abs-diff is 0.0 across 8 weight seeds (PACE and HF
        # agree bit-for-bit in fp32), so 1e-5 leaves ample headroom while
        # catching regressions three orders of magnitude smaller than before.
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_modernbert_model_matches_hf_long_sequence(self):
        """Parity where the local window genuinely constrains attention.

        With ``local_attention=16`` over 64 positions, sliding layers see a
        +/-8 neighbourhood rather than the whole sequence, so the window mask
        and the two RoPE bases are both meaningfully exercised.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        cfg = self._long_config()
        _, hf, pace = self._models(ModernBertModel, HF, cfg=cfg)
        ids, am = self._inputs(seq_len=self.LONG_SEQ, pad_from=self.LONG_SEQ - 8)
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
            got = pace(input_ids=ids, attention_mask=am).last_hidden_state
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_modernbert_classifier_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForSequenceClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForSequenceClassification

        _, hf, pace = self._models(
            ModernBertForSequenceClassification, HF, num_labels=3
        )
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_modernbert_mean_pooling_classifier_matches_hf(self):
        """The classifier parity tests pin classifier_pooling='cls' only."""
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForSequenceClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForSequenceClassification

        _, hf, pace = self._models(
            ModernBertForSequenceClassification,
            HF,
            num_labels=3,
            classifier_pooling="mean",
        )
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_modernbert_mean_pooling_without_a_mask_matches_hf(self):
        """No mask means all-ones, not 'give up and use CLS'.

        The head required a non-None mask before it would mean-pool and
        returned the CLS row otherwise, so one checkpoint produced two
        different sets of logits depending on whether a mask was passed.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForSequenceClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForSequenceClassification

        _, hf, pace = self._models(
            ModernBertForSequenceClassification,
            HF,
            num_labels=3,
            classifier_pooling="mean",
        )
        ids, _ = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=None).logits
            got = pace(input_ids=ids, attention_mask=None).logits
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_modernbert_token_classifier_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForTokenClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForTokenClassification

        _, hf, pace = self._models(ModernBertForTokenClassification, HF, num_labels=5)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        self.assertEqual(tuple(got.shape), (2, 12, 5))
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    # Mutation guards. Each asserts that a specific attention-path input the
    # parity tests above depend on measurably changes the output -- i.e. that
    # those tests would fail if it were wrong.

    def test_attention_is_load_bearing(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        _, _, intact = self._models(ModernBertModel, HF)
        _, _, ablated = self._models(ModernBertModel, HF)
        with torch.no_grad():
            for layer in ablated.layers:
                layer.attn.Wo.weight.zero_()

        ids, am = self._inputs()
        self.assert_attention_is_load_bearing(intact, ablated, ids, am)

    def test_rope_base_is_not_silently_ignored(self):
        """A wrong RoPE base must move the output.

        Directly guards the regression that motivated these tests: a 16x error
        in the global RoPE base on most layers shifted ``last_hidden_state`` by
        4.3e-5 under HuggingFace's initialiser -- comfortably inside the old
        1e-3 tolerance -- so every ModernBERT parity test passed with the
        attention mechanism substantially broken. With the rescaled weights the
        same error moves it by ~3.0.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        _, _, correct = self._models(ModernBertModel, HF)
        _, _, wrong = self._models(ModernBertModel, HF, global_rope_theta=160000.0 / 16)
        ids, am = self._inputs()
        diff = (
            (self._encode(correct, ids, am) - self._encode(wrong, ids, am))
            .abs()
            .max()
            .item()
        )
        self.assertGreater(
            diff,
            1e-1,
            f"a 16x error in the global RoPE base moved the output by only "
            f"{diff:.3e}; the parity tests cannot see the RoPE path",
        )

    def test_sliding_window_is_not_silently_ignored(self):
        """Changing the local window size must move the output.

        The companion guard to the RoPE one. Under HuggingFace's initialiser a
        sliding-window fault was effectively unreadable: per-token cosine
        similarity measured 0.999996 whether or not the window was applied, and
        while the old parity tests did fail on a window forced off, they did so
        by only 3.7e-3 against a 1e-3 tolerance -- under 4x of passing, i.e. at
        the mercy of the weight seed. Measured here at seq 64, widening the
        window from 16 to 64 (i.e. to full attention) moves the output by ~2.9,
        so the same fault now fails the long-sequence parity test by ~700x
        more than it used to.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        _, _, windowed = self._models(ModernBertModel, HF, cfg=self._long_config())
        _, _, wide = self._models(
            ModernBertModel, HF, cfg=self._long_config(local_attention=self.LONG_SEQ)
        )
        ids, am = self._inputs(seq_len=self.LONG_SEQ, pad_from=self.LONG_SEQ - 8)
        got = self._encode(windowed, ids, am)
        ref = self._encode(wide, ids, am)
        diff = (got - ref).abs().max().item()
        mean_cos, min_cos, _ = _parity_stats(got, ref, am)
        self.assertGreater(
            diff,
            1e-1,
            f"widening the local window from {self.LONG_WINDOW} to "
            f"{self.LONG_SEQ} moved the output by only {diff:.3e}; the local "
            "window is not being applied",
        )
        # Also pin the metric itself: mean cosine must separate these two, or a
        # cosine-based check would report a broken window as identical.
        self.assertLess(
            mean_cos,
            0.99,
            f"mean per-token cosine {mean_cos:.6f} does not distinguish a "
            f"windowed encoder from a full-attention one (min {min_cos:.6f})",
        )


def _modernbert_rope_parameters_supported() -> bool:
    """True when ``ModernBertConfig`` keys RoPE settings by layer type.

    transformers >= 5 replaced ModernBERT's flat ``global_rope_theta`` /
    ``local_rope_theta`` pair with ``layer_types`` + ``rope_parameters``; only
    then can a checkpoint request YaRN scaling per attention type.
    """
    if not _modernbert_available():
        return False
    try:
        from transformers import ModernBertConfig

        cfg = ModernBertConfig(num_hidden_layers=2)
        return isinstance(getattr(cfg, "rope_parameters", None), dict) and bool(
            getattr(cfg, "layer_types", None)
        )
    except Exception:
        return False


@unittest.skipUnless(
    _modernbert_rope_parameters_supported(),
    "transformers ModernBert without per-layer-type rope_parameters",
)
class TestModernBertRopeScaling(unittest.TestCase, _ParityAssertions):
    """ModernBERT must honour the RoPE settings its config asks for.

    ``mmbert32k-*`` checkpoints reach a 32K context via YaRN scaling, and
    ModernBERT's global and local layers legitimately use different RoPE bases
    (and may use different ``rope_type``s), so the per-layer-type distinction
    has to survive the plumbing.
    """

    LAYER_TYPES = [
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]
    # Pretrained context; YaRN stretches it by FACTOR to MAX_POSITIONS.
    ORIGINAL_MAX_POSITIONS = 256
    FACTOR = 4.0
    MAX_POSITIONS = 1024
    SEQ_LEN = 512  # > ORIGINAL_MAX_POSITIONS, so scaling changes the result

    GLOBAL_THETA = 160000.0
    LOCAL_THETA = 10000.0

    def _default_rope(self):
        return {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": self.GLOBAL_THETA,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": self.LOCAL_THETA,
            },
        }

    def _yarn_rope(self, theta):
        return {
            "rope_type": "yarn",
            "rope_theta": theta,
            "factor": self.FACTOR,
            "original_max_position_embeddings": self.ORIGINAL_MAX_POSITIONS,
        }

    def _config(self, rope_parameters):
        """Build a fresh config.

        ``ModernBertConfig`` is a strict dataclass whose ``rope_parameters`` are
        validated on assignment and normalised in ``__post_init__``, so RoPE
        settings are passed at construction rather than patched onto an
        existing config.
        """
        from transformers import ModernBertConfig

        return ModernBertConfig(
            vocab_size=200,
            hidden_size=64,
            num_hidden_layers=len(self.LAYER_TYPES),
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=self.MAX_POSITIONS,
            layer_types=list(self.LAYER_TYPES),
            rope_parameters=rope_parameters,
            local_attention=64,
            hidden_activation="gelu",
            norm_bias=False,
            attention_bias=False,
            mlp_bias=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
        )

    def _models(self, rope_parameters, seed=0):
        """HF + PACE ModernBERT sharing one set of weights.

        Weights are redrawn at a realistic scale so that a wrong RoPE actually
        shows up in the output; see :func:`_rescale_weight_matrices`.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        cfg = self._config(rope_parameters)
        torch.manual_seed(seed)
        hf = HF(cfg).eval()
        _rescale_weight_matrices(hf, seed=seed + 1)
        pace = _build_pace(ModernBertModel, cfg)
        _copy_weights(pace, hf)
        return cfg, hf, pace

    def _inputs(self):
        torch.manual_seed(7)
        ids = torch.randint(3, 200, (2, self.SEQ_LEN))
        am = torch.ones(2, self.SEQ_LEN, dtype=torch.long)
        am[0, self.SEQ_LEN - 32 :] = 0
        return ids, am

    def test_yarn_rope_matches_hf(self):
        rope = {
            "full_attention": self._yarn_rope(self.GLOBAL_THETA),
            "sliding_attention": self._yarn_rope(self.LOCAL_THETA),
        }
        _, hf, pace = self._models(rope)
        ids, am = self._inputs()
        torch.testing.assert_close(
            self._encode(pace, ids, am),
            self._encode(hf, ids, am),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_unscaled_rope_matches_hf(self):
        # Regression guard: models without scaling must behave as before.
        _, hf, pace = self._models(self._default_rope())
        ids, am = self._inputs()
        torch.testing.assert_close(
            self._encode(pace, ids, am),
            self._encode(hf, ids, am),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_mixed_rope_types_per_layer_match_hf(self):
        # Only the global layers are stretched; the sliding layers keep their
        # short-context RoPE. Both must be honoured independently.
        rope = {
            "full_attention": self._yarn_rope(self.GLOBAL_THETA),
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": self.LOCAL_THETA,
            },
        }
        _, hf, pace = self._models(rope)

        built = [
            (rope.rope_type, rope.rope_theta)
            for rope in (pace.rotary_emb_global, pace.rotary_emb_local)
        ]
        self.assertEqual(
            built,
            [("yarn", self.GLOBAL_THETA), ("default", self.LOCAL_THETA)],
        )
        self.assertEqual(
            [layer.attn.is_global for layer in pace.layers],
            [True, False, True, False],
        )

        ids, am = self._inputs()
        torch.testing.assert_close(
            self._encode(pace, ids, am),
            self._encode(hf, ids, am),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_yarn_is_not_silently_ignored(self):
        # Guards the parity checks above from passing vacuously: if the RoPE
        # type were pinned to "default" again, the two encoders would agree.
        rope = {
            "full_attention": self._yarn_rope(self.GLOBAL_THETA),
            "sliding_attention": self._yarn_rope(self.LOCAL_THETA),
        }
        _, _, scaled = self._models(rope)
        _, _, unscaled = self._models(self._default_rope())
        ids, am = self._inputs()
        diff = (
            (self._encode(scaled, ids, am) - self._encode(unscaled, ids, am))
            .abs()
            .max()
        )
        self.assertGreater(diff.item(), 1e-2)

    def test_per_layer_rope_parameters_are_plumbed_through(self):
        from pace.llm.models.modernbert import _layer_rope_parameters

        cfg = self._config(
            {
                "full_attention": self._yarn_rope(self.GLOBAL_THETA),
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": self.LOCAL_THETA,
                },
            }
        )
        full = _layer_rope_parameters(cfg, "full_attention")
        sliding = _layer_rope_parameters(cfg, "sliding_attention")

        self.assertEqual(full["rope_type"], "yarn")
        self.assertEqual(full["rope_theta"], self.GLOBAL_THETA)
        self.assertEqual(full["factor"], self.FACTOR)
        self.assertEqual(
            full["original_max_position_embeddings"], self.ORIGINAL_MAX_POSITIONS
        )
        self.assertEqual(sliding["rope_type"], "default")
        self.assertEqual(sliding["rope_theta"], self.LOCAL_THETA)

    def test_transformers_normalises_legacy_config(self):
        """A pre-v5 checkpoint config still arrives with the modern fields.

        ``_layer_attention_type`` and ``_layer_rope_parameters`` read
        ``layer_types`` and ``rope_parameters`` directly, with no fallback for
        the older ``global_attn_every_n_layers`` / ``global_rope_theta``
        spelling. That is only safe because the config class derives the modern
        fields from the older ones when loading. If a future transformers bump
        stops doing so, this fails and those fallbacks have to come back.
        """
        import json
        import tempfile

        from transformers import AutoConfig

        legacy = {
            "model_type": "modernbert",
            "num_hidden_layers": 4,
            "global_attn_every_n_layers": 2,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "intermediate_size": 128,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as handle:
                json.dump(legacy, handle)
            cfg = AutoConfig.from_pretrained(tmp)

        self.assertEqual(
            list(cfg.layer_types),
            [
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            "layer_types was not derived from global_attn_every_n_layers",
        )
        self.assertEqual(cfg.rope_parameters["full_attention"]["rope_theta"], 160000.0)
        self.assertEqual(
            cfg.rope_parameters["sliding_attention"]["rope_theta"], 10000.0
        )


class TestBertBfloat16Parity(TestBertParity):
    """BERT parity in bfloat16 -- the dtype ``pace-vllm`` actually serves.

    ``pace-vllm`` requires bfloat16 and refuses any other dtype, so the fp32
    suite above validates a configuration that never runs in production. The
    inherited fp32 tests are re-run here in bf16 by overriding ``_models``;
    exact-comparison assertions are replaced by the relative/distributional
    criterion described at :data:`_BF16_MEAN_COS`.
    """

    DTYPE = torch.bfloat16

    def _models(self, *args, **kwargs):
        kwargs.setdefault("dtype", self.DTYPE)
        return super()._models(*args, **kwargs)

    def test_bert_model_matches_hf(self):
        from transformers.models.bert.modeling_bert import BertModel as HFBert

        from pace.llm.models.bert import BertModel

        _, hf, pace = self._models(BertModel, HFBert, add_pooling_layer=False)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
            got = pace(input_ids=ids, attention_mask=am).last_hidden_state
        self.assertEqual(got.dtype, self.DTYPE)
        # BERT has no RoPE, so PACE and HF run the same ops in the same order
        # and agree bit-for-bit: the measured max-abs-diff is 0.0 over 6 seeds.
        # Asserted through the shared bf16 criterion anyway, so a future
        # divergence is reported in the same terms as ModernBERT's.
        self.assert_bf16_parity(got, ref, am, label="bert bf16 last_hidden_state")

    def test_bert_classifier_matches_hf(self):
        from transformers.models.bert.modeling_bert import (
            BertForSequenceClassification as HF,
        )

        from pace.llm.models.bert import BertForSequenceClassification

        _, hf, pace = self._models(BertForSequenceClassification, HF, num_labels=3)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        self.assert_bf16_parity(got, ref, label="bert bf16 logits")

    def test_bert_token_classifier_matches_hf(self):
        from transformers.models.bert.modeling_bert import (
            BertForTokenClassification as HF,
        )

        from pace.llm.models.bert import BertForTokenClassification

        _, hf, pace = self._models(BertForTokenClassification, HF, num_labels=5)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        self.assertEqual(tuple(got.shape), (2, 10, 5))
        self.assert_bf16_parity(got, ref, am, label="bert bf16 token logits")


@unittest.skipUnless(
    _modernbert_available(), "transformers ModernBert not available in this env"
)
class TestModernBertBfloat16Parity(TestModernBertParity):
    """ModernBERT parity in bfloat16.

    Beyond covering the served dtype, this is the only place the suite reaches
    ``torch.ops.pace.fused_rope_apply``: that kernel is gated on bf16 q/k/cos/
    sin, so the fp32 tests above exercise the Python RoPE fallback exclusively.
    See :meth:`test_fused_rope_kernel_is_exercised`.
    """

    DTYPE = torch.bfloat16

    def _models(self, *args, **kwargs):
        kwargs.setdefault("dtype", self.DTYPE)
        return super()._models(*args, **kwargs)

    def test_modernbert_model_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        _, hf, pace = self._models(ModernBertModel, HF)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
            got = pace(input_ids=ids, attention_mask=am).last_hidden_state
        self.assertEqual(got.dtype, self.DTYPE)
        self.assert_bf16_parity(got, ref, am, label="modernbert bf16")

    def test_modernbert_model_matches_hf_long_sequence(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        _, hf, pace = self._models(ModernBertModel, HF, cfg=self._long_config())
        ids, am = self._inputs(seq_len=self.LONG_SEQ, pad_from=self.LONG_SEQ - 8)
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).last_hidden_state
            got = pace(input_ids=ids, attention_mask=am).last_hidden_state
        self.assert_bf16_parity(got, ref, am, label="modernbert bf16 seq64")

    def test_modernbert_classifier_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForSequenceClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForSequenceClassification

        _, hf, pace = self._models(
            ModernBertForSequenceClassification, HF, num_labels=3
        )
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        self.assert_bf16_parity(got, ref, label="modernbert bf16 logits")

    def test_modernbert_mean_pooling_classifier_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForSequenceClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForSequenceClassification

        _, hf, pace = self._models(
            ModernBertForSequenceClassification,
            HF,
            num_labels=3,
            classifier_pooling="mean",
        )
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        self.assert_bf16_parity(got, ref, label="modernbert bf16 mean-pooled logits")

    def test_modernbert_mean_pooling_without_a_mask_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForSequenceClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForSequenceClassification

        _, hf, pace = self._models(
            ModernBertForSequenceClassification,
            HF,
            num_labels=3,
            classifier_pooling="mean",
        )
        ids, _ = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=None).logits
            got = pace(input_ids=ids, attention_mask=None).logits
        self.assert_bf16_parity(
            got, ref, label="modernbert bf16 mean-pooled logits, no mask"
        )

    def test_modernbert_token_classifier_matches_hf(self):
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertForTokenClassification as HF,
        )

        from pace.llm.models.modernbert import ModernBertForTokenClassification

        _, hf, pace = self._models(ModernBertForTokenClassification, HF, num_labels=5)
        ids, am = self._inputs()
        with torch.inference_mode():
            ref = hf(input_ids=ids, attention_mask=am).logits
            got = pace(input_ids=ids, attention_mask=am).logits
        self.assertEqual(tuple(got.shape), (2, 12, 5))
        self.assert_bf16_parity(got, ref, am, label="modernbert bf16 token logits")

    def test_fused_rope_kernel_is_exercised(self):
        """bf16 must reach the fused RoPE kernel; fp32 must not.

        ``RotaryEmbedding.apply_rotary_emb`` dispatches to
        ``torch.ops.pace.fused_rope_apply`` only when q, k, cos and sin are all
        bfloat16, and swallows any exception from it with a silent fallback to
        the Python path. Both halves matter: the assertion on bf16 is what
        makes the parity tests in this class cover the kernel that runs in
        production, and the assertion on fp32 documents that the rest of the
        suite does not.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        ids, am = self._inputs()
        calls = {}
        for dtype in (torch.float32, self.DTYPE):
            real = torch.ops.pace.fused_rope_apply
            with mock.patch.object(
                torch.ops.pace, "fused_rope_apply", side_effect=real
            ) as spy:
                _, _, pace = self._models(ModernBertModel, HF, dtype=dtype)
                self._encode(pace, ids, am)
                calls[dtype] = spy.call_count

        num_layers = self._config().num_hidden_layers
        self.assertEqual(
            calls[self.DTYPE],
            num_layers,
            "bf16 ModernBERT did not reach pace::fused_rope_apply on every "
            f"layer (got {calls[self.DTYPE]} calls for {num_layers} layers); "
            "the fused kernel is silently falling back",
        )
        self.assertEqual(
            calls[torch.float32],
            0,
            "fp32 unexpectedly called the bf16-only fused RoPE kernel",
        )

    def test_bf16_is_no_less_accurate_than_hf_bf16(self):
        """PACE's bf16 error must be of the same order as HF's own.

        The absolute PACE-vs-HF gap in bf16 (~1e-1 on activations of magnitude
        ~3) is far too large to bound at 1e-3, but on its own that number says
        nothing about which side is right. Measuring both against a shared fp32
        reference does: PACE's deviation was 0.95x-1.32x HF's over 6 weight
        seeds, i.e. the gap is bf16 rounding, not a PACE defect. ``3x`` leaves
        room for reduction-order luck while still failing an implementation
        that is genuinely losing precision -- note the fused kernel accumulates
        in fp32 internally, so PACE has no structural reason to be worse.
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        ids, am = self._inputs()
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                _, hf32, _ = self._models(
                    ModernBertModel, HF, seed=seed, dtype=torch.float32
                )
                _, hf16, pace16 = self._models(ModernBertModel, HF, seed=seed)
                ref = self._encode(hf32, ids, am).float()
                pace_err = (self._encode(pace16, ids, am).float() - ref).abs().max()
                hf_err = (self._encode(hf16, ids, am).float() - ref).abs().max()
                ratio = (pace_err / hf_err.clamp(min=1e-9)).item()
                self.assertLess(
                    ratio,
                    3.0,
                    f"PACE bf16 deviates from the fp32 reference {ratio:.2f}x "
                    f"more than HF bf16 does (PACE {pace_err.item():.3e} vs "
                    f"HF {hf_err.item():.3e})",
                )

    def test_bf16_criterion_detects_attention_bugs(self):
        """The bf16 thresholds must separate rounding noise from real bugs.

        A loose criterion is only defensible if it still fails on the faults it
        is meant to catch. Both mutations below are injected in bf16 and must
        be rejected by :meth:`assert_bf16_parity`, which the intact model
        passes -- the measured margin is ~500x in ``1 - mean cosine``
        (noise 1.7e-4, bugs ~1.3e-1).
        """
        from transformers.models.modernbert.modeling_modernbert import (
            ModernBertModel as HF,
        )

        from pace.llm.models.modernbert import ModernBertModel

        ids, am = self._inputs()
        _, hf, pace = self._models(ModernBertModel, HF)
        ref = self._encode(hf, ids, am)
        self.assert_bf16_parity(self._encode(pace, ids, am), ref, am)

        mutations = {
            "rope base /16": dict(global_rope_theta=160000.0 / 16),
            "local window 8 -> 64": dict(local_attention=64),
        }
        for label, overrides in mutations.items():
            with self.subTest(mutation=label):
                _, _, broken = self._models(ModernBertModel, HF, **overrides)
                with self.assertRaises(
                    AssertionError, msg=f"bf16 criterion did not catch {label}"
                ):
                    self.assert_bf16_parity(
                        self._encode(broken, ids, am), ref, am, label=label
                    )


@unittest.skipUnless(
    _modernbert_available(), "transformers ModernBert not available in this env"
)
class TestModernBertRopeCacheSharing(unittest.TestCase):
    """ModernBERT has two RoPE configurations, so it needs two cos/sin caches.

    Each cache is ``[max_position_embeddings, rotary_dim / 2]``; on a 32K
    context that is 8 MiB per layer, which is worth not duplicating 22 times per
    model when several encoders are resident at once.
    """

    def _config(self):
        from transformers import ModernBertConfig

        return ModernBertConfig(
            vocab_size=200,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=64,
            global_attn_every_n_layers=2,
            local_attention=8,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            hidden_activation="gelu",
            norm_bias=False,
            attention_bias=False,
            mlp_bias=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
        )

    def _model(self):
        from pace.llm.models.modernbert import ModernBertModel

        return _build_pace(ModernBertModel, self._config())

    @staticmethod
    def _rope_modules(model):
        return (model.rotary_emb_global, model.rotary_emb_local)

    @classmethod
    def _storages(cls, model):
        return {
            tensor.untyped_storage().data_ptr()
            for rope in cls._rope_modules(model)
            for tensor in (rope.cos_cache, rope.sin_cache)
        }

    def test_one_cache_per_configuration_not_per_layer(self):
        model = self._model()
        # cos + sin for the global config, cos + sin for the local one, however
        # many layers there are.
        self.assertEqual(len(self._storages(model)), 4)

    def test_each_configuration_keeps_its_own_rope_theta(self):
        # Guard against collapsing the global and local configs into one, or
        # wiring the layers to the wrong one of the two.
        model = self._model()
        self.assertEqual(model.rotary_emb_global.rope_theta, 160000.0)
        self.assertEqual(model.rotary_emb_local.rope_theta, 10000.0)
        self.assertEqual(
            [layer.attn.is_global for layer in model.layers],
            [True, False, True, False],
        )

    def test_rope_is_evaluated_once_per_configuration(self):
        # The point of hoisting the modules onto the model: the cos/sin gather
        # happens twice per forward, not once for each of the layers.
        model = self._model()
        ids = torch.randint(3, 200, (2, 12))
        mask = torch.ones(2, 12, dtype=torch.long)
        with (
            mock.patch.object(
                model.rotary_emb_global,
                "forward",
                wraps=model.rotary_emb_global.forward,
            ) as global_rope,
            mock.patch.object(
                model.rotary_emb_local, "forward", wraps=model.rotary_emb_local.forward
            ) as local_rope,
        ):
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask)
        global_rope.assert_called_once()
        local_rope.assert_called_once()

    def test_two_models_do_not_share_caches(self):
        # Each model owns its caches, so unloading either frees what it
        # allocated and neither is handed tensors owned by a dead model.
        first, second = self._model(), self._model()
        self.assertFalse(self._storages(first) & self._storages(second))

    def test_forward_does_not_write_into_the_caches(self):
        # Sharing is only safe while the caches stay read-only.
        model = self._model()
        before = [
            (rope.cos_cache.clone(), rope.sin_cache.clone())
            for rope in self._rope_modules(model)
        ]
        ids = torch.randint(3, 200, (2, 12))
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=torch.ones(2, 12, dtype=torch.long))
        for rope, (cos, sin) in zip(self._rope_modules(model), before):
            torch.testing.assert_close(rope.cos_cache, cos, rtol=0, atol=0)
            torch.testing.assert_close(rope.sin_cache, sin, rtol=0, atol=0)


class TestBidirectionalAttention(unittest.TestCase):
    def _backend(self, is_causal):
        from pace.llm.attention.contiguous.backend import ContiguousAttentionBackend

        return ContiguousAttentionBackend(
            num_heads=2,
            num_kv_heads=2,
            head_dim=8,
            attn_backend=AttentionBackendType.NATIVE,
            is_causal=is_causal,
        )

    def test_encoder_is_non_causal(self):
        # An encoder attends to future tokens; a causal backend does not, so
        # the two outputs must differ for a multi-token sequence.
        torch.manual_seed(0)
        Q = torch.randn(1, 5, 2, 8)
        K = torch.randn(1, 5, 2, 8)
        V = torch.randn(1, 5, 2, 8)

        enc = self._backend(is_causal=False)
        out_enc = enc(Q, K, V, None, None)
        # First token under bidirectional attention sees the whole sequence,
        # so it should not equal V[:, 0] (which causal-at-position-0 returns).
        self.assertFalse(torch.allclose(out_enc[:, 0], V[:, 0], atol=1e-4))

    def test_padding_mask_excludes_padded_keys(self):
        torch.manual_seed(0)
        Q = torch.randn(1, 4, 2, 8)
        K = torch.randn(1, 4, 2, 8)
        V = torch.randn(1, 4, 2, 8)
        enc = self._backend(is_causal=False)

        full = enc(Q, K, V, None, None, attention_mask=torch.ones(1, 4))
        # Mask out the last key; output must change and must match attention
        # computed over only the first 3 positions.
        masked = enc(Q, K, V, None, None, attention_mask=torch.tensor([[1, 1, 1, 0]]))
        self.assertFalse(torch.allclose(full, masked, atol=1e-4))

        trimmed = enc(Q[:, :, :, :], K[:, :3], V[:, :3], None, None)
        # Query rows for the (still-present) tokens should match the trimmed
        # computation since the padded key contributes nothing.
        torch.testing.assert_close(masked, trimmed, rtol=1e-4, atol=1e-4)


class TestPooling(unittest.TestCase):
    def test_cls_pool(self):
        x = torch.randn(2, 5, 8)
        torch.testing.assert_close(pool(x, "cls"), x[:, 0])

    def test_mean_pool_respects_mask(self):
        x = torch.randn(1, 4, 8)
        am = torch.tensor([[1, 1, 0, 0]])
        torch.testing.assert_close(pool(x, "mean", am), x[:, :2].mean(dim=1))

    def test_last_pool_respects_mask(self):
        x = torch.randn(1, 4, 8)
        am = torch.tensor([[1, 1, 1, 0]])
        torch.testing.assert_close(pool(x, "last", am), x[:, 2])

    def test_normalize_unit_norm(self):
        x = torch.randn(3, 5, 8)
        out = pool(x, "cls", normalize=True)
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=0)


class TestModelFamilyLookup(unittest.TestCase):
    def test_decoder_lookup_rejects_bert(self):
        from pace.llm.models.model_list import get_model_class

        with self.assertRaises(ModuleNotFoundError) as ctx:
            get_model_class("BertModel")
        self.assertIn("encoder", str(ctx.exception).lower())

    def test_encoder_lookup_loads_masked_lm_as_the_encoder_body(self):
        from pace.llm.models.model_list import get_encoder_model_class
        from pace.llm.models.modernbert import ModernBertModel

        self.assertIs(get_encoder_model_class("ModernBertForMaskedLM"), ModernBertModel)

    def test_encoder_lookup_rejects_llama(self):
        from pace.llm.models.model_list import get_encoder_model_class

        with self.assertRaises(ModuleNotFoundError) as ctx:
            get_encoder_model_class("LlamaForCausalLM")
        self.assertIn("decoder", str(ctx.exception).lower())

    def test_models_supported_is_decoder_only(self):
        from pace.llm.models.model_list import (
            encoder_models_supported,
            models_supported,
        )

        self.assertIn("LlamaForCausalLM", models_supported())
        self.assertNotIn("BertModel", models_supported())
        self.assertIn("BertModel", encoder_models_supported())
        self.assertIn("ModernBertForMaskedLM", encoder_models_supported())


class TestEncoderOpconfig(unittest.TestCase):
    def test_partial_config_keeps_override_and_encoder_defaults(self):
        from pace.llm.attention import AttentionBackendType as Attn
        from pace.llm.embedding import finalize_encoder_opconfig
        from pace.llm.ops import LLMBackendType

        op = OperatorConfig()
        op[LLMOperatorType.Attention] = Attn.NATIVE
        finalized = finalize_encoder_opconfig(op)
        self.assertEqual(finalized.attention, Attn.NATIVE)
        self.assertEqual(finalized.qkv_projection, LLMBackendType.TPP)
        self.assertEqual(finalized.mlp, LLMBackendType.TPP)
        self.assertNotEqual(finalized.attention, Attn.SLAB)

    def test_none_gets_full_encoder_defaults(self):
        from pace.llm.embedding import (
            default_encoder_opconfig,
            finalize_encoder_opconfig,
        )

        got = finalize_encoder_opconfig(None)
        ref = default_encoder_opconfig().finalize()
        self.assertEqual(got.attention, ref.attention)
        self.assertEqual(got.mlp, ref.mlp)


if __name__ == "__main__":
    unittest.main()
