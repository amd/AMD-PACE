# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Optional

import torch
from torch.library import register_fake

# Adding support for fake ops to be used in compile
# https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.ahugy69p2jmz


def compute_linear_out(inputs: torch.Tensor, weights: torch.Tensor):
    """Output shape for linear ops with PyTorch-style weights [out_features, in_features]."""
    out_shape = list(inputs.size())
    if weights.dim() == 2:
        out_shape[-1] = weights.size(0)
    elif weights.dim() == 5:
        out_shape[-1] = weights.size(0) * weights.size(3)
    return torch.empty(out_shape, dtype=inputs.dtype, device=inputs.device)


def compute_linear_out_aocl_dlp(inputs: torch.Tensor, weights: torch.Tensor):
    """Output shape for AOCL-DLP linear ops. Weights are preprocessed [K, N] (in_features, out_features)."""
    out_shape = list(inputs.size())
    if weights.dim() == 2:
        out_shape[-1] = weights.size(1)  # N = out_features
    else:
        out_shape[-1] = weights.size(0)  # fallback
    return torch.empty(out_shape, dtype=inputs.dtype, device=inputs.device)


# float‐output versions (no quant params)
# quantized versions cannot be registered as fake ops
# because they require qtensor, and they are not supported
@register_fake("pace::linear")
@register_fake("pace::linear_relu")
@register_fake("pace::libxsmmlinear_plain")
@register_fake("pace::libxsmmlinear_gelu")
@register_fake("pace::libxsmmlinear_relu")
@register_fake("pace::libxsmmlinear_silu")
def _fake_linear(
    inputs: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None
):
    return compute_linear_out(inputs, weights)


@register_fake("pace::libxsmmlinear_mul")
def _fake_libxsmmlinear_mul(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    return compute_linear_out(input1, weight)


# AOCL-DLP linear ops (torch.compile support). Weights are preprocessed [K, N].
@register_fake("pace::aocl_dlp_linear_plain")
@register_fake("pace::aocl_dlp_linear_gelu")
@register_fake("pace::aocl_dlp_linear_relu")
@register_fake("pace::aocl_dlp_linear_silu")
def _fake_aocl_dlp_linear(
    inputs: torch.Tensor, weights: torch.Tensor, bias: Optional[torch.Tensor] = None
):
    return compute_linear_out_aocl_dlp(inputs, weights)


@register_fake("pace::aocl_dlp_linear_mul")
def _fake_aocl_dlp_linear_mul(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    return compute_linear_out_aocl_dlp(input1, weight)


@register_fake("pace::aocl_dlp_reshape_weights")
def _fake_aocl_dlp_reshape_weights(weight: torch.Tensor):
    return torch.empty_like(weight)


@register_fake("pace::multi_head_attention")
def _fake_attention(
    input_Q: torch.Tensor,
    input_K: torch.Tensor,
    input_V: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_KQ: Optional[float] = None,
):
    return torch.empty_like(input_Q)


@register_fake("pace::grouped_query_attention")
def _fake_grouped_query_attention(
    input_Q: torch.Tensor,
    input_K: torch.Tensor,
    input_V: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
):
    return torch.empty_like(input_Q)


@register_fake("pace::mlp_mlp_fusion")
def _fake_mlp_mlp_fusion(
    src: torch.Tensor,
    weights1: list[torch.Tensor],
    bias1: Optional[list[torch.Tensor]],
    weights2: list[torch.Tensor],
    bias2: Optional[torch.Tensor],
    nlf: str,
    weights_gateProj: Optional[list[torch.Tensor]] = None,
    bias_gateProj: Optional[list[torch.Tensor]] = None,
):
    return torch.empty_like(src)


@register_fake("pace::libxsmm_fused_mlp")
def _fake_libxsmm_fused_mlp(
    src: torch.Tensor,
    wt_gate: Optional[torch.Tensor],
    wt_up: torch.Tensor,
    wt_down: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
    activation: str,
):
    out_shape = list(src.size())
    out_shape[-1] = wt_down.size(0) * wt_down.size(3)
    return torch.empty(out_shape, dtype=src.dtype, device=src.device)


# Note: ops registered with the `m.def(schema, fn)` form are
# CompositeImplicitAutograd and automatically handle meta /
# FakeTensorMode by running the impl. No `register_fake` needed --
# adding one would raise because CompositeImplicitAutograd ops
# decompose rather than dispatch. This applies to:
#   - pace::slab_autotune_block_size (csrc/ops/slab_attention.cpp)
#   - pace::thread_bind, pace::log (csrc/core/core_ops.cpp)
# The three core helpers are also side-effecting void ops with no tensor
# arguments, so there is nothing for a fake to compute anyway.


@register_fake("pace::fused_rope_apply")
def _fake_fused_rope_apply(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
):
    return (torch.empty_like(query), torch.empty_like(key))


@register_fake("pace::fused_add_layernorm")
def _fake_fused_add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
):
    torch._check(x.shape == residual.shape)
    return (torch.empty_like(x), torch.empty_like(x))


@register_fake("pace::layernorm")
def _fake_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
):
    return torch.empty_like(x)


@register_fake("pace::fused_add_rmsnorm")
def _fake_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    torch._check(x.shape == residual.shape)
    return (torch.empty_like(x), torch.empty_like(x))


@register_fake("pace::rmsnorm")
def _fake_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    return torch.empty_like(x)


@register_fake("pace::rmsnorm_gated")
def _fake_rmsnorm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    torch._check(x.shape == gate.shape)
    return x.new_empty(x.shape)


@register_fake("pace::recurrent_gated_delta_rule")
@register_fake("pace::chunk_gated_delta_rule")
def _fake_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
):
    # One output row per value head; `state` is advanced in place.
    return query.new_empty((query.size(0), query.size(1), value.size(2), value.size(3)))


@register_fake("pace::causal_conv1d_update")
def _fake_causal_conv1d_update(
    input: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
):
    return torch.empty_like(input)


@register_fake("pace::causal_conv1d_prefill")
def _fake_causal_conv1d_prefill(
    input: torch.Tensor,
    conv_state: Optional[torch.Tensor],
    weight: torch.Tensor,
):
    batch, channels, seq_len = input.shape
    # The kernel writes token-major [B, T, C] and returns its [B, C, T] view.
    out = input.new_empty((batch, seq_len, channels)).transpose(1, 2)
    new_state = input.new_empty((batch, channels, weight.size(1) - 1))
    return out, new_state


@register_fake("pace::moe_route_softmax")
def _fake_moe_route_softmax(
    router_logits: torch.Tensor, topk: int, renormalize: bool = True
):
    num_tokens = router_logits.size(0)
    ids = torch.empty((num_tokens, topk), dtype=torch.long, device=router_logits.device)
    weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=router_logits.device
    )
    return ids, weights


@register_fake("pace::libxsmm_moe_engine")
def _fake_moe_engine(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    wt_gate: Optional[torch.Tensor],
    wt_up: torch.Tensor,
    wt_down: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
    activation: str,
    num_experts: int,
):
    # Void in-place op (schema `-> ()`): writes into `output`, returns nothing.
    return None
