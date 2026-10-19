/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_GATED_DELTANET_KERNEL_H
#define PACE_GATED_DELTANET_KERNEL_H

#include <ATen/ATen.h>
#include <cstdint>

namespace pace {
namespace kernels {

/**
 * Fused recurrent gated delta rule — single-step decode kernel.
 *
 * Computes one time-step of the gated delta recurrence:
 *   state = state * exp(g) + k^T * (v - state^T k) * beta
 *   output = state^T * q
 *
 * With L2-norm on Q/K and query scaling (1/sqrt(k_dim)). GQA handled
 * internally (value head h reads query/key head h / (Hv/Hq)).
 *
 * @param query   [B, 1, Hq, Kd]  bf16
 * @param key     [B, 1, Hq, Kd]  bf16
 * @param value   [B, 1, Hv, Vd]  bf16
 * @param g       [B, 1, Hv]      fp32 or bf16 (log-decay)
 * @param beta    [B, 1, Hv]      fp32 or bf16 (update gate, already sigmoided)
 * @param state   [B, Hv, Kd, Vd] fp32 (recurrent state, updated in-place)
 * @return output [B, 1, Hv, Vd]  bf16
 */
at::Tensor recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state);

/**
 * Multi-token (prefill) gated delta rule — sequential C++ scan.
 *
 * Mathematically identical to the per-token / chunked recurrence run over
 * `T` tokens, advancing the fp32 recurrent state in place. Replaces the pure
 * PyTorch `_chunk_gated_delta_rule` (which materializes large [B,H,T,d] fp32
 * intermediates and many bmm/mul ops). Q/K L2-normalized and Q scaled by
 * 1/sqrt(Kd) internally; GQA handled internally (value head h reads
 * query/key head h / (Hv/Hq)).
 *
 * @param query   [B, T, Hq, Kd]  bf16
 * @param key     [B, T, Hq, Kd]  bf16
 * @param value   [B, T, Hv, Vd]  bf16
 * @param g       [B, T, Hv]      fp32 or bf16 (log-decay)
 * @param beta    [B, T, Hv]      fp32 or bf16 (update gate, already sigmoided)
 * @param state   [B, Hv, Kd, Vd] fp32 (initial state, advanced in-place)
 * @return output [B, T, Hv, Vd]  bf16
 */
at::Tensor chunk_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state);

/**
 * Causal conv1d state update — single-step decode kernel.
 *
 * For seq_len=1 decode: prepends conv_state, applies depthwise conv1d + SiLU.
 * Updates conv_state in-place.
 *
 * @param input      [B, C, 1]           bf16
 * @param conv_state [B, C, kernel-1]    bf16 (updated in-place)
 * @param weight     [C, kernel]         bf16 (depthwise conv weight)
 * @return output    [B, C, 1]           bf16
 */
at::Tensor causal_conv1d_update(
    const at::Tensor& input,
    at::Tensor& conv_state,
    const at::Tensor& weight);

/**
 * Causal depthwise conv1d + SiLU over a whole sequence (prefill and chunked
 * continuation). Token-major so each tap is a contiguous channel vector.
 *
 * @param x       [B, T, C]    bf16, channel-contiguous (token stride free)
 * @param prev    [B, K-1, C]  bf16 cached tokens preceding x, or none (zeros)
 * @param weight  [C, K]       bf16
 * @return        [B, T, C]    bf16
 */
at::Tensor causal_conv1d_prefill(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& prev,
    const at::Tensor& weight);

} // namespace kernels
} // namespace pace

#endif // PACE_GATED_DELTANET_KERNEL_H
