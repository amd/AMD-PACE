// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#ifndef LIBXSMM_MOE_H
#define LIBXSMM_MOE_H

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <string>
#include <tuple>

namespace pace {

/**
 * @brief Standalone top-k softmax router.
 *
 * Computes top-k expert selection + softmax over the selected logits, returning
 * the routing decision as tensors for consumption by libxsmm_moe_engine.
 *
 * @param router_logits  Router output [num_tokens, num_experts] (bf16/fp32)
 * @param topk           Number of experts to select per token
 * @return tuple (topk_ids [num_tokens, topk] int64,
 *                topk_weights [num_tokens, topk] float32)
 */
std::tuple<at::Tensor, at::Tensor> moe_route_softmax(
    const at::Tensor& router_logits,
    int64_t topk);
std::tuple<at::Tensor, at::Tensor> moe_route_softmax(
    const at::Tensor& router_logits,
    int64_t topk,
    bool renormalize);

/**
 * @brief Routing-agnostic MoE expert engine.
 *
 * Consumes a pre-computed routing decision (topk_ids / topk_weights) and runs
 * the expert FFN: gather -> gate/up + activation -> down -> weighted scatter.
 * Shared experts, if any, are encoded directly as additional columns in
 * topk_ids/topk_weights by the caller (no special handling here).
 *
 * @param output         Pre-allocated output [num_tokens, hidden_size] (bf16),
 *                       initialized by caller (zeros or shared-expert result).
 * @param hidden_states  Input [num_tokens, hidden_size] (bf16)
 * @param topk_ids       Selected expert ids [num_tokens, topk] (int64)
 * @param topk_weights   Scatter weights [num_tokens, topk] (float32/bf16)
 * @param wt_gate        Optional gate weights [E, Nk, Nc, 32, Hk, 2]
 * @param wt_up          Up-projection weights [E, Nk, Nc, 32, Hk, 2]
 * @param wt_down        Down-projection weights [E, Nk_d, Nc_d, 32, Hk_d, 2]
 * @param gate_bias      Optional gate bias [E, intermediate_size]
 * @param up_bias        Optional up bias [E, intermediate_size]
 * @param down_bias      Optional down bias [E, hidden_size]
 * @param activation     One of "silu", "gelu", "relu", "gptoss_glu"
 * @param num_experts    Total experts in the weight stack
 * Void in-place op (schema `-> ()`): writes the result into `output`.
 */
void libxsmm_moe_engine(
    at::Tensor& output,
    const at::Tensor& hidden_states,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    const c10::optional<at::Tensor>& wt_gate,
    const at::Tensor& wt_up,
    const at::Tensor& wt_down,
    const c10::optional<at::Tensor>& gate_bias,
    const c10::optional<at::Tensor>& up_bias,
    const c10::optional<at::Tensor>& down_bias,
    const std::string& activation,
    int64_t num_experts);

} // namespace pace

#endif // LIBXSMM_MOE_H
