/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef GATED_DELTANET_H
#define GATED_DELTANET_H

#include <ATen/ATen.h>

namespace pace {

at::Tensor recurrent_gated_delta_rule_op(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state);

at::Tensor chunk_gated_delta_rule_op(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state);

at::Tensor causal_conv1d_update_op(
    const at::Tensor& input,
    at::Tensor& conv_state,
    const at::Tensor& weight);

std::tuple<at::Tensor, at::Tensor> causal_conv1d_prefill_op(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& conv_state,
    const at::Tensor& weight);

} // namespace pace

#endif // GATED_DELTANET_H
