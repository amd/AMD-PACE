/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LIBXSMM_MLP_H
#define LIBXSMM_MLP_H

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace pace {

/**
 * @brief Performs a fully connected layer operation with optional bias
 *
 * Computes the output of a fully connected layer by performing matrix
 * multiplication between the input tensor and the weight tensor, optionally
 * adding a bias tensor.
 *
 * @tparam T Data type of the input, weight, and output tensors
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */

at::Tensor libxsmmlinear_plain(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with GELU activation
 *
 * Computes the output of a fully connected layer followed by the GELU
 * activation function, optionally adding a bias tensor.
 *
 * @tparam T Data type of the input, weight, and output tensors
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */

at::Tensor libxsmmlinear_gelu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with SiLU activation
 *
 * Computes the output of a fully connected layer followed by the SiLU (Sigmoid
 * Linear Unit) activation function, optionally adding a bias tensor.
 *
 * @tparam T Data type of the input, weight, and output tensors
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */

at::Tensor libxsmmlinear_silu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with ReLU activation
 *
 * Computes the output of a fully connected layer followed by the ReLU
 * activation function, optionally adding a bias tensor.
 *
 * @tparam T Data type of the input, weight, and output tensors
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */
at::Tensor libxsmmlinear_relu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with an additional
 * multiplication step
 *
 * Computes the output of a fully connected layer by performing matrix
 * multiplication between the input tensor and an multiplier tensor, followed
 * by another multiplication with a weight tensor. Optionally, a bias tensor can
 * be added.
 *
 * @param input Input tensor
 * @param multiplier multiplier tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */
at::Tensor libxsmmlinear_mul(
    at::Tensor& input,
    at::Tensor& multiplier,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Fused MLP: gate+activation+up+mul+down in a single OMP region.
 *
 * Performs all three MLP projections (gate, up, down) fused into one call,
 * reducing OMP region overhead and enabling L1 tile reuse.
 *
 * @param input      Input tensor [B, S, C] or [S, C] (bfloat16)
 * @param wt_gate    Optional gate weight (packed 5D)
 * @param wt_up      Up-projection weight (packed 5D)
 * @param wt_down    Down-projection weight (packed 5D)
 * @param gate_bias  Optional gate bias
 * @param up_bias    Optional up-projection bias
 * @param down_bias  Optional down-projection bias
 * @param activation One of "silu", "gelu", "relu"
 * @return Output tensor with same batch dims as input
 */
at::Tensor libxsmm_fused_mlp(
    const at::Tensor& input,
    c10::optional<at::Tensor> wt_gate,
    const at::Tensor& wt_up,
    const at::Tensor& wt_down,
    c10::optional<at::Tensor> gate_bias,
    c10::optional<at::Tensor> up_bias,
    c10::optional<at::Tensor> down_bias,
    const std::string& activation);

} // namespace pace

#endif // LIBXSMM_MLP_H
