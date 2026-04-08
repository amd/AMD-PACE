/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef AOCL_DLP_LINEAR_H
#define AOCL_DLP_LINEAR_H

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
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */

at::Tensor aocl_dlp_linear_plain(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with GELU activation
 *
 * Computes the output of a fully connected layer followed by the GELU
 * activation function, optionally adding a bias tensor.
 *
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */

at::Tensor aocl_dlp_linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with SiLU activation
 *
 * Computes the output of a fully connected layer followed by the SiLU (Sigmoid
 * Linear Unit) activation function, optionally adding a bias tensor.
 *
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */

at::Tensor aocl_dlp_linear_silu(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with ReLU activation
 *
 * Computes the output of a fully connected layer followed by the ReLU
 * activation function, optionally adding a bias tensor.
 *
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */
at::Tensor aocl_dlp_linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Performs a fully connected layer operation with an additional
 * multiplication step
 *
 * Computes the output of a fully connected layer by performing matrix
 * multiplication between the input tensor and a weight tensor, followed
 * by multiplication with a multiplier tensor. Optionally, a bias tensor can
 * be added.
 *
 * @param input Input tensor
 * @param multiplier multiplier tensor
 * @param weight Weight tensor
 * @param bias_opt Optional bias tensor
 * @return Output tensor
 */
at::Tensor aocl_dlp_linear_mul(
    const at::Tensor& input,
    const at::Tensor& multiplier,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt);

/**
 * @brief Reshapes weight tensor for AOCL-DLP operations
 *
 * Uses AOCL-DLP's aocl_dlp_reshape_bf16 kernel to reorder and transpose
 * the weight tensor from PyTorch format [N, K] to AOCL-DLP optimized format [K,
 * N]. The output buffer is allocated by AOCL-DLP and managed by PyTorch. Only
 * supports 2D weight tensors in bfloat16 format.
 *
 * @param weight Weight tensor to reshape (must be 2D and bfloat16)
 * @return Reshaped weight tensor in AOCL-DLP optimized format
 */
at::Tensor aocl_dlp_reshape_weights(const at::Tensor& weight);

} // namespace pace

#endif // AOCL_DLP_LINEAR_H
