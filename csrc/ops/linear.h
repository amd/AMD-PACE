/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LINEAR_H
#define LINEAR_H

#include <ATen/ATen.h>
#include <ops/onednn.h>
#include <ops/onednn_utils.h>

namespace pace {

/**
 * @brief Get the bias memory if available
 *
 * @param bias The bias tensor to be parsed
 * @return memory
 */
inline memory get_bias_memory(const at::Tensor& bias) {
  memory bias_mem = memory({}, cpu_eng(), DNNL_MEMORY_NONE);
  if (bias.dim() != 0) {
    bias_mem = view_tensor_as_memory(
        bias, /*transpose*/ transpose_dims(), /*ndims*/ 2);
  }
  return bias_mem;
}

/**
 * @brief fp32 -> fp32 and bf16 -> bf16 kernel for 2D matrix multiplication
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @return at::Tensor
 */
at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

/**
 * @brief fp32 -> fp32 and bf16 -> bf16 kernel for 2D matrix multiplication
 * with post-ops: ReLU
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @return at::Tensor
 */
at::Tensor linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

} // namespace pace

#endif // LINEAR_H
