/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/
#ifndef PACE_ATTENTION_KERNEL_H
#define PACE_ATTENTION_KERNEL_H

#include <ATen/ATen.h>
#include <ops/jit_helper.h>

namespace pace {

namespace kernels {

/**
 * @brief MHA kernel implementation with 4 different approaches
 * MHA supported for F32 and BF16 input data types
 *
 * @param input_Q_mem
 * @param input_K_mem
 * @param input_V_mem
 * @param output_mem
 * @param input_mask_mem
 * @param use_KQ
 */
void multi_head_attention_kernel(
    at::Tensor& output,
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const int& use_KQ,
    const c10::optional<at::Tensor>& input_mask);

} // namespace kernels

} // namespace pace

#endif // PACE_ATTENTION_KERNEL_H
