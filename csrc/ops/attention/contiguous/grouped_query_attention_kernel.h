/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef PACE_GQA_KERNEL_H
#define PACE_GQA_KERNEL_H

#include <ATen/ATen.h>

namespace pace {
namespace kernels {

/**
 * @brief Grouped Query Attention - GQA kernel implementation
 * GQA supported for F32 and BF16 input data types
 *
 * @param query
 * @param key
 * @param value
 * @param output
 * @param attn_mask
 */

void grouped_query_attention_kernel(
    at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor> attn_mask);

} // namespace kernels
} // namespace pace
#endif // PACE_GQA_KERNEL_H
