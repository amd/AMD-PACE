/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef ROPE_H
#define ROPE_H

#include <ATen/ATen.h>

namespace pace {

/**
 * Fused Rotary Position Embedding (RoPE) for query and key tensors.
 *
 * Applies neox-style RoPE to both Q and K in a single OMP-parallel pass
 * per tensor, avoiding 6 intermediate tensor allocations per Q/K that
 * the Python chunk/mul/cat approach requires.
 *
 * @param query         Query tensor, 4D BFloat16
 * @param key           Key tensor, 4D BFloat16
 * @param cos           Cosine tensor [BS, seq_len, head_dim // 2]
 * @param sin           Sine tensor [BS, seq_len, head_dim // 2]
 * @param unsqueeze_dim 1 for BNSH layout, 2 for BSNH layout
 * @return (query_out, key_out) with RoPE applied
 */
std::tuple<at::Tensor, at::Tensor> fused_rope_apply(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t unsqueeze_dim);

} // namespace pace

#endif // ROPE_H
