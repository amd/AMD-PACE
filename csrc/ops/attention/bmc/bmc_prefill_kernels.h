/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * BMC BRGeMM tiled prefill kernel declarations.
 *
 * Separated from slab kernels because BMC uses:
 *   - Separate K and V tensors (not interleaved in a slab pool)
 *   - Head-major Q/O layout with flexible strides
 *
 * Reuses slab tiling constants and dpbf16 SIMD primitives.
 ******************************************************************************/

#ifndef PACE_BMC_PREFILL_KERNELS_H
#define PACE_BMC_PREFILL_KERNELS_H

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

namespace pace {
namespace kernels {

constexpr int64_t BMC_Q_TILE = 64;
constexpr int64_t BMC_N_MAX = 64;
constexpr int64_t BMC_MAX_REP = 16;

// BmcKernelCtx: per-call constants for BMC prefill.
//
// K is at: k_ptr + kv_h * kv_head_stride + block * blk_size * head_dim
// V is at: v_ptr + kv_h * kv_head_stride + block * blk_size * head_dim
//
// Q is at: q_ptr + qh * q_head_stride + token * q_token_stride
// O is at: o_ptr + qh * q_head_stride + token * q_token_stride
struct BmcKernelCtx {
  const at::BFloat16* k_ptr;
  const at::BFloat16* v_ptr;
  const at::BFloat16* q_ptr;
  at::BFloat16* o_ptr;
  int64_t num_q_heads, head_dim, num_kv, n_rep, blk_size;
  int64_t kv_head_stride; // elements between KV heads
  int64_t q_head_stride; // elements between Q/O heads
  int64_t q_token_stride; // elements between Q/O tokens
  float scale_f;
};

// BRGeMM tiled prefill for one (kv_head, q_tile) work item.
// Manages its own thread-local BrgemmCache internally.
void bmc_prefill_tile(
    const BmcKernelCtx& ctx,
    int64_t kv_len,
    int64_t q_len,
    int64_t q_offset,
    const std::vector<int64_t>& block_indices,
    int64_t kv_h,
    int64_t qt);

} // namespace kernels
} // namespace pace

#endif // PACE_BMC_PREFILL_KERNELS_H
