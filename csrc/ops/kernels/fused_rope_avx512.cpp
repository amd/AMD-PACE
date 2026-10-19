/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ATen/ATen.h>
#include <immintrin.h>
#include <omp.h>
#include <ops/kernels/avx512_bf16_utils.h>
#include "ops/kernels/fused_rope.h"

namespace pace {

namespace kernels {

namespace impl {

static at::Tensor fused_rope_apply_impl(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const bool is_bnsh) {
  const int64_t BS = x.size(0);
  const int64_t dim1 = x.size(1);
  const int64_t dim2 = x.size(2);
  const int64_t head_dim = x.size(3);
  const int64_t half = head_dim / 2;
  const int64_t seq_len = is_bnsh ? dim2 : dim1;

  auto x_contig = x.contiguous();
  auto cos_contig = cos.contiguous();
  auto sin_contig = sin.contiguous();

  auto output = at::empty_like(x_contig);

  const at::BFloat16* x_ptr = x_contig.data_ptr<at::BFloat16>();
  const at::BFloat16* cos_ptr = cos_contig.data_ptr<at::BFloat16>();
  const at::BFloat16* sin_ptr = sin_contig.data_ptr<at::BFloat16>();
  at::BFloat16* out_ptr = output.data_ptr<at::BFloat16>();

  const int64_t cs_seq_stride = half;
  const int64_t cs_batch_stride = seq_len * half;
  const int64_t total_rows = BS * dim1 * dim2;

  constexpr int64_t VEC_WIDTH = 16; // 512 bits / 32 bits per float

#pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < total_rows; ++idx) {
    const int64_t b = idx / (dim1 * dim2);
    const int64_t s =
        is_bnsh ? (idx % seq_len) : ((idx % (dim1 * dim2)) / dim2);

    const at::BFloat16* x_row = x_ptr + idx * head_dim;
    at::BFloat16* o_row = out_ptr + idx * head_dim;
    const at::BFloat16* c_row =
        cos_ptr + b * cs_batch_stride + s * cs_seq_stride;
    const at::BFloat16* s_row =
        sin_ptr + b * cs_batch_stride + s * cs_seq_stride;

    int64_t i = 0;
    for (; i + VEC_WIDTH <= half; i += VEC_WIDTH) {
      __m512 vx1 = load_bf16_fp32(x_row + i);
      __m512 vx2 = load_bf16_fp32(x_row + i + half);
      __m512 vc = load_bf16_fp32(c_row + i);
      __m512 vs = load_bf16_fp32(s_row + i);

      // out[i]        = x1 * cos - x2 * sin
      // out[i + half] = x2 * cos + x1 * sin
      __m512 o1 = _mm512_fmsub_ps(vx1, vc, _mm512_mul_ps(vx2, vs));
      __m512 o2 = _mm512_fmadd_ps(vx1, vs, _mm512_mul_ps(vx2, vc));

      store_fp32_bf16(o_row + i, o1);
      store_fp32_bf16(o_row + i + half, o2);
    }

    for (; i < half; ++i) {
      float x1 = static_cast<float>(x_row[i]);
      float x2 = static_cast<float>(x_row[i + half]);
      float c = static_cast<float>(c_row[i]);
      float sv = static_cast<float>(s_row[i]);

      o_row[i] = at::BFloat16(x1 * c - x2 * sv);
      o_row[i + half] = at::BFloat16(x2 * c + x1 * sv);
    }
  }

  return output;
}

} // namespace impl

/**
 * Fused Rotary Position Embedding (RoPE) kernel.
 *
 * Applies RoPE to Q and K in a single pass per tensor using AVX-512
 * intrinsics, avoiding the intermediate tensor allocations of the
 * Python chunk/mul/cat approach.
 *
 * Supports both BNSH [BS, num_heads, seq_len, head_dim] and
 *                BSNH [BS, seq_len, num_heads, head_dim] layouts.
 *
 * Neox-style layout: x = [x1 | x2] where x1 = x[..., :half], x2 = x[...,
 * half:]
 *   out[..., i]        = x1[i] * cos[i] - x2[i] * sin[i]
 *   out[..., i + half] = x2[i] * cos[i] + x1[i] * sin[i]
 */
std::tuple<at::Tensor, at::Tensor> fused_rope_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const int64_t unsqueeze_dim) {
  auto cos_u = cos.unsqueeze(unsqueeze_dim);
  auto sin_u = sin.unsqueeze(unsqueeze_dim);

  // unsqueeze_dim==1 → BNSH [BS, num_heads, seq_len, head_dim]
  // unsqueeze_dim==2 → BSNH [BS, seq_len, num_heads, head_dim]
  const bool is_bnsh = (unsqueeze_dim == 1);

  auto q_out = impl::fused_rope_apply_impl(query, cos_u, sin_u, is_bnsh);
  auto k_out = impl::fused_rope_apply_impl(key, cos_u, sin_u, is_bnsh);

  return {q_out, k_out};
}

} // namespace kernels
} // namespace pace
