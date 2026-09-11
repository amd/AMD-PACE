// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

// Compiled with AVX-512 flags (filename ends in _avx512.cpp).
// SIMD scatter-accumulate for FusedMoE Phase 3.
// Uses dpbf16_kernels.h bf16_to_fp32 (single _mm512_cvtpbh_ps instruction).

#include <immintrin.h>
#include "ops/attention/slab/dpbf16_kernels.h"

namespace pace {
namespace kernels {

void moe_scatter_accumulate_bf16(
    at::BFloat16* dst,
    const at::BFloat16* src,
    float weight,
    int64_t n) {
  __m512 w_vec = _mm512_set1_ps(weight);
  int64_t j = 0;

  for (; j + 16 <= n; j += 16) {
    // dpbf16::detail::bf16_to_fp32: single _mm512_cvtpbh_ps instruction
    __m512 dst_fp32 =
        dpbf16::detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(dst + j)));
    __m512 src_fp32 =
        dpbf16::detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(src + j)));
    dst_fp32 = _mm512_fmadd_ps(w_vec, src_fp32, dst_fp32);
    _mm256_storeu_si256(
        (__m256i*)(dst + j), (__m256i)_mm512_cvtneps_pbh(dst_fp32));
  }

  for (; j < n; j++) {
    float val =
        static_cast<float>(dst[j]) + static_cast<float>(src[j]) * weight;
    dst[j] = static_cast<at::BFloat16>(val);
  }
}

} // namespace kernels
} // namespace pace
