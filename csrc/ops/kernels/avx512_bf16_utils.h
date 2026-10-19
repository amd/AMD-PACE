/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Register-level bf16 <-> fp32 conversion helpers shared by the AVX-512
 * kernels.
 *
 * REQUIRES: -mavx512f -mavx512bf16 compiler flags, so include this only from
 * *_avx512.cpp translation units.
 ******************************************************************************/

#ifndef PACE_AVX512_BF16_UTILS_H
#define PACE_AVX512_BF16_UTILS_H

#include <ATen/ATen.h>
#include <immintrin.h>
#include <cstdint>

namespace pace {
namespace kernels {
namespace impl {

// Load 16 bf16 values and widen them to fp32.
inline __m512 load_bf16_fp32(const at::BFloat16* p) {
  __m256bh raw =
      (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_cvtpbh_ps(raw);
}

// Round 16 fp32 values to bf16 (round-to-nearest-even) and store them.
inline void store_fp32_bf16(at::BFloat16* p, __m512 v) {
  __m256bh packed = _mm512_cvtneps_pbh(v);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), (__m256i)packed);
}

// Widen `dim` bf16 values from `src` into the fp32 buffer `dst`.
inline void load_bf16_to_fp32(
    float* dst,
    const at::BFloat16* src,
    int64_t dim) {
  const int64_t vec_end = dim & ~15LL;
  for (int64_t j = 0; j < vec_end; j += 16) {
    _mm512_storeu_ps(dst + j, load_bf16_fp32(src + j));
  }
  for (int64_t j = vec_end; j < dim; ++j) {
    dst[j] = static_cast<float>(src[j]);
  }
}

} // namespace impl
} // namespace kernels
} // namespace pace

#endif // PACE_AVX512_BF16_UTILS_H
