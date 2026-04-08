/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * DPBF16 Optimized Kernels for SlabPool Attention
 *
 * AVX512-BF16 (DPBF16) optimized implementations:
 * - Dot product: BF16 input, FP32 accumulation, FP32 output
 * - Weighted accumulation: BF16 input, FP32 accumulation, FP32 output
 *
 * DPBF16 instruction (_mm512_dpbf16_ps):
 *   acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]
 *   Processes 32 BF16 elements -> 16 FP32 results per instruction
 *   2x throughput vs manual BF16->FP32 conversion + FMA
 *
 * Supported: AMD Zen4+ (EPYC Genoa/Turin), Intel Sapphire Rapids+
 * REQUIRES: -mavx512bf16 compiler flag. No fallback.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <immintrin.h>
#include <ops/exp_approx.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef __AVX512BF16__
#error "dpbf16_kernels.h requires AVX512-BF16. Compile with -mavx512bf16 flag."
#endif

namespace pace {
namespace kernels {
namespace dpbf16 {

namespace detail {

inline __m512 bf16_to_fp32(__m256i bf16) {
  return _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(bf16));
}

} // namespace detail

// QK: Dot product (DPBF16 for 32-element chunks, FMA for 16-element tail)
inline float dot_product(
    const at::BFloat16* a,
    const at::BFloat16* b,
    int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;

  for (; i + 32 <= n; i += 32) {
    __m512bh a_bf16 = (__m512bh)_mm512_loadu_si512(a + i);
    __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(b + i);
    acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
  }

  if (i + 16 <= n) {
    __m256i a_bf16 = _mm256_loadu_si256((__m256i*)(a + i));
    __m256i b_bf16 = _mm256_loadu_si256((__m256i*)(b + i));
    __m512 a_fp32 = detail::bf16_to_fp32(a_bf16);
    __m512 b_fp32 = detail::bf16_to_fp32(b_bf16);
    acc = _mm512_fmadd_ps(a_fp32, b_fp32, acc);
    i += 16;
  }

  float result = _mm512_reduce_add_ps(acc);

  for (; i < n; ++i) {
    result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }

  return result;
}

// SV: Weighted V accumulation (output += weight * V[t] for each token)
inline void accumulate_weighted(
    float* output,
    const at::BFloat16* v,
    float weight,
    int64_t n) {
  __m512 w_vec = _mm512_set1_ps(weight);
  int64_t i = 0;

  for (; i + 16 <= n; i += 16) {
    __m256i v_bf16 = _mm256_loadu_si256((__m256i*)(v + i));
    __m512 v_fp32 = detail::bf16_to_fp32(v_bf16);
    __m512 out_vec = _mm512_loadu_ps(output + i);
    out_vec = _mm512_fmadd_ps(w_vec, v_fp32, out_vec);
    _mm512_storeu_ps(output + i, out_vec);
  }

  for (; i < n; ++i) {
    output[i] += weight * static_cast<float>(v[i]);
  }
}

// Batched: accumulate 4 weighted V rows per iteration to reduce
// output load/store traffic by 4x vs per-token accumulation.
inline void accumulate_weighted_4x(
    float* __restrict__ output,
    const at::BFloat16* v_base,
    const float* weights,
    int64_t n_tokens,
    int64_t head_dim) {
  int64_t t = 0;
  for (; t + 4 <= n_tokens; t += 4) {
    const __m512 vw0 = _mm512_set1_ps(weights[t + 0]);
    const __m512 vw1 = _mm512_set1_ps(weights[t + 1]);
    const __m512 vw2 = _mm512_set1_ps(weights[t + 2]);
    const __m512 vw3 = _mm512_set1_ps(weights[t + 3]);
    const at::BFloat16* v0 = v_base + (t + 0) * head_dim;
    const at::BFloat16* v1 = v_base + (t + 1) * head_dim;
    const at::BFloat16* v2 = v_base + (t + 2) * head_dim;
    const at::BFloat16* v3 = v_base + (t + 3) * head_dim;
    int64_t d = 0;
    for (; d + 16 <= head_dim; d += 16) {
      __m512 out = _mm512_loadu_ps(output + d);
      out = _mm512_fmadd_ps(
          vw0,
          detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(v0 + d))),
          out);
      out = _mm512_fmadd_ps(
          vw1,
          detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(v1 + d))),
          out);
      out = _mm512_fmadd_ps(
          vw2,
          detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(v2 + d))),
          out);
      out = _mm512_fmadd_ps(
          vw3,
          detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(v3 + d))),
          out);
      _mm512_storeu_ps(output + d, out);
    }
    for (; d < head_dim; ++d) {
      output[d] += weights[t + 0] * static_cast<float>(v0[d]) +
          weights[t + 1] * static_cast<float>(v1[d]) +
          weights[t + 2] * static_cast<float>(v2[d]) +
          weights[t + 3] * static_cast<float>(v3[d]);
    }
  }
  for (; t < n_tokens; ++t) {
    accumulate_weighted(output, v_base + t * head_dim, weights[t], head_dim);
  }
}

// Template-based AVX-512 copy (replaces copy_256B + memcpy fallback)
// N = number of 64-byte (512-bit) chunks. Guaranteed unrolled at compile time.
// head_dim=64 → N=2, head_dim=128 → N=4, head_dim=256 → N=8
template <int N>
static inline void copy_avx512(
    void* __restrict__ dst,
    const void* __restrict__ src) {
  auto* d = reinterpret_cast<__m512i*>(dst);
  const auto* s = reinterpret_cast<const __m512i*>(src);
  for (int i = 0; i < N; ++i)
    _mm512_storeu_si512(d + i, _mm512_loadu_si512(s + i));
}

// Dispatch helper: switch on head_dim at runtime, call template at compile
// time. The switch is evaluated once per call; branch predictor resolves
// immediately since head_dim is constant across all calls for a given model.
static inline void copy_bf16_avx512(
    void* __restrict__ dst,
    const void* __restrict__ src,
    int64_t head_dim) {
  switch (head_dim) {
    case 32:
      copy_avx512<1>(dst, src);
      break;
    case 64:
      copy_avx512<2>(dst, src);
      break;
    case 128:
      copy_avx512<4>(dst, src);
      break;
    case 256:
      copy_avx512<8>(dst, src);
      break;
    case 512:
      copy_avx512<16>(dst, src);
      break;
    default:
      std::memcpy(dst, src, head_dim * sizeof(at::BFloat16));
      break;
  }
}

// Template-based weighted V accumulation (replaces regblock + fallback)
// N_CHUNKS = head_dim / 16 (number of 16-element FP32 chunks).
// Output accumulators live in registers across all tokens, loaded once,
// stored once. Guaranteed unrolled at compile time.
// head_dim=64 → N_CHUNKS=4, head_dim=128 → N_CHUNKS=8, head_dim=256 →
// N_CHUNKS=16
template <int N_CHUNKS>
static inline void accumulate_weighted_regblock_t(
    float* __restrict__ output,
    const at::BFloat16* v_base,
    const float* weights,
    int64_t n_tokens,
    int64_t head_dim) {
  // Load all accumulator chunks from output
  __m512 o[N_CHUNKS];
  for (int c = 0; c < N_CHUNKS; ++c)
    o[c] = _mm512_loadu_ps(output + c * 16);

  // Accumulate weighted V for each token
  for (int64_t t = 0; t < n_tokens; ++t) {
    const __m512 w = _mm512_set1_ps(weights[t]);
    const at::BFloat16* vt = v_base + t * head_dim;
    for (int c = 0; c < N_CHUNKS; ++c)
      o[c] = _mm512_fmadd_ps(
          w,
          detail::bf16_to_fp32(_mm256_loadu_si256((__m256i*)(vt + c * 16))),
          o[c]);
  }

  // Store accumulators back
  for (int c = 0; c < N_CHUNKS; ++c)
    _mm512_storeu_ps(output + c * 16, o[c]);
}

// Dispatch helper: switch on head_dim, call the right template instantiation.
inline void accumulate_weighted_regblock(
    float* __restrict__ output,
    const at::BFloat16* v_base,
    const float* weights,
    int64_t n_tokens,
    int64_t head_dim) {
  switch (head_dim) {
    case 64:
      accumulate_weighted_regblock_t<4>(
          output, v_base, weights, n_tokens, head_dim);
      break;
    case 128:
      accumulate_weighted_regblock_t<8>(
          output, v_base, weights, n_tokens, head_dim);
      break;
    case 256:
      accumulate_weighted_regblock_t<16>(
          output, v_base, weights, n_tokens, head_dim);
      break;
    default:
      accumulate_weighted_4x(output, v_base, weights, n_tokens, head_dim);
      break;
  }
}

} // namespace dpbf16
} // namespace kernels
} // namespace pace
