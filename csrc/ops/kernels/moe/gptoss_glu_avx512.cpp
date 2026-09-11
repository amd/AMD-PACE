// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

// Compiled with AVX-512 flags (filename ends in _avx512.cpp).
// Vectorized GptOss GLU: gate.clamp(7)*sigmoid(gate*1.702),
// (up.clamp(±7)+1)*glu Processes 16 BF16 elements per iteration using AVX-512 +
// EXP_APPROX_AVX512.

#include <ATen/ATen.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include "ops/attention/slab/dpbf16_kernels.h"
#include "ops/exp_approx.h"

namespace pace {
namespace kernels {

static constexpr float GPTOSS_ALPHA = 1.702f;
static constexpr float GPTOSS_LIMIT = 7.0f;

void gptoss_glu_activation(
    at::BFloat16* tile_gate,
    at::BFloat16* tile_up,
    long rows,
    long cols,
    long ld) {
  EXP_APPROX_AVX512_CONSTANTS();
  const __m512 v_alpha = _mm512_set1_ps(GPTOSS_ALPHA);
  const __m512 v_one = _mm512_set1_ps(1.0f);
  const __m512 v_limit = _mm512_set1_ps(GPTOSS_LIMIT);
  const __m512 v_neglimit = _mm512_set1_ps(-GPTOSS_LIMIT);

  for (long r = 0; r < rows; r++) {
    at::BFloat16* gp = tile_gate + r * ld;
    at::BFloat16* up = tile_up + r * ld;
    long c = 0;

    for (; c + 16 <= cols; c += 16) {
      // Load gate and up as BF16, convert to FP32
      __m512 g = dpbf16::detail::bf16_to_fp32(
          _mm256_loadu_si256((const __m256i*)(gp + c)));
      __m512 u = dpbf16::detail::bf16_to_fp32(
          _mm256_loadu_si256((const __m256i*)(up + c)));

      // gate: clamp(max=7)
      g = _mm512_min_ps(g, v_limit);
      // up: clamp(min=-7, max=7)
      u = _mm512_max_ps(u, v_neglimit);
      u = _mm512_min_ps(u, v_limit);

      // sigmoid(g * alpha) = 1 / (1 + exp(-g * alpha))
      __m512 neg_g_alpha =
          _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), g), v_alpha);
      __m512 exp_val;
      EXP_APPROX_AVX512(neg_g_alpha, exp_val);
      __m512 sigmoid_val = _mm512_div_ps(v_one, _mm512_add_ps(v_one, exp_val));

      // glu = g * sigmoid(g * alpha)
      __m512 glu = _mm512_mul_ps(g, sigmoid_val);

      // result = (u + 1) * glu — stored back into gate tile
      __m512 result = _mm512_mul_ps(_mm512_add_ps(u, v_one), glu);

      // Store as BF16 back into gate tile
      _mm256_storeu_si256(
          (__m256i*)(gp + c), (__m256i)_mm512_cvtneps_pbh(result));
    }

    // Scalar tail
    for (; c < cols; c++) {
      float g = static_cast<float>(gp[c]);
      float u = static_cast<float>(up[c]);
      g = std::min(g, GPTOSS_LIMIT);
      u = std::max(-GPTOSS_LIMIT, std::min(u, GPTOSS_LIMIT));
      float exp_val;
      EXP_APPROX_SCALAR(-g * GPTOSS_ALPHA, exp_val);
      float glu = g / (1.0f + exp_val);
      gp[c] = static_cast<at::BFloat16>((u + 1.0f) * glu);
    }
  }
}

} // namespace kernels
} // namespace pace
