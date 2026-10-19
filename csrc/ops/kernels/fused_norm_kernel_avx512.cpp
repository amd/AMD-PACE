/******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/
/* Inspired by:
 * https://github.com/vllm-project/vllm/blob/a0c70816956298f7dd1d0cf47cfa1a169a413692/csrc/cpu/layernorm.cpp
 * Copyright 2023-2025 The vLLM Authors.
 * Licensed under the Apache License, Version 2.0 */

#include <ATen/ATen.h>
#include <immintrin.h>
#include <omp.h>
#include <ops/exp_approx.h>
#include <ops/kernels/avx512_bf16_utils.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pace {
namespace kernels {
namespace impl {

/**
 * Unified AVX-512 normalization kernel.
 *
 * Template parameters:
 *   IsRMSNorm       true  -> y = x / sqrt(mean(x^2) + eps) * w
 *                   false -> y = (x - mean) / sqrt(var + eps) * w + b
 *   IsFusedResidual true  -> x = input + residual before norm
 *                   false -> x = input directly
 *
 * Two passes per row, all arithmetic stays in fp32:
 *   Pass 1: accumulate stats; if fused, add residual and store the fp32
 *           sum into a thread-local scratch buffer (no bf16 round-trip).
 *   Pass 2: read the fp32 scratch (or original input), normalize, scale,
 *           write bf16 output (+ bf16 residual_out from the scratch).
 */
template <bool IsRMSNorm, bool IsFusedResidual>
static void norm_impl(
    at::BFloat16* __restrict__ out,
    at::BFloat16* __restrict__ res_out,
    const at::BFloat16* __restrict__ input,
    const at::BFloat16* __restrict__ residual,
    const at::BFloat16* __restrict__ weight,
    const at::BFloat16* __restrict__ bias,
    float eps,
    int64_t num_rows,
    int64_t C) {
  const int64_t vec_end = C & ~15LL;
  const float inv_C = 1.0f / static_cast<float>(C);

#pragma omp parallel for schedule(static) if (num_rows > 1)
  for (int64_t i = 0; i < num_rows; ++i) {
    const at::BFloat16* x_row = input + i * C;
    at::BFloat16* o_row = out + i * C;

    // Thread-local fp32 scratch for fused residual sum (avoids bf16 round-trip)
    static thread_local std::vector<float> scratch;
    float* fp32_buf = nullptr;
    if constexpr (IsFusedResidual) {
      if (static_cast<int64_t>(scratch.size()) < C)
        scratch.resize(C);
      fp32_buf = scratch.data();
    }

    const at::BFloat16* r_row = nullptr;
    if constexpr (IsFusedResidual) {
      r_row = residual + i * C;
    }

    // Pass 1: accumulate stats (fused add stays in fp32)
    __m512 var_acc = _mm512_setzero_ps();
    __m512 sum_acc = _mm512_setzero_ps();

    for (int64_t j = 0; j < vec_end; j += 16) {
      __m512 x = load_bf16_fp32(x_row + j);

      if constexpr (IsFusedResidual) {
        x = _mm512_add_ps(x, load_bf16_fp32(r_row + j));
        _mm512_storeu_ps(fp32_buf + j, x);
      }

      if constexpr (!IsRMSNorm) {
        sum_acc = _mm512_add_ps(sum_acc, x);
      }
      var_acc = _mm512_fmadd_ps(x, x, var_acc);
    }

    float variance = _mm512_reduce_add_ps(var_acc);
    float sum = 0.0f;
    if constexpr (!IsRMSNorm) {
      sum = _mm512_reduce_add_ps(sum_acc);
    }

    for (int64_t j = vec_end; j < C; ++j) {
      float v = static_cast<float>(x_row[j]);
      if constexpr (IsFusedResidual) {
        v += static_cast<float>(r_row[j]);
        fp32_buf[j] = v;
      }
      if constexpr (!IsRMSNorm) {
        sum += v;
      }
      variance += v * v;
    }

    // Compute normalization constants
    float inv_denom, mean_val = 0.0f;
    if constexpr (IsRMSNorm) {
      inv_denom = 1.0f / std::sqrt(variance * inv_C + eps);
    } else {
      mean_val = sum * inv_C;
      float var = std::max(0.0f, variance * inv_C - mean_val * mean_val);
      inv_denom = 1.0f / std::sqrt(var + eps);
    }
    __m512 inv_denom_v = _mm512_set1_ps(inv_denom);

    // Pass 2: normalize, scale, write bf16 output
    // For fused: read fp32 scratch, also write bf16 residual_out.
    at::BFloat16* ro_row = IsFusedResidual ? (res_out + i * C) : nullptr;

    if constexpr (IsRMSNorm) {
      for (int64_t j = 0; j < vec_end; j += 16) {
        __m512 val;
        if constexpr (IsFusedResidual) {
          val = _mm512_loadu_ps(fp32_buf + j);
          store_fp32_bf16(ro_row + j, val);
        } else {
          val = load_bf16_fp32(x_row + j);
        }
        __m512 w = load_bf16_fp32(weight + j);
        store_fp32_bf16(
            o_row + j, _mm512_mul_ps(_mm512_mul_ps(val, inv_denom_v), w));
      }
      for (int64_t j = vec_end; j < C; ++j) {
        float v = IsFusedResidual ? fp32_buf[j] : static_cast<float>(x_row[j]);
        if constexpr (IsFusedResidual) {
          ro_row[j] = at::BFloat16(v);
        }
        o_row[j] = at::BFloat16(v * inv_denom * static_cast<float>(weight[j]));
      }
    } else {
      __m512 mean_v = _mm512_set1_ps(mean_val);
      for (int64_t j = 0; j < vec_end; j += 16) {
        __m512 val;
        if constexpr (IsFusedResidual) {
          val = _mm512_loadu_ps(fp32_buf + j);
          store_fp32_bf16(ro_row + j, val);
        } else {
          val = load_bf16_fp32(x_row + j);
        }
        val = _mm512_sub_ps(val, mean_v);
        __m512 w = load_bf16_fp32(weight + j);
        __m512 b = load_bf16_fp32(bias + j);
        store_fp32_bf16(
            o_row + j, _mm512_fmadd_ps(_mm512_mul_ps(val, inv_denom_v), w, b));
      }
      for (int64_t j = vec_end; j < C; ++j) {
        float v = IsFusedResidual ? fp32_buf[j] : static_cast<float>(x_row[j]);
        if constexpr (IsFusedResidual) {
          ro_row[j] = at::BFloat16(v);
        }
        v = (v - mean_val) * inv_denom * static_cast<float>(weight[j]) +
            static_cast<float>(bias[j]);
        o_row[j] = at::BFloat16(v);
      }
    }
  }
}

} // namespace impl

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  const auto orig_shape = x.sizes();
  const int64_t C = orig_shape.back();
  const int64_t num_rows = x.numel() / C;
  auto x_contig = x.contiguous();
  auto out = at::empty({num_rows, C}, x.options());
  impl::norm_impl<true, false>(
      out.data_ptr<at::BFloat16>(),
      nullptr,
      x_contig.data_ptr<at::BFloat16>(),
      nullptr,
      weight.data_ptr<at::BFloat16>(),
      nullptr,
      static_cast<float>(eps),
      num_rows,
      C);
  return out.reshape(orig_shape);
}

at::Tensor rmsnorm_gated(
    const at::Tensor& input,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps) {
  const auto orig_shape = input.sizes();
  const int64_t D = orig_shape.back();
  const int64_t N = input.numel() / D;

  auto x = input.contiguous();
  auto g = gate.contiguous();
  auto output = at::empty({N, D}, input.options());

  const at::BFloat16* __restrict__ x_ptr = x.data_ptr<at::BFloat16>();
  const at::BFloat16* __restrict__ g_ptr = g.data_ptr<at::BFloat16>();
  const at::BFloat16* __restrict__ w_ptr = weight.data_ptr<at::BFloat16>();
  at::BFloat16* __restrict__ o_ptr = output.data_ptr<at::BFloat16>();

  const int64_t vec_end = D & ~15LL;
  const float eps_f = static_cast<float>(eps);

  EXP_APPROX_AVX512_CONSTANTS();
  const __m512 one_v = _mm512_set1_ps(1.0f);
  const __m512 zero_v = _mm512_setzero_ps();
  // The shared exp approximation only clamps its input from below, so
  // exp(-gate) overflows for gate < -88. The gate itself is clamped, not just
  // exp's argument, so SiLU stays ~0 even for huge negative gates (SiLU(-88)
  // is already -5e-37).
  constexpr float kMinSiluArg = -88.0f;
  const __m512 min_silu_arg_v = _mm512_set1_ps(kMinSiluArg);

#pragma omp parallel for schedule(static) if (N > 1)
  for (int64_t i = 0; i < N; ++i) {
    const at::BFloat16* xi = x_ptr + i * D;
    const at::BFloat16* gi = g_ptr + i * D;
    at::BFloat16* oi = o_ptr + i * D;

    // Pass 1: variance
    __m512 var_acc = _mm512_setzero_ps();
    for (int64_t j = 0; j < vec_end; j += 16) {
      __m512 v = impl::load_bf16_fp32(xi + j);
      var_acc = _mm512_fmadd_ps(v, v, var_acc);
    }
    float variance = _mm512_reduce_add_ps(var_acc);
    for (int64_t j = vec_end; j < D; ++j) {
      float v = static_cast<float>(xi[j]);
      variance += v * v;
    }

    float inv_rms = 1.0f / std::sqrt(variance / static_cast<float>(D) + eps_f);
    __m512 inv_rms_v = _mm512_set1_ps(inv_rms);

    // Pass 2: normalize, scale by weight, multiply by SiLU(gate)
    for (int64_t j = 0; j < vec_end; j += 16) {
      __m512 xv = impl::load_bf16_fp32(xi + j);
      __m512 wv = impl::load_bf16_fp32(w_ptr + j);
      __m512 gv = impl::load_bf16_fp32(gi + j);

      // RMSNorm: weight * (x * inv_rms)
      __m512 normed = _mm512_mul_ps(_mm512_mul_ps(xv, inv_rms_v), wv);

      // SiLU(gate) = gate / (1 + exp(-gate)), fully vectorized through the
      // shared exp approximation (same one the attention softmax uses).
      // MAXPS returns its second operand when either is NaN, so the gate goes
      // second to keep a NaN gate NaN.
      const __m512 gc = _mm512_max_ps(min_silu_arg_v, gv);
      __m512 exp_neg_g;
      EXP_APPROX_AVX512(_mm512_sub_ps(zero_v, gc), exp_neg_g);
      __m512 silu_v = _mm512_div_ps(gc, _mm512_add_ps(one_v, exp_neg_g));

      impl::store_fp32_bf16(oi + j, _mm512_mul_ps(normed, silu_v));
    }

    // Scalar tail
    for (int64_t j = vec_end; j < D; ++j) {
      float xv = static_cast<float>(xi[j]);
      float wv = static_cast<float>(w_ptr[j]);
      float gv = static_cast<float>(gi[j]);
      float normed = xv * inv_rms * wv;
      const float gc = std::max(gv, kMinSiluArg);
      float exp_neg_g;
      EXP_APPROX_SCALAR(-gc, exp_neg_g);
      oi[j] = at::BFloat16(normed * gc / (1.0f + exp_neg_g));
    }
  }

  return output.reshape(orig_shape);
}

std::tuple<at::Tensor, at::Tensor> fused_add_rmsnorm(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double eps) {
  const auto orig_shape = x.sizes();
  const int64_t C = orig_shape.back();
  const int64_t num_rows = x.numel() / C;
  auto x_contig = x.contiguous();
  auto res_contig = residual.contiguous();
  auto out = at::empty({num_rows, C}, x.options());
  auto res_out = at::empty({num_rows, C}, x.options());
  impl::norm_impl<true, true>(
      out.data_ptr<at::BFloat16>(),
      res_out.data_ptr<at::BFloat16>(),
      x_contig.data_ptr<at::BFloat16>(),
      res_contig.data_ptr<at::BFloat16>(),
      weight.data_ptr<at::BFloat16>(),
      nullptr,
      static_cast<float>(eps),
      num_rows,
      C);
  return {out.reshape(orig_shape), res_out.reshape(orig_shape)};
}

at::Tensor layernorm(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  const auto orig_shape = x.sizes();
  const int64_t C = orig_shape.back();
  const int64_t num_rows = x.numel() / C;
  auto x_contig = x.contiguous();
  auto out = at::empty({num_rows, C}, x.options());
  impl::norm_impl<false, false>(
      out.data_ptr<at::BFloat16>(),
      nullptr,
      x_contig.data_ptr<at::BFloat16>(),
      nullptr,
      weight.data_ptr<at::BFloat16>(),
      bias.data_ptr<at::BFloat16>(),
      static_cast<float>(eps),
      num_rows,
      C);
  return out.reshape(orig_shape);
}

std::tuple<at::Tensor, at::Tensor> fused_add_layernorm(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  const auto orig_shape = x.sizes();
  const int64_t C = orig_shape.back();
  const int64_t num_rows = x.numel() / C;
  auto x_contig = x.contiguous();
  auto res_contig = residual.contiguous();
  auto out = at::empty({num_rows, C}, x.options());
  auto res_out = at::empty({num_rows, C}, x.options());
  impl::norm_impl<false, true>(
      out.data_ptr<at::BFloat16>(),
      res_out.data_ptr<at::BFloat16>(),
      x_contig.data_ptr<at::BFloat16>(),
      res_contig.data_ptr<at::BFloat16>(),
      weight.data_ptr<at::BFloat16>(),
      bias.data_ptr<at::BFloat16>(),
      static_cast<float>(eps),
      num_rows,
      C);
  return {out.reshape(orig_shape), res_out.reshape(orig_shape)};
}

} // namespace kernels
} // namespace pace
