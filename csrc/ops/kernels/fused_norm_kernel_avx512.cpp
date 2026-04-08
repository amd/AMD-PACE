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
#include <cmath>
#include <cstdint>
#include <vector>

namespace pace {
namespace kernels {
namespace impl {

static inline __m512 load_bf16_fp32(const at::BFloat16* p) {
  __m256bh raw =
      (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_cvtpbh_ps(raw);
}

static inline void store_fp32_bf16(at::BFloat16* p, __m512 v) {
  __m256bh packed = _mm512_cvtneps_pbh(v);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), (__m256i)packed);
}

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
