/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ops/attention/slab/dpbf16_kernels.h>
#include <ops/attention/slab/slab_kernels.h>
#include <ops/exp_approx.h>
#include <cstring>
#include <limits>

#ifndef __AVX512F__
#error "slab_decode_avx512.cpp requires AVX512F."
#endif

#include <immintrin.h>

namespace pace {
namespace kernels {
namespace impl {

void decode_one_head(
    const KernelCtx& ctx,
    const at::BFloat16* q_ptr,
    const std::vector<int64_t>& block_indices,
    int64_t seq_len,
    int64_t kv_h,
    float sink_bias,
    at::BFloat16* out_ptr) {
  const int64_t num_blocks = (seq_len + ctx.blk_size - 1) / ctx.blk_size;
  float output_fp32[512];
  float block_scores[256];
  std::fill(output_fp32, output_fp32 + ctx.head_dim, 0.0f);
  float max_score = -std::numeric_limits<float>::infinity();
  float sum_exp = 0.0f;

  // Sink: prepend a virtual score before softmax
  if (sink_bias != 0.0f) {
    max_score = sink_bias;
    sum_exp = 1.0f;
  }

  // Sliding window: skip blocks entirely before the window
  int64_t start_blk = 0;
  if (ctx.sliding_window > 0 && seq_len > ctx.sliding_window) {
    start_blk =
        std::max(int64_t(0), (seq_len - ctx.sliding_window) / ctx.blk_size);
  }

  const int64_t q_pos = seq_len - 1;

  for (int64_t blk_idx = start_blk; blk_idx < num_blocks; ++blk_idx) {
    const int64_t pool_blk = block_indices[blk_idx];
    const int64_t tokens_in_blk = (blk_idx == num_blocks - 1)
        ? (seq_len - blk_idx * ctx.blk_size)
        : ctx.blk_size;

    const at::BFloat16* k_base =
        ctx.pool_ptr + kv_h * ctx.ph + pool_blk * ctx.pb;
    const at::BFloat16* v_base = k_base + ctx.pvo;

    if (blk_idx + 1 < num_blocks) {
      const int64_t next_blk = block_indices[blk_idx + 1];
      const at::BFloat16* next_k =
          ctx.pool_ptr + kv_h * ctx.ph + next_blk * ctx.pb;
      _mm_prefetch(reinterpret_cast<const char*>(next_k), _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(next_k + ctx.pvo), _MM_HINT_T0);
    }

    float block_max = -std::numeric_limits<float>::infinity();
    for (int64_t t = 0; t < tokens_in_blk; ++t) {
      const int64_t kv_pos = blk_idx * ctx.blk_size + t;
      if (ctx.sliding_window > 0 && q_pos - kv_pos >= ctx.sliding_window) {
        block_scores[t] = -std::numeric_limits<float>::infinity();
        continue;
      }
      float s =
          dpbf16::dot_product(q_ptr, k_base + t * ctx.head_dim, ctx.head_dim) *
          ctx.scale_f;
      block_scores[t] = s;
      block_max = std::max(block_max, s);
    }

    if (block_max == -std::numeric_limits<float>::infinity())
      continue;

    const float new_max = std::max(max_score, block_max);
    if (new_max > max_score) {
      float correction;
      EXP_APPROX_SCALAR(max_score - new_max, correction);
      sum_exp *= correction;
      const __m512 vc = _mm512_set1_ps(correction);
      int64_t d = 0;
      for (; d + 16 <= ctx.head_dim; d += 16)
        _mm512_storeu_ps(
            output_fp32 + d,
            _mm512_mul_ps(_mm512_loadu_ps(output_fp32 + d), vc));
      for (; d < ctx.head_dim; ++d)
        output_fp32[d] *= correction;
    }

    // Vectorized exp: compute all weights in one pass (16 at a time)
    float exp_weights[256];
    int64_t t = 0;
    const __m512 vnmax = _mm512_set1_ps(-new_max);
    __m512 vsum = _mm512_setzero_ps();
    {
      EXP_APPROX_AVX512_CONSTANTS();
      for (; t + 16 <= tokens_in_blk; t += 16) {
        __m512 s = _mm512_loadu_ps(block_scores + t);
        __m512 e;
        EXP_APPROX_AVX512(_mm512_add_ps(s, vnmax), e);
        _mm512_storeu_ps(exp_weights + t, e);
        vsum = _mm512_add_ps(vsum, e);
      }
    }
    float local_sum = _mm512_reduce_add_ps(vsum);
    for (; t < tokens_in_blk; ++t) {
      EXP_APPROX_SCALAR(block_scores[t] - new_max, exp_weights[t]);
      local_sum += exp_weights[t];
    }
    sum_exp += local_sum;

    dpbf16::accumulate_weighted_regblock(
        output_fp32, v_base, exp_weights, tokens_in_blk, ctx.head_dim);
    max_score = new_max;
  }

  if (sum_exp > 0.0f) {
    const __m512 vinv = _mm512_set1_ps(1.0f / sum_exp);
    int64_t d = 0;
    for (; d + 16 <= ctx.head_dim; d += 16) {
      __m512 v = _mm512_mul_ps(_mm512_loadu_ps(output_fp32 + d), vinv);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(out_ptr + d),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
    }
    const float inv_sum = 1.0f / sum_exp;
    for (; d < ctx.head_dim; ++d)
      out_ptr[d] = at::BFloat16(output_fp32[d] * inv_sum);
  } else {
    std::memset(out_ptr, 0, ctx.head_dim * sizeof(at::BFloat16));
  }
}

// Process n_rep Q heads sharing one KV head in a single pass over blocks.
// Reads each KV block once instead of n_rep times.
void decode_gqa_group(
    const KernelCtx& ctx,
    const at::BFloat16* const* q_ptrs,
    int64_t n_rep,
    const std::vector<int64_t>& block_indices,
    int64_t seq_len,
    int64_t kv_h,
    const float* sink_biases,
    at::BFloat16* const* out_ptrs) {
  const int64_t num_blocks = (seq_len + ctx.blk_size - 1) / ctx.blk_size;

  // ~48KB stack per call with SLAB_MAX_REP=16 (32KB output + 16KB scores).
  // Safe with default OMP stack (2-8MB). Revisit if SLAB_MAX_REP grows.
  float output_fp32[SLAB_MAX_REP][512];
  float max_score[SLAB_MAX_REP];
  float sum_exp[SLAB_MAX_REP];
  float block_scores[SLAB_MAX_REP][256];

  for (int64_t r = 0; r < n_rep; ++r) {
    std::fill(output_fp32[r], output_fp32[r] + ctx.head_dim, 0.0f);
    max_score[r] = -std::numeric_limits<float>::infinity();
    sum_exp[r] = 0.0f;
    if (sink_biases && sink_biases[r] != 0.0f) {
      max_score[r] = sink_biases[r];
      sum_exp[r] = 1.0f;
    }
  }

  int64_t start_blk = 0;
  if (ctx.sliding_window > 0 && seq_len > ctx.sliding_window) {
    start_blk =
        std::max(int64_t(0), (seq_len - ctx.sliding_window) / ctx.blk_size);
  }

  const int64_t q_pos = seq_len - 1;

  for (int64_t blk_idx = start_blk; blk_idx < num_blocks; ++blk_idx) {
    const int64_t pool_blk = block_indices[blk_idx];
    const int64_t tokens_in_blk = (blk_idx == num_blocks - 1)
        ? (seq_len - blk_idx * ctx.blk_size)
        : ctx.blk_size;

    const at::BFloat16* k_base =
        ctx.pool_ptr + kv_h * ctx.ph + pool_blk * ctx.pb;
    const at::BFloat16* v_base = k_base + ctx.pvo;

    if (blk_idx + 1 < num_blocks) {
      const int64_t next_blk = block_indices[blk_idx + 1];
      const at::BFloat16* next_k =
          ctx.pool_ptr + kv_h * ctx.ph + next_blk * ctx.pb;
      _mm_prefetch(reinterpret_cast<const char*>(next_k), _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(next_k + ctx.pvo), _MM_HINT_T0);
    }

    // Compute scores for all n_rep Q heads against the same K block
    for (int64_t r = 0; r < n_rep; ++r) {
      float blk_max = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t < tokens_in_blk; ++t) {
        const int64_t kv_pos = blk_idx * ctx.blk_size + t;
        if (ctx.sliding_window > 0 && q_pos - kv_pos >= ctx.sliding_window) {
          block_scores[r][t] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float s = dpbf16::dot_product(
                      q_ptrs[r], k_base + t * ctx.head_dim, ctx.head_dim) *
            ctx.scale_f;
        block_scores[r][t] = s;
        blk_max = std::max(blk_max, s);
      }

      if (blk_max == -std::numeric_limits<float>::infinity())
        continue;

      const float new_max = std::max(max_score[r], blk_max);
      if (new_max > max_score[r]) {
        float correction;
        EXP_APPROX_SCALAR(max_score[r] - new_max, correction);
        sum_exp[r] *= correction;
        const __m512 vc = _mm512_set1_ps(correction);
        int64_t d = 0;
        for (; d + 16 <= ctx.head_dim; d += 16)
          _mm512_storeu_ps(
              output_fp32[r] + d,
              _mm512_mul_ps(_mm512_loadu_ps(output_fp32[r] + d), vc));
        for (; d < ctx.head_dim; ++d)
          output_fp32[r][d] *= correction;
      }

      float exp_weights[256];
      int64_t t = 0;
      const __m512 vnmax = _mm512_set1_ps(-new_max);
      __m512 vsum = _mm512_setzero_ps();
      {
        EXP_APPROX_AVX512_CONSTANTS();
        for (; t + 16 <= tokens_in_blk; t += 16) {
          __m512 s = _mm512_loadu_ps(block_scores[r] + t);
          __m512 e;
          EXP_APPROX_AVX512(_mm512_add_ps(s, vnmax), e);
          _mm512_storeu_ps(exp_weights + t, e);
          vsum = _mm512_add_ps(vsum, e);
        }
      }
      float local_sum = _mm512_reduce_add_ps(vsum);
      for (; t < tokens_in_blk; ++t) {
        EXP_APPROX_SCALAR(block_scores[r][t] - new_max, exp_weights[t]);
        local_sum += exp_weights[t];
      }
      sum_exp[r] += local_sum;
      dpbf16::accumulate_weighted_regblock(
          output_fp32[r], v_base, exp_weights, tokens_in_blk, ctx.head_dim);
      max_score[r] = new_max;
    }
  }

  // Normalize and write outputs
  for (int64_t r = 0; r < n_rep; ++r) {
    if (sum_exp[r] > 0.0f) {
      const __m512 vinv = _mm512_set1_ps(1.0f / sum_exp[r]);
      int64_t d = 0;
      for (; d + 16 <= ctx.head_dim; d += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(output_fp32[r] + d), vinv);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(out_ptrs[r] + d),
            reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
      }
      const float inv_sum = 1.0f / sum_exp[r];
      for (; d < ctx.head_dim; ++d)
        out_ptrs[r][d] = at::BFloat16(output_fp32[r][d] * inv_sum);
    } else {
      std::memset(out_ptrs[r], 0, ctx.head_dim * sizeof(at::BFloat16));
    }
  }
}

// Split-K partial: process blocks [blk_start, blk_end) for a GQA group.
// Writes partial online softmax state into partials[0..n_rep-1].
void decode_gqa_group_partial(
    const KernelCtx& ctx,
    const at::BFloat16* const* q_ptrs,
    int64_t n_rep,
    const std::vector<int64_t>& block_indices,
    int64_t seq_len,
    int64_t kv_h,
    int64_t blk_start,
    int64_t blk_end,
    PartialSoftmax* partials) {
  const int64_t q_pos = seq_len - 1;
  const int64_t num_blocks = (seq_len + ctx.blk_size - 1) / ctx.blk_size;

  for (int64_t r = 0; r < n_rep; ++r) {
    partials[r].max_score = -std::numeric_limits<float>::infinity();
    partials[r].sum_exp = 0.0f;
    std::fill(
        partials[r].output_fp32, partials[r].output_fp32 + ctx.head_dim, 0.0f);
  }

  float block_scores_buf[SLAB_MAX_REP][256];

  for (int64_t blk_idx = blk_start; blk_idx < blk_end; ++blk_idx) {
    const int64_t pool_blk = block_indices[blk_idx];
    const int64_t tokens_in_blk = (blk_idx == num_blocks - 1)
        ? (seq_len - blk_idx * ctx.blk_size)
        : ctx.blk_size;

    const at::BFloat16* k_base =
        ctx.pool_ptr + kv_h * ctx.ph + pool_blk * ctx.pb;
    const at::BFloat16* v_base = k_base + ctx.pvo;

    if (blk_idx + 1 < blk_end) {
      const int64_t next_blk = block_indices[blk_idx + 1];
      const at::BFloat16* next_k =
          ctx.pool_ptr + kv_h * ctx.ph + next_blk * ctx.pb;
      _mm_prefetch(reinterpret_cast<const char*>(next_k), _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(next_k + ctx.pvo), _MM_HINT_T0);
    }

    for (int64_t r = 0; r < n_rep; ++r) {
      float blk_max = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t < tokens_in_blk; ++t) {
        const int64_t kv_pos = blk_idx * ctx.blk_size + t;
        if (ctx.sliding_window > 0 && q_pos - kv_pos >= ctx.sliding_window) {
          block_scores_buf[r][t] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float s = dpbf16::dot_product(
                      q_ptrs[r], k_base + t * ctx.head_dim, ctx.head_dim) *
            ctx.scale_f;
        block_scores_buf[r][t] = s;
        blk_max = std::max(blk_max, s);
      }

      if (blk_max == -std::numeric_limits<float>::infinity())
        continue;

      auto& p = partials[r];
      const float new_max = std::max(p.max_score, blk_max);
      if (new_max > p.max_score) {
        float correction;
        EXP_APPROX_SCALAR(p.max_score - new_max, correction);
        p.sum_exp *= correction;
        const __m512 vc = _mm512_set1_ps(correction);
        int64_t d = 0;
        for (; d + 16 <= ctx.head_dim; d += 16)
          _mm512_storeu_ps(
              p.output_fp32 + d,
              _mm512_mul_ps(_mm512_loadu_ps(p.output_fp32 + d), vc));
        for (; d < ctx.head_dim; ++d)
          p.output_fp32[d] *= correction;
      }

      float exp_weights[256];
      int64_t t = 0;
      const __m512 vnmax = _mm512_set1_ps(-new_max);
      __m512 vsum = _mm512_setzero_ps();
      {
        EXP_APPROX_AVX512_CONSTANTS();
        for (; t + 16 <= tokens_in_blk; t += 16) {
          __m512 s = _mm512_loadu_ps(block_scores_buf[r] + t);
          __m512 e;
          EXP_APPROX_AVX512(_mm512_add_ps(s, vnmax), e);
          _mm512_storeu_ps(exp_weights + t, e);
          vsum = _mm512_add_ps(vsum, e);
        }
      }
      float local_sum = _mm512_reduce_add_ps(vsum);
      for (; t < tokens_in_blk; ++t) {
        EXP_APPROX_SCALAR(block_scores_buf[r][t] - new_max, exp_weights[t]);
        local_sum += exp_weights[t];
      }
      p.sum_exp += local_sum;
      dpbf16::accumulate_weighted_regblock(
          p.output_fp32, v_base, exp_weights, tokens_in_blk, ctx.head_dim);
      p.max_score = new_max;
    }
  }
}

// Reduce num_splits partial results per rep into final BF16 output.
void reduce_gqa_partials(
    const PartialSoftmax* partials,
    int64_t num_splits,
    int64_t n_rep,
    int64_t head_dim,
    at::BFloat16* const* out_ptrs) {
  for (int64_t r = 0; r < n_rep; ++r) {
    float max_score = -std::numeric_limits<float>::infinity();
    for (int64_t s = 0; s < num_splits; ++s)
      max_score = std::max(max_score, partials[s * n_rep + r].max_score);

    float sum_exp = 0.0f;
    float output_fp32[512];
    std::fill(output_fp32, output_fp32 + head_dim, 0.0f);

    for (int64_t s = 0; s < num_splits; ++s) {
      const auto& p = partials[s * n_rep + r];
      if (p.sum_exp == 0.0f)
        continue;
      float corr;
      EXP_APPROX_SCALAR(p.max_score - max_score, corr);
      sum_exp += p.sum_exp * corr;
      const __m512 vc = _mm512_set1_ps(corr);
      int64_t d = 0;
      for (; d + 16 <= head_dim; d += 16) {
        __m512 o = _mm512_loadu_ps(output_fp32 + d);
        __m512 pp = _mm512_loadu_ps(p.output_fp32 + d);
        _mm512_storeu_ps(output_fp32 + d, _mm512_fmadd_ps(vc, pp, o));
      }
      for (; d < head_dim; ++d)
        output_fp32[d] += corr * p.output_fp32[d];
    }

    if (sum_exp > 0.0f) {
      const __m512 vinv = _mm512_set1_ps(1.0f / sum_exp);
      int64_t d = 0;
      for (; d + 16 <= head_dim; d += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(output_fp32 + d), vinv);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(out_ptrs[r] + d),
            reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
      }
      const float inv = 1.0f / sum_exp;
      for (; d < head_dim; ++d)
        out_ptrs[r][d] = at::BFloat16(output_fp32[d] * inv);
    } else {
      std::memset(out_ptrs[r], 0, head_dim * sizeof(at::BFloat16));
    }
  }
}

// Single-row multi-token decode: processes one query row at position q_pos
// attending to kv_len KV positions with causal + sliding window masking.
void multi_token_decode_one_row(
    const KernelCtx& ctx,
    const at::BFloat16* q_row,
    const std::vector<int64_t>& block_indices,
    int64_t kv_len,
    int64_t q_pos,
    int64_t kv_h,
    float sink_bias,
    at::BFloat16* out_ptr) {
  const int64_t n_blks = (kv_len + ctx.blk_size - 1) / ctx.blk_size;

  float output_fp32[512];
  float block_scores[256];
  std::fill(output_fp32, output_fp32 + ctx.head_dim, 0.0f);
  float max_score = -std::numeric_limits<float>::infinity();
  float sum_exp = 0.0f;

  // Sink: prepend virtual score
  if (sink_bias != 0.0f) {
    max_score = sink_bias;
    sum_exp = 1.0f;
  }

  // Sliding window: compute start block
  int64_t start_bi = 0;
  if (ctx.sliding_window > 0 && q_pos >= ctx.sliding_window) {
    start_bi =
        std::max(int64_t(0), (q_pos - ctx.sliding_window + 1) / ctx.blk_size);
  }

  for (int64_t bi = start_bi; bi < n_blks; ++bi) {
    const int64_t kv_start = bi * ctx.blk_size;
    if (kv_start > q_pos)
      break;

    const int64_t pool_blk = block_indices[bi];
    const int64_t tokens = std::min(ctx.blk_size, kv_len - kv_start);
    int64_t valid = std::min(tokens, q_pos - kv_start + 1);
    // Sliding window: further restrict valid tokens within this block
    int64_t t_start = 0;
    if (ctx.sliding_window > 0 && q_pos >= ctx.sliding_window) {
      const int64_t window_start = q_pos - ctx.sliding_window + 1;
      if (window_start > kv_start)
        t_start = window_start - kv_start;
    }

    const at::BFloat16* k_base =
        ctx.pool_ptr + kv_h * ctx.ph + pool_blk * ctx.pb;
    const at::BFloat16* v_base = k_base + ctx.pvo;

    if (bi + 1 < n_blks) {
      const int64_t next_blk = block_indices[bi + 1];
      const at::BFloat16* next_k =
          ctx.pool_ptr + kv_h * ctx.ph + next_blk * ctx.pb;
      _mm_prefetch(reinterpret_cast<const char*>(next_k), _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(next_k + ctx.pvo), _MM_HINT_T0);
    }

    float block_max = -std::numeric_limits<float>::infinity();
    for (int64_t t = t_start; t < valid; ++t) {
      float s =
          dpbf16::dot_product(q_row, k_base + t * ctx.head_dim, ctx.head_dim) *
          ctx.scale_f;
      block_scores[t] = s;
      block_max = std::max(block_max, s);
    }

    if (block_max == -std::numeric_limits<float>::infinity())
      continue;

    const float new_max = std::max(max_score, block_max);
    if (new_max > max_score) {
      float corr;
      EXP_APPROX_SCALAR(max_score - new_max, corr);
      sum_exp *= corr;
      const __m512 vc = _mm512_set1_ps(corr);
      int64_t cd = 0;
      for (; cd + 16 <= ctx.head_dim; cd += 16)
        _mm512_storeu_ps(
            output_fp32 + cd,
            _mm512_mul_ps(_mm512_loadu_ps(output_fp32 + cd), vc));
      for (; cd < ctx.head_dim; ++cd)
        output_fp32[cd] *= corr;
    }

    {
      const int64_t n_valid = valid - t_start;
      const at::BFloat16* vb = v_base + t_start * ctx.head_dim;
      const float* sc = block_scores + t_start;
      float exp_weights[256];
      int64_t et = 0;
      const __m512 vnmax = _mm512_set1_ps(-new_max);
      __m512 vsum = _mm512_setzero_ps();
      {
        EXP_APPROX_AVX512_CONSTANTS();
        for (; et + 16 <= n_valid; et += 16) {
          __m512 s = _mm512_loadu_ps(sc + et);
          __m512 e;
          EXP_APPROX_AVX512(_mm512_add_ps(s, vnmax), e);
          _mm512_storeu_ps(exp_weights + et, e);
          vsum = _mm512_add_ps(vsum, e);
        }
      }
      float local_sum = _mm512_reduce_add_ps(vsum);
      for (; et < n_valid; ++et) {
        EXP_APPROX_SCALAR(sc[et] - new_max, exp_weights[et]);
        local_sum += exp_weights[et];
      }
      sum_exp += local_sum;
      dpbf16::accumulate_weighted_regblock(
          output_fp32, vb, exp_weights, n_valid, ctx.head_dim);
    }
    max_score = new_max;
  }

  if (sum_exp > 0.0f) {
    const __m512 vinv = _mm512_set1_ps(1.0f / sum_exp);
    int64_t d = 0;
    for (; d + 16 <= ctx.head_dim; d += 16) {
      __m512 v = _mm512_mul_ps(_mm512_loadu_ps(output_fp32 + d), vinv);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(out_ptr + d),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
    }
    const float inv = 1.0f / sum_exp;
    for (; d < ctx.head_dim; ++d)
      out_ptr[d] = at::BFloat16(output_fp32[d] * inv);
  }
}

} // namespace impl
} // namespace kernels
} // namespace pace
