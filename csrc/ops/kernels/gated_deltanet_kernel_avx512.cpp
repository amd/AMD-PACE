/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>
#include <immintrin.h>
#include <omp.h>
#include <ops/exp_approx.h>
#include <ops/kernels/avx512_bf16_utils.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pace {
namespace kernels {
namespace impl {

// L2-normalize a vector in-place (fp32), returns the inverse norm
static inline float l2norm_and_scale(
    float* vec,
    int64_t dim,
    float eps = 1e-6f) {
  float sum_sq = 0.0f;
  int64_t vec_end = dim & ~15LL;

  __m512 acc = _mm512_setzero_ps();
  for (int64_t j = 0; j < vec_end; j += 16) {
    __m512 v = _mm512_loadu_ps(vec + j);
    acc = _mm512_fmadd_ps(v, v, acc);
  }
  sum_sq = _mm512_reduce_add_ps(acc);
  for (int64_t j = vec_end; j < dim; ++j) {
    sum_sq += vec[j] * vec[j];
  }

  float inv_norm = 1.0f / std::sqrt(sum_sq + eps);

  __m512 inv_v = _mm512_set1_ps(inv_norm);
  for (int64_t j = 0; j < vec_end; j += 16) {
    __m512 v = _mm512_loadu_ps(vec + j);
    _mm512_storeu_ps(vec + j, _mm512_mul_ps(v, inv_v));
  }
  for (int64_t j = vec_end; j < dim; ++j) {
    vec[j] *= inv_norm;
  }

  return inv_norm;
}

// Dot product of two fp32 vectors
static inline float dot_product(const float* a, const float* b, int64_t dim) {
  int64_t vec_end = dim & ~15LL;
  __m512 acc = _mm512_setzero_ps();
  for (int64_t j = 0; j < vec_end; j += 16) {
    __m512 va = _mm512_loadu_ps(a + j);
    __m512 vb = _mm512_loadu_ps(b + j);
    acc = _mm512_fmadd_ps(va, vb, acc);
  }
  float result = _mm512_reduce_add_ps(acc);
  for (int64_t j = vec_end; j < dim; ++j) {
    result += a[j] * b[j];
  }
  return result;
}

// Scale a fp32 vector: dst[j] *= scale
static inline void vec_scale(float* dst, float scale, int64_t dim) {
  int64_t vec_end = dim & ~15LL;
  __m512 s = _mm512_set1_ps(scale);
  for (int64_t j = 0; j < vec_end; j += 16) {
    __m512 v = _mm512_loadu_ps(dst + j);
    _mm512_storeu_ps(dst + j, _mm512_mul_ps(v, s));
  }
  for (int64_t j = vec_end; j < dim; ++j) {
    dst[j] *= scale;
  }
}

// dst[j] += a * src[j]
static inline void vec_axpy(
    float* dst,
    float a,
    const float* src,
    int64_t dim) {
  int64_t vec_end = dim & ~15LL;
  __m512 av = _mm512_set1_ps(a);
  for (int64_t j = 0; j < vec_end; j += 16) {
    __m512 d = _mm512_loadu_ps(dst + j);
    __m512 s = _mm512_loadu_ps(src + j);
    _mm512_storeu_ps(dst + j, _mm512_fmadd_ps(av, s, d));
  }
  for (int64_t j = vec_end; j < dim; ++j) {
    dst[j] += a * src[j];
  }
}

// One gated-delta recurrence step for a single (value) head.
//   state  : [Kd, Vd] row-major fp32, decayed and rank-1 updated in place
//   q_ptr/k_ptr : fp32, already L2-normalized (q also pre-scaled by 1/sqrt(Kd))
//   kq     : dot(k, q) (independent of the value head, computed once per group)
//   v_ptr  : bf16 value row, converted to fp32 in-register
//   o_ptr  : bf16 output row, written via fp32->bf16 in-register
//   kv_mem_buf / out_buf : per-thread fp32 scratch of length Vd
// Uses the fused identity out = (q.S_decayed) + delta*(k.q) so the readout
// needs no separate pass over the updated state.
static inline void delta_rule_step(
    const float* q_ptr,
    const float* k_ptr,
    float kq,
    const at::BFloat16* v_ptr,
    float g_val,
    float beta_val,
    float* s_ptr,
    at::BFloat16* o_ptr,
    int64_t Kd,
    int64_t Vd,
    float* kv_mem_buf,
    float* out_buf) {
  // Decay: state *= exp(g)
  const float decay = std::exp(g_val);
  vec_scale(s_ptr, decay, Kd * Vd);

  for (int64_t vi = 0; vi < Vd; ++vi) {
    kv_mem_buf[vi] = 0.0f;
    out_buf[vi] = 0.0f;
  }
  for (int64_t ki = 0; ki < Kd; ++ki) {
    const float* srow = s_ptr + ki * Vd;
    vec_axpy(kv_mem_buf, k_ptr[ki], srow, Vd);
    vec_axpy(out_buf, q_ptr[ki], srow, Vd);
  }

  const int64_t vec_end = Vd & ~15LL;
  const __m512 beta_v = _mm512_set1_ps(beta_val);
  const __m512 kq_v = _mm512_set1_ps(kq);
  for (int64_t vi = 0; vi < vec_end; vi += 16) {
    __m512 vv = load_bf16_fp32(v_ptr + vi);
    __m512 kvm = _mm512_loadu_ps(kv_mem_buf + vi);
    __m512 delta = _mm512_mul_ps(_mm512_sub_ps(vv, kvm), beta_v);
    _mm512_storeu_ps(kv_mem_buf + vi, delta);
    __m512 outp = _mm512_loadu_ps(out_buf + vi);
    store_fp32_bf16(o_ptr + vi, _mm512_fmadd_ps(delta, kq_v, outp));
  }
  for (int64_t vi = vec_end; vi < Vd; ++vi) {
    const float delta =
        (static_cast<float>(v_ptr[vi]) - kv_mem_buf[vi]) * beta_val;
    kv_mem_buf[vi] = delta;
    o_ptr[vi] = at::BFloat16(out_buf[vi] + delta * kq);
  }

  for (int64_t ki = 0; ki < Kd; ++ki) {
    vec_axpy(s_ptr + ki * Vd, k_ptr[ki], kv_mem_buf, Vd);
  }
}

// fp32 scratch used by one thread: q and k (Kd each), kv_mem and out (Vd
// each). All threads' slices live in one buffer allocated before the parallel
// region; each slice is padded to a whole number of 64-byte cache lines so
// neighbouring threads never write to the same line.
struct DeltaScratch {
  float* q;
  float* k;
  float* kv_mem;
  float* out;
};

static inline int64_t delta_scratch_stride(int64_t Kd, int64_t Vd) {
  return (2 * (Kd + Vd) + 15) & ~15LL;
}

static inline at::Tensor alloc_delta_scratch(int64_t Kd, int64_t Vd) {
  return at::empty(
      {static_cast<int64_t>(omp_get_max_threads()),
       delta_scratch_stride(Kd, Vd)},
      at::TensorOptions().dtype(at::kFloat));
}

static inline DeltaScratch thread_delta_scratch(
    float* scratch,
    int64_t Kd,
    int64_t Vd) {
  float* base = scratch + omp_get_thread_num() * delta_scratch_stride(Kd, Vd);
  return {base, base + Kd, base + 2 * Kd, base + 2 * Kd + Vd};
}

static inline __m512 load_bf16_fp32_mask(const at::BFloat16* p, __mmask16 m) {
  return _mm512_cvtpbh_ps((__m256bh)_mm256_maskz_loadu_epi16(m, p));
}

static inline void store_fp32_bf16_mask(
    at::BFloat16* p,
    __m512 v,
    __mmask16 m) {
  _mm256_mask_storeu_epi16(p, m, (__m256i)_mm512_cvtneps_pbh(v));
}

// x * sigmoid(x). The shared exp approximation only clamps its input from
// below, so exp(-x) overflows for x < -88. x itself is clamped, not just exp's
// argument, so SiLU stays ~0 even for huge negative x (SiLU(-88) is already
// -5e-37). MAXPS returns its second operand when either is NaN, so x goes
// second to keep a NaN input NaN.
static inline __m512 silu_ps(__m512 x) {
  EXP_APPROX_AVX512_CONSTANTS();
  const __m512 xc = _mm512_max_ps(_mm512_set1_ps(-88.0f), x);
  __m512 exp_neg_x;
  EXP_APPROX_AVX512(_mm512_sub_ps(_mm512_setzero_ps(), xc), exp_neg_x);
  return _mm512_div_ps(xc, _mm512_add_ps(_ea_one, exp_neg_x));
}

constexpr int64_t kMaxConvKernel = 8;

// Token-major depthwise causal conv + SiLU. Work is split into (batch, token
// block, channel block) tiles; inside a tile each 16-channel vector keeps its
// K tap weights in registers and walks the tokens. Token t reads inputs
// t-K+1..t; indices < 0 come from `prev` (the cached tail), or are zero.
template <int KT>
static void causal_conv1d_prefill_impl(
    const at::BFloat16* x,
    int64_t x_bs,
    int64_t x_ts,
    const at::BFloat16* prev,
    const float* w,
    at::BFloat16* out,
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t K_rt) {
  const int64_t K = KT > 0 ? KT : K_rt;
  const int64_t S = K - 1;
  constexpr int64_t kCBlock = 64;
  constexpr int64_t kTBlock = 64;
  const int64_t n_cb = (C + kCBlock - 1) / kCBlock;
  const int64_t n_tb = (T + kTBlock - 1) / kTBlock;

#pragma omp parallel for collapse(3) schedule(static)
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t tb = 0; tb < n_tb; ++tb) {
      for (int64_t cb = 0; cb < n_cb; ++cb) {
        const int64_t t0 = tb * kTBlock;
        const int64_t t1 = std::min(T, t0 + kTBlock);
        const int64_t c1 = std::min(C, (cb + 1) * kCBlock);
        for (int64_t c = cb * kCBlock; c < c1; c += 16) {
          const int64_t n = std::min<int64_t>(16, c1 - c);
          const __mmask16 m = static_cast<__mmask16>((1u << n) - 1);
          __m512 wv[kMaxConvKernel];
          for (int64_t k = 0; k < K; ++k) {
            wv[k] = _mm512_maskz_loadu_ps(m, w + k * C + c);
          }
          for (int64_t t = t0; t < t1; ++t) {
            __m512 acc = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; ++k) {
              const int64_t src = t - S + k;
              const at::BFloat16* row;
              if (src >= 0) {
                row = x + b * x_bs + src * x_ts;
              } else if (prev != nullptr) {
                row = prev + (b * S + S + src) * C;
              } else {
                continue;
              }
              acc =
                  _mm512_fmadd_ps(wv[k], load_bf16_fp32_mask(row + c, m), acc);
            }
            store_fp32_bf16_mask(out + (b * T + t) * C + c, silu_ps(acc), m);
          }
        }
      }
    }
  }
}

} // namespace impl

at::Tensor recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state) {
  // value: [B, 1, H, Vd] (H = number of value heads, also the state/output and
  // g/beta head dim). query/key: [B, 1, H_q, Kd] where H_q may be < H for
  // grouped-query attention; each value head h reads key/query head h/group.
  // This lets the caller pass the UNEXPANDED 16-head q/k instead of
  // materializing a repeat_interleave'd 32-head copy, eliminating the largest
  // DeltaNet decode copy bucket and halving the q/k float conversion volume.
  const int64_t B = query.size(0);
  const int64_t H = value.size(2);
  const int64_t H_q = query.size(2);
  const int64_t Kd = query.size(3);
  const int64_t Vd = value.size(3);
  // value heads are an integer multiple of query heads (validated in the op
  // wrapper); group = value_heads / query_heads is the GQA broadcast factor.
  const int64_t group = H / H_q;

  // q/k/v are kept in bf16 in memory and converted to fp32 at register level
  // (load_bf16_fp32) inside the loops, avoiding whole-tensor .to(kFloat)
  // copies. The recurrent state stays fp32 (precision-sensitive, read/written
  // many times per step). The output is written straight to bf16 via
  // register-level fp32->bf16 conversion, removing the trailing output.to(bf16)
  // copy. g/beta are just H per-head scalars (negligible), normalized to fp32
  // here so the kernel is agnostic to their incoming dtype (g is fp32, beta is
  // bf16 in the Qwen3.5 model).
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto g_f = g.to(at::kFloat).contiguous();
  auto beta_f = beta.to(at::kFloat).contiguous();

  const at::BFloat16* query_p = query_c.data_ptr<at::BFloat16>();
  const at::BFloat16* key_p = key_c.data_ptr<at::BFloat16>();
  const at::BFloat16* value_p = value_c.data_ptr<at::BFloat16>();
  const float* g_p = g_f.data_ptr<float>();
  const float* beta_p = beta_f.data_ptr<float>();

  const float scale = 1.0f / std::sqrt(static_cast<float>(Kd));

  auto output = at::empty({B, 1, H, Vd}, query.options());
  at::BFloat16* output_p = output.data_ptr<at::BFloat16>();

  auto scratch = impl::alloc_delta_scratch(Kd, Vd);
  float* scratch_p = scratch.data_ptr<float>();

#pragma omp parallel
  {
    // q/k scratch holds the L2-normalized (and, for q, scaled) key/query in
    // fp32. With GQA the key/query head is shared by `group` value heads, so
    // we normalize it ONCE per key/query head and broadcast it across the
    // group's value heads (matching the slab decode GQA pattern: KV head
    // loaded/processed once, reused across the group).
    const impl::DeltaScratch buf =
        impl::thread_delta_scratch(scratch_p, Kd, Vd);

    // Parallelize over (batch, key/query head). Each key/query head owns a
    // contiguous group of `group` value heads (h = hk*group + r).
#pragma omp for collapse(2) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t hk = 0; hk < H_q; ++hk) {
        const at::BFloat16* q_src = query_p + (b * H_q + hk) * Kd;
        const at::BFloat16* k_src = key_p + (b * H_q + hk) * Kd;

        float* q_ptr = buf.q;
        float* k_ptr = buf.k;
        // Register-level bf16->fp32 into scratch (replaces .to(kFloat)+memcpy).
        impl::load_bf16_to_fp32(q_ptr, q_src, Kd);
        impl::load_bf16_to_fp32(k_ptr, k_src, Kd);

        // L2-normalize q/k ONCE for this shared head.
        impl::l2norm_and_scale(q_ptr, Kd);
        impl::l2norm_and_scale(k_ptr, Kd);
        impl::vec_scale(q_ptr, scale, Kd);

        // k.q is independent of the value head, compute once and reuse.
        const float kq = impl::dot_product(k_ptr, q_ptr, Kd);

        // Broadcast this key/query head across its group of value heads.
        for (int64_t r = 0; r < group; ++r) {
          const int64_t h = hk * group + r;
          const at::BFloat16* v_ptr = value_p + (b * H + h) * Vd;
          const float g_val = g_p[b * H + h];
          const float beta_val = beta_p[b * H + h];
          float* s_ptr = state.data_ptr<float>() + (b * H + h) * Kd * Vd;
          at::BFloat16* o_ptr = output_p + (b * H + h) * Vd;

          impl::delta_rule_step(
              q_ptr,
              k_ptr,
              kq,
              v_ptr,
              g_val,
              beta_val,
              s_ptr,
              o_ptr,
              Kd,
              Vd,
              buf.kv_mem,
              buf.out);
        }
      }
    }
  } // omp parallel

  // output is already in query's (bf16) dtype.
  return output;
}

at::Tensor chunk_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state) {
  // Multi-token (prefill) sequential scan. Mathematically identical to the
  // per-token recurrence run over T tokens, advancing the fp32 state. Replaces
  // the pure-PyTorch _chunk_gated_delta_rule (large [B,H,T,d] fp32
  // intermediates
  // + many bmm/mul). q/k stay bf16 (register-level fp32 convert), state fp32,
  // output bf16. GQA handled internally (value head h reads q/k head h/group).
  //   query/key: [B, T, Hq, Kd] bf16 ; value: [B, T, H, Vd] bf16
  //   g/beta:    [B, T, H] fp32       ; state: [B, H, Kd, Vd] fp32 (in-place)
  const int64_t B = query.size(0);
  const int64_t T = query.size(1);
  const int64_t H_q = query.size(2);
  const int64_t H = value.size(2);
  const int64_t Kd = query.size(3);
  const int64_t Vd = value.size(3);
  const int64_t group = H / H_q;

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto g_f = g.to(at::kFloat).contiguous();
  auto beta_f = beta.to(at::kFloat).contiguous();

  const at::BFloat16* query_p = query_c.data_ptr<at::BFloat16>();
  const at::BFloat16* key_p = key_c.data_ptr<at::BFloat16>();
  const at::BFloat16* value_p = value_c.data_ptr<at::BFloat16>();
  const float* g_p = g_f.data_ptr<float>();
  const float* beta_p = beta_f.data_ptr<float>();
  float* state_p = state.data_ptr<float>();

  const float scale = 1.0f / std::sqrt(static_cast<float>(Kd));

  auto output = at::empty({B, T, H, Vd}, query.options());
  at::BFloat16* output_p = output.data_ptr<at::BFloat16>();

  auto scratch = impl::alloc_delta_scratch(Kd, Vd);
  float* scratch_p = scratch.data_ptr<float>();

#pragma omp parallel
  {
    const impl::DeltaScratch buf =
        impl::thread_delta_scratch(scratch_p, Kd, Vd);

    // Parallelize over (batch, key/query head). The time scan is sequential
    // (state recurrence), but each (b, head) chain is independent.
#pragma omp for collapse(2) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t hk = 0; hk < H_q; ++hk) {
        for (int64_t t = 0; t < T; ++t) {
          const at::BFloat16* q_src = query_p + ((b * T + t) * H_q + hk) * Kd;
          const at::BFloat16* k_src = key_p + ((b * T + t) * H_q + hk) * Kd;

          float* q_ptr = buf.q;
          float* k_ptr = buf.k;
          impl::load_bf16_to_fp32(q_ptr, q_src, Kd);
          impl::load_bf16_to_fp32(k_ptr, k_src, Kd);

          impl::l2norm_and_scale(q_ptr, Kd);
          impl::l2norm_and_scale(k_ptr, Kd);
          impl::vec_scale(q_ptr, scale, Kd);

          const float kq = impl::dot_product(k_ptr, q_ptr, Kd);

          for (int64_t r = 0; r < group; ++r) {
            const int64_t h = hk * group + r;
            const at::BFloat16* v_ptr = value_p + ((b * T + t) * H + h) * Vd;
            const float g_val = g_p[(b * T + t) * H + h];
            const float beta_val = beta_p[(b * T + t) * H + h];
            float* s_ptr = state_p + (b * H + h) * Kd * Vd;
            at::BFloat16* o_ptr = output_p + ((b * T + t) * H + h) * Vd;

            impl::delta_rule_step(
                q_ptr,
                k_ptr,
                kq,
                v_ptr,
                g_val,
                beta_val,
                s_ptr,
                o_ptr,
                Kd,
                Vd,
                buf.kv_mem,
                buf.out);
          }
        }
      }
    }
  } // omp parallel

  return output;
}

at::Tensor causal_conv1d_update(
    const at::Tensor& input,
    at::Tensor& conv_state,
    const at::Tensor& weight) {
  // input: [B, C, 1], conv_state: [B, C, K-1], weight: [C, K]
  // Python reference semantics:
  //   concat = cat([conv_state, input], dim=-1)   → [B, C, K]
  //   conv_state[:] = concat[:, :, -(K-1):]       → shift left, append new
  //   out = conv1d(concat, weight, groups=C)       → dot per channel
  //   out = silu(out)
  const int64_t B = input.size(0);
  const int64_t C = input.size(1);
  const int64_t K = weight.size(1);
  const int64_t state_len = conv_state.size(2); // K-1

  TORCH_CHECK(
      state_len == K - 1, "conv_state last dim must be kernel_size - 1");

  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();
  auto output = at::empty_like(input);

  const at::BFloat16* in_ptr = input_contig.data_ptr<at::BFloat16>();
  at::BFloat16* state_ptr = conv_state.data_ptr<at::BFloat16>();
  const at::BFloat16* w_ptr = weight_contig.data_ptr<at::BFloat16>();
  at::BFloat16* out_ptr = output.data_ptr<at::BFloat16>();

  // Process 16 channels at a time with AVX512 for the hot path (K=4)
  const int64_t total_channels = B * C;
  const int64_t vec_end = total_channels & ~15LL;

  if (K == 4 && state_len == 3) {
    // Specialized fast path for kernel_size=4 (the Qwen3.5 default)
#pragma omp parallel for schedule(static)
    for (int64_t bc = 0; bc < vec_end; bc += 16) {
      // Load 16 channels of state [s0, s1, s2] and input [new]
      // State layout: for channel bc+i, state is at state_ptr[(bc+i)*3 +
      // {0,1,2}] This is strided, so we gather per channel

      float conv_results[16];
      for (int i = 0; i < 16; ++i) {
        int64_t idx = bc + i;
        at::BFloat16* s = state_ptr + idx * 3;
        const at::BFloat16* w = w_ptr + (idx % C) * 4;
        float new_val = static_cast<float>(in_ptr[idx]);

        // Compute conv: dot([s0, s1, s2, new], [w0, w1, w2, w3])
        float sum = static_cast<float>(s[0]) * static_cast<float>(w[0]) +
            static_cast<float>(s[1]) * static_cast<float>(w[1]) +
            static_cast<float>(s[2]) * static_cast<float>(w[2]) +
            new_val * static_cast<float>(w[3]);

        // Update state: [s0, s1, s2] → [s1, s2, new]
        s[0] = s[1];
        s[1] = s[2];
        s[2] = at::BFloat16(new_val);

        conv_results[i] = sum;
      }

      // SiLU on 16 results using AVX512
      __m512 conv_v = _mm512_loadu_ps(conv_results);
      __m512 neg_v = _mm512_sub_ps(_mm512_setzero_ps(), conv_v);

      // sigmoid: compute per-element (exp is not vectorized without svml)
      float sig_buf[16];
      _mm512_storeu_ps(sig_buf, neg_v);
      for (int i = 0; i < 16; ++i) {
        sig_buf[i] = 1.0f / (1.0f + std::exp(sig_buf[i]));
      }
      __m512 sig_v = _mm512_loadu_ps(sig_buf);
      __m512 silu_v = _mm512_mul_ps(conv_v, sig_v);

      impl::store_fp32_bf16(out_ptr + bc, silu_v);
    }

    // Scalar tail
#pragma omp parallel for schedule(static)
    for (int64_t bc = vec_end; bc < total_channels; ++bc) {
      at::BFloat16* s = state_ptr + bc * 3;
      const at::BFloat16* w = w_ptr + (bc % C) * 4;
      float new_val = static_cast<float>(in_ptr[bc]);

      float sum = static_cast<float>(s[0]) * static_cast<float>(w[0]) +
          static_cast<float>(s[1]) * static_cast<float>(w[1]) +
          static_cast<float>(s[2]) * static_cast<float>(w[2]) +
          new_val * static_cast<float>(w[3]);

      s[0] = s[1];
      s[1] = s[2];
      s[2] = at::BFloat16(new_val);

      float sig = 1.0f / (1.0f + std::exp(-sum));
      out_ptr[bc] = at::BFloat16(sum * sig);
    }
  } else {
    // Generic path for any kernel size
#pragma omp parallel for schedule(static)
    for (int64_t bc = 0; bc < total_channels; ++bc) {
      at::BFloat16* s = state_ptr + bc * state_len;
      const at::BFloat16* w = w_ptr + (bc % C) * K;
      float new_val = static_cast<float>(in_ptr[bc]);

      // Conv: dot([s0..s_{K-2}, new], [w0..w_{K-1}])
      float sum = 0.0f;
      for (int64_t k = 0; k < state_len; ++k) {
        sum += static_cast<float>(s[k]) * static_cast<float>(w[k]);
      }
      sum += new_val * static_cast<float>(w[K - 1]);

      // Update state: shift left, append new
      for (int64_t k = 0; k < state_len - 1; ++k) {
        s[k] = s[k + 1];
      }
      s[state_len - 1] = at::BFloat16(new_val);

      // SiLU
      float sig = 1.0f / (1.0f + std::exp(-sum));
      out_ptr[bc] = at::BFloat16(sum * sig);
    }
  }

  return output;
}

at::Tensor causal_conv1d_prefill(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& prev,
    const at::Tensor& weight) {
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t C = x.size(2);
  const int64_t K = weight.size(1);
  TORCH_CHECK(
      K <= impl::kMaxConvKernel,
      "causal_conv1d_prefill: kernel size ",
      K,
      " exceeds the supported maximum of ",
      impl::kMaxConvKernel);

  // [K, C] fp32 so each tap loads as a contiguous channel vector.
  const auto w_kc = weight.t().to(at::kFloat).contiguous();
  auto out = at::empty({B, T, C}, x.options());

  const at::BFloat16* x_p = x.data_ptr<at::BFloat16>();
  const at::BFloat16* prev_p =
      prev.has_value() ? prev->data_ptr<at::BFloat16>() : nullptr;
  const float* w_p = w_kc.data_ptr<float>();
  at::BFloat16* out_p = out.data_ptr<at::BFloat16>();

  if (K == 4) {
    impl::causal_conv1d_prefill_impl<4>(
        x_p, x.stride(0), x.stride(1), prev_p, w_p, out_p, B, T, C, K);
  } else {
    impl::causal_conv1d_prefill_impl<0>(
        x_p, x.stride(0), x.stride(1), prev_p, w_p, out_p, B, T, C, K);
  }
  return out;
}

} // namespace kernels
} // namespace pace
