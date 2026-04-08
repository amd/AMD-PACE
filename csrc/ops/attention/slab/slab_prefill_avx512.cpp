/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * SLAB BRGeMM tiled prefill kernel.
 *
 * - oneDNN BRGeMM ukernels for QK^T and S@V matmuls
 * - Per-block K/V packed once via oneDNN transform, reused across n_rep heads
 * - Online softmax (block-by-block) with FP32 accumulation
 ******************************************************************************/

#include <ops/attention/slab/dpbf16_kernels.h>
#include <ops/attention/slab/slab_kernels.h>
#include <ops/exp_approx.h>
#include <cstring>
#include <limits>

#ifndef __AVX512F__
#error "slab_prefill_avx512.cpp requires AVX512F."
#endif

#include <immintrin.h>
#include "oneapi/dnnl/dnnl_ukernel.hpp"

namespace pace {
namespace kernels {

using namespace dnnl;
using namespace dnnl::ukernel;

namespace {

// BrgemmCache: thread-local lazy-init cache for oneDNN BRGeMM ukernels.
// File-local to slab_prefill_avx512.cpp -- not exposed in headers.
struct BrgemmCache {
  using dim = dnnl::memory::dim;

  int64_t tile_q = 0, tokens = 0, head_dim = 0;
  int64_t n_sub = 0, sub_n = 0; // token sub-blocking (K packing)
  int64_t n_hd_sub = 0, hd_sub = 0, hd_padded = 0; // head_dim sub-blocking
  int64_t n_rep_alloc = 0; // reps allocated in rep_mem

  dnnl::ukernel::brgemm brg_qkt;
  dnnl::ukernel::brgemm brg_sv;
  dnnl::ukernel::transform pack_k;
  dnnl::ukernel::transform pack_v;

  std::vector<uint8_t> scratchpad;
  std::vector<uint8_t> k_packed;
  std::vector<uint8_t> v_packed;
  std::vector<std::pair<dim, dim>> unit_offset;
  std::vector<std::pair<dim, dim>> sv_offsets;

  // Per-rep state (persists across blocks within a tile)
  std::vector<uint8_t> rep_mem;
  float* out_fp32[SLAB_MAX_REP];
  float* rmax_arr[SLAB_MAX_REP];
  float* rsum_arr[SLAB_MAX_REP];
  at::BFloat16* q_tile_buf[SLAB_MAX_REP];
  at::BFloat16* weights_bf16[SLAB_MAX_REP];

  // Shared across reps (reused sequentially per block, NOT thread-safe).
  // Do not parallelize the rep loop without giving each rep its own buffer.
  float* scores_fp32 = nullptr;
  at::BFloat16* k_padded = nullptr;
  at::BFloat16* v_padded = nullptr;

  bool valid = false;
};

static thread_local BrgemmCache tl_prefill_cache;

void init_brgemm_cache(
    BrgemmCache& c,
    int64_t tile_q,
    int64_t tokens,
    int64_t head_dim,
    int64_t n_rep) {
  if (c.valid && c.tile_q == tile_q && c.tokens == tokens &&
      c.head_dim == head_dim && n_rep <= c.n_rep_alloc)
    return;

  auto dt = memory::data_type::bf16;
  auto acc_dt = memory::data_type::f32;

  int64_t n_sub = (tokens + SLAB_N_MAX - 1) / SLAB_N_MAX;
  int64_t sub_n = (tokens + n_sub - 1) / n_sub;
  sub_n = ((sub_n + 15) / 16) * 16;
  if (sub_n > SLAB_N_MAX)
    sub_n = SLAB_N_MAX;
  n_sub = (tokens + sub_n - 1) / sub_n;

  c.brg_qkt = brgemm(
      tile_q, sub_n, head_dim, 1, head_dim, sub_n, tokens, dt, dt, acc_dt);
  c.brg_qkt.set_add_C(false);
  c.brg_qkt.finalize();
  c.brg_qkt.generate();

  // Head_dim sub-blocking for V packing (transform out_ld must be <= 64)
  int64_t n_hd_sub = (head_dim + SLAB_N_MAX - 1) / SLAB_N_MAX;
  int64_t hd_sub = (head_dim + n_hd_sub - 1) / n_hd_sub;
  hd_sub = ((hd_sub + 15) / 16) * 16;
  if (hd_sub > SLAB_N_MAX)
    hd_sub = SLAB_N_MAX;
  n_hd_sub = (head_dim + hd_sub - 1) / hd_sub;
  int64_t hd_padded = n_hd_sub * hd_sub;

  c.brg_sv = brgemm(
      tile_q, hd_sub, sub_n, n_sub, tokens, hd_sub, hd_padded, dt, dt, acc_dt);
  c.brg_sv.set_add_C(true);
  c.brg_sv.finalize();
  c.brg_sv.generate();

  c.pack_k =
      transform(head_dim, sub_n, pack_type::trans, head_dim, sub_n, dt, dt);
  c.pack_k.generate();
  c.pack_v =
      transform(sub_n, hd_sub, pack_type::no_trans, head_dim, hd_sub, dt, dt);
  c.pack_v.generate();

  size_t sz =
      std::max(c.brg_qkt.get_scratchpad_size(), c.brg_sv.get_scratchpad_size());
  c.scratchpad.resize(sz + 64);
  c.k_packed.resize(n_sub * head_dim * sub_n * sizeof(at::BFloat16) + 64);
  c.v_packed.resize(n_sub * sub_n * hd_sub * sizeof(at::BFloat16) + 64);
  c.unit_offset = {{0, 0}};

  int64_t v_sub_packed_bytes = sub_n * hd_sub * sizeof(at::BFloat16);
  c.sv_offsets.resize(n_sub);
  for (int64_t s = 0; s < n_sub; ++s) {
    c.sv_offsets[s] = {
        static_cast<memory::dim>(
            s * sub_n * static_cast<int64_t>(sizeof(at::BFloat16))),
        static_cast<memory::dim>(s * v_sub_packed_bytes)};
  }

  const size_t al = 64;
  auto align = [al](size_t x) { return (x + al - 1) & ~(al - 1); };

  // Per-rep buffers: out_fp32, rmax, rsum, q_tile_buf, weights_bf16
  size_t per_rep = align(tile_q * hd_padded * sizeof(float)) +
      align(tile_q * sizeof(float)) + align(tile_q * sizeof(float)) +
      align(tile_q * head_dim * sizeof(at::BFloat16)) +
      align(tile_q * tokens * sizeof(at::BFloat16));

  // Shared buffers: scores, k_padded, v_padded
  size_t shared = align(tile_q * tokens * sizeof(float)) +
      align(tokens * head_dim * sizeof(at::BFloat16)) +
      align(tokens * head_dim * sizeof(at::BFloat16));

  c.rep_mem.resize(n_rep * per_rep + shared);
  uint8_t* base = c.rep_mem.data();
  for (int64_t r = 0; r < n_rep; ++r) {
    uint8_t* rb = base + r * per_rep;
    size_t off = 0;
    c.out_fp32[r] = reinterpret_cast<float*>(rb + off);
    off += align(tile_q * hd_padded * sizeof(float));
    c.rmax_arr[r] = reinterpret_cast<float*>(rb + off);
    off += align(tile_q * sizeof(float));
    c.rsum_arr[r] = reinterpret_cast<float*>(rb + off);
    off += align(tile_q * sizeof(float));
    c.q_tile_buf[r] = reinterpret_cast<at::BFloat16*>(rb + off);
    off += align(tile_q * head_dim * sizeof(at::BFloat16));
    c.weights_bf16[r] = reinterpret_cast<at::BFloat16*>(rb + off);
  }

  uint8_t* sb = base + n_rep * per_rep;
  size_t off = 0;
  c.scores_fp32 = reinterpret_cast<float*>(sb + off);
  off += align(tile_q * tokens * sizeof(float));
  c.k_padded = reinterpret_cast<at::BFloat16*>(sb + off);
  off += align(tokens * head_dim * sizeof(at::BFloat16));
  c.v_padded = reinterpret_cast<at::BFloat16*>(sb + off);

  c.n_sub = n_sub;
  c.sub_n = sub_n;
  c.n_hd_sub = n_hd_sub;
  c.hd_sub = hd_sub;
  c.hd_padded = hd_padded;
  c.n_rep_alloc = n_rep;
  c.tile_q = tile_q;
  c.tokens = tokens;
  c.head_dim = head_dim;
  c.valid = true;
}

} // anonymous namespace

void prefill_tile(
    const KernelCtx& ctx,
    int64_t kv_len,
    int64_t q_len,
    int64_t q_offset,
    const std::vector<int64_t>& block_indices,
    int64_t kv_h,
    int64_t qt) {
  auto& c = tl_prefill_cache;
  const int64_t aq = std::min(q_len, kv_len);
  const int64_t qs = qt * SLAB_Q_TILE;
  if (qs >= aq)
    return;
  const int64_t qe = std::min(qs + SLAB_Q_TILE, aq);
  const int64_t tile_q = qe - qs;
  const int64_t padded_tile_q = SLAB_Q_TILE;
  const int64_t n_blks = (kv_len + ctx.blk_size - 1) / ctx.blk_size;
  const int64_t n_rep = ctx.n_rep;
  const int64_t head_dim = ctx.head_dim;
  const int64_t blk_size = ctx.blk_size;

  init_brgemm_cache(c, padded_tile_q, blk_size, head_dim, n_rep);
  c.brg_qkt.set_hw_context();

  // Init per-rep accumulators (with optional sink bias)
  for (int64_t r = 0; r < n_rep; ++r) {
    std::fill(c.out_fp32[r], c.out_fp32[r] + padded_tile_q * c.hd_padded, 0.0f);
    const int64_t qh = kv_h * n_rep + r;
    const float sink_bias = ctx.has_sinks ? ctx.sinks_ptr[qh] : 0.0f;
    if (sink_bias != 0.0f) {
      std::fill(c.rmax_arr[r], c.rmax_arr[r] + padded_tile_q, sink_bias);
      std::fill(c.rsum_arr[r], c.rsum_arr[r] + padded_tile_q, 1.0f);
    } else {
      std::fill(
          c.rmax_arr[r],
          c.rmax_arr[r] + padded_tile_q,
          -std::numeric_limits<float>::infinity());
      std::fill(c.rsum_arr[r], c.rsum_arr[r] + padded_tile_q, 0.0f);
    }
  }

  // Copy Q tiles for all reps with pre-applied scale
  const int64_t q_base_offset = q_offset + qs;
  const __m512 vscale = _mm512_set1_ps(ctx.scale_f);
  const int64_t hd_vec = head_dim & ~15;
  for (int64_t r = 0; r < n_rep; ++r) {
    const int64_t qh = kv_h * n_rep + r;
    for (int64_t qi = 0; qi < tile_q; ++qi) {
      const at::BFloat16* src =
          ctx.q_ptr + (q_base_offset + qi) * ctx.token_stride + qh * head_dim;
      at::BFloat16* dst = c.q_tile_buf[r] + qi * head_dim;
      int64_t d = 0;
      for (; d < hd_vec; d += 16) {
        __m256i raw =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + d));
        __m512 fp32 = _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(raw));
        fp32 = _mm512_mul_ps(fp32, vscale);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst + d),
            reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(fp32)));
      }
      for (; d < head_dim; ++d)
        dst[d] = at::BFloat16(static_cast<float>(src[d]) * ctx.scale_f);
    }
    if (tile_q < padded_tile_q)
      std::memset(
          c.q_tile_buf[r] + tile_q * head_dim,
          0,
          (padded_tile_q - tile_q) * head_dim * sizeof(at::BFloat16));
  }

  // Sliding window: compute the first block that could be visible to any Q row.
  const int64_t min_q_pos = qs + (kv_len - aq);
  const int64_t max_q_pos = (qs + tile_q - 1) + (kv_len - aq);
  int64_t start_bi = 0;
  if (ctx.sliding_window > 0 && min_q_pos >= ctx.sliding_window) {
    start_bi = (min_q_pos - ctx.sliding_window + 1) / blk_size;
  }

  // Block loop: pack K, QK^T, mask, softmax, pack V, SV
  for (int64_t bi = start_bi; bi < n_blks; ++bi) {
    const int64_t pool_blk = block_indices[bi];
    const int64_t kv_start = bi * blk_size;
    const int64_t tokens = std::min(blk_size, kv_len - kv_start);

    if (kv_start > max_q_pos)
      break;
    const bool fully_unmasked =
        (kv_start + tokens - 1 <= min_q_pos) && (tokens == blk_size);

    const at::BFloat16* k_blk =
        ctx.pool_ptr + kv_h * ctx.ph + pool_blk * ctx.pb;
    const at::BFloat16* k_src = k_blk;
    if (tokens < blk_size) {
      std::memcpy(c.k_padded, k_blk, tokens * head_dim * sizeof(at::BFloat16));
      std::memset(
          c.k_padded + tokens * head_dim,
          0,
          (blk_size - tokens) * head_dim * sizeof(at::BFloat16));
      k_src = c.k_padded;
    }

    const int64_t k_sub_sz = head_dim * c.sub_n * sizeof(at::BFloat16);
    for (int64_t s = 0; s < c.n_sub; ++s)
      c.pack_k.execute(
          k_src + s * c.sub_n * head_dim, c.k_packed.data() + s * k_sub_sz);

    // For each rep: QK^T (all subs) + mask + online softmax
    for (int64_t r = 0; r < n_rep; ++r) {
      float* scores = c.scores_fp32;
      at::BFloat16* wbf16 = c.weights_bf16[r];

      for (int64_t s = 0; s < c.n_sub; ++s)
        c.brg_qkt.execute(
            c.q_tile_buf[r],
            c.k_packed.data() + s * k_sub_sz,
            c.unit_offset,
            scores + s * c.sub_n,
            c.scratchpad.data());

      // Apply causal mask + optional sliding window mask
      {
        const float neginf = -std::numeric_limits<float>::infinity();
        if (!fully_unmasked || ctx.sliding_window > 0) {
          for (int64_t qi = 0; qi < tile_q; ++qi) {
            const int64_t q_pos = (qs + qi) + (kv_len - aq);
            float* srow = scores + qi * blk_size;
            for (int64_t t = 0; t < blk_size; ++t) {
              const int64_t kv_pos = kv_start + t;
              const bool causal_ok = kv_pos <= q_pos && t < tokens;
              const bool window_ok = (ctx.sliding_window <= 0) ||
                  (q_pos - kv_pos < ctx.sliding_window);
              if (!(causal_ok && window_ok))
                srow[t] = neginf;
            }
          }
        }
      }

      // Online softmax per Q row: find block max, rescale running state,
      // compute exp weights as BF16 for the SV BRGeMM.
      for (int64_t qi = 0; qi < tile_q; ++qi) {
        const float* srow = scores + qi * blk_size;

        // Vectorized block-max over valid tokens
        __m512 vbmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        int64_t mt = 0;
        for (; mt + 16 <= tokens; mt += 16)
          vbmax = _mm512_max_ps(vbmax, _mm512_loadu_ps(srow + mt));
        float bmax = _mm512_reduce_max_ps(vbmax);
        for (; mt < tokens; ++mt)
          bmax = std::max(bmax, srow[mt]);

        // All scores masked — skip this Q row for this block
        if (bmax == -std::numeric_limits<float>::infinity()) {
          std::memset(
              wbf16 + qi * blk_size, 0, blk_size * sizeof(at::BFloat16));
          continue;
        }

        // Rescale accumulated output if block max exceeds running max
        float nmax = std::max(c.rmax_arr[r][qi], bmax);
        if (nmax > c.rmax_arr[r][qi]) {
          float corr;
          EXP_APPROX_SCALAR(c.rmax_arr[r][qi] - nmax, corr);
          c.rsum_arr[r][qi] *= corr;
          float* orow = c.out_fp32[r] + qi * c.hd_padded;
          const __m512 vcorr = _mm512_set1_ps(corr);
          int64_t cd = 0;
          for (; cd + 16 <= head_dim; cd += 16)
            _mm512_storeu_ps(
                orow + cd, _mm512_mul_ps(_mm512_loadu_ps(orow + cd), vcorr));
          for (; cd < head_dim; ++cd)
            orow[cd] *= corr;
        }

        // Exp weights: vertical accumulate for sum, convert to BF16 for SV
        const __m512 vnmax = _mm512_set1_ps(-nmax);
        __m512 vsum = _mm512_setzero_ps();
        int64_t t = 0;
        {
          EXP_APPROX_AVX512_CONSTANTS();
          for (; t + 16 <= tokens; t += 16) {
            __m512 s16 = _mm512_loadu_ps(srow + t);
            __m512 e16;
            EXP_APPROX_AVX512(_mm512_add_ps(s16, vnmax), e16);
            vsum = _mm512_add_ps(vsum, e16);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(wbf16 + qi * blk_size + t),
                reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(e16)));
          }
        }
        float rsum_local = _mm512_reduce_add_ps(vsum);
        for (; t < tokens; ++t) {
          float w;
          EXP_APPROX_SCALAR(srow[t] - nmax, w);
          rsum_local += w;
          wbf16[qi * blk_size + t] = at::BFloat16(w);
        }
        c.rsum_arr[r][qi] += rsum_local;
        // Zero padding weights so BRGeMM SV reads zeros for unused positions
        if (tokens < blk_size)
          std::memset(
              wbf16 + qi * blk_size + tokens,
              0,
              (blk_size - tokens) * sizeof(at::BFloat16));
        c.rmax_arr[r][qi] = nmax;
      }
    }

    // S@V: pack V and accumulate weighted sum via BRGeMM
    const at::BFloat16* v_blk = k_blk + ctx.pvo;
    const at::BFloat16* v_src = v_blk;
    if (tokens < blk_size) {
      std::memcpy(c.v_padded, v_blk, tokens * head_dim * sizeof(at::BFloat16));
      std::memset(
          c.v_padded + tokens * head_dim,
          0,
          (blk_size - tokens) * head_dim * sizeof(at::BFloat16));
      v_src = c.v_padded;
    }

    const int64_t v_sub_sz = c.sub_n * c.hd_sub * sizeof(at::BFloat16);
    const bool hd_needs_pad = (c.n_hd_sub * c.hd_sub > head_dim);
    for (int64_t hs = 0; hs < c.n_hd_sub; ++hs) {
      const bool last_hd = hd_needs_pad && (hs == c.n_hd_sub - 1);
      for (int64_t s = 0; s < c.n_sub; ++s) {
        const at::BFloat16* vs = v_src + s * c.sub_n * head_dim + hs * c.hd_sub;
        if (last_hd) {
          const int64_t valid_hd = head_dim - hs * c.hd_sub;
          at::BFloat16* vpad = c.v_padded;
          for (int64_t t = 0; t < c.sub_n; ++t) {
            std::memcpy(
                vpad + t * head_dim,
                vs + t * head_dim,
                valid_hd * sizeof(at::BFloat16));
            std::memset(
                vpad + t * head_dim + valid_hd,
                0,
                (c.hd_sub - valid_hd) * sizeof(at::BFloat16));
          }
          vs = vpad;
        }
        c.pack_v.execute(vs, c.v_packed.data() + s * v_sub_sz);
      }
      for (int64_t r = 0; r < n_rep; ++r)
        c.brg_sv.execute(
            c.weights_bf16[r],
            c.v_packed.data(),
            c.sv_offsets,
            c.out_fp32[r] + hs * c.hd_sub,
            c.scratchpad.data());
    }
  }

  brgemm::release_hw_context();

  // Normalize and write output to ragged positions
  for (int64_t r = 0; r < n_rep; ++r) {
    const int64_t qh = kv_h * n_rep + r;
    for (int64_t qi = 0; qi < tile_q; ++qi) {
      at::BFloat16* out =
          ctx.o_ptr + (q_base_offset + qi) * ctx.token_stride + qh * head_dim;
      const float* src = c.out_fp32[r] + qi * c.hd_padded;
      if (c.rsum_arr[r][qi] > 0.0f) {
        const __m512 vinv = _mm512_set1_ps(1.0f / c.rsum_arr[r][qi]);
        int64_t d = 0;
        for (; d + 16 <= head_dim; d += 16) {
          __m512 v = _mm512_mul_ps(_mm512_loadu_ps(src + d), vinv);
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(out + d),
              reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
        }
        const float inv = 1.0f / c.rsum_arr[r][qi];
        for (; d < head_dim; ++d)
          out[d] = at::BFloat16(src[d] * inv);
      }
    }
  }
}

} // namespace kernels
} // namespace pace
