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

struct BrgemmKey {
  int64_t tile_q = 0, tokens = 0, head_dim = 0;

  bool operator==(const BrgemmKey& o) const {
    return tile_q == o.tile_q && tokens == o.tokens && head_dim == o.head_dim;
  }
};

// Cached per geometry because generate() is JIT codegen. Scratch memory is
// deliberately kept out of here so extra slots stay cheap.
struct BrgemmKernels {
  using dim = dnnl::memory::dim;

  BrgemmKey key;
  int64_t n_sub = 0, sub_n = 0; // token sub-blocking (K packing)
  int64_t n_hd_sub = 0, hd_sub = 0, hd_padded = 0; // head_dim sub-blocking

  dnnl::ukernel::brgemm brg_qkt;
  dnnl::ukernel::brgemm brg_sv;
  dnnl::ukernel::transform pack_k;
  dnnl::ukernel::transform pack_v;

  std::vector<std::pair<dim, dim>> unit_offset;
  std::vector<std::pair<dim, dim>> sv_offsets;

  bool valid = false;
  uint64_t last_used = 0;
};

// Shared across geometries, grown to the high-water mark and laid out for one
// geometry at a time (layout_key).
struct BrgemmBuffers {
  std::vector<uint8_t> scratchpad;
  std::vector<uint8_t> k_packed;
  std::vector<uint8_t> v_packed;

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

  BrgemmKey layout_key;
  int64_t n_rep_layout = 0;
  bool valid = false;
};

// Two cover the sliding/global head_dim mix; the rest is headroom.
constexpr int SLAB_BRGEMM_SLOTS = 4;

static thread_local BrgemmKernels tl_kernels[SLAB_BRGEMM_SLOTS];
static thread_local BrgemmBuffers tl_buffers;
static thread_local uint64_t tl_use_tick = 0;

void generate_kernels(BrgemmKernels& c, const BrgemmKey& key) {
  const int64_t tile_q = key.tile_q;
  const int64_t tokens = key.tokens;
  const int64_t head_dim = key.head_dim;

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

  c.unit_offset = {{0, 0}};

  int64_t v_sub_packed_bytes = sub_n * hd_sub * sizeof(at::BFloat16);
  c.sv_offsets.resize(n_sub);
  for (int64_t s = 0; s < n_sub; ++s) {
    c.sv_offsets[s] = {
        static_cast<memory::dim>(
            s * sub_n * static_cast<int64_t>(sizeof(at::BFloat16))),
        static_cast<memory::dim>(s * v_sub_packed_bytes)};
  }

  c.n_sub = n_sub;
  c.sub_n = sub_n;
  c.n_hd_sub = n_hd_sub;
  c.hd_sub = hd_sub;
  c.hd_padded = hd_padded;
  c.key = key;
  c.valid = true;
}

BrgemmKernels& acquire_kernels(const BrgemmKey& key) {
  for (auto& c : tl_kernels) {
    if (c.valid && c.key == key) {
      c.last_used = ++tl_use_tick;
      return c;
    }
  }

  BrgemmKernels* victim = &tl_kernels[0];
  for (auto& c : tl_kernels) {
    if (!c.valid) {
      victim = &c;
      break;
    }
    if (c.last_used < victim->last_used)
      victim = &c;
  }

  generate_kernels(*victim, key);
  victim->last_used = ++tl_use_tick;
  return *victim;
}

void ensure_buffers(BrgemmBuffers& b, const BrgemmKernels& k, int64_t n_rep) {
  TORCH_CHECK(
      n_rep > 0 && n_rep <= SLAB_MAX_REP,
      "slab prefill: n_rep=",
      n_rep,
      " must be in [1, SLAB_MAX_REP=",
      SLAB_MAX_REP,
      "]");

  const bool same_geometry = b.valid && b.layout_key == k.key;
  if (same_geometry && n_rep <= b.n_rep_layout)
    return;

  const int64_t tile_q = k.key.tile_q;
  const int64_t tokens = k.key.tokens;
  const int64_t head_dim = k.key.head_dim;
  // Widest layout seen, so a smaller n_rep does not relocate the per-rep
  // regions.
  const int64_t n_rep_layout =
      same_geometry ? std::max(n_rep, b.n_rep_layout) : n_rep;

  size_t sz =
      std::max(k.brg_qkt.get_scratchpad_size(), k.brg_sv.get_scratchpad_size());
  if (b.scratchpad.size() < sz + 64)
    b.scratchpad.resize(sz + 64);

  size_t k_need = k.n_sub * head_dim * k.sub_n * sizeof(at::BFloat16) + 64;
  if (b.k_packed.size() < k_need)
    b.k_packed.resize(k_need);

  size_t v_need = k.n_sub * k.sub_n * k.hd_sub * sizeof(at::BFloat16) + 64;
  if (b.v_packed.size() < v_need)
    b.v_packed.resize(v_need);

  const size_t al = 64;
  auto align = [al](size_t x) { return (x + al - 1) & ~(al - 1); };

  // Per-rep buffers: out_fp32, rmax, rsum, q_tile_buf, weights_bf16
  size_t per_rep = align(tile_q * k.hd_padded * sizeof(float)) +
      align(tile_q * sizeof(float)) + align(tile_q * sizeof(float)) +
      align(tile_q * head_dim * sizeof(at::BFloat16)) +
      align(tile_q * tokens * sizeof(at::BFloat16));

  // Shared buffers: scores, k_padded, v_padded
  size_t shared = align(tile_q * tokens * sizeof(float)) +
      align(tokens * head_dim * sizeof(at::BFloat16)) +
      align(tokens * head_dim * sizeof(at::BFloat16));

  size_t need = n_rep_layout * per_rep + shared;
  if (b.rep_mem.size() < need)
    b.rep_mem.resize(need);

  uint8_t* base = b.rep_mem.data();
  for (int64_t r = 0; r < n_rep_layout; ++r) {
    uint8_t* rb = base + r * per_rep;
    size_t off = 0;
    b.out_fp32[r] = reinterpret_cast<float*>(rb + off);
    off += align(tile_q * k.hd_padded * sizeof(float));
    b.rmax_arr[r] = reinterpret_cast<float*>(rb + off);
    off += align(tile_q * sizeof(float));
    b.rsum_arr[r] = reinterpret_cast<float*>(rb + off);
    off += align(tile_q * sizeof(float));
    b.q_tile_buf[r] = reinterpret_cast<at::BFloat16*>(rb + off);
    off += align(tile_q * head_dim * sizeof(at::BFloat16));
    b.weights_bf16[r] = reinterpret_cast<at::BFloat16*>(rb + off);
  }

  uint8_t* sb = base + n_rep_layout * per_rep;
  size_t off = 0;
  b.scores_fp32 = reinterpret_cast<float*>(sb + off);
  off += align(tile_q * tokens * sizeof(float));
  b.k_padded = reinterpret_cast<at::BFloat16*>(sb + off);
  off += align(tokens * head_dim * sizeof(at::BFloat16));
  b.v_padded = reinterpret_cast<at::BFloat16*>(sb + off);

  b.layout_key = k.key;
  b.n_rep_layout = n_rep_layout;
  b.valid = true;
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
  auto& c = tl_buffers;
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

  auto& kern = acquire_kernels(BrgemmKey{padded_tile_q, blk_size, head_dim});
  ensure_buffers(c, kern, n_rep);
  kern.brg_qkt.set_hw_context();

  // Init per-rep accumulators (with optional sink bias)
  for (int64_t r = 0; r < n_rep; ++r) {
    std::fill(
        c.out_fp32[r], c.out_fp32[r] + padded_tile_q * kern.hd_padded, 0.0f);
    const int64_t qh = kv_h * n_rep + r;
    const float sink_bias = ctx.has_sinks ? ctx.sinks_ptr[qh] : 0.0f;
    if (ctx.has_sinks) {
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

  // First block visible to any Q row in the tile (sliding + chunked window).
  // The visible lower bound is non-decreasing in q_pos, so the smallest q_pos
  // (min_q_pos) yields the earliest visible block for the whole tile.
  const int64_t min_q_pos = qs + (kv_len - aq);
  const int64_t max_q_pos = (qs + tile_q - 1) + (kv_len - aq);
  const int64_t start_bi = kv_visible_start(ctx, min_q_pos) / blk_size;

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

    const int64_t k_sub_sz = head_dim * kern.sub_n * sizeof(at::BFloat16);
    for (int64_t s = 0; s < kern.n_sub; ++s)
      kern.pack_k.execute(
          k_src + s * kern.sub_n * head_dim, c.k_packed.data() + s * k_sub_sz);

    // For each rep: QK^T (all subs) + mask + online softmax
    for (int64_t r = 0; r < n_rep; ++r) {
      float* scores = c.scores_fp32;
      at::BFloat16* wbf16 = c.weights_bf16[r];

      for (int64_t s = 0; s < kern.n_sub; ++s)
        kern.brg_qkt.execute(
            c.q_tile_buf[r],
            c.k_packed.data() + s * k_sub_sz,
            kern.unit_offset,
            scores + s * kern.sub_n,
            c.scratchpad.data());

      // Apply causal mask + optional sliding window mask
      {
        const float neginf = -std::numeric_limits<float>::infinity();
        if (!fully_unmasked || ctx.sliding_window > 0 || ctx.chunk_size > 0) {
          for (int64_t qi = 0; qi < tile_q; ++qi) {
            const int64_t q_pos = (qs + qi) + (kv_len - aq);
            const int64_t vis_start = kv_visible_start(ctx, q_pos);
            float* srow = scores + qi * blk_size;
            for (int64_t t = 0; t < blk_size; ++t) {
              const int64_t kv_pos = kv_start + t;
              const bool causal_ok = kv_pos <= q_pos && t < tokens;
              const bool window_ok = kv_pos >= vis_start;
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
          float* orow = c.out_fp32[r] + qi * kern.hd_padded;
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

    const int64_t v_sub_sz = kern.sub_n * kern.hd_sub * sizeof(at::BFloat16);
    const bool hd_needs_pad = (kern.n_hd_sub * kern.hd_sub > head_dim);
    for (int64_t hs = 0; hs < kern.n_hd_sub; ++hs) {
      const bool last_hd = hd_needs_pad && (hs == kern.n_hd_sub - 1);
      for (int64_t s = 0; s < kern.n_sub; ++s) {
        const at::BFloat16* vs =
            v_src + s * kern.sub_n * head_dim + hs * kern.hd_sub;
        if (last_hd) {
          const int64_t valid_hd = head_dim - hs * kern.hd_sub;
          at::BFloat16* vpad = c.v_padded;
          for (int64_t t = 0; t < kern.sub_n; ++t) {
            std::memcpy(
                vpad + t * head_dim,
                vs + t * head_dim,
                valid_hd * sizeof(at::BFloat16));
            std::memset(
                vpad + t * head_dim + valid_hd,
                0,
                (kern.hd_sub - valid_hd) * sizeof(at::BFloat16));
          }
          vs = vpad;
        }
        kern.pack_v.execute(vs, c.v_packed.data() + s * v_sub_sz);
      }
      for (int64_t r = 0; r < n_rep; ++r)
        kern.brg_sv.execute(
            c.weights_bf16[r],
            c.v_packed.data(),
            kern.sv_offsets,
            c.out_fp32[r] + hs * kern.hd_sub,
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
      const float* src = c.out_fp32[r] + qi * kern.hd_padded;
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
