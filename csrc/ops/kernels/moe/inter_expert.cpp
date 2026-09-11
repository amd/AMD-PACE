// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#include "moe.h"

namespace pace {
namespace kernels {

// Method 2: Flat inter-expert tiles (decode) — 3 barriers
// Flatten (e, nk, s1) tiles across ALL experts; all threads share all work.
// Wins when per-expert token count is small and barrier cost dominates.
// Buffers are compact 1D [total_assignments, dim], indexed by
// expert_offsets[e].
template <typename GateActivationTPP>
void run_flat_tiles(MoEKernelContext& ctx) {
  // Per-expert buffer offsets using expert_offsets (compact 1D layout).
  // Buffers are [total_assignments, dim] — no padding between experts.
  auto perm_e = [&](int64_t e) {
    return ctx.perm_ptr + (*ctx.expert_offsets)[e] * ctx.C;
  };
  auto inter_e = [&](int64_t e) {
    return ctx.inter_base + (*ctx.expert_offsets)[e] * ctx.N;
  };
  auto out_e = [&](int64_t e) {
    return ctx.out_e_base + (*ctx.expert_offsets)[e] * ctx.K_out;
  };

  // Build flat lists of (e, nk, s1) tiles across all experts — outside OMP
  struct ETile {
    int64_t e;
    long nk;
    long s1;
  };
  std::vector<ETile> p1_tiles, p2_tiles;
  for (int64_t e = 0; e < ctx.num_experts; e++) {
    long BS_e = (*ctx.expert_counts)[e];
    if (BS_e == 0)
      continue;
    long blocks = (BS_e + ctx.BSb - 1) / ctx.BSb;
    for (long b = 0; b < blocks; b++) {
      long s1 = b * ctx.BSb;
      for (long nk = 0; nk < ctx.Nk; nk++)
        p1_tiles.push_back({e, nk, s1});
      for (long nk_d = 0; nk_d < ctx.Nk_d; nk_d++)
        p2_tiles.push_back({e, nk_d, s1});
    }
  }
  int np1 = (int)p1_tiles.size(), np2 = (int)p2_tiles.size();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    DataType* tile_a =
        ctx.tile_buf_ptr + tid * (ctx.gated ? 2L : 1L) * ctx.tile_elems;
    DataType* tile_b = ctx.gated ? tile_a + ctx.tile_elems : nullptr;

    // Gather — all experts simultaneously
    for (int64_t e = 0; e < ctx.num_experts; e++) {
      long BS_e = (*ctx.expert_counts)[e];
      long e_off = (*ctx.expert_offsets)[e];
#pragma omp for schedule(static) nowait
      for (int64_t i = 0; i < BS_e; i++)
        std::memcpy(
            perm_e(e) + i * ctx.C,
            ctx.hidden_ptr + (*ctx.sorted_token_idx)[e_off + i] * ctx.C,
            ctx.C * sizeof(DataType));
    }
#pragma omp barrier

    // Phase 1 — all (e, nk, s1) tiles across all experts
#pragma omp for schedule(static) nowait
    for (int ti = 0; ti < np1; ti++) {
      auto& t = p1_tiles[ti];
      long BS_e = (*ctx.expert_counts)[t.e];
      bool is_rem = (t.s1 + ctx.BSb > BS_e);
      auto& tpps = *(*ctx.expert_tpps)[t.e];
      DataType* inter_base = inter_e(t.e);
      // Prefetch next tile's weights → L2 (imminent, private to this thread)
      if (ti + 1 < np1) {
        auto& nx = p1_tiles[ti + 1];
        if (ctx.gated)
          _mm_prefetch(
              (const char*)ctx.wt_gate_vla[nx.e][nx.nk][0], _MM_HINT_T1);
        _mm_prefetch((const char*)ctx.wt_up_vla[nx.e][nx.nk][0], _MM_HINT_T1);
      }
      MOE_P1_TILE(
          t.e,
          t.nk,
          t.s1,
          is_rem,
          tile_a,
          tile_b,
          tpps,
          (perm_e(t.e) + t.s1 * ctx.C),
          ctx);
    }
#pragma omp barrier

    // Phase 2 — all (e, nk_d, s1) with nc_d L2 sub-batching
    for (long nc_d = 0; nc_d < ctx.Nc_d; nc_d += ctx.Ncb_d) {
      auto cnt = std::min(ctx.Ncb_d, ctx.Nc_d - nc_d);
#pragma omp for schedule(static) nowait
      for (int ti = 0; ti < np2; ti++) {
        auto& t = p2_tiles[ti];
        long BS_e = (*ctx.expert_counts)[t.e];
        bool is_rem = (t.s1 + ctx.BSb > BS_e);
        auto& tpps = *(*ctx.expert_tpps)[t.e];
        auto& brg_d = is_rem ? tpps.brgemm_down_rem : tpps.brgemm_down;
        DataType* out_ptr = out_e(t.e) + t.s1 * ctx.K_out + t.nk * ctx.Hk_d;
        DataType* int_ptr = inter_e(t.e) + t.s1 * ctx.N + nc_d * ctx.Hc_d;
        // Prefetch next nc_d sub-batch → L2
        if (nc_d + ctx.Ncb_d < ctx.Nc_d) {
          _mm_prefetch(
              (const char*)(inter_e(t.e) + t.s1 * ctx.N +
                            (nc_d + ctx.Ncb_d) * ctx.Hc_d),
              _MM_HINT_T1);
          _mm_prefetch(
              (const char*)ctx.wt_down_vla[t.e][t.nk][nc_d + ctx.Ncb_d],
              _MM_HINT_T1);
        }
        if (nc_d == 0) {
          if (ctx.with_down_bias) {
            auto& cb = is_rem ? tpps.cpbias_d_rem : tpps.cpbias_d;
            cb(ctx.dbias[t.e][t.nk], out_ptr);
          } else {
            auto& z = is_rem ? tpps.zero_d_rem : tpps.zero_d;
            z(out_ptr);
          }
        }
        brg_d(int_ptr, ctx.wt_down_vla[t.e][t.nk][nc_d], out_ptr, cnt, !is_rem);
      }
    }
#pragma omp barrier

    // Phase 3 — scatter: topk passes over num_tokens, one slot per pass.
    // Within each pass every token appears exactly once → no write-write
    // conflict. schedule(static) gives each thread the same token range across
    // all k passes, so each thread exclusively owns its rows of output_ptr. No
    // barrier needed between k-passes since different threads write to disjoint
    // token rows.
    for (int64_t k = 0; k < ctx.topk; k++) {
#pragma omp for schedule(static) nowait
      for (int64_t tok = 0; tok < ctx.num_tokens; tok++) {
        int64_t e = (*ctx.topk_ids)[tok * ctx.topk + k];
        int64_t local = (*ctx.token_local)[tok * ctx.topk + k];
        float w = (*ctx.routing_w)[tok * ctx.topk + k];
        moe_scatter_accumulate_bf16(
            ctx.output_ptr + tok * ctx.K_out,
            out_e(e) + local * ctx.K_out,
            w,
            ctx.K_out);
      }
    }
  } // omp parallel
}

#define INSTANTIATE_FLAT(ActType) \
  template void run_flat_tiles<ActType>(MoEKernelContext&);

INSTANTIATE_FLAT(SiLUFwdTPP<float>)
INSTANTIATE_FLAT(GeluFwdTPP<float>)
INSTANTIATE_FLAT(ReLUFwdTPP<float>)
INSTANTIATE_FLAT(GptOssGLUTag)
#undef INSTANTIATE_FLAT

} // namespace kernels
} // namespace pace
