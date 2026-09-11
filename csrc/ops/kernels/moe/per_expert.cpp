// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#include "moe.h"

namespace pace {
namespace kernels {

// Method 1: Expert-sequential (prefill) — 4*E barriers
// One expert processed at a time; all threads share each expert's tiles.
// Benefits: L3 weight reuse (all threads read same expert weights), clean
// L2 nc_d sub-batching for down projection.
template <typename GateActivationTPP>
void run_expert_seq(MoEKernelContext& ctx) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthr = omp_get_num_threads();
    DataType* tile_a =
        ctx.tile_buf_ptr + tid * (ctx.gated ? 2L : 1L) * ctx.tile_elems;
    DataType* tile_b = ctx.gated ? tile_a + ctx.tile_elems : nullptr;

    for (int64_t e = 0; e < ctx.num_experts; e++) {
      long BS_e = (*ctx.expert_counts)[e];
      if (BS_e == 0)
        continue;
      long e_off = (*ctx.expert_offsets)[e];
      long BS_blocks_e = (BS_e + ctx.BSb - 1) / ctx.BSb;
      auto& tpps = *(*ctx.expert_tpps)[e];

      // Next active expert for weight prefetch
      int64_t next_e = -1;
      for (int64_t ne = e + 1; ne < ctx.num_experts; ne++)
        if ((*ctx.expert_counts)[ne] > 0) {
          next_e = ne;
          break;
        }

      // Gather
#pragma omp for schedule(static) nowait
      for (int64_t i = 0; i < BS_e; i++)
        std::memcpy(
            ctx.perm_ptr + i * ctx.C,
            ctx.hidden_ptr + (*ctx.sorted_token_idx)[e_off + i] * ctx.C,
            ctx.C * sizeof(DataType));
#pragma omp barrier

      // Phase 1: distribute (nk, s1_block) tiles across threads
      int total_tiles = (int)(BS_blocks_e * ctx.Nk);
      int tpt = (total_tiles + nthr - 1) / nthr;
      int t0 = tid * tpt, t1 = std::min(t0 + tpt, total_tiles);

      // inter_base resolves MOE_P1_TILE macro: inter_base + s1*N + nk*Hk
      DataType* inter_base = ctx.inter_base;

      for (int tile = t0; tile < t1; tile++) {
        long nk = tile / BS_blocks_e;
        long s1 = (tile % (int)BS_blocks_e) * ctx.BSb;
        bool is_rem = (s1 + ctx.BSb > BS_e);
        // Weight prefetch: next nk block → L2 (imminent, private to this
        // thread)
        if (tile + 1 < t1) {
          long nx = (tile + 1) / (int)BS_blocks_e;
          if (nx != nk) {
            if (ctx.gated)
              _mm_prefetch((const char*)ctx.wt_gate_vla[e][nx][0], _MM_HINT_T1);
            _mm_prefetch((const char*)ctx.wt_up_vla[e][nx][0], _MM_HINT_T1);
          }
        }
        if (tile == t0 && next_e >= 0) {
          if (ctx.gated)
            _mm_prefetch(
                (const char*)ctx.wt_gate_vla[next_e][0][0], _MM_HINT_T2);
          _mm_prefetch((const char*)ctx.wt_up_vla[next_e][0][0], _MM_HINT_T2);
        }
        MOE_P1_TILE(
            e,
            nk,
            s1,
            is_rem,
            tile_a,
            tile_b,
            tpps,
            (ctx.perm_ptr + s1 * ctx.C),
            ctx);
      }
#pragma omp barrier

      // Phase 2: down projection with L2-aware nc_d sub-batching
      if (next_e >= 0)
        _mm_prefetch((const char*)ctx.wt_down_vla[next_e][0][0], _MM_HINT_T2);
      for (long nc_d = 0; nc_d < ctx.Nc_d; nc_d += ctx.Ncb_d) {
        auto cnt = std::min(ctx.Ncb_d, ctx.Nc_d - nc_d);
#pragma omp for collapse(2) schedule(static) nowait
        for (long nk_d = 0; nk_d < ctx.Nk_d; nk_d++) {
          for (long s1 = 0; s1 < BS_e; s1 += ctx.BSb) {
            bool is_rem = (s1 + ctx.BSb > BS_e);
            auto& brg_d = is_rem ? tpps.brgemm_down_rem : tpps.brgemm_down;
            DataType* out_ptr =
                ctx.out_e_base + s1 * ctx.K_out + nk_d * ctx.Hk_d;
            if (nc_d + ctx.Ncb_d < ctx.Nc_d) {
              _mm_prefetch(
                  (const char*)(ctx.inter_base + s1 * ctx.N +
                                (nc_d + ctx.Ncb_d) * ctx.Hc_d),
                  _MM_HINT_T1);
              _mm_prefetch(
                  (const char*)ctx.wt_down_vla[e][nk_d][nc_d + ctx.Ncb_d],
                  _MM_HINT_T1);
            }
            if (nc_d == 0) {
              if (ctx.with_down_bias) {
                auto& cb = is_rem ? tpps.cpbias_d_rem : tpps.cpbias_d;
                cb(ctx.dbias[e][nk_d], out_ptr);
              } else {
                auto& z = is_rem ? tpps.zero_d_rem : tpps.zero_d;
                z(out_ptr);
              }
            }
            brg_d(
                ctx.inter_base + s1 * ctx.N + nc_d * ctx.Hc_d,
                ctx.wt_down_vla[e][nk_d][nc_d],
                out_ptr,
                cnt,
                !is_rem);
          }
        }
      }
#pragma omp barrier

      // Phase 3: scatter weighted outputs into final output
#pragma omp for schedule(static) nowait
      for (int64_t i = 0; i < BS_e; i++) {
        int64_t tok = (*ctx.sorted_token_idx)[e_off + i];
        float w = (*ctx.routing_w)
            [tok * ctx.topk + (*ctx.sorted_topk_slot)[e_off + i]];
        moe_scatter_accumulate_bf16(
            ctx.output_ptr + tok * ctx.K_out,
            ctx.out_e_base + i * ctx.K_out,
            w,
            ctx.K_out);
      }
#pragma omp barrier
    } // expert loop
  } // omp parallel
}

// Explicit instantiations — must be in same TU as the template definition
#define INSTANTIATE_SEQ(ActType) \
  template void run_expert_seq<ActType>(MoEKernelContext&);

INSTANTIATE_SEQ(SiLUFwdTPP<float>)
INSTANTIATE_SEQ(GeluFwdTPP<float>)
INSTANTIATE_SEQ(ReLUFwdTPP<float>)
INSTANTIATE_SEQ(GptOssGLUTag)
#undef INSTANTIATE_SEQ

} // namespace kernels
} // namespace pace
