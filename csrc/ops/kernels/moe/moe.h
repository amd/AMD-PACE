// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

// Single header for all MoE kernel internals.
// External code only includes libxsmm_moe.h (the public API).

#pragma once

#include <ATen/ATen.h>
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ops/libxsmm_dependency/tensor_helper.h"
#include "ops/libxsmm_dependency/xsmm_functors.h"

namespace pace {
namespace kernels {

using DataType = at::BFloat16;

// Tuning knobs (defined in tpp_jit_cache.cpp)
extern int PACE_MLP_BSB;
extern int PACE_DOWN_NCB;

// Helper (defined in tpp_jit_cache.cpp)
bool has_bias(const c10::optional<at::Tensor>& b);

// TPP cache key / struct — defined in tpp_jit_cache.cpp
inline size_t moe_hash_combine(size_t seed, size_t v) {
  return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

struct MoEMLPKey {
  long BSb, Hk, Hc, C, N, K_out, Nc, Nc_d, Hc_d, Hk_d, Ncb_d, rem;
  bool operator==(const MoEMLPKey& o) const {
    return BSb == o.BSb && Hk == o.Hk && Hc == o.Hc && C == o.C && N == o.N &&
        K_out == o.K_out && Nc == o.Nc && Nc_d == o.Nc_d && Hc_d == o.Hc_d &&
        Hk_d == o.Hk_d && Ncb_d == o.Ncb_d && rem == o.rem;
  }
};

struct MoEMLPKeyHash {
  size_t operator()(const MoEMLPKey& k) const {
    size_t h = 0;
    for (auto v :
         {k.BSb,
          k.Hk,
          k.Hc,
          k.C,
          k.N,
          k.K_out,
          k.Nc,
          k.Nc_d,
          k.Hc_d,
          k.Hk_d,
          k.Ncb_d,
          k.rem})
      h = moe_hash_combine(h, std::hash<long>()(v));
    return h;
  }
};

struct MoEMLPTPPs {
  BrgemmTPP<DataType, DataType> brgemm_gu, brgemm_gu_rem;
  BrgemmTPP<DataType, DataType> brgemm_gu_bias, brgemm_gu_bias_rem;
  CpyBiasTPP<DataType> cpbias_gu, cpbias_gu_rem;
  SiLUFwdTPP<DataType> silu, silu_rem;
  GeluFwdTPP<DataType> gelu, gelu_rem;
  ReLUFwdTPP<DataType> relu, relu_rem;
  MulTPP<DataType, DataType> mul, mul_rem;
  CpyTPP<DataType> cpy, cpy_rem;
  BrgemmTPP<DataType, DataType> brgemm_down, brgemm_down_rem;
  SetZeroTPP<DataType> zero_d, zero_d_rem;
  CpyBiasTPP<DataType> cpbias_d, cpbias_d_rem;
};

MoEMLPTPPs& get_moe_mlp_tpps(const MoEMLPKey& key);

// Activation dispatch
template <typename T>
struct MoEActSelector {};

template <>
struct MoEActSelector<SiLUFwdTPP<float>> {
  using type = SiLUFwdTPP<DataType>;
  static type& get(MoEMLPTPPs& t, bool r) {
    return r ? t.silu_rem : t.silu;
  }
};
template <>
struct MoEActSelector<GeluFwdTPP<float>> {
  using type = GeluFwdTPP<DataType>;
  static type& get(MoEMLPTPPs& t, bool r) {
    return r ? t.gelu_rem : t.gelu;
  }
};
template <>
struct MoEActSelector<ReLUFwdTPP<float>> {
  using type = ReLUFwdTPP<DataType>;
  static type& get(MoEMLPTPPs& t, bool r) {
    return r ? t.relu_rem : t.relu;
  }
};

template <typename ActTPP>
static inline void moe_apply_act(ActTPP& act, DataType* g, DataType* o) {
  if constexpr (std::is_same_v<ActTPP, ReLUFwdTPP<DataType>>)
    act(g, o, nullptr);
  else
    act(g, o);
}

// GptOss GLU activation: gate.clamp(7)*sigmoid(gate*1.702),
// (up.clamp(±7)+1)*glu. Defined in gptoss_glu_avx512.cpp.
void gptoss_glu_activation(
    DataType* tile_gate,
    DataType* tile_up,
    long rows,
    long cols,
    long ld);

// GptOss GLU activation tag + implementation
struct GptOssGLUTag {};

template <>
struct MoEActSelector<GptOssGLUTag> {
  using type = SiLUFwdTPP<DataType>;
  static type& get(MoEMLPTPPs& t, bool r) {
    return r ? t.silu_rem : t.silu;
  }
};

// Inline top-k + softmax (no PyTorch dispatch) — defined in expert_routing.cpp
void inline_topk_softmax(
    const at::Tensor& t_router_logits,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t topk,
    bool renormalize,
    std::vector<int64_t>& topk_ids,
    std::vector<float>& routing_w);

// VLA type aliases
using WtVLA3 =
    decltype(GetVLAPtr<DataType>(std::declval<at::Tensor&>(), {1L, 1L, 1L}));
using BiasVLA2 =
    decltype(GetVLAPtr<DataType>(std::declval<at::Tensor&>(), {1L, 1L}));

// MoEKernelContext — groups all parameters shared between run_moe_engine,
// run_expert_seq, and run_flat_tiles into a single struct.
struct MoEKernelContext {
  // Shape / config (set once from weight dims)
  bool gated;
  int64_t num_experts, topk;
  long C, Nk, Nc, Hc, Hk, N;
  long Nk_d, Nc_d, Hk_d, K_out, Hc_d;
  long BSb, Ncb_d;

  // Routing results (from the provided topk_ids / topk_weights)
  int64_t num_tokens, total_assignments, max_count;
  const std::vector<int64_t>* expert_counts;
  const std::vector<int64_t>* expert_offsets;
  const std::vector<int64_t>* sorted_token_idx;
  const std::vector<int64_t>* sorted_topk_slot;
  const std::vector<int64_t>*
      topk_ids; // [num_tokens * topk] expert id per (tok, k)
  const std::vector<int64_t>*
      token_local; // [num_tokens * topk] local offset within expert buf
  const std::vector<float>* routing_w;
  const std::vector<MoEMLPTPPs*>* expert_tpps;

  // Buffer pointers
  DataType* hidden_ptr;
  DataType* perm_ptr;
  DataType* inter_base;
  DataType* out_e_base;
  DataType* output_ptr;
  DataType* tile_buf_ptr;
  long tile_elems;

  // Weight VLAs
  WtVLA3 wt_gate_vla, wt_up_vla, wt_down_vla;
  BiasVLA2 gbias, ubias, dbias;
  bool with_gate_bias, with_up_bias, with_down_bias;
};

// run_expert_seq — expert-sequential, 4*E barriers (prefill)
// Defined in per_expert.cpp
template <typename GateActivationTPP>
void run_expert_seq(MoEKernelContext& ctx);

// run_flat_tiles — inter-expert parallelism, 3 barriers total (decode).
// Defined in inter_expert.cpp
template <typename GateActivationTPP>
void run_flat_tiles(MoEKernelContext& ctx);

// Phase-1 tile macros — used in per_expert.cpp and inter_expert.cpp
// Defined here so they're visible across all kernel files.

// Project (gate or up) weight block into tile_dst.
// IN_S1 = pointer to the first row of this s1-block in the gather buffer.
// ctx   = MoEKernelContext& providing Nc and TPP objects.
#define MOE_PROJECT(                                                         \
    tile_dst, wt_nk, bias_nk, use_bias, is_rem, tpps, IN_S1, ctx)            \
  do {                                                                       \
    if (use_bias) {                                                          \
      auto& _cb = (is_rem) ? tpps.cpbias_gu_rem : tpps.cpbias_gu;            \
      auto& _brg = (is_rem) ? tpps.brgemm_gu_bias_rem : tpps.brgemm_gu_bias; \
      _cb(bias_nk, tile_dst);                                                \
      _brg(IN_S1, wt_nk, tile_dst, ctx.Nc, !(is_rem));                       \
    } else {                                                                 \
      auto& _brg = (is_rem) ? tpps.brgemm_gu_rem : tpps.brgemm_gu;           \
      _brg(IN_S1, wt_nk, tile_dst, ctx.Nc, !(is_rem));                       \
    }                                                                        \
  } while (0)

// Complete Phase-1 for one (e, nk, s1) tile: project → activate → copy to
// inter. ctx = MoEKernelContext& providing all weight VLAs, bias flags, and
// dims. inter_base must be a local DataType* pointing to this expert's
// intermediate buf.
#define MOE_P1_TILE(e, nk, s1, is_rem, tile_a, tile_b, tpps, in_s1, ctx)  \
  do {                                                                    \
    auto& _act = MoEActSelector<GateActivationTPP>::get(tpps, is_rem);    \
    auto& _cp = (is_rem) ? tpps.cpy_rem : tpps.cpy;                       \
    if constexpr (std::is_same_v<GateActivationTPP, GptOssGLUTag>) {      \
      MOE_PROJECT(                                                        \
          tile_a,                                                         \
          ctx.wt_gate_vla[e][nk][0],                                      \
          ctx.with_gate_bias ? ctx.gbias[e][nk] : nullptr,                \
          ctx.with_gate_bias,                                             \
          is_rem,                                                         \
          tpps,                                                           \
          in_s1,                                                          \
          ctx);                                                           \
      MOE_PROJECT(                                                        \
          tile_b,                                                         \
          ctx.wt_up_vla[e][nk][0],                                        \
          ctx.with_up_bias ? ctx.ubias[e][nk] : nullptr,                  \
          ctx.with_up_bias,                                               \
          is_rem,                                                         \
          tpps,                                                           \
          in_s1,                                                          \
          ctx);                                                           \
      long _rows = (is_rem) ? ((*ctx.expert_counts)[e] - (s1)) : ctx.BSb; \
      gptoss_glu_activation(tile_a, tile_b, _rows, ctx.Hk, ctx.Hk);       \
    } else if (ctx.gated) {                                               \
      MOE_PROJECT(                                                        \
          tile_a,                                                         \
          ctx.wt_gate_vla[e][nk][0],                                      \
          ctx.with_gate_bias ? ctx.gbias[e][nk] : nullptr,                \
          ctx.with_gate_bias,                                             \
          is_rem,                                                         \
          tpps,                                                           \
          in_s1,                                                          \
          ctx);                                                           \
      MOE_PROJECT(                                                        \
          tile_b,                                                         \
          ctx.wt_up_vla[e][nk][0],                                        \
          ctx.with_up_bias ? ctx.ubias[e][nk] : nullptr,                  \
          ctx.with_up_bias,                                               \
          is_rem,                                                         \
          tpps,                                                           \
          in_s1,                                                          \
          ctx);                                                           \
      moe_apply_act(_act, tile_a, tile_a);                                \
      auto& _m = (is_rem) ? tpps.mul_rem : tpps.mul;                      \
      _m(tile_b, tile_a, tile_a);                                         \
    } else {                                                              \
      MOE_PROJECT(                                                        \
          tile_a,                                                         \
          ctx.wt_up_vla[e][nk][0],                                        \
          ctx.with_up_bias ? ctx.ubias[e][nk] : nullptr,                  \
          ctx.with_up_bias,                                               \
          is_rem,                                                         \
          tpps,                                                           \
          in_s1,                                                          \
          ctx);                                                           \
      moe_apply_act(_act, tile_a, tile_a);                                \
    }                                                                     \
    _cp(tile_a, inter_base + (s1) * ctx.N + (nk) * ctx.Hk);               \
  } while (0)

// Engine dispatcher (routing-agnostic) — defined in expert_dispatch.cpp
at::Tensor moe_engine_dispatch(
    at::Tensor& t_output,
    const at::Tensor& t_hidden,
    const at::Tensor& t_topk_ids,
    const at::Tensor& t_topk_weights,
    const c10::optional<at::Tensor>& t_wt_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_wt_down,
    const c10::optional<at::Tensor>& t_gate_bias,
    const c10::optional<at::Tensor>& t_up_bias,
    const c10::optional<at::Tensor>& t_down_bias,
    const std::string& activation,
    int64_t num_experts);

// Standalone top-k softmax router — defined in expert_routing.cpp.
// Returns (topk_ids [num_tokens, topk] int64, topk_weights [num_tokens, topk]
// float32).
std::tuple<at::Tensor, at::Tensor> moe_route_softmax_dispatch(
    const at::Tensor& t_router_logits,
    int64_t topk,
    bool renormalize);

// Scatter-accumulate: dst[j] += src[j] * weight, AVX-512 BF16 optimized
// Implemented in expert_scatter_avx512.cpp
void moe_scatter_accumulate_bf16(
    at::BFloat16* dst,
    const at::BFloat16* src,
    float weight,
    int64_t n);

} // namespace kernels
} // namespace pace
