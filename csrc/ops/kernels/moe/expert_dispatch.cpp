// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#include "moe.h"

namespace pace {
namespace kernels {

// Routing-agnostic expert engine: histogram-sort + gather + FFN + scatter.
// Consumes pre-computed routing (topk_ids / routing_w vectors, `topk` columns
// per token).
template <typename GateActivationTPP>
static at::Tensor run_moe_engine(
    at::Tensor& t_output,
    const at::Tensor& t_hidden,
    const c10::optional<at::Tensor>& t_wt_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_wt_down,
    const c10::optional<at::Tensor>& t_gate_bias,
    const c10::optional<at::Tensor>& t_up_bias,
    const c10::optional<at::Tensor>& t_down_bias,
    int64_t num_experts,
    int64_t topk,
    std::vector<int64_t>& topk_ids,
    std::vector<float>& routing_w) {
  bool gated = t_wt_gate.has_value() && t_wt_gate.value().numel() > 0;

  int64_t num_tokens = t_hidden.size(0);
  int64_t C = t_hidden.size(1);

  // Extract dims from 6D weights: [E, Nk, Nc, 32, Hk, 2]
  auto wt_gu_sizes = (gated ? t_wt_gate.value() : t_wt_up).sizes();
  long Nk = wt_gu_sizes[1];
  long Nc = wt_gu_sizes[2];
  long Hc = C / Nc;
  long Hk = wt_gu_sizes[4];
  long N = Nk * Hk;

  auto wt_d_sizes = t_wt_down.sizes();
  long Nk_d = wt_d_sizes[1];
  long Nc_d = wt_d_sizes[2];
  long Hk_d = wt_d_sizes[4];
  long K_out = Nk_d * Hk_d;
  long Hc_d = N / Nc_d;

  long Ncb_d = std::min(Nc_d, (long)PACE_DOWN_NCB);

  int64_t total_assignments = num_tokens * topk;

  // Histogram-sort: count + prefix-sum + scatter
  std::vector<int64_t> expert_counts(num_experts, 0);
  for (int64_t i = 0; i < total_assignments; i++) {
    expert_counts[topk_ids[i]]++;
  }

  int64_t max_count =
      *std::max_element(expert_counts.begin(), expert_counts.end());
  if (max_count == 0)
    return t_output; // output already initialized by caller

  // Adaptive BSb: for decode (small max_count), size BSb to actual workload.
  // Avoids oversized tiles and unnecessary remainder handling.
  long BSb_default = (PACE_MLP_BSB > 0) ? (long)PACE_MLP_BSB
      : (max_count >= 512)              ? 128L
                                        : 64L;
  long BSb = std::min(BSb_default, max_count);

  std::vector<int64_t> expert_offsets(num_experts + 1, 0);
  for (int64_t e = 0; e < num_experts; e++) {
    expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e];
  }

  std::vector<int64_t> sorted_token_idx(total_assignments);
  std::vector<int64_t> sorted_topk_slot(total_assignments);
  // token_local[t*topk + k] = local offset of (tok=t, slot=k) within its
  // expert's output buf. Used by the flat-path scatter to look up
  // out_e(e)[local * K_out] per (token, k) without scanning sorted_token_idx in
  // reverse.
  std::vector<int64_t> token_local(total_assignments);
  {
    std::vector<int64_t> wp(num_experts, 0);
    for (int64_t t = 0; t < num_tokens; t++) {
      for (int64_t k = 0; k < topk; k++) {
        int64_t eid = topk_ids[t * topk + k];
        int64_t local = wp[eid]++;
        int64_t pos = expert_offsets[eid] + local;
        sorted_token_idx[pos] = t;
        sorted_topk_slot[pos] = k;
        token_local[t * topk + k] = local;
      }
    }
  }

  // =========================================================================
  // Allocate shared buffers
  // =========================================================================

  // Per-expert sequential buffers ([max_count, *]) are allocated only on the
  // sequential path below; the flat path uses its own compact buffers, so
  // allocating these unconditionally would waste peak memory.
  // t_output is caller-provided (pre-allocated zeros or shared expert result).
  at::Tensor t_perm_input, t_inter, t_expert_out;

  // Tile buffers: per-thread
  int max_threads = omp_get_max_threads();
  long tile_elems = BSb * Hk;
  auto t_tile_buf =
      at::empty({max_threads, gated ? 2L : 1L, tile_elems}, t_hidden.options());

  auto* hidden_ptr = t_hidden.data_ptr<DataType>();
  auto* output_ptr = t_output.data_ptr<DataType>();
  auto* tile_buf_ptr = t_tile_buf.data_ptr<DataType>();

  bool with_gate_bias = gated && has_bias(t_gate_bias);
  bool with_up_bias = has_bias(t_up_bias);
  bool with_down_bias = has_bias(t_down_bias);

  // VLA pointers for 6D weights
  auto wt_up_vla = GetVLAPtr<DataType>(t_wt_up, {Nk, Nc, Hc * Hk});
  auto wt_down_vla = GetVLAPtr<DataType>(t_wt_down, {Nk_d, Nc_d, Hc_d * Hk_d});

  auto dummy_6d = at::empty({1, Nk, Nc, 32, Hk, 2}, t_hidden.options());
  auto wt_gate_t = gated ? t_wt_gate.value() : dummy_6d;
  auto wt_gate_vla = GetVLAPtr<DataType>(wt_gate_t, {Nk, Nc, Hc * Hk});

  auto dummy_bias_gu = at::empty({1, N}, t_hidden.options());
  auto dummy_bias_d = at::empty({1, K_out}, t_hidden.options());
  auto gb_t = with_gate_bias ? t_gate_bias.value() : dummy_bias_gu;
  auto ub_t = with_up_bias ? t_up_bias.value() : dummy_bias_gu;
  auto db_t = with_down_bias ? t_down_bias.value() : dummy_bias_d;
  auto gbias = GetVLAPtr<DataType>(gb_t, {Nk, Hk});
  auto ubias = GetVLAPtr<DataType>(ub_t, {Nk, Hk});
  auto dbias = GetVLAPtr<DataType>(db_t, {Nk_d, Hk_d});

  // Pre-compute TPPs (JIT outside OMP)
  std::vector<MoEMLPTPPs*> expert_tpps(num_experts, nullptr);
  for (int64_t e = 0; e < num_experts; e++) {
    if (expert_counts[e] == 0)
      continue;
    long rem_e = expert_counts[e] % BSb;
    MoEMLPKey key{BSb, Hk, Hc, C, N, K_out, Nc, Nc_d, Hc_d, Hk_d, Ncb_d, rem_e};
    expert_tpps[e] = &get_moe_mlp_tpps(key);
  }

  // =========================================================================
  // Single OMP region
  // =========================================================================

  // Aggregate-initialize ctx with all fields at once (avoids VLAPtr default
  // constructor which does not exist). perm_ptr/inter_base/out_e_base are
  // path-dependent and filled in below before each call.
  MoEKernelContext ctx{
      gated,
      num_experts,
      topk,
      C,
      Nk,
      Nc,
      Hc,
      Hk,
      N,
      Nk_d,
      Nc_d,
      Hk_d,
      K_out,
      Hc_d,
      BSb,
      Ncb_d,
      num_tokens,
      total_assignments,
      max_count,
      &expert_counts,
      &expert_offsets,
      &sorted_token_idx,
      &sorted_topk_slot,
      &topk_ids,
      &token_local,
      &routing_w,
      &expert_tpps,
      hidden_ptr,
      nullptr, // perm_ptr  — set per-path below
      nullptr, // inter_base — set per-path below
      nullptr, // out_e_base — set per-path below
      output_ptr,
      tile_buf_ptr,
      tile_elems,
      wt_gate_vla,
      wt_up_vla,
      wt_down_vla,
      gbias,
      ubias,
      dbias,
      with_gate_bias,
      with_up_bias,
      with_down_bias,
  };

  // FLAT vs SEQ by tiles-per-thread-per-expert. SEQ (one expert at a time, 4*E
  // barriers) only wins once an expert has enough tiles to fill the cores and
  // amortize barriers via L3 weight reuse; else FLAT wins. _tpte divides by nt,
  // so this fixed threshold auto-scales with core count: SEQ engages only at
  // very large per-expert token counts (big prefill on many cores).
  static int FLAT_THRESH = env2int("PACE_MOE_FLAT_THRESHOLD", 16);
  int _nt = omp_get_max_threads();
  int _tpte = (int)(((max_count + BSb - 1) / BSb) * Nk + _nt - 1) / _nt;
  bool use_flat = (_tpte < FLAT_THRESH);

  if (use_flat) {
    // Flat path: compact 1D buffers sized to total_assignments (exact), indexed
    // via expert_offsets[e] — eliminates waste from per-expert load imbalance.
    auto t_perm_flat = at::empty({total_assignments, C}, t_hidden.options());
    auto t_inter_flat = at::empty({total_assignments, N}, t_hidden.options());
    auto t_out_flat = at::empty({total_assignments, K_out}, t_hidden.options());
    ctx.perm_ptr = t_perm_flat.data_ptr<DataType>();
    ctx.inter_base = t_inter_flat.data_ptr<DataType>();
    ctx.out_e_base = t_out_flat.data_ptr<DataType>();
    run_flat_tiles<GateActivationTPP>(ctx);
  } else {
    t_perm_input = at::empty({max_count, C}, t_hidden.options());
    t_inter = at::empty({max_count, N}, t_hidden.options());
    t_expert_out = at::empty({max_count, K_out}, t_hidden.options());
    ctx.perm_ptr = t_perm_input.data_ptr<DataType>();
    ctx.inter_base = t_inter.data_ptr<DataType>();
    ctx.out_e_base = t_expert_out.data_ptr<DataType>();
    run_expert_seq<GateActivationTPP>(ctx);
  }

  return t_output;
}

// Engine kernel: routing-agnostic. Routing is provided as topk_ids/topk_weights
// (any scheme, any shared-expert handling already encoded in the arrays).
template <typename GateActivationTPP>
static at::Tensor moe_engine_kernel(
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
    int64_t num_experts) {
  int64_t num_tokens = t_topk_ids.size(0);
  int64_t topk = t_topk_ids.size(1);
  int64_t total = num_tokens * topk;

  auto ids = t_topk_ids.to(at::kLong).contiguous();
  auto wts = t_topk_weights.to(at::kFloat).contiguous();
  const int64_t* ids_ptr = ids.data_ptr<int64_t>();
  const float* wts_ptr = wts.data_ptr<float>();
  std::vector<int64_t> topk_ids(ids_ptr, ids_ptr + total);
  std::vector<float> routing_w(wts_ptr, wts_ptr + total);

  return run_moe_engine<GateActivationTPP>(
      t_output,
      t_hidden,
      t_wt_gate,
      t_wt_up,
      t_wt_down,
      t_gate_bias,
      t_up_bias,
      t_down_bias,
      num_experts,
      topk,
      topk_ids,
      routing_w);
}

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
    int64_t num_experts) {
#define CALL_ENGINE(ActType)         \
  return moe_engine_kernel<ActType>( \
      t_output,                      \
      t_hidden,                      \
      t_topk_ids,                    \
      t_topk_weights,                \
      t_wt_gate,                     \
      t_wt_up,                       \
      t_wt_down,                     \
      t_gate_bias,                   \
      t_up_bias,                     \
      t_down_bias,                   \
      num_experts)

  if (activation == "silu") {
    CALL_ENGINE(SiLUFwdTPP<float>);
  } else if (activation == "gelu") {
    CALL_ENGINE(GeluFwdTPP<float>);
  } else if (activation == "relu") {
    CALL_ENGINE(ReLUFwdTPP<float>);
  } else if (activation == "gptoss_glu") {
    CALL_ENGINE(GptOssGLUTag);
  } else {
    TORCH_CHECK(
        false,
        "moe_engine_dispatch: unsupported activation '",
        activation,
        "', must be one of: silu, gelu, relu, gptoss_glu");
  }
#undef CALL_ENGINE
}

} // namespace kernels
} // namespace pace
