// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#include "libxsmm_moe.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include "core/logging.h"
#include "kernels/moe/moe.h"

namespace pace {

static inline bool has_value(const c10::optional<at::Tensor>& t) {
  return t.has_value() && t.value().numel() > 0 && t.value().dim() > 0;
}

std::tuple<at::Tensor, at::Tensor> moe_route_softmax(
    const at::Tensor& router_logits,
    int64_t topk,
    bool renormalize) {
  PROFILE_PACE_FUNCTION("moe_route_softmax");
  TORCH_CHECK(
      router_logits.scalar_type() == at::kBFloat16 ||
          router_logits.scalar_type() == at::kFloat,
      "moe_route_softmax: router_logits must be bfloat16 or float32");
  return kernels::moe_route_softmax_dispatch(router_logits, topk, renormalize);
}

std::tuple<at::Tensor, at::Tensor> moe_route_softmax(
    const at::Tensor& router_logits,
    int64_t topk) {
  return moe_route_softmax(router_logits, topk, true);
}

void libxsmm_moe_engine(
    at::Tensor& output,
    const at::Tensor& hidden_states,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    const c10::optional<at::Tensor>& wt_gate,
    const at::Tensor& wt_up,
    const at::Tensor& wt_down,
    const c10::optional<at::Tensor>& gate_bias,
    const c10::optional<at::Tensor>& up_bias,
    const c10::optional<at::Tensor>& down_bias,
    const std::string& activation,
    int64_t num_experts) {
  PROFILE_PACE_FUNCTION("libxsmm_moe_engine");

  // --- Input validation ---
  TORCH_CHECK(
      hidden_states.dim() == 2,
      "moe_engine: hidden_states must be 2D [num_tokens, hidden_size]");
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kBFloat16,
      "moe_engine: hidden_states must be bfloat16");
  TORCH_CHECK(
      hidden_states.is_contiguous(),
      "moe_engine: hidden_states must be contiguous");

  int64_t num_tokens = hidden_states.size(0);

  TORCH_CHECK(
      output.dim() == 2 && output.sizes() == hidden_states.sizes(),
      "moe_engine: output shape must match hidden_states shape");
  TORCH_CHECK(
      output.scalar_type() == at::kBFloat16,
      "moe_engine: output must be bfloat16");
  TORCH_CHECK(output.is_contiguous(), "moe_engine: output must be contiguous");

  // Routing inputs: [num_tokens, topk]
  TORCH_CHECK(
      topk_ids.dim() == 2 && topk_ids.size(0) == num_tokens,
      "moe_engine: topk_ids must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kLong || topk_ids.scalar_type() == at::kInt,
      "moe_engine: topk_ids must be an integer tensor (int32/int64)");
  TORCH_CHECK(
      topk_weights.dim() == 2 && topk_weights.size(0) == num_tokens &&
          topk_weights.size(1) == topk_ids.size(1),
      "moe_engine: topk_weights must match topk_ids shape [num_tokens, topk]");
  // Validate expert ids are in range BEFORE the kernel histograms them: an
  // out-of-range id would be an out-of-bounds write into expert_counts.
  if (topk_ids.numel() > 0) {
    auto ids_long = topk_ids.to(at::kLong);
    int64_t lo = ids_long.min().item<int64_t>();
    int64_t hi = ids_long.max().item<int64_t>();
    TORCH_CHECK(
        lo >= 0 && hi < num_experts,
        "moe_engine: topk_ids out of range [0, num_experts=",
        num_experts,
        "); got [",
        lo,
        ", ",
        hi,
        "]");
  }

  // Weights: 6D [E, Nk, Nc, 32, Hk, 2]
  TORCH_CHECK(wt_up.dim() == 6, "moe_engine: wt_up must be 6D");
  TORCH_CHECK(wt_down.dim() == 6, "moe_engine: wt_down must be 6D");
  TORCH_CHECK(
      wt_down.size(1) * wt_down.size(4) == output.size(1),
      "moe_engine: output hidden dimension must match wt_down output size");
  TORCH_CHECK(
      wt_up.size(0) == num_experts,
      "moe_engine: wt_up first dim must equal num_experts");
  TORCH_CHECK(
      wt_down.size(0) == num_experts,
      "moe_engine: wt_down first dim must equal num_experts");
  if (has_value(wt_gate)) {
    auto& g = wt_gate.value();
    TORCH_CHECK(g.dim() == 6, "moe_engine: wt_gate must be 6D");
    TORCH_CHECK(
        g.size(0) == num_experts,
        "moe_engine: wt_gate first dim must equal num_experts");
    for (int d = 1; d < 6; d++) {
      TORCH_CHECK(
          g.size(d) == wt_up.size(d),
          "moe_engine: wt_gate and wt_up shape mismatch at dim ",
          d);
    }
  }
  TORCH_CHECK(
      activation == "silu" || activation == "gelu" || activation == "relu" ||
          activation == "gptoss_glu",
      "moe_engine: unsupported activation '",
      activation,
      "', must be silu/gelu/relu/gptoss_glu");
  TORCH_CHECK(
      activation != "gptoss_glu" || has_value(wt_gate),
      "moe_engine: activation 'gptoss_glu' requires wt_gate to be provided");

  // Void in-place op (schema `-> ()`): writes into `output`. See the note on
  // libxsmm_fused_moe re: torch.compile functionalization.
  kernels::moe_engine_dispatch(
      output,
      hidden_states,
      topk_ids,
      topk_weights,
      wt_gate,
      wt_up,
      wt_down,
      gate_bias,
      up_bias,
      down_bias,
      activation,
      num_experts);
}

} // namespace pace

// Torch library registration
TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "moe_route_softmax("
      "Tensor router_logits, "
      "int topk, "
      "bool renormalize=True"
      ") -> (Tensor, Tensor)");
  m.def(
      "libxsmm_moe_engine("
      "Tensor(a!) output, "
      "Tensor hidden_states, "
      "Tensor topk_ids, "
      "Tensor topk_weights, "
      "Tensor? wt_gate, "
      "Tensor wt_up, "
      "Tensor wt_down, "
      "Tensor? gate_bias, "
      "Tensor? up_bias, "
      "Tensor? down_bias, "
      "str activation, "
      "int num_experts"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  using MoeRouteSoftmaxFn =
      std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&, int64_t, bool);
  m.impl(
      "moe_route_softmax",
      static_cast<MoeRouteSoftmaxFn>(&pace::moe_route_softmax));
  m.impl("libxsmm_moe_engine", pace::libxsmm_moe_engine);
}
