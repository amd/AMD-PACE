// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#include "moe.h"
#include "ops/exp_approx.h"

namespace pace {
namespace kernels {

void inline_topk_softmax(
    const at::Tensor& t_router_logits,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t topk,
    bool renormalize,
    std::vector<int64_t>& topk_ids, // [num_tokens * topk]
    std::vector<float>& routing_w) { // [num_tokens * topk]
  // Convert router logits to fp32 row-by-row and find topk + softmax inline.
  bool is_bf16 = (t_router_logits.scalar_type() == at::kBFloat16);
  const DataType* bf16_ptr =
      is_bf16 ? t_router_logits.data_ptr<DataType>() : nullptr;
  const float* fp32_ptr = is_bf16 ? nullptr : t_router_logits.data_ptr<float>();

  // Stack-allocated heap for general topk (k <= 16 avoids malloc)
  struct KV {
    float val;
    int64_t idx;
  };

  // Each token is independent — parallelize across tokens.
  // Thread-local heap_vec avoids allocation contention for topk > 16.
#pragma omp parallel
  {
    std::vector<KV> heap_vec;

#pragma omp for schedule(static)
    for (int64_t t = 0; t < num_tokens; t++) {
      KV heap[16]; // stack buffer for topk <= 16
      KV* hp;
      if (topk <= 16) {
        hp = heap;
      } else {
        heap_vec.resize(topk);
        hp = heap_vec.data();
      }

      float global_max = -std::numeric_limits<float>::infinity();
      float global_sum = 0.0f;
      auto accumulate_global_softmax = [&](float value) {
        if (renormalize)
          return;
        if (value > global_max) {
          float correction;
          EXP_APPROX_SCALAR(global_max - value, correction);
          global_sum = global_sum * correction + 1.0f;
          global_max = value;
        } else {
          float exp_value;
          EXP_APPROX_SCALAR(value - global_max, exp_value);
          global_sum += exp_value;
        }
      };

      // Initialize heap with first topk elements
      for (int64_t e = 0; e < topk; e++) {
        float v = is_bf16 ? static_cast<float>(bf16_ptr[t * num_experts + e])
                          : fp32_ptr[t * num_experts + e];
        hp[e] = {v, e};
        accumulate_global_softmax(v);
      }

      // Build min-heap (root = smallest of topk largest)
      auto cmp = [](const KV& a, const KV& b) { return a.val > b.val; };
      std::make_heap(hp, hp + topk, cmp);

      // Scan remaining experts
      for (int64_t e = topk; e < num_experts; e++) {
        float v = is_bf16 ? static_cast<float>(bf16_ptr[t * num_experts + e])
                          : fp32_ptr[t * num_experts + e];
        accumulate_global_softmax(v);
        if (v > hp[0].val) {
          std::pop_heap(hp, hp + topk, cmp);
          hp[topk - 1] = {v, e};
          std::push_heap(hp, hp + topk, cmp);
        }
      }

      // Sort descending for deterministic output
      std::sort(hp, hp + topk, [](const KV& a, const KV& b) {
        return a.val > b.val;
      });

      if (renormalize) {
        float max_v = hp[0].val;
        float sum = 0.0f;
        for (int64_t k = 0; k < topk; k++) {
          float e_val;
          EXP_APPROX_SCALAR(hp[k].val - max_v, e_val);
          hp[k].val = e_val;
          sum += e_val;
        }
        for (int64_t k = 0; k < topk; k++) {
          topk_ids[t * topk + k] = hp[k].idx;
          routing_w[t * topk + k] = hp[k].val / sum;
        }
      } else {
        for (int64_t k = 0; k < topk; k++) {
          float weight;
          EXP_APPROX_SCALAR(hp[k].val - global_max, weight);
          topk_ids[t * topk + k] = hp[k].idx;
          routing_w[t * topk + k] = weight / global_sum;
        }
      }
    }
  } // omp parallel
}

std::tuple<at::Tensor, at::Tensor> moe_route_softmax_dispatch(
    const at::Tensor& t_router_logits,
    int64_t topk,
    bool renormalize) {
  TORCH_CHECK(
      t_router_logits.dim() == 2,
      "moe_route_softmax: router_logits must be 2D [num_tokens, num_experts]");
  int64_t num_tokens = t_router_logits.size(0);
  int64_t num_experts = t_router_logits.size(1);
  TORCH_CHECK(
      topk > 0 && topk <= num_experts,
      "moe_route_softmax: topk must be in [1, num_experts]");

  auto logits = t_router_logits.contiguous();
  std::vector<int64_t> topk_ids(num_tokens * topk);
  std::vector<float> routing_w(num_tokens * topk);
  inline_topk_softmax(
      logits, num_tokens, num_experts, topk, renormalize, topk_ids, routing_w);

  auto t_ids =
      at::empty({num_tokens, topk}, t_router_logits.options().dtype(at::kLong));
  auto t_wts = at::empty(
      {num_tokens, topk}, t_router_logits.options().dtype(at::kFloat));
  std::memcpy(
      t_ids.data_ptr<int64_t>(),
      topk_ids.data(),
      sizeof(int64_t) * topk_ids.size());
  std::memcpy(
      t_wts.data_ptr<float>(),
      routing_w.data(),
      sizeof(float) * routing_w.size());
  return std::make_tuple(t_ids, t_wts);
}

} // namespace kernels
} // namespace pace
