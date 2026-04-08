/******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Inspired by vLLM's paged attention design
 * (https://github.com/vllm-project/vllm)
 *
 * CPU Paged Attention implementation using vLLM's optimized CPU attention
 * kernels. This replaces the previous naive implementation with vLLM's
 * L2-cache-aware, SIMD-optimized attention that supports tiled GEMM,
 * online softmax, KV-split scheduling, and works for both prefill and decode.
 ******************************************************************************/

#include <core/logging.h>
#include <ops/paged_attention.h>
#include <torch/library.h>
#include <utils/utils.h>

#include <omp.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#include "attention/paged/cpu_attn.h"

namespace pace {

static std::string map_isa_hint(const std::string& pace_isa_hint) {
  if (pace_isa_hint == "vec" || pace_isa_hint == "vec16") {
    return pace_isa_hint;
  } else if (
      pace_isa_hint == "avx512" || pace_isa_hint == "avx2" ||
      pace_isa_hint == "auto") {
    return "vec";
  } else {
    return "vec16";
  }
}

std::string get_optimal_attention_isa(
    at::ScalarType dtype,
    int64_t block_size,
    int64_t head_dim) {
  if (head_dim > 0 && head_dim % 32 != 0 && head_dim % 16 == 0) {
    return "vec16";
  }
  if (block_size % 32 == 0) {
    return "vec";
  }
  return "vec16";
}

at::Tensor get_paged_attention_scheduler_metadata(
    int64_t num_reqs,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const at::Tensor& seq_lens,
    at::ScalarType dtype,
    const at::Tensor& query_start_loc,
    bool causal,
    int64_t sliding_window_size,
    const std::string& isa_hint,
    bool enable_kv_split) {
  PROFILE_PACE_FUNCTION("get_paged_attention_scheduler_metadata");

  TORCH_CHECK(
      seq_lens.dim() == 1,
      "seq_lens must be 1D tensor, got ",
      seq_lens.dim(),
      "D");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt,
      "seq_lens must be int32, got ",
      seq_lens.scalar_type());
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
  TORCH_CHECK(
      query_start_loc.dim() == 1,
      "query_start_loc must be 1D tensor, got ",
      query_start_loc.dim(),
      "D");
  TORCH_CHECK(
      query_start_loc.scalar_type() == at::kInt,
      "query_start_loc must be int32, got ",
      query_start_loc.scalar_type());
  TORCH_CHECK(
      query_start_loc.is_contiguous(), "query_start_loc must be contiguous");

  std::string vllm_isa = map_isa_hint(isa_hint);

  return get_scheduler_metadata(
      num_reqs,
      num_heads_q,
      num_heads_kv,
      head_dim,
      seq_lens,
      dtype,
      query_start_loc,
      causal,
      sliding_window_size,
      vllm_isa,
      enable_kv_split);
}

void paged_attention_reshape_and_cache(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    const std::string& isa) {
  PROFILE_PACE_FUNCTION("paged_attention_reshape_and_cache");

  TORCH_CHECK(
      key.dim() == 3,
      "key must be 3D [num_tokens, num_kv_heads, head_dim], got ",
      key.dim(),
      "D");
  TORCH_CHECK(
      value.dim() == 3,
      "value must be 3D [num_tokens, num_kv_heads, head_dim], got ",
      value.dim(),
      "D");
  TORCH_CHECK(
      key_cache.dim() == 4,
      "key_cache must be 4D [num_blocks, num_kv_heads, block_size, head_dim], got ",
      key_cache.dim(),
      "D");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must be 4D [num_blocks, num_kv_heads, block_size, head_dim], got ",
      value_cache.dim(),
      "D");
  TORCH_CHECK(
      slot_mapping.dim() == 1,
      "slot_mapping must be 1D, got ",
      slot_mapping.dim(),
      "D");
  TORCH_CHECK(
      key.stride(2) == 1 && value.stride(2) == 1,
      "key and value must be contiguous in the last dimension");

  std::string vllm_isa = map_isa_hint(isa);

  at::Tensor slot_mapping_i64 = slot_mapping;
  if (slot_mapping.scalar_type() != at::kLong) {
    slot_mapping_i64 = slot_mapping.to(at::kLong);
  }

  cpu_attn_reshape_and_cache(
      key, value, key_cache, value_cache, slot_mapping_i64, vllm_isa);
}

void paged_attention_with_kv_cache(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& output,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    double scale,
    bool causal,
    const c10::optional<at::Tensor>& alibi_slopes,
    int64_t sliding_window_left,
    int64_t sliding_window_right,
    const at::Tensor& block_table,
    double softcap,
    const at::Tensor& scheduler_metadata,
    const c10::optional<at::Tensor>& s_aux) {
  PROFILE_PACE_FUNCTION("paged_attention_with_kv_cache");

  TORCH_CHECK(
      query.dim() == 3,
      "query must be 3D [num_tokens, num_heads, head_dim], got ",
      query.dim(),
      "D");
  TORCH_CHECK(
      key_cache.dim() == 4,
      "key_cache must be 4D [num_blocks, num_kv_heads, block_size, head_dim]");
  TORCH_CHECK(
      value_cache.dim() == 4,
      "value_cache must be 4D [num_blocks, num_kv_heads, block_size, head_dim]");
  TORCH_CHECK(
      output.dim() == 3, "output must be 3D [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(
      query.stride(2) == 1, "query must be contiguous in the last dimension");

  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt,
      "seq_lens must be int32, got ",
      seq_lens.scalar_type());
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
  TORCH_CHECK(
      query_start_loc.scalar_type() == at::kInt,
      "query_start_loc must be int32, got ",
      query_start_loc.scalar_type());
  TORCH_CHECK(
      query_start_loc.is_contiguous(), "query_start_loc must be contiguous");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt,
      "block_table must be int32, got ",
      block_table.scalar_type());
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(
      scheduler_metadata.is_contiguous(),
      "scheduler_metadata must be contiguous");

  cpu_attention_with_kv_cache(
      query,
      key_cache,
      value_cache,
      output,
      query_start_loc,
      seq_lens,
      scale,
      causal,
      alibi_slopes,
      sliding_window_left,
      sliding_window_right,
      block_table,
      softcap,
      scheduler_metadata,
      s_aux);
}

std::vector<int64_t> get_paged_kv_cache_shape(
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  return {num_blocks, num_kv_heads, block_size, head_dim};
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "get_paged_attention_scheduler_metadata("
      "int num_reqs, int num_heads_q, int num_heads_kv, int head_dim, "
      "Tensor seq_lens, ScalarType dtype, Tensor query_start_loc, "
      "bool causal, int sliding_window_size, str isa_hint, "
      "bool enable_kv_split) -> Tensor");
  m.def(
      "paged_attention_reshape_and_cache("
      "Tensor key, Tensor value, Tensor(a!) key_cache, "
      "Tensor(b!) value_cache, Tensor slot_mapping, str isa) -> ()");
  m.def(
      "paged_attention_with_kv_cache("
      "Tensor query, Tensor key_cache, Tensor value_cache, "
      "Tensor(a!) output, Tensor query_start_loc, Tensor seq_lens, "
      "float scale, bool causal, Tensor? alibi_slopes, "
      "int sliding_window_left, int sliding_window_right, "
      "Tensor block_table, float softcap, Tensor scheduler_metadata, "
      "Tensor? s_aux) -> ()");
  m.def(
      "get_paged_kv_cache_shape("
      "int num_blocks, int block_size, int num_kv_heads, int head_dim"
      ") -> int[]",
      pace::get_paged_kv_cache_shape);
  m.def(
      "get_optimal_attention_isa("
      "ScalarType dtype, int block_size, int head_dim=0) -> str",
      pace::get_optimal_attention_isa);
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl(
      "get_paged_attention_scheduler_metadata",
      pace::get_paged_attention_scheduler_metadata);
  m.impl(
      "paged_attention_reshape_and_cache",
      pace::paged_attention_reshape_and_cache);
  m.impl("paged_attention_with_kv_cache", pace::paged_attention_with_kv_cache);
}

} // namespace
