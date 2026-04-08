/******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Inspired by vLLM (https://github.com/vllm-project/vllm)
 ******************************************************************************/

#ifndef PACE_PAGED_ATTENTION_CPU_ATTN_H
#define PACE_PAGED_ATTENTION_CPU_ATTN_H

#include <ATen/ATen.h>
#include <string>

at::Tensor get_scheduler_metadata(
    const int64_t num_req,
    const int64_t num_heads_q,
    const int64_t num_heads_kv,
    const int64_t head_dim,
    const at::Tensor& seq_lens,
    at::ScalarType dtype,
    const at::Tensor& query_start_loc,
    const bool causal,
    const int64_t window_size,
    const std::string& isa_hint,
    const bool enable_kv_split);

void cpu_attn_reshape_and_cache(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    const std::string& isa);

void cpu_attention_with_kv_cache(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& output,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    const double scale,
    const bool causal,
    const c10::optional<at::Tensor>& alibi_slopes,
    const int64_t sliding_window_left,
    const int64_t sliding_window_right,
    const at::Tensor& block_table,
    const double softcap,
    const at::Tensor& scheduler_metadata,
    const c10::optional<at::Tensor>& s_aux);

#endif // PACE_PAGED_ATTENTION_CPU_ATTN_H
