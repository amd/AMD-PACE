/******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Inspired by vLLM's paged attention design
 * (https://github.com/vllm-project/vllm)
 *
 * CPU Paged Attention APIs using vLLM's optimized CPU attention kernels.
 ******************************************************************************/

#ifndef PACE_PAGED_ATTENTION_H
#define PACE_PAGED_ATTENTION_H

#include <ATen/ATen.h>
#include <string>

namespace pace {

/**
 * @brief Get scheduler metadata for CPU attention
 *
 * Uses vLLM's L2-cache-aware scheduler that distributes work items across
 * threads with KV-split support for long sequences.
 */
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
    bool enable_kv_split);

/**
 * @brief Reshape and cache key/value tensors into paged KV cache
 *
 * Uses vLLM's ISA-dispatched reshape_and_cache for optimal memory layout.
 */
void paged_attention_reshape_and_cache(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    const std::string& isa);

/**
 * @brief CPU paged attention with KV cache
 *
 * Uses vLLM's tiled GEMM attention kernel with online softmax, KV-split
 * scheduling, and ISA-specific optimizations (AVX512).
 * Works for both prefill (multi-token) and decode (single-token) paths.
 */
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
    const c10::optional<at::Tensor>& s_aux);

/**
 * @brief Get the shape for paged KV cache allocation
 */
std::vector<int64_t> get_paged_kv_cache_shape(
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim);

/**
 * @brief Determine the optimal ISA for the current CPU
 */
std::string get_optimal_attention_isa(
    at::ScalarType dtype,
    int64_t block_size,
    int64_t head_dim = 0);

} // namespace pace

#endif // PACE_PAGED_ATTENTION_H
