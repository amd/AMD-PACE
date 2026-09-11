/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * SlabPool: Global KV Cache Pool for Serving
 *
 * Implemented as a torch::CustomClassHolder for proper lifetime management,
 * zero-overhead state access, and clean Python API via
 * torch.classes.pace.SlabPool.
 *
 * Pool layout (configurable via SLAB_LAYOUT env var):
 *   head_major (default): [num_kv_heads, total_blocks, 2, block_size, head_dim]
 *   block_major:          [total_blocks, num_kv_heads, 2, block_size, head_dim]
 * Input/output layout (BSHD): [batch, seq_len, heads, head_dim]
 *
 * Natively supports:
 *   - Causal attention (sliding_window=0, sinks empty)
 *   - Sliding window attention (sliding_window>0)
 *   - Sinking attention (sinks tensor non-empty)
 *   - Combined sliding window + sinks
 ******************************************************************************/

#ifndef PACE_SLAB_POOL_H
#define PACE_SLAB_POOL_H

#include <ATen/ATen.h>
#include <ops/attention/slab/slab_kernels.h>
#include <torch/custom_class.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pace {
namespace kernels {

struct BlockMetadata {
  int64_t sequence_id;
  int64_t block_index;
};

struct SequenceState {
  int64_t sequence_id;
  int64_t seq_len;
  int64_t max_seq_len;
  std::vector<int64_t> block_indices;
};

struct SlabPool : torch::CustomClassHolder {
  at::Tensor pool_tensor;

  int64_t total_blocks;
  int64_t block_size;
  int64_t num_kv_heads;
  int64_t head_dim;

  std::vector<BlockMetadata> blocks;
  std::vector<int64_t> free_list;
  int64_t allocated_blocks;

  std::unordered_map<int64_t, SequenceState> sequences;

  bool block_major;
  int64_t pool_head_stride;
  int64_t pool_blk_stride;
  int64_t pool_kv_offset;

  int64_t splitk_max_splits;

  // Pre-allocated Split-K partial buffers (avoids 1MB heap alloc per call)
  std::vector<PartialSoftmax> splitk_partials_buf;

  std::mutex pool_mutex;
  std::mutex sequence_mutex;

  // Constructor
  SlabPool(
      int64_t total_blocks,
      int64_t num_kv_heads,
      int64_t head_dim,
      int64_t block_size);

  // Sequence management (seq_id assigned by Python SlabPoolManager)
  void create_sequence(int64_t seq_id, int64_t max_seq_len);
  void remove_sequence(int64_t sequence_id);
  void truncate_sequence(int64_t sequence_id, int64_t remove_len);
  int64_t get_sequence_length(int64_t sequence_id);
  int64_t get_free_block_count();

  // Cache update: accepts 3D [total_tokens, KV, D] (ragged) or
  // 4D [B, S, KV, D] (batched). 4D is auto-reshaped to 3D internally.
  // Includes single-token fast-path for decode.
  void cache_update(
      const std::vector<int64_t>& sequence_ids,
      const at::Tensor& keys,
      const at::Tensor& values,
      const std::vector<int64_t>& token_counts);

  // Unified attention: accepts 3D [total_tokens, H, D] (ragged) or
  // 4D [B, S, H, D] (batched). 4D is auto-reshaped to 3D internally.
  // Per-sequence 4-way dispatch: decode_gqa, decode_head, mtd, BRGeMM prefill.
  // sliding_window: 0 = full causal, >0 = window size
  // sinks: empty tensor = no sinks, [num_heads] = sink biases
  // chunk_size: 0 = disabled, >0 = chunked (local) attention window
  at::Tensor attention(
      const std::vector<int64_t>& sequence_ids,
      const at::Tensor& query,
      const std::vector<int64_t>& query_lens,
      const std::vector<int64_t>& q_start_offsets,
      double scale,
      int64_t sliding_window = 0,
      const at::Tensor& sinks = {},
      int64_t chunk_size = 0);

  // Expose physical block indices for a sequence (for prefix caching).
  std::vector<int64_t> get_block_indices(int64_t sequence_id);

  // Eagerly allocate blocks (without advancing seq_len) for prefix preinsert.
  void reserve_blocks(int64_t sequence_id, int64_t num_blocks);

  // Assign existing pool blocks to a sequence and advance seq_len.
  // Zero-copy: the destination simply references the same physical
  // blocks — no data is moved.  Used by prefix caching so a new
  // sequence can reuse cached KV data without re-computing it.
  void assign_prefix_blocks(
      int64_t dst_sequence_id,
      const std::vector<int64_t>& pool_block_indices,
      int64_t num_tokens);

  // Repoint one logical block of a sequence at another pool block (prefix
  // dedup).
  void reassign_block(
      int64_t sequence_id,
      int64_t block_pos,
      int64_t new_pool_block);

  // Mark blocks as "cached" so remove_sequence skips freeing them.
  // Returns how many blocks were newly marked.
  int64_t retain_blocks(const std::vector<int64_t>& pool_block_indices);

  // Release previously retained blocks back to the free list.
  void release_retained_blocks(const std::vector<int64_t>& pool_block_indices);

 private:
  int64_t allocate_block(int64_t sequence_id, int64_t block_index);
  std::unordered_set<int64_t> retained_blocks_;
};

int64_t autotune_block_size(int64_t num_kv_heads, int64_t head_dim);

} // namespace kernels
} // namespace pace

#endif // PACE_SLAB_POOL_H
