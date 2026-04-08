/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * SlabPool: Pool management, sequence management, and cache update.
 *
 * attention() lives in slab_attention_ragged_avx512.cpp and handles
 * both 3D [T,H,D] and 4D [B,S,H,D] input (auto-reshape).
 * cache_update() lives here and handles both 3D and 4D likewise.
 ******************************************************************************/

#include <omp.h>
#include <ops/attention/slab/dpbf16_kernels.h>
#include <ops/attention/slab/slab_kernels.h>
#include <ops/attention/slab/slab_pool.h>
#include <cstring>
#include <fstream>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifndef __AVX512F__
#error "slab_pool_avx512.cpp requires AVX512F."
#endif

#include <immintrin.h>

namespace pace {
namespace kernels {

// L2 Cache Detection
namespace {

int64_t read_sysfs_cache_size(const char* path) {
  std::ifstream f(path);
  if (!f.is_open())
    return 0;
  std::string s;
  f >> s;
  if (s.empty())
    return 0;
  try {
    int64_t sz = std::stoll(s);
    char c = s.back();
    if (c == 'K' || c == 'k')
      sz *= 1024;
    else if (c == 'M' || c == 'm')
      sz *= 1024 * 1024;
    return sz > 0 ? sz : 0;
  } catch (...) {
    return 0;
  }
}

#ifdef __x86_64__
// cpuid leaf 0x04: deterministic cache parameters.
// index: 0=L1d, 1=L1i, 2=L2, 3=L3
int64_t read_cpuid_cache_size(int index) {
  unsigned int eax, ebx, ecx, edx;
  __cpuid(0, eax, ebx, ecx, edx);
  if (eax < 0x04)
    return 0;
  __cpuid_count(0x04, index, eax, ebx, ecx, edx);
  if ((eax & 0x1F) == 0)
    return 0;
  int64_t line_size = (ebx & 0xFFF) + 1;
  int64_t partitions = ((ebx >> 12) & 0x3FF) + 1;
  int64_t ways = ((ebx >> 22) & 0x3FF) + 1;
  int64_t sets = ecx + 1;
  int64_t sz = line_size * partitions * ways * sets;
  return sz > 0 ? sz : 0;
}
#endif

} // namespace

int64_t autotune_block_size(int64_t num_kv_heads, int64_t head_dim) {
  int64_t l2_size =
      read_sysfs_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size");
#ifdef __x86_64__
  if (l2_size == 0)
    l2_size = read_cpuid_cache_size(2);
#endif
  if (l2_size == 0)
    return 64;
  int64_t bytes_per_token = 2 * num_kv_heads * head_dim * sizeof(at::BFloat16);
  int64_t target_bytes = l2_size / 4;

  for (int64_t block_size : {256, 128, 64, 32}) {
    int64_t block_bytes = block_size * bytes_per_token;
    if (block_bytes <= target_bytes) {
      return block_size;
    }
  }

  return 32;
}

// Constructor
SlabPool::SlabPool(
    int64_t total_blocks,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
  TORCH_CHECK(
      block_size > 0 && block_size <= 256 && block_size % 16 == 0,
      "SlabPool: block_size=",
      block_size,
      " must be in {16, 32, 64, 128, 256} (multiple of 16, BRGeMM sub-blocking)");

  TORCH_CHECK(
      head_dim <= 512,
      "SlabPool: head_dim=",
      head_dim,
      " exceeds PartialSoftmax buffer size (512) and is currently not supported");

  this->total_blocks = total_blocks;
  this->block_size = block_size;
  this->num_kv_heads = num_kv_heads;
  this->head_dim = head_dim;
  this->allocated_blocks = 0;

  const char* layout_env = std::getenv("SLAB_LAYOUT");
  this->block_major =
      (layout_env != nullptr && std::string(layout_env) == "block_major");

  auto options = at::TensorOptions()
                     .dtype(at::kBFloat16)
                     .device(at::kCPU)
                     .requires_grad(false);

  if (this->block_major) {
    this->pool_tensor = at::zeros(
        {total_blocks, num_kv_heads, 2, block_size, head_dim}, options);
    this->pool_head_stride = 2 * block_size * head_dim;
    this->pool_blk_stride = num_kv_heads * 2 * block_size * head_dim;
  } else {
    this->pool_tensor = at::zeros(
        {num_kv_heads, total_blocks, 2, block_size, head_dim}, options);
    this->pool_head_stride = total_blocks * 2 * block_size * head_dim;
    this->pool_blk_stride = 2 * block_size * head_dim;
  }
  this->pool_kv_offset = block_size * head_dim;

  {
    int64_t num_threads = omp_get_max_threads();
    const char* env_ms = std::getenv("SLAB_MAX_SPLITS");
    this->splitk_max_splits = env_ms
        ? std::atoi(env_ms)
        : std::min(
              num_threads / std::max(num_kv_heads, int64_t(1)),
              int64_t(SLAB_MAX_SPLITS));
    if (this->splitk_max_splits < 2)
      this->splitk_max_splits = 2;
  }

  // Pre-allocate Split-K partials: num_threads * SLAB_MAX_REP entries.
  // Avoids 1MB heap allocation per attention call.
  // Invariant: the Split-K heuristic (sp >= 4, gqa_items < num_threads)
  // ensures total_partials <= num_threads * n_rep <= num_threads *
  // SLAB_MAX_REP. The TORCH_CHECK in attention() catches overflow at runtime.
  // If splitk_max_splits or the sp threshold changes, re-evaluate this sizing.
  {
    int64_t num_threads = omp_get_max_threads();
    this->splitk_partials_buf.resize(
        static_cast<size_t>(num_threads * SLAB_MAX_REP));
  }

  this->blocks.resize(total_blocks);
  this->free_list.reserve(total_blocks);

  for (int64_t i = 0; i < total_blocks; ++i) {
    this->blocks[i].sequence_id = -1;
    this->blocks[i].block_index = -1;
    this->free_list.push_back(total_blocks - 1 - i);
  }
}

// Pool Query
int64_t SlabPool::get_free_block_count() {
  std::lock_guard<std::mutex> lock(pool_mutex);
  return static_cast<int64_t>(free_list.size());
}

// Sequence Management
void SlabPool::create_sequence(int64_t seq_id, int64_t max_seq_len) {
  int64_t blocks_needed = (max_seq_len + block_size - 1) / block_size;

  SequenceState seq_state;
  seq_state.sequence_id = seq_id;
  seq_state.seq_len = 0;
  seq_state.max_seq_len = max_seq_len;
  seq_state.block_indices.reserve(blocks_needed);

  {
    std::lock_guard<std::mutex> lock(sequence_mutex);
    TORCH_CHECK(
        sequences.find(seq_id) == sequences.end(),
        "SlabPool: sequence ",
        seq_id,
        " already exists");
    sequences[seq_id] = std::move(seq_state);
  }
}

void SlabPool::remove_sequence(int64_t sequence_id) {
  std::lock_guard<std::mutex> seq_lock(sequence_mutex);

  auto it = sequences.find(sequence_id);
  TORCH_CHECK(
      it != sequences.end(), "SlabPool: sequence ", sequence_id, " not found");

  SequenceState& seq = it->second;

  {
    std::lock_guard<std::mutex> pool_lock(pool_mutex);
    for (int64_t block_idx : seq.block_indices) {
      blocks[block_idx].sequence_id = -1;
      blocks[block_idx].block_index = -1;
      free_list.push_back(block_idx);
    }

    allocated_blocks -= static_cast<int64_t>(seq.block_indices.size());
  }

  sequences.erase(it);
}

void SlabPool::truncate_sequence(int64_t sequence_id, int64_t remove_len) {
  TORCH_CHECK(remove_len >= 0, "SlabPool: remove_len must be non-negative");

  std::lock_guard<std::mutex> seq_lock(sequence_mutex);

  auto it = sequences.find(sequence_id);
  TORCH_CHECK(
      it != sequences.end(), "SlabPool: sequence ", sequence_id, " not found");

  SequenceState& seq = it->second;

  TORCH_CHECK(
      remove_len <= seq.seq_len,
      "SlabPool: cannot remove ",
      remove_len,
      " tokens from sequence with ",
      seq.seq_len,
      " tokens");

  int64_t new_len = seq.seq_len - remove_len;
  int64_t old_blocks_needed = (seq.seq_len + block_size - 1) / block_size;
  int64_t new_blocks_needed =
      (new_len > 0) ? (new_len + block_size - 1) / block_size : 0;

  if (new_blocks_needed < old_blocks_needed) {
    std::lock_guard<std::mutex> pool_lock(pool_mutex);

    for (int64_t i = new_blocks_needed; i < old_blocks_needed; ++i) {
      int64_t block_idx = seq.block_indices[i];
      blocks[block_idx].sequence_id = -1;
      blocks[block_idx].block_index = -1;
      free_list.push_back(block_idx);
    }

    allocated_blocks -= (old_blocks_needed - new_blocks_needed);
    seq.block_indices.resize(new_blocks_needed);
  }

  seq.seq_len = new_len;
}

int64_t SlabPool::get_sequence_length(int64_t sequence_id) {
  std::lock_guard<std::mutex> lock(sequence_mutex);
  auto it = sequences.find(sequence_id);
  TORCH_CHECK(
      it != sequences.end(), "SlabPool: sequence ", sequence_id, " not found");
  return it->second.seq_len;
}

// Block Allocation (private)
int64_t SlabPool::allocate_block(int64_t sequence_id, int64_t block_index) {
  std::lock_guard<std::mutex> lock(pool_mutex);

  TORCH_CHECK(!free_list.empty(), "SlabPool: no free blocks available");

  int64_t block_idx = free_list.back();
  free_list.pop_back();

  blocks[block_idx].sequence_id = sequence_id;
  blocks[block_idx].block_index = block_index;

  allocated_blocks++;

  return block_idx;
}

// Unified Cache Update -- handles 3D [T, KV, D] and 4D [B, S, KV, D]
void SlabPool::cache_update(
    const std::vector<int64_t>& sequence_ids,
    const at::Tensor& keys,
    const at::Tensor& values,
    const std::vector<int64_t>& token_counts) {
  const int64_t n_seq = static_cast<int64_t>(sequence_ids.size());
  const int64_t bsize = this->block_size;

  // Detect input layout and extract dimensions
  const bool is_4d = (keys.dim() == 4);
  const int64_t n_kv_heads = is_4d ? keys.size(2) : keys.size(1);
  const int64_t hdim = is_4d ? keys.size(3) : keys.size(2);
  const int64_t new_tokens_max =
      is_4d ? keys.size(1) : 0; // only meaningful for 4D

  // Detect single-token decode: check tensor shape (4D) or token_counts (3D)
  bool all_single = false;
  if (is_4d && new_tokens_max == 1) {
    all_single = true;
  } else if (!is_4d && !token_counts.empty()) {
    all_single = true;
    for (int64_t i = 0; i < n_seq && all_single; ++i)
      if (token_counts[i] != 1)
        all_single = false;
  }

  if (all_single) {
    // DECODE FAST-PATH: single token per sequence
    struct DecodeInfo {
      SequenceState* seq;
      int64_t pool_blk;
      int64_t pos_in_blk;
    };
    std::vector<DecodeInfo> dinfos(static_cast<size_t>(n_seq));

    {
      std::lock_guard<std::mutex> lock(sequence_mutex);
      for (int64_t i = 0; i < n_seq; ++i) {
        auto it = sequences.find(sequence_ids[i]);
        TORCH_CHECK(it != sequences.end(), "SlabPool: sequence not found");
        SequenceState& seq = it->second;
        TORCH_CHECK(
            seq.seq_len + 1 <= seq.max_seq_len,
            "SlabPool: sequence would exceed max_seq_len");
        int64_t target_blk = seq.seq_len / bsize;
        if (target_blk >= static_cast<int64_t>(seq.block_indices.size())) {
          int64_t new_block = allocate_block(sequence_ids[i], target_blk);
          seq.block_indices.push_back(new_block);
        }
        dinfos[i].seq = &seq;
        dinfos[i].pool_blk = seq.block_indices[target_blk];
        dinfos[i].pos_in_blk = seq.seq_len % bsize;
      }
    }

    const at::BFloat16* key_ptr = keys.data_ptr<at::BFloat16>();
    const at::BFloat16* val_ptr = values.data_ptr<at::BFloat16>();
    at::BFloat16* pool_ptr = pool_tensor.data_ptr<at::BFloat16>();

    const int64_t ph = this->pool_head_stride;
    const int64_t pb = this->pool_blk_stride;
    const int64_t pvo = this->pool_kv_offset;

    const int64_t key_batch_stride = is_4d ? keys.stride(0) : n_kv_heads * hdim;
    const int64_t key_head_stride = is_4d ? keys.stride(2) : hdim;
    const int64_t val_batch_stride =
        is_4d ? values.stride(0) : n_kv_heads * hdim;
    const int64_t val_head_stride = is_4d ? values.stride(2) : hdim;

    const int64_t total_work = n_seq * n_kv_heads;
    const int64_t total_bytes =
        total_work * hdim * int64_t(sizeof(at::BFloat16)) * 2;

#pragma omp parallel for schedule(static) if (total_bytes > 64 * 1024)
    for (int64_t wi = 0; wi < total_work; ++wi) {
      const int64_t i = wi / n_kv_heads;
      const int64_t h = wi % n_kv_heads;
      const DecodeInfo& di = dinfos[i];

      at::BFloat16* dst_k =
          pool_ptr + h * ph + di.pool_blk * pb + di.pos_in_blk * hdim;
      at::BFloat16* dst_v = dst_k + pvo;

      const at::BFloat16* src_k =
          key_ptr + i * key_batch_stride + h * key_head_stride;
      const at::BFloat16* src_v =
          val_ptr + i * val_batch_stride + h * val_head_stride;

      dpbf16::copy_bf16_avx512(dst_k, src_k, hdim);
      dpbf16::copy_bf16_avx512(dst_v, src_v, hdim);
    }

    {
      std::lock_guard<std::mutex> lock(sequence_mutex);
      for (int64_t i = 0; i < n_seq; ++i)
        dinfos[i].seq->seq_len += 1;
    }
    return;
  }

  // GENERAL MULTI-TOKEN PATH
  // Derive per-sequence token counts
  const int64_t uniform_tokens = is_4d ? keys.size(1) : 0;
  const bool use_uniform = is_4d && token_counts.empty();

  if (!is_4d && token_counts.empty()) {
    TORCH_CHECK(
        keys.size(0) % n_seq == 0,
        "SlabPool: 3D input size(0)=",
        keys.size(0),
        " not divisible by n_seq=",
        n_seq,
        "; provide explicit token_counts");
  }

  struct SeqInfo {
    SequenceState* seq;
    int64_t old_seq_len;
    int64_t new_tokens;
    int64_t src_offset;
  };
  std::vector<SeqInfo> infos(static_cast<size_t>(n_seq));

  {
    std::lock_guard<std::mutex> lock(sequence_mutex);
    for (int64_t i = 0; i < n_seq; ++i) {
      auto it = sequences.find(sequence_ids[i]);
      TORCH_CHECK(it != sequences.end(), "SlabPool: sequence not found");
      SequenceState* seq = &it->second;
      int64_t nt = use_uniform   ? uniform_tokens
          : token_counts.empty() ? (keys.size(is_4d ? 1 : 0) / n_seq)
                                 : token_counts[i];
      TORCH_CHECK(
          seq->seq_len + nt <= seq->max_seq_len,
          "SlabPool: sequence would exceed max_seq_len");

      infos[i].seq = seq;
      infos[i].old_seq_len = seq->seq_len;
      infos[i].new_tokens = nt;
      infos[i].src_offset = is_4d ? (keys.size(1) - nt) : 0;

      int64_t new_seq_len = seq->seq_len + nt;
      int64_t first_blk = seq->seq_len / bsize;
      int64_t last_blk = (new_seq_len - 1) / bsize;
      for (int64_t blk = first_blk; blk <= last_blk; ++blk) {
        if (blk >= static_cast<int64_t>(seq->block_indices.size())) {
          int64_t new_block = allocate_block(sequence_ids[i], blk);
          seq->block_indices.push_back(new_block);
        }
      }
    }
  }

  const at::BFloat16* key_ptr = keys.data_ptr<at::BFloat16>();
  const at::BFloat16* val_ptr = values.data_ptr<at::BFloat16>();
  at::BFloat16* pool_ptr = pool_tensor.data_ptr<at::BFloat16>();

  const int64_t ph = this->pool_head_stride;
  const int64_t pb = this->pool_blk_stride;
  const int64_t pvo = this->pool_kv_offset;

  // Source strides: use tensor strides for 4D, computed for 3D
  const int64_t src_batch_stride = is_4d ? keys.stride(0) : 0;
  const int64_t src_seq_stride = is_4d ? keys.stride(1) : n_kv_heads * hdim;
  const int64_t src_head_stride = is_4d ? keys.stride(2) : hdim;
  const int64_t vsrc_batch_stride = is_4d ? values.stride(0) : 0;
  const int64_t vsrc_seq_stride = is_4d ? values.stride(1) : n_kv_heads * hdim;
  const int64_t vsrc_head_stride = is_4d ? values.stride(2) : hdim;

  // For 3D, compute per-sequence source token offsets via prefix sum
  std::vector<int64_t> seq_tok_offset(static_cast<size_t>(n_seq), 0);
  if (!is_4d) {
    int64_t off = 0;
    for (int64_t i = 0; i < n_seq; ++i) {
      seq_tok_offset[i] = off;
      off += infos[i].new_tokens;
    }
  }

  int64_t total_tokens = 0;
  for (int64_t i = 0; i < n_seq; ++i)
    total_tokens += infos[i].new_tokens;
  const int64_t prefill_bytes =
      total_tokens * n_kv_heads * hdim * int64_t(sizeof(at::BFloat16)) * 2;

#pragma omp parallel for schedule(static) if (prefill_bytes > 64 * 1024)
  for (int64_t wi = 0; wi < n_seq * n_kv_heads; ++wi) {
    const int64_t i = wi / n_kv_heads;
    const int64_t h = wi % n_kv_heads;
    const auto& info = infos[i];

    // Source base pointer for this sequence and head.
    // src_offset skips left-padding tokens in 4D input.
    const at::BFloat16* k_seq_base = is_4d
        ? (key_ptr + i * src_batch_stride + info.src_offset * src_seq_stride +
           h * src_head_stride)
        : (key_ptr + seq_tok_offset[i] * src_seq_stride + h * src_head_stride);
    const at::BFloat16* v_seq_base = is_4d
        ? (val_ptr + i * vsrc_batch_stride + info.src_offset * vsrc_seq_stride +
           h * vsrc_head_stride)
        : (val_ptr + seq_tok_offset[i] * vsrc_seq_stride +
           h * vsrc_head_stride);

    for (int64_t t = 0; t < info.new_tokens; ++t) {
      int64_t global_pos = info.old_seq_len + t;
      int64_t blk_idx = global_pos / bsize;
      int64_t pos_in_blk = global_pos % bsize;
      int64_t pool_blk = info.seq->block_indices[blk_idx];

      at::BFloat16* dst_k =
          pool_ptr + h * ph + pool_blk * pb + pos_in_blk * hdim;
      at::BFloat16* dst_v = dst_k + pvo;

      dpbf16::copy_bf16_avx512(dst_k, k_seq_base + t * src_seq_stride, hdim);
      dpbf16::copy_bf16_avx512(dst_v, v_seq_base + t * vsrc_seq_stride, hdim);
    }
  }

  {
    std::lock_guard<std::mutex> lock(sequence_mutex);
    for (int64_t i = 0; i < n_seq; ++i)
      infos[i].seq->seq_len += infos[i].new_tokens;
  }
}

} // namespace kernels
} // namespace pace
