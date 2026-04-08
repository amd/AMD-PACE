/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * BRGeMM tiled prefill attention op for BMC cache layout.
 *
 * Takes contiguous Q [B, N_q, S_padded, H], K [B, N_kv, KV_padded, H],
 * V [B, N_kv, KV_padded, H] with per-sequence q_offsets and q_lens
 * to handle left/right padding.
 *
 * Per-sequence dispatch: each batch element is processed with its own
 * q_offset, q_len, and K/V base pointer offset so prefill_tile only
 * sees real tokens — padding is skipped entirely.
 *
 * All pointers and per-batch metadata are precomputed outside the OMP
 * loop — zero ATen dispatch calls inside the parallel region.
 ******************************************************************************/

#include <omp.h>
#include <ops/attention/bmc/bmc_prefill_kernels.h>
#include <torch/library.h>
#include <cmath>
#include <cstring>

namespace pace {

namespace {

constexpr int64_t DEFAULT_BLOCK_SIZE = 64;
constexpr int64_t OMP_SERIAL_THRESHOLD = 8;

int64_t choose_block_size(int64_t kv_len) {
  if (kv_len <= 64)
    return ((kv_len + 15) / 16) * 16;
  return DEFAULT_BLOCK_SIZE;
}

} // namespace

at::Tensor prefill_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::vector<int64_t>& q_offsets,
    const std::vector<int64_t>& q_lens) {
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "prefill_attention requires 4D inputs [B, N, S, H]");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16,
      "prefill_attention requires BFloat16 inputs");
  TORCH_CHECK(
      key.scalar_type() == at::kBFloat16 &&
          value.scalar_type() == at::kBFloat16,
      "prefill_attention requires BFloat16 K/V");
  TORCH_CHECK(
      key.sizes() == value.sizes(),
      "prefill_attention: K and V must have same shape");

  const int64_t B = query.size(0);
  const int64_t num_q_heads = query.size(1);
  const int64_t q_padded = query.size(2);
  const int64_t head_dim = query.size(3);
  const int64_t num_kv_heads = key.size(1);
  const int64_t kv_padded = key.size(2);

  TORCH_CHECK(
      num_q_heads % num_kv_heads == 0,
      "prefill_attention: num_q_heads must be divisible by num_kv_heads");
  const bool has_per_seq = !q_offsets.empty();
  if (has_per_seq) {
    TORCH_CHECK(
        static_cast<int64_t>(q_offsets.size()) == B &&
            static_cast<int64_t>(q_lens.size()) == B,
        "prefill_attention: q_offsets and q_lens must match batch size");
  }

  const int64_t n_rep = num_q_heads / num_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  at::Tensor q_contig = query.contiguous();
  at::Tensor k_contig = key.contiguous();
  at::Tensor v_contig = value.contiguous();

  at::Tensor output = at::empty(
      {B, num_q_heads, q_padded, head_dim},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));

  const int64_t q_head_stride = q_padded * head_dim;
  const int64_t q_token_stride = head_dim;
  const int64_t kv_head_stride = kv_padded * head_dim;

  struct BatchInfo {
    const at::BFloat16* k_ptr;
    const at::BFloat16* v_ptr;
    const at::BFloat16* q_ptr;
    at::BFloat16* o_ptr;
    int64_t q_len;
    int64_t q_offset;
    int64_t kv_len;
    int64_t blk_size;
    int64_t n_blocks;
    int64_t n_qt;
  };
  std::vector<BatchInfo> batches(B);
  int64_t total_work = 0;
  int64_t max_n_blocks = 0;

  for (int64_t b = 0; b < B; ++b) {
    auto& bi = batches[b];
    int64_t ql = has_per_seq ? q_lens[b] : q_padded;
    int64_t qo = has_per_seq ? q_offsets[b] : 0;
    bi.q_len = ql;
    bi.q_offset = qo;
    bi.kv_len = ql;
    bi.blk_size = choose_block_size(ql);
    bi.n_blocks = ql > 0 ? (ql + bi.blk_size - 1) / bi.blk_size : 0;
    bi.n_qt = ql > 0 ? (ql + kernels::BMC_Q_TILE - 1) / kernels::BMC_Q_TILE : 0;
    total_work += num_kv_heads * bi.n_qt;
    if (bi.n_blocks > max_n_blocks)
      max_n_blocks = bi.n_blocks;

    const at::BFloat16* k_base = k_contig[b].data_ptr<at::BFloat16>();
    const at::BFloat16* v_base = v_contig[b].data_ptr<at::BFloat16>();
    bi.k_ptr = k_base + qo * head_dim;
    bi.v_ptr = v_base + qo * head_dim;
    bi.q_ptr = q_contig[b].data_ptr<at::BFloat16>();
    bi.o_ptr = output[b].data_ptr<at::BFloat16>();
  }

  if (total_work == 0) {
    output.zero_();
    return output;
  }

  if (has_per_seq) {
    for (int64_t b = 0; b < B; ++b) {
      const auto& bi = batches[b];
      if (bi.q_offset > 0)
        output[b].narrow(1, 0, bi.q_offset).zero_();
      int64_t end = bi.q_offset + bi.q_len;
      if (end < q_padded)
        output[b].narrow(1, end, q_padded - end).zero_();
    }
  }

  std::vector<int64_t> block_indices(max_n_blocks);
  for (int64_t i = 0; i < max_n_blocks; ++i)
    block_indices[i] = i;

  struct WorkItem {
    int64_t b, kv_h, qt;
  };
  std::vector<WorkItem> items(total_work);
  int64_t wi = 0;
  for (int64_t b = 0; b < B; ++b)
    for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h)
      for (int64_t qt = 0; qt < batches[b].n_qt; ++qt)
        items[wi++] = {b, kv_h, qt};

  auto dispatch = [&](int64_t w) {
    const auto& item = items[w];
    const auto& bi = batches[item.b];
    if (bi.q_len == 0)
      return;

    kernels::BmcKernelCtx ctx;
    ctx.k_ptr = bi.k_ptr;
    ctx.v_ptr = bi.v_ptr;
    ctx.q_ptr = bi.q_ptr;
    ctx.o_ptr = bi.o_ptr;
    ctx.num_q_heads = num_q_heads;
    ctx.head_dim = head_dim;
    ctx.num_kv = num_kv_heads;
    ctx.n_rep = n_rep;
    ctx.blk_size = bi.blk_size;
    ctx.kv_head_stride = kv_head_stride;
    ctx.q_head_stride = q_head_stride;
    ctx.q_token_stride = q_token_stride;
    ctx.scale_f = scale;

    kernels::bmc_prefill_tile(
        ctx,
        bi.kv_len,
        bi.q_len,
        bi.q_offset,
        block_indices,
        item.kv_h,
        item.qt);
  };

  if (total_work <= OMP_SERIAL_THRESHOLD) {
    for (int64_t w = 0; w < total_work; ++w)
      dispatch(w);
  } else {
#pragma omp parallel for schedule(dynamic)
    for (int64_t w = 0; w < total_work; ++w)
      dispatch(w);
  }

  return output;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "prefill_attention(Tensor query, Tensor key, Tensor value,"
      " int[] q_offsets, int[] q_lens) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("prefill_attention", pace::prefill_attention);
}

} // namespace
