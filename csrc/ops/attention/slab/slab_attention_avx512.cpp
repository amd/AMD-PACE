/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * SlabPool Unified Attention Kernel (ragged continuous batching).
 *
 * Single entry point for ALL attention: attention() handles variable
 * query_lens per sequence. The batched 4D API delegates here by
 * reshaping [B,S,H,D] -> [B*S,H,D].
 *
 * Single OMP region with 4-way dispatch per work item:
 *   - DECODE_GQA:  GQA group (all reps, single KV pass)
 *   - DECODE_HEAD: per-head (more OMP parallelism when decode seqs sparse)
 *   - MTD:         multi-token decode (1 < q_len <= Q_TILE, dot-product)
 *   - PREFILL:     BRGeMM tiled (q_len > Q_TILE, oneDNN ukernels)
 *
 * OMP scheduling: adaptive static/dynamic based on work item count.
 * When total_work <= num_threads (each thread gets at most 1 item),
 * schedule(static) avoids atomic contention. Otherwise schedule(dynamic)
 * balances uneven work. SLAB_SCHEDULE env var overrides for testing.
 ******************************************************************************/

#include <omp.h>
#include <ops/attention/slab/slab_kernels.h>
#include <ops/attention/slab/slab_pool.h>

namespace pace {
namespace kernels {

namespace {

enum class SeqType : int8_t {
  DECODE_GQA,
  DECODE_HEAD,
  DECODE_SPLITK,
  MTD,
  PREFILL
};

struct SeqWorkInfo {
  int64_t kv_len;
  int64_t q_len;
  int64_t q_offset;
  int64_t work_start;
  int64_t n_work;
  int64_t num_splits;
  SeqType type;
  std::vector<int64_t> blocks;
};

// Tagged work item -- meaning of arg0/arg1 depends on sequence type:
//   DECODE_GQA:    arg0=kv_h
//   DECODE_HEAD:   arg0=qh
//   DECODE_SPLITK: arg0=kv_h, arg1=split_idx
//   MTD:           arg0=qh, arg1=qr
//   PREFILL:       arg0=kv_h, arg1=qt
struct WorkItem {
  int64_t seq_idx;
  int64_t arg0, arg1;
};

} // anonymous namespace

at::Tensor SlabPool::attention(
    const std::vector<int64_t>& sequence_ids,
    const at::Tensor& query,
    const std::vector<int64_t>& query_lens,
    const std::vector<int64_t>& q_start_offsets,
    double scale,
    int64_t sliding_window,
    const at::Tensor& sinks) {
  // 4D [B, S, H, D] -> flat reshape with per-sequence offsets
  if (query.dim() == 4) {
    const int64_t B = query.size(0), S = query.size(1);
    const int64_t H = query.size(2), D = query.size(3);
    auto qlens = query_lens.empty()
        ? std::vector<int64_t>(static_cast<size_t>(B), S)
        : query_lens;

    // Compute per-sequence Q start offsets into the flat [B*S, H, D] buffer.
    // Each offset skips left-padding tokens for that sequence.
    std::vector<int64_t> offsets(static_cast<size_t>(B));
    for (int64_t i = 0; i < B; ++i) {
      TORCH_CHECK(
          qlens[i] >= 0 && qlens[i] <= S,
          "SlabPool: query_lens[",
          i,
          "]=",
          qlens[i],
          " out of range [0, ",
          S,
          "]");
      offsets[i] = i * S + (S - qlens[i]);
    }

    auto out = attention(
        sequence_ids,
        query.reshape({B * S, H, D}),
        qlens,
        offsets,
        scale,
        sliding_window,
        sinks);
    return out.reshape({B, S, H, D});
  }

  const int64_t n_seq = static_cast<int64_t>(sequence_ids.size());
  const int64_t num_q_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_kv = this->num_kv_heads;
  TORCH_CHECK(
      num_q_heads % num_kv == 0,
      "SlabPool: num_q_heads (",
      num_q_heads,
      ") must be divisible by num_kv_heads (",
      num_kv,
      ")");
  const int64_t n_rep = num_q_heads / num_kv;
  const int64_t blk_size = this->block_size;
  const int64_t total_tokens = query.size(0);

  TORCH_CHECK(
      n_rep <= SLAB_MAX_REP,
      "SlabPool: n_rep=",
      n_rep,
      " exceeds SLAB_MAX_REP=",
      SLAB_MAX_REP);

  const bool has_sinks = sinks.defined() && sinks.numel() > 0;

  // Gather per-sequence info
  std::vector<SeqWorkInfo> seq_infos(static_cast<size_t>(n_seq));
  int64_t total_work = 0;
  int64_t n_decode_seqs = 0;
  {
    int64_t q_off = 0;
    std::lock_guard<std::mutex> lock(sequence_mutex);
    for (int64_t i = 0; i < n_seq; ++i) {
      auto& si = seq_infos[i];
      si.q_len = query_lens[i];
      si.q_offset = q_start_offsets.empty() ? q_off : q_start_offsets[i];
      q_off += si.q_len;

      auto it = sequences.find(sequence_ids[i]);
      TORCH_CHECK(
          it != sequences.end(),
          "SlabPool::attention: sequence_id ",
          sequence_ids[i],
          " not found");
      if (it->second.seq_len > 0) {
        si.kv_len = it->second.seq_len;
        si.blocks = it->second.block_indices;
      } else {
        si.kv_len = 0;
      }

      if (si.q_len == 1)
        ++n_decode_seqs;
    }
  }

  const int64_t num_threads = omp_get_max_threads();
  const int64_t gqa_items = n_decode_seqs * num_kv;
  const int64_t max_splits = this->splitk_max_splits;

  for (int64_t i = 0; i < n_seq; ++i) {
    auto& si = seq_infos[i];
    si.num_splits = 0;
    if (si.q_len == 1) {
      const int64_t num_blocks_i = (si.kv_len + blk_size - 1) / blk_size;

      if (n_rep <= 1) {
        si.type = SeqType::DECODE_HEAD;
        si.n_work = num_q_heads;
      } else {
        const bool prefer_gqa = gqa_items >= num_threads / 2;

        if (prefer_gqa && gqa_items < num_threads &&
            num_blocks_i >= max_splits) {
          int64_t sp = std::min(
              (num_threads + gqa_items - 1) / std::max(gqa_items, int64_t(1)),
              std::min(num_blocks_i, max_splits));
          if (sp >= 4) {
            si.type = SeqType::DECODE_SPLITK;
            si.num_splits = sp;
            si.n_work = num_kv * sp;
          } else {
            si.type = SeqType::DECODE_GQA;
            si.n_work = num_kv;
          }
        } else if (prefer_gqa) {
          si.type = SeqType::DECODE_GQA;
          si.n_work = num_kv;
        } else {
          si.type = SeqType::DECODE_HEAD;
          si.n_work = num_q_heads;
        }
      }
    } else if (si.q_len <= SLAB_Q_TILE) {
      si.type = SeqType::MTD;
      si.n_work = num_q_heads * si.q_len;
    } else {
      si.type = SeqType::PREFILL;
      const int64_t n_qt = (si.q_len + SLAB_Q_TILE - 1) / SLAB_Q_TILE;
      si.n_work = num_kv * n_qt;
    }
    si.work_start = total_work;
    total_work += si.n_work;
  }

  at::Tensor output = at::zeros(
      {total_tokens, num_q_heads, head_dim},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));

  if (total_work == 0)
    return output;

  KernelCtx ctx;
  ctx.pool_ptr = pool_tensor.data_ptr<at::BFloat16>();
  ctx.q_ptr = query.data_ptr<at::BFloat16>();
  ctx.o_ptr = output.data_ptr<at::BFloat16>();
  ctx.num_q_heads = num_q_heads;
  ctx.head_dim = head_dim;
  ctx.num_kv = num_kv;
  ctx.n_rep = n_rep;
  ctx.blk_size = blk_size;
  ctx.ph = pool_head_stride;
  ctx.pb = pool_blk_stride;
  ctx.pvo = pool_kv_offset;
  ctx.token_stride = num_q_heads * head_dim;
  ctx.scale_f = static_cast<float>(scale);
  ctx.sliding_window = sliding_window;
  ctx.has_sinks = has_sinks;
  ctx.sinks_ptr = has_sinks ? sinks.data_ptr<float>() : nullptr;

  // Build flat work-item table
  std::vector<WorkItem> work_items(static_cast<size_t>(total_work));
  for (int64_t i = 0; i < n_seq; ++i) {
    const auto& si = seq_infos[i];
    int64_t wi = si.work_start;
    switch (si.type) {
      case SeqType::DECODE_GQA:
        for (int64_t kv_h = 0; kv_h < num_kv; ++kv_h)
          work_items[wi++] = {i, kv_h, 0};
        break;
      case SeqType::DECODE_SPLITK:
        for (int64_t kv_h = 0; kv_h < num_kv; ++kv_h)
          for (int64_t sp = 0; sp < si.num_splits; ++sp)
            work_items[wi++] = {i, kv_h, sp};
        break;
      case SeqType::DECODE_HEAD:
        for (int64_t qh = 0; qh < num_q_heads; ++qh)
          work_items[wi++] = {i, qh, 0};
        break;
      case SeqType::MTD:
        for (int64_t qh = 0; qh < num_q_heads; ++qh)
          for (int64_t qr = 0; qr < si.q_len; ++qr)
            work_items[wi++] = {i, qh, qr};
        break;
      case SeqType::PREFILL: {
        const int64_t n_qt = (si.q_len + SLAB_Q_TILE - 1) / SLAB_Q_TILE;
        for (int64_t kv_h = 0; kv_h < num_kv; ++kv_h)
          for (int64_t qt = 0; qt < n_qt; ++qt)
            work_items[wi++] = {i, kv_h, qt};
        break;
      }
    }
  }

  // Allocate Split-K partial buffers
  int64_t total_partials = 0;
  std::vector<int64_t> splitk_base(static_cast<size_t>(n_seq), -1);
  for (int64_t i = 0; i < n_seq; ++i) {
    if (seq_infos[i].type == SeqType::DECODE_SPLITK) {
      splitk_base[i] = total_partials;
      total_partials += num_kv * seq_infos[i].num_splits * n_rep;
    }
  }
  // Use pool's pre-allocated partials buffer (avoids 1MB heap alloc per call)
  TORCH_CHECK(
      total_partials <= static_cast<int64_t>(this->splitk_partials_buf.size()),
      "SlabPool: need ",
      total_partials,
      " partials but buffer has ",
      this->splitk_partials_buf.size());
  PartialSoftmax* partials_buf = this->splitk_partials_buf.data();

  // Unified OMP dispatch
  // When total_work <= num_threads, each thread gets 0 or 1 items.
  // schedule(static) assigns deterministically with zero atomic overhead.
  // schedule(dynamic) adds per-item atomic contention with no benefit
  // (nothing to steal when every thread has at most 1 item).
  // When total_work > num_threads, threads process 2+ items and dynamic
  // helps balance micro-variations in per-item compute time.
  // Mixed workloads (decode + prefill) always need dynamic because
  // prefill items take 10-100x longer than decode items.
  // SLAB_SCHEDULE env: "static" or "dynamic" to force; unset = auto.
  bool use_static = false;
  {
    static const char* sched_env = std::getenv("SLAB_SCHEDULE");
    if (sched_env) {
      use_static = (std::string(sched_env) == "static");
    } else {
      const bool mixed = (n_decode_seqs > 0 && n_decode_seqs < n_seq);
      use_static = !mixed && (total_work <= num_threads);
    }
  }

  auto dispatch_one = [&](int64_t wi) {
    const auto& item = work_items[wi];
    const auto& si = seq_infos[item.seq_idx];
    if (si.blocks.empty() || si.kv_len == 0)
      return;

    if (si.type == SeqType::DECODE_SPLITK) {
      const int64_t kv_h = item.arg0;
      const int64_t split_idx = item.arg1;
      const int64_t num_blocks_total = (si.kv_len + blk_size - 1) / blk_size;
      const int64_t blocks_per_split =
          (num_blocks_total + si.num_splits - 1) / si.num_splits;
      const int64_t blk_start = split_idx * blocks_per_split;
      const int64_t blk_end =
          std::min(blk_start + blocks_per_split, num_blocks_total);

      const at::BFloat16* q_seq = ctx.q_ptr + si.q_offset * ctx.token_stride;
      const int64_t first_qh = kv_h * n_rep;
      const at::BFloat16* q_ptrs[SLAB_MAX_REP];
      for (int64_t r = 0; r < n_rep; ++r)
        q_ptrs[r] = q_seq + (first_qh + r) * head_dim;

      int64_t pbase = splitk_base[item.seq_idx] + kv_h * si.num_splits * n_rep +
          split_idx * n_rep;

      impl::decode_gqa_group_partial(
          ctx,
          q_ptrs,
          n_rep,
          si.blocks,
          si.kv_len,
          kv_h,
          blk_start,
          blk_end,
          &partials_buf[static_cast<size_t>(pbase)]);

    } else if (si.type == SeqType::DECODE_GQA) {
      // DECODE GQA: all reps share single KV pass
      const int64_t kv_h = item.arg0;
      const at::BFloat16* q_seq = ctx.q_ptr + si.q_offset * ctx.token_stride;
      at::BFloat16* o_seq = ctx.o_ptr + si.q_offset * ctx.token_stride;

      const int64_t first_qh = kv_h * n_rep;
      const at::BFloat16* q_ptrs[SLAB_MAX_REP];
      at::BFloat16* o_ptrs[SLAB_MAX_REP];
      float rep_sink_biases[SLAB_MAX_REP] = {};
      for (int64_t r = 0; r < n_rep; ++r) {
        q_ptrs[r] = q_seq + (first_qh + r) * head_dim;
        o_ptrs[r] = o_seq + (first_qh + r) * head_dim;
        if (ctx.has_sinks)
          rep_sink_biases[r] = ctx.sinks_ptr[first_qh + r];
      }
      impl::decode_gqa_group(
          ctx,
          q_ptrs,
          n_rep,
          si.blocks,
          si.kv_len,
          kv_h,
          ctx.has_sinks ? rep_sink_biases : nullptr,
          o_ptrs);

    } else if (si.type == SeqType::DECODE_HEAD) {
      // DECODE PER-HEAD: one Q head per work item
      const int64_t qh = item.arg0;
      const int64_t kv_h = qh / n_rep;
      const float sb = ctx.has_sinks ? ctx.sinks_ptr[qh] : 0.0f;
      impl::decode_one_head(
          ctx,
          ctx.q_ptr + si.q_offset * ctx.token_stride + qh * head_dim,
          si.blocks,
          si.kv_len,
          kv_h,
          sb,
          ctx.o_ptr + si.q_offset * ctx.token_stride + qh * head_dim);

    } else if (si.type == SeqType::MTD) {
      // MULTI-TOKEN DECODE: one (q_head, q_row) per work item
      const int64_t qh = item.arg0;
      const int64_t qr = item.arg1;
      const int64_t kv_h = qh / n_rep;
      const int64_t aq = std::min(si.q_len, si.kv_len);
      if (qr >= aq)
        return;
      const int64_t q_pos = qr + (si.kv_len - aq);
      const float sb = ctx.has_sinks ? ctx.sinks_ptr[qh] : 0.0f;
      impl::multi_token_decode_one_row(
          ctx,
          ctx.q_ptr + (si.q_offset + qr) * ctx.token_stride + qh * head_dim,
          si.blocks,
          si.kv_len,
          q_pos,
          kv_h,
          sb,
          ctx.o_ptr + (si.q_offset + qr) * ctx.token_stride + qh * head_dim);

    } else {
      // TILED PREFILL: BRGeMM
      prefill_tile(
          ctx,
          si.kv_len,
          si.q_len,
          si.q_offset,
          si.blocks,
          item.arg0,
          item.arg1);
    }
  };

  if (use_static) {
#pragma omp parallel for schedule(static)
    for (int64_t wi = 0; wi < total_work; ++wi)
      dispatch_one(wi);
  } else {
#pragma omp parallel for schedule(dynamic)
    for (int64_t wi = 0; wi < total_work; ++wi)
      dispatch_one(wi);
  }

  // Split-K reduction
  if (total_partials > 0) {
    for (int64_t i = 0; i < n_seq; ++i) {
      const auto& si = seq_infos[i];
      if (si.type != SeqType::DECODE_SPLITK)
        continue;

      at::BFloat16* o_seq = ctx.o_ptr + si.q_offset * ctx.token_stride;

      for (int64_t kv_h = 0; kv_h < num_kv; ++kv_h) {
        const int64_t first_qh = kv_h * n_rep;
        at::BFloat16* o_ptrs[SLAB_MAX_REP];
        for (int64_t r = 0; r < n_rep; ++r)
          o_ptrs[r] = o_seq + (first_qh + r) * head_dim;

        int64_t pbase = splitk_base[i] + kv_h * si.num_splits * n_rep;

        impl::reduce_gqa_partials(
            &partials_buf[static_cast<size_t>(pbase)],
            si.num_splits,
            n_rep,
            head_dim,
            o_ptrs);
      }
    }
  }

  return output;
}

} // namespace kernels
} // namespace pace
