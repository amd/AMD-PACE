/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * torch.classes.pace.SlabPool registration with input validation wrappers.
 ******************************************************************************/

#include <ATen/ATen.h>
#include <ops/attention/slab/slab_pool.h>
#include <torch/custom_class.h>

namespace {

using Pool = pace::kernels::SlabPool;

// Validation wrappers
void checked_cache_update(
    const c10::intrusive_ptr<Pool>& self,
    const std::vector<int64_t>& sequence_ids,
    const at::Tensor& keys,
    const at::Tensor& values,
    const std::vector<int64_t>& token_counts) {
  TORCH_CHECK(
      keys.dim() == 3 || keys.dim() == 4,
      "SlabPool: keys must be 3D [T, KV, D] or 4D [B, S, KV, D]");
  TORCH_CHECK(
      keys.scalar_type() == at::kBFloat16, "SlabPool: keys must be BFloat16");
  TORCH_CHECK(
      values.scalar_type() == at::kBFloat16,
      "SlabPool: values must be BFloat16");
  TORCH_CHECK(
      values.dim() == keys.dim(),
      "SlabPool: values dim (",
      values.dim(),
      ") must match keys dim (",
      keys.dim(),
      ")");
  for (int64_t d = 0; d < keys.dim(); ++d) {
    TORCH_CHECK(
        values.size(d) == keys.size(d),
        "SlabPool: values.size(",
        d,
        ")=",
        values.size(d),
        " must match keys.size(",
        d,
        ")=",
        keys.size(d));
  }

  const int64_t kv_dim = keys.dim() == 4 ? 2 : 1;
  const int64_t hd_dim = keys.dim() == 4 ? 3 : 2;
  TORCH_CHECK(
      keys.size(hd_dim) == self->head_dim,
      "SlabPool: keys head_dim is ",
      keys.size(hd_dim),
      " but pool has ",
      self->head_dim);
  TORCH_CHECK(
      keys.size(kv_dim) == self->num_kv_heads,
      "SlabPool: keys num_kv_heads is ",
      keys.size(kv_dim),
      " but pool has ",
      self->num_kv_heads);

  const int64_t n_seq = static_cast<int64_t>(sequence_ids.size());
  if (keys.dim() == 4) {
    TORCH_CHECK(
        keys.size(0) == n_seq,
        "SlabPool: 4D keys batch size (",
        keys.size(0),
        ") must match sequence_ids size (",
        n_seq,
        ")");
  }
  TORCH_CHECK(
      token_counts.empty() ||
          static_cast<int64_t>(token_counts.size()) == n_seq,
      "SlabPool: token_counts size (",
      token_counts.size(),
      ") must match sequence_ids size (",
      n_seq,
      ")");

  self->cache_update(sequence_ids, keys, values, token_counts);
}

at::Tensor checked_attention(
    const c10::intrusive_ptr<Pool>& self,
    const std::vector<int64_t>& sequence_ids,
    const at::Tensor& query,
    const std::vector<int64_t>& query_lens,
    const std::vector<int64_t>& q_start_offsets,
    double scale,
    int64_t sliding_window,
    const at::Tensor& sinks) {
  TORCH_CHECK(
      query.dim() == 3 || query.dim() == 4,
      "SlabPool: query must be 3D [T, H, D] or 4D [B, S, H, D]");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16, "SlabPool: query must be BFloat16");

  const int64_t hd_dim = query.dim() == 4 ? 3 : 2;
  TORCH_CHECK(
      query.size(hd_dim) == self->head_dim,
      "SlabPool: query head_dim is ",
      query.size(hd_dim),
      " but pool has ",
      self->head_dim);

  const int64_t n_seq = static_cast<int64_t>(sequence_ids.size());
  if (query.dim() == 3) {
    TORCH_CHECK(
        !query_lens.empty(),
        "SlabPool: query_lens is required for 3D query input");
    TORCH_CHECK(
        static_cast<int64_t>(query_lens.size()) == n_seq,
        "SlabPool: query_lens size (",
        query_lens.size(),
        ") must match sequence_ids size (",
        n_seq,
        ")");
  } else {
    TORCH_CHECK(
        query.size(0) == n_seq,
        "SlabPool: 4D query batch size (",
        query.size(0),
        ") must match sequence_ids size (",
        n_seq,
        ")");
    TORCH_CHECK(
        query_lens.empty() || static_cast<int64_t>(query_lens.size()) == n_seq,
        "SlabPool: query_lens size (",
        query_lens.size(),
        ") must match sequence_ids size (",
        n_seq,
        ")");
  }
  TORCH_CHECK(
      q_start_offsets.empty() ||
          static_cast<int64_t>(q_start_offsets.size()) == n_seq,
      "SlabPool: q_start_offsets size (",
      q_start_offsets.size(),
      ") must match sequence_ids size (",
      n_seq,
      ")");
  if (sinks.defined() && sinks.numel() > 0) {
    const int64_t num_q_heads =
        query.dim() == 4 ? query.size(2) : query.size(1);
    TORCH_CHECK(
        sinks.numel() == num_q_heads,
        "SlabPool: sinks size (",
        sinks.numel(),
        ") must match num_q_heads (",
        num_q_heads,
        ")");
  }

  at::Tensor sinks_f32 = sinks;
  if (sinks.defined() && sinks.numel() > 0 &&
      sinks.scalar_type() != at::kFloat) {
    sinks_f32 = sinks.to(at::kFloat);
  }
  return self->attention(
      sequence_ids,
      query,
      query_lens,
      q_start_offsets,
      scale,
      sliding_window,
      sinks_f32);
}

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.class_<Pool>("SlabPool")
      .def(torch::init<int64_t, int64_t, int64_t, int64_t>())
      .def("create_sequence", &Pool::create_sequence)
      .def("remove_sequence", &Pool::remove_sequence)
      .def("truncate_sequence", &Pool::truncate_sequence)
      .def("get_sequence_length", &Pool::get_sequence_length)
      .def("get_free_block_count", &Pool::get_free_block_count)
      .def("cache_update", checked_cache_update)
      .def("attention", checked_attention);
}

} // namespace
