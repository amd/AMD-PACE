/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ATen/ATen.h>
#include <core/logging.h>
#include <ops/kernels/fused_rope.h>
#include <torch/library.h>

namespace pace {

std::tuple<at::Tensor, at::Tensor> fused_rope_apply(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t unsqueeze_dim) {
  PROFILE_PACE_FUNCTION("fused_rope");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4,
      "fused_rope: query and key must be 4D (BNSH or BSNH)");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 &&
          key.scalar_type() == at::kBFloat16,
      "fused_rope: query and key must be BFloat16");
  TORCH_CHECK(
      cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16,
      "fused_rope: cos and sin must be BFloat16");
  TORCH_CHECK(
      unsqueeze_dim == 1 || unsqueeze_dim == 2,
      "fused_rope: unsqueeze_dim must be 1 (BNSH) or 2 (BSNH)");

  TORCH_CHECK(query.size(3) % 2 == 0, "fused_rope: head_dim must be even");
  TORCH_CHECK(
      query.size(3) == key.size(3),
      "fused_rope: query and key must have the same head_dim");
  TORCH_CHECK(
      query.size(0) == key.size(0),
      "fused_rope: query and key must have the same batch size");

  TORCH_CHECK(
      cos.dim() == 3, "fused_rope: cos must be 3D [batch, seq_len, half_dim]");
  TORCH_CHECK(
      cos.sizes() == sin.sizes(),
      "fused_rope: cos and sin must have the same shape");

  const auto half = query.size(3) / 2;
  TORCH_CHECK(
      cos.size(-1) == half,
      "fused_rope: cos/sin last dim must be half of head_dim");
  TORCH_CHECK(
      cos.size(0) == query.size(0),
      "fused_rope: cos batch dim must match query batch size");

  const auto seq_dim = (unsqueeze_dim == 1) ? 2 : 1;
  TORCH_CHECK(
      cos.size(1) == query.size(seq_dim),
      "fused_rope: cos seq_len must match query sequence length");
  TORCH_CHECK(
      cos.size(1) == key.size(seq_dim),
      "fused_rope: cos seq_len must match key sequence length");

  return kernels::fused_rope_forward(query, key, cos, sin, unsqueeze_dim);
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "fused_rope_apply(Tensor query, Tensor key, Tensor cos, Tensor sin, int unsqueeze_dim) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("fused_rope_apply", pace::fused_rope_apply);
}

} // namespace
