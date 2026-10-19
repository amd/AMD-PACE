/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>
#include <core/logging.h>
#include <ops/gated_deltanet.h>
#include <ops/kernels/gated_deltanet_kernel.h>
#include <torch/library.h>
#include <utils/utils.h>
#include <algorithm>

namespace pace {
namespace {

inline void check_shape(
    const at::Tensor& t,
    at::IntArrayRef expected,
    const char* name,
    const char* op) {
  TORCH_CHECK(
      t.sizes() == expected,
      op,
      ": ",
      name,
      " must have shape ",
      expected,
      ", got ",
      t.sizes());
}

// Shared by the decode and prefill ops. The kernels derive every offset from
// the query/value shapes and walk raw pointers, so any mismatch here would
// read or write out of bounds instead of failing.
inline void check_gated_delta_inputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& state,
    const char* op) {
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 &&
          key.scalar_type() == at::kBFloat16 &&
          value.scalar_type() == at::kBFloat16,
      op,
      ": query, key and value must be BFloat16");
  TORCH_CHECK(
      dtype_supported(g.scalar_type(), {at::kFloat, at::kBFloat16}) &&
          dtype_supported(beta.scalar_type(), {at::kFloat, at::kBFloat16}),
      op,
      ": g and beta must be Float32 or BFloat16");
  TORCH_CHECK(state.scalar_type() == at::kFloat, op, ": state must be Float32");

  TORCH_CHECK(
      query.dim() == 4,
      op,
      ": query must be 4D [B,T,Hq,Kd], got ",
      query.sizes());
  TORCH_CHECK(
      value.dim() == 4,
      op,
      ": value must be 4D [B,T,Hv,Vd], got ",
      value.sizes());

  const int64_t B = query.size(0);
  const int64_t T = query.size(1);
  const int64_t Hq = query.size(2);
  const int64_t Kd = query.size(3);
  const int64_t Hv = value.size(2);
  const int64_t Vd = value.size(3);

  check_shape(key, query.sizes(), "key", op);
  TORCH_CHECK(
      value.size(0) == B && value.size(1) == T,
      op,
      ": value batch/sequence dims ",
      value.sizes(),
      " must match query ",
      query.sizes());
  // Grouped-query attention: each query/key head is broadcast across its group
  // of value heads, so value heads must be an integer multiple of query heads.
  TORCH_CHECK(
      Hq > 0 && Hv % Hq == 0,
      op,
      ": value heads (",
      Hv,
      ") must be an integer multiple of query heads (",
      Hq,
      ")");
  check_shape(g, {B, T, Hv}, "g", op);
  check_shape(beta, {B, T, Hv}, "beta", op);
  check_shape(state, {B, Hv, Kd, Vd}, "state", op);
  // state is advanced in place through a raw pointer, so a non-contiguous
  // view would silently read/write the wrong elements.
  TORCH_CHECK(state.is_contiguous(), op, ": state must be contiguous");
}

void check_conv1d_prefill_inputs(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& conv_state,
    const at::Tensor& weight,
    const char* op) {
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 &&
          weight.scalar_type() == at::kBFloat16,
      op,
      ": input and weight must be BFloat16");
  TORCH_CHECK(
      input.dim() == 3, op, ": input must be 3D [B,C,T], got ", input.sizes());
  TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == input.size(1) &&
          weight.size(1) >= 2,
      op,
      ": weight must be [C,K] with C = ",
      input.size(1),
      " and K >= 2, got ",
      weight.sizes());
  if (conv_state.has_value()) {
    TORCH_CHECK(
        conv_state->scalar_type() == at::kBFloat16,
        op,
        ": conv_state must be BFloat16");
    check_shape(
        *conv_state,
        {input.size(0), input.size(1), weight.size(1) - 1},
        "conv_state",
        op);
  }
}

} // namespace

at::Tensor recurrent_gated_delta_rule_op(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state) {
  PROFILE_PACE_FUNCTION("recurrent_gated_delta_rule");
  check_gated_delta_inputs(
      query, key, value, g, beta, state, "recurrent_gated_delta_rule");
  TORCH_CHECK(
      query.size(1) == 1,
      "recurrent_gated_delta_rule: seq_len must be 1 for recurrent step");

  return kernels::recurrent_gated_delta_rule(query, key, value, g, beta, state);
}

at::Tensor chunk_gated_delta_rule_op(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& state) {
  PROFILE_PACE_FUNCTION("chunk_gated_delta_rule");
  check_gated_delta_inputs(
      query, key, value, g, beta, state, "chunk_gated_delta_rule");

  return kernels::chunk_gated_delta_rule(query, key, value, g, beta, state);
}

at::Tensor causal_conv1d_update_op(
    const at::Tensor& input,
    at::Tensor& conv_state,
    const at::Tensor& weight) {
  PROFILE_PACE_FUNCTION("causal_conv1d_update");

  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16,
      "causal_conv1d_update: input must be BFloat16");
  TORCH_CHECK(
      input.dim() == 3, "causal_conv1d_update: input must be 3D [B,C,1]");
  TORCH_CHECK(input.size(2) == 1, "causal_conv1d_update: last dim must be 1");
  TORCH_CHECK(
      conv_state.scalar_type() == at::kBFloat16 &&
          weight.scalar_type() == at::kBFloat16,
      "causal_conv1d_update: conv_state and weight must be BFloat16");
  // conv_state is shifted in place through a raw pointer.
  TORCH_CHECK(
      conv_state.is_contiguous(),
      "causal_conv1d_update: conv_state must be contiguous");

  return kernels::causal_conv1d_update(input, conv_state, weight);
}

std::tuple<at::Tensor, at::Tensor> causal_conv1d_prefill_op(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& conv_state,
    const at::Tensor& weight) {
  PROFILE_PACE_FUNCTION("causal_conv1d_prefill");
  check_conv1d_prefill_inputs(
      input, conv_state, weight, "causal_conv1d_prefill");

  const int64_t B = input.size(0);
  const int64_t C = input.size(1);
  const int64_t T = input.size(2);
  const int64_t S = weight.size(1) - 1;

  // The model passes a transpose of the token-major in_proj output, so this
  // view is normally channel-contiguous already and costs no copy.
  auto x = input.transpose(1, 2);
  if (x.stride(2) != 1) {
    x = x.contiguous();
  }
  c10::optional<at::Tensor> prev;
  if (conv_state.has_value()) {
    prev = conv_state->transpose(1, 2).contiguous();
  }

  auto out = kernels::causal_conv1d_prefill(x, prev, weight);

  // Next cache: the last S tokens of (cached state ++ input), left-padded
  // with zeros when the sequence is shorter than S.
  const auto history = at::cat(
      {prev.has_value() ? *prev : at::zeros({B, S, C}, input.options()),
       x.narrow(1, std::max<int64_t>(0, T - S), std::min(T, S))},
      1);
  auto new_state =
      history.narrow(1, history.size(1) - S, S).transpose(1, 2).contiguous();

  // [B, C, T] view of the token-major result; the model transposes it back.
  return {out.transpose(1, 2), new_state};
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, "
      "Tensor g, Tensor beta, Tensor(a!) state) -> Tensor");
  m.def(
      "chunk_gated_delta_rule(Tensor query, Tensor key, Tensor value, "
      "Tensor g, Tensor beta, Tensor(a!) state) -> Tensor");
  m.def(
      "causal_conv1d_update(Tensor input, Tensor(a!) conv_state, "
      "Tensor weight) -> Tensor");
  m.def(
      "causal_conv1d_prefill(Tensor input, Tensor? conv_state, "
      "Tensor weight) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("recurrent_gated_delta_rule", pace::recurrent_gated_delta_rule_op);
  m.impl("chunk_gated_delta_rule", pace::chunk_gated_delta_rule_op);
  m.impl("causal_conv1d_update", pace::causal_conv1d_update_op);
  m.impl("causal_conv1d_prefill", pace::causal_conv1d_prefill_op);
}

} // namespace
