/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ATen/ATen.h>
#include <core/logging.h>
#include <ops/kernels/fused_norm_kernel.h>
#include <ops/norm.h>
#include <torch/library.h>

namespace pace {
namespace {

inline void check_norm_inputs(
    const at::Tensor& x,
    const at::Tensor& weight,
    const char* op) {
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
      op,
      ": x and weight must be BFloat16");
  TORCH_CHECK(weight.dim() == 1, op, ": weight must be 1D");
  TORCH_CHECK(weight.is_contiguous(), op, ": weight must be contiguous");
  TORCH_CHECK(
      weight.size(0) == x.size(-1),
      op,
      ": weight size must match last dim of x");
}

inline void check_residual(
    const at::Tensor& x,
    const at::Tensor& residual,
    const char* op) {
  TORCH_CHECK(
      residual.scalar_type() == at::kBFloat16,
      op,
      ": residual must be BFloat16");
  TORCH_CHECK(
      x.sizes() == residual.sizes(),
      op,
      ": x and residual must have the same shape");
}

inline void check_bias(
    const at::Tensor& x,
    const at::Tensor& bias,
    const char* op) {
  TORCH_CHECK(
      bias.scalar_type() == at::kBFloat16, op, ": bias must be BFloat16");
  TORCH_CHECK(bias.dim() == 1, op, ": bias must be 1D");
  TORCH_CHECK(bias.is_contiguous(), op, ": bias must be contiguous");
  TORCH_CHECK(
      bias.size(0) == x.size(-1), op, ": bias size must match last dim of x");
}

} // namespace

std::tuple<at::Tensor, at::Tensor> fused_add_rmsnorm_op(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double eps) {
  PROFILE_PACE_FUNCTION("fused_add_rmsnorm");
  check_norm_inputs(x, weight, "fused_add_rmsnorm");
  check_residual(x, residual, "fused_add_rmsnorm");
  return kernels::fused_add_rmsnorm(x, residual, weight, eps);
}

at::Tensor rmsnorm_op(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {
  PROFILE_PACE_FUNCTION("rmsnorm");
  check_norm_inputs(x, weight, "rmsnorm");
  return kernels::rmsnorm(x, weight, eps);
}

std::tuple<at::Tensor, at::Tensor> fused_add_layernorm_op(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  PROFILE_PACE_FUNCTION("fused_add_layernorm");
  check_norm_inputs(x, weight, "fused_add_layernorm");
  check_residual(x, residual, "fused_add_layernorm");
  check_bias(x, bias, "fused_add_layernorm");
  return kernels::fused_add_layernorm(x, residual, weight, bias, eps);
}

at::Tensor layernorm_op(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  PROFILE_PACE_FUNCTION("layernorm");
  check_norm_inputs(x, weight, "layernorm");
  check_bias(x, bias, "layernorm");
  return kernels::layernorm(x, weight, bias, eps);
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "fused_add_rmsnorm(Tensor x, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
  m.def(
      "fused_add_layernorm(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps) -> (Tensor, Tensor)");
  m.def("layernorm(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("fused_add_rmsnorm", pace::fused_add_rmsnorm_op);
  m.impl("rmsnorm", pace::rmsnorm_op);
  m.impl("fused_add_layernorm", pace::fused_add_layernorm_op);
  m.impl("layernorm", pace::layernorm_op);
}

} // namespace
