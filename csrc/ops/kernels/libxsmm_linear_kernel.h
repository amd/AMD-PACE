/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LIBXSMM_MLP_KERNEL_H
#define LIBXSMM_MLP_KERNEL_H

#include <ATen/ATen.h>
#include <ops/libxsmm_dependency/tensor_helper.h>

namespace pace {
namespace kernels {

class NoOpActivation {
 public:
  NoOpActivation(long BSb, long Hk, long K, long K2) {}

  template <typename T>
  void operator()(T* in, T* out) const {}
};

using ReLUActivation = ReLUFwdTPP<at::BFloat16>;
using GeluActivation = GeluFwdTPP<at::BFloat16>;
using SiLUActivation = SiLUFwdTPP<at::BFloat16>;
using MulActivation = MulTPP<at::BFloat16, at::BFloat16>;

/**
 * @brief Performs a fully connected layer operation with multiplication.
 *
 * Performs matrix multiplication involving input, intermediate, and weight
 * tensors, optionally applying a bias and an activation function.
 *
 * @tparam ActivationTPP Type of the activation function to apply
 * @param t_in Input tensor
 * @param t_in1 Intermediate tensor
 * @param t_wt Weight tensor
 * @param t_bias Bias tensor
 * @param t_out Output tensor
 */
template <typename ActivationTPP>
void libxsmmlinear_kernel(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out);

/**
 * @brief Fused MLP: gate+activation+up+mul+down in a single OMP region.
 *
 * Dispatches to the templated kernel implementation based on the activation
 * string. Supports "silu", "gelu", and "relu".
 *
 * @param t_in         Input tensor [B, S, C] (bfloat16)
 * @param t_wt_gate    Gate weight (packed 5D), or empty for non-gated MLP
 * @param t_wt_up      Up-projection weight (packed 5D)
 * @param t_wt_down    Down-projection weight (packed 5D)
 * @param t_gate_bias  Optional gate bias
 * @param t_up_bias    Optional up-projection bias
 * @param t_down_bias  Optional down-projection bias
 * @param activation   One of "silu", "gelu", "relu"
 */
at::Tensor fused_mlp_dispatch(
    const at::Tensor& t_in,
    const c10::optional<at::Tensor>& t_wt_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_wt_down,
    const c10::optional<at::Tensor>& t_gate_bias,
    const c10::optional<at::Tensor>& t_up_bias,
    const c10::optional<at::Tensor>& t_down_bias,
    const std::string& activation);

} // namespace kernels
} // namespace pace

#endif // LIBXSMM_MLP_KERNEL_H
