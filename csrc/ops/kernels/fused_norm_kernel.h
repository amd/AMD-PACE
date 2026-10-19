/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_FUSED_NORM_KERNEL_H
#define PACE_FUSED_NORM_KERNEL_H

#include <ATen/ATen.h>
#include <cstdint>

namespace pace {
namespace kernels {

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps);

/**
 * Fused RMSNorm + SiLU gating.
 *
 * Computes: output = (weight * RMSNorm(x)) * SiLU(gate)
 *
 * @param input  [..., D] bf16
 * @param gate   same shape as input, bf16
 * @param weight [D]      bf16
 * @param eps    float
 * @return       same shape as input, bf16
 */
at::Tensor rmsnorm_gated(
    const at::Tensor& input,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps);

std::tuple<at::Tensor, at::Tensor> fused_add_rmsnorm(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double eps);

at::Tensor layernorm(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

std::tuple<at::Tensor, at::Tensor> fused_add_layernorm(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

} // namespace kernels
} // namespace pace

#endif // PACE_FUSED_NORM_KERNEL_H
