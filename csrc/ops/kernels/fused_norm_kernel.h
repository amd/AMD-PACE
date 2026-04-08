/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef PACE_FUSED_NORM_KERNEL_H
#define PACE_FUSED_NORM_KERNEL_H

#include <ATen/ATen.h>
#include <cstdint>

namespace pace {
namespace kernels {

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps);

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
