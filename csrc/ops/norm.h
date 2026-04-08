/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef NORM_H
#define NORM_H

#include <ATen/ATen.h>

namespace pace {

std::tuple<at::Tensor, at::Tensor> fused_add_rmsnorm_op(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double eps);

at::Tensor rmsnorm_op(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps);

std::tuple<at::Tensor, at::Tensor> fused_add_layernorm_op(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

at::Tensor layernorm_op(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

} // namespace pace

#endif // NORM_H
