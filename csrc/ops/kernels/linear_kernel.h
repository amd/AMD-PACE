/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LINEAR_KERNEL_H
#define LINEAR_KERNEL_H

#include <ATen/ATen.h>
#include <ops/onednn.h>

namespace pace {

namespace kernels {

/**
 * @brief Wraps the matmul primitive with the correct arguments
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias bias to be added to AxB
 * @param output C matrix
 * @param attr any post-ops and extra attributes to be used
 * @param with_reorder whether weights and biases should be reordered
 */
void linear_kernel(
    const memory& input,
    memory& weight,
    memory& bias,
    memory& output,
    primitive_attr& attr,
    bool with_reorder = true);

} // namespace kernels

} // namespace pace

#endif // LINEAR_KERNEL_H
