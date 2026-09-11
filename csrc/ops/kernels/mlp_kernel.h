/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/
#ifndef PACE_MLP_KERNEL_H
#define PACE_MLP_KERNEL_H

#include <ATen/ATen.h>
#include <ops/onednn.h>

namespace pace {

namespace kernels {

/**
 * @brief Creates a mlp block of a transformer network using the IMBPS
 * implementation with weights already split.(uses ZenDNN/oneDNN matmuls)
 *
 * @param src Src tensor to the first matmul opearation
 * @param weight Array of weight tensors to the first matmul opearation
 * @param bias Array of Bias tensors to be added to first matmul operation
 * @param weights2 Array of weights tensors of the second matmul operation
 * @param bias2 Bias tensor to the second matmul operation
 * @param nlf Non linearity function
 * @param dst2_mem Memory object to store the output of the second matmul
 * operation
 * @param weight_gateProj Optional weight tensor for the gate projection(For
 * LLama based models)
 */
void IMBPS_MLP(
    const at::Tensor& src,
    const std::vector<at::Tensor>& weights1,
    const c10::optional<std::vector<at::Tensor>>& bias1,
    const std::vector<at::Tensor>& weights2,
    const c10::optional<at::Tensor>& bias2,
    std::string& nlf,
    memory& dst2_mem,
    const c10::optional<std::vector<at::Tensor>>& weight_gateProj,
    const c10::optional<std::vector<at::Tensor>>& bias_gateProj);

} // namespace kernels

} // namespace pace

#endif // PACE_MLP_KERNEL_H
