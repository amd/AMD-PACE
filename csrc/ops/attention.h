/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_ATTENTION_H
#define PACE_ATTENTION_H

#include <ATen/ATen.h>

namespace pace {

/*
  Scaled Dot Product Attention (SDPA) functions -
  1) MHA - multi_head_attention
  2) GQA - grouped_query_attention
  Performs the ops: softmax ( ( add_mask( Q.K') ) ) . V
*/

/**
 * @brief Implementation for MHA
 * MHA supported for F32 and BF16 input data types
 *
 * @param input_Q Tensor Q
 * @param input_K Tensor K
 * @param input_V Tensor V
 * @param input_mask Tensor mask
 * @param use_KQ Scalar Flag
 */
at::Tensor multi_head_attention(
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const c10::optional<at::Tensor>& input_mask,
    const c10::optional<at::Scalar>& use_KQ);
/**
 * @brief Implementation for SDPA attention on a list of tensors
 * Performs the ops: softmax ( ( add_mask( Q.K') ) ) . V
 * SDPA supported for F32 and BF16 input data types
 *
 * @param input_Q list of Tensor Q
 * @param input_K list of Tensor K
 * @param input_V list of Tensor V
 * @param input_mask list of Tensor mask
 * @param use_KQ Scalar Flag
 */
at::Tensor multi_head_attention_list(
    const std::vector<at::Tensor>& input_Q,
    const std::vector<at::Tensor>& input_K,
    const std::vector<at::Tensor>& input_V,
    const std::vector<at::Tensor>& input_mask,
    const c10::optional<at::Scalar>& use_KQ);

/**
 * @brief Implementation for GQA
 * GQA supported for F32 and BF16 input data types
 *
 * @param input_Q Tensor Q
 * @param input_K Tensor K
 * @param input_V Tensor V
 * @param input_mask Tensor mask
 */
at::Tensor grouped_query_attention(
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const c10::optional<at::Tensor>& input_mask);

/**
 * @brief Implementation for GQA on a list of tensors
 * GQA supported for F32 and BF16 input data types
 *
 * @param list_Q list of Tensor Q
 * @param list_K list of Tensor K
 * @param list_V list of Tensor V
 * @param list_mask list of Tensor mask
 */
at::Tensor grouped_query_attention_list(
    const std::vector<at::Tensor>& list_Q,
    const std::vector<at::Tensor>& list_K,
    const std::vector<at::Tensor>& list_V,
    const std::vector<at::Tensor>& list_mask);
} // namespace pace

#endif // PACE_ATTENTION_H
