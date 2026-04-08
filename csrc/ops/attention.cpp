/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/attention.h>
#include <ops/cpu.h>
#include <ops/jit_helper.h>
#include <pace_tensor/pace_aten_interface.h>
#include <torch/library.h>
#include <utils/utils.h>

#include <ops/attention/contiguous/grouped_query_attention_kernel.h>
#include <ops/attention/contiguous/multi_head_attention_kernel.h>

namespace pace {

at::Tensor multi_head_attention(
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const c10::optional<at::Tensor>& input_mask,
    const c10::optional<at::Scalar>& use_KQ) {
  PROFILE_PACE_FUNCTION("multi_head_attention");

  TORCH_CHECK(
      input_Q.dim() == 4 && input_K.dim() == 4 && input_V.dim() == 4,
      "pace::MHA attention requires 4D inputs, but received: ",
      " input_Q - ",
      input_Q.sizes(),
      ", ",
      " input_K - ",
      input_K.sizes(),
      ", ",
      " input_V - ",
      input_V.sizes());

  TORCH_CHECK(
      (input_K.sizes() == input_V.sizes()),
      "pace::MHA attention requires Key and Value sizes to be of "
      "same shape, but received: ",
      " input_K - ",
      input_K.sizes(),
      ", ",
      " input_V - ",
      input_V.sizes());

  TORCH_CHECK(
      dtype_supported(input_Q.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::MHA attention only support the dtypes Float and BF16 types for input Query");

  TORCH_CHECK(
      dtype_supported(input_K.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::MHA attention only support the dtypes Float and BF16 types for input Key");

  TORCH_CHECK(
      dtype_supported(input_V.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::MHA attention only support the dtypes Float and BF16 types for input Value");

  if (input_mask.has_value()) {
    TORCH_CHECK(
        input_mask.value().dim() == 4,
        "pace::MHA attention requires 4D attention mask, but received: ",
        input_mask.value().sizes());

    TORCH_CHECK(
        dtype_supported(
            input_mask.value().scalar_type(), {at::kFloat, at::kBFloat16}),
        "pace::MHA attention only support the dtypes Float and BF16 types for input Attention mask");
  }

  // By default KQ = 0
  int KQ = 0;
  if (use_KQ.has_value())
    KQ = use_KQ.value().toInt();

  TORCH_CHECK(
      (KQ == 0 || KQ == 1),
      "pace::MHA attention requires use_KQ to be 0 or 1 "
      "but received: ",
      " use_KQ - ",
      KQ);

  // Create output tensor memory
  at::Tensor output = at::empty(input_Q.sizes(), input_Q.scalar_type());

  kernels::multi_head_attention_kernel(
      output, input_Q, input_K, input_V, KQ, input_mask);

  return output; // B,N,S,H Format
}

at::Tensor grouped_query_attention(
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const c10::optional<at::Tensor>& input_mask) {
  PROFILE_PACE_FUNCTION("grouped_query_attention");

  TORCH_CHECK(
      input_Q.dim() == 4 && input_K.dim() == 4 && input_V.dim() == 4,
      "pace::GQA requires 4D inputs, but received: ",
      " input_Q - ",
      input_Q.sizes(),
      ", ",
      " input_K - ",
      input_K.sizes(),
      ", ",
      " input_V - ",
      input_V.sizes());

  TORCH_CHECK(
      (input_K.sizes() == input_V.sizes()),
      "pace::GQA requires Key and Value sizes to be of "
      "same shape, but received: ",
      " input_K - ",
      input_K.sizes(),
      ", ",
      " input_V - ",
      input_V.sizes());

  TORCH_CHECK(
      (input_Q.size(1) % input_K.size(1) == 0),
      "pace::GQA requires that the number of Q heads be divisible by the number of KV heads");

  TORCH_CHECK(
      dtype_supported(input_Q.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::GQA attention supports only the dtypes Float and BF16 types for input Query");

  TORCH_CHECK(
      dtype_supported(input_K.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::GQA attention supports only the dtypes Float and BF16 types for input Key");

  TORCH_CHECK(
      dtype_supported(input_V.scalar_type(), {at::kFloat, at::kBFloat16}),
      "pace::GQA attention supports only the dtypes Float and BF16 types for input Value");

  if (input_mask.has_value()) {
    TORCH_CHECK(
        input_mask.value().dim() == 4,
        "pace::GQA requires 4D attention mask, but received: ",
        input_mask.value().sizes());

    TORCH_CHECK(
        dtype_supported(
            input_mask.value().scalar_type(), {at::kFloat, at::kBFloat16}),
        "pace::GQA attention supports only the dtypes Float and BF16 types for input Attention mask");
  }

  // Create output tensor memory
  at::Tensor output = at::empty_like(input_Q);

  kernels::grouped_query_attention_kernel(
      output, input_Q, input_K, input_V, input_mask);

  return output; // B,N,S,H Format
}

at::Tensor multi_head_attention_list(
    const std::vector<at::Tensor>& list_Q,
    const std::vector<at::Tensor>& list_K,
    const std::vector<at::Tensor>& list_V,
    const std::vector<at::Tensor>& list_mask,
    const c10::optional<at::Scalar>& use_KQ) {
  PROFILE_PACE_FUNCTION("multi_head_attention_list");
  size_t len = list_Q.size();

  std::vector<int64_t> sizes(
      list_Q[0].sizes().begin(), list_Q[0].sizes().end());
  sizes[0] = len;
  at::Tensor output_tensor = at::empty(sizes, list_Q[0].scalar_type());

#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    at::Tensor input_Q = list_Q[i].contiguous();
    at::Tensor input_K = list_K[i].contiguous();
    at::Tensor input_V = list_V[i].contiguous();
    at::Tensor input_mask = list_mask[i].contiguous();

    // Get the output slice
    at::Tensor output = output_tensor.slice(0, i, i + 1);

    // Prepare inputs for attention function
    c10::optional<at::Tensor> mask_optional = c10::make_optional(input_mask);

    // By default KQ = 0
    int KQ = 0;
    if (use_KQ.has_value())
      KQ = use_KQ.value().toInt();

    // Call the kernel directly with the pre-allocated output slice
    kernels::multi_head_attention_kernel(
        output, input_Q, input_K, input_V, KQ, mask_optional);
  }

  return output_tensor;
}

at::Tensor grouped_query_attention_list(
    const std::vector<at::Tensor>& list_Q,
    const std::vector<at::Tensor>& list_K,
    const std::vector<at::Tensor>& list_V,
    const std::vector<at::Tensor>& list_mask) {
  PROFILE_PACE_FUNCTION("grouped_query_attention_list");
  size_t len = list_Q.size();
  std::vector<int64_t> sizes(
      list_Q[0].sizes().begin(), list_Q[0].sizes().end());
  sizes[0] = len;
  at::Tensor output_tensor = at::empty(sizes, list_Q[0].scalar_type());
#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    at::Tensor input_Q = list_Q[i].contiguous();
    at::Tensor input_K = list_K[i].contiguous();
    at::Tensor input_V = list_V[i].contiguous();
    at::Tensor input_mask = list_mask[i].contiguous();
    c10::optional<at::Tensor> mask_optional = c10::make_optional(input_mask);
    at::Tensor output = output_tensor.slice(0, i, i + 1);
    // Call the kernel directly with the pre-allocated output slice
    kernels::grouped_query_attention_kernel(
        output, input_Q, input_K, input_V, input_mask);
  }

  return output_tensor;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "multi_head_attention(Tensor input_Q, Tensor input_K, Tensor input_V, Tensor ? input_mask, Scalar ? use_KQ) -> Tensor");
  m.def(
      "grouped_query_attention(Tensor input_Q, Tensor input_K, Tensor input_V, Tensor ? input_mask) -> Tensor");
  m.def(
      "multi_head_attention_list( Tensor[] list_Q, Tensor[] list_K, Tensor[] list_V, Tensor[] list_mask, Scalar ? use_KQ) -> Tensor");
  m.def(
      "grouped_query_attention_list( Tensor[] list_Q, Tensor[] list_K, Tensor[] list_V, Tensor[] list_mask) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("multi_head_attention", pace::multi_head_attention);
  m.impl("grouped_query_attention", pace::grouped_query_attention);
  m.impl("multi_head_attention_list", pace::multi_head_attention_list);
  m.impl("grouped_query_attention_list", pace::grouped_query_attention_list);
}

} // namespace
