/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/kernels/linear_kernel.h>
#include <ops/linear.h>
#include <ops/onednn.h>
#include <ops/onednn_utils.h>
#include <torch/library.h>
#include <utils/utils.h>
#include <array>
#include <vector>

namespace pace {

at::Tensor flatten_input(at::Tensor input) {
  at::Tensor input_reshaped = input;
  if (input.dim() != 2) {
    const at::SymIntArrayRef input_sizes = input.sym_sizes();
    c10::SymInt flattened_dim = 1;
    for (int64_t i = 0, ndim = input_sizes.size(); i < ndim - 1; ++i) {
      flattened_dim = flattened_dim * input_sizes[i];
    }
    input_reshaped = input.reshape_symint(
        {flattened_dim, input_sizes.at(input_sizes.size() - 1)});
  }
  return input_reshaped;
}

void verify_linear_dims(
    std::string method_name,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor bias) {
  TORCH_CHECK(
      (input.dim() >= 1),
      "pace::" + method_name + " got dims less than 1, please check input.");
  TORCH_CHECK(
      (weight.dim() == 2),
      "pace::" + method_name +
          " only supports 2D tensors for weight"
          ", got " +
          std::to_string(weight.dim()) + "D weight.");
  int64_t input_K = input.size(input.dim() - 1);
  int64_t weight_K = weight.size(1);
  TORCH_CHECK(
      (input_K == weight_K),
      "pace::" + method_name + " got incompatible input and weight, got ",
      input.sizes(),
      " for input and ",
      weight.sizes(),
      " for weight.");
  if (bias.dim() != 0) {
    TORCH_CHECK(
        (bias.dim() == 1),
        "pace::" + method_name +
            " only supports 1D tensors for bias "
            ", got " +
            std::to_string(bias.dim()) + "D bias.");
    int64_t weight_N = weight.size(0);
    int64_t bias_N = bias.size(0);
    TORCH_CHECK(
        (weight_N == bias_N),
        "pace::" + method_name + " got incompatible weight and bias, got ",
        weight.sizes(),
        " for weights and ",
        bias.sizes(),
        " for bias.");
  }
}

// Wrapper method which takes care of fp32/bf16 inputs
at::Tensor _linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const post_ops& post_ops) {
  verify_linear_dims("_linear", input, weight, bias);
  TORCH_CHECK(
      (input.scalar_type() == weight.scalar_type()),
      "pace::_linear got mismatched types, input: ",
      input.scalar_type(),
      ", weight: ",
      weight.scalar_type(),
      ".");
  TORCH_CHECK(
      dtype_supported(input.scalar_type(), {at::kBFloat16, at::kFloat}),
      "pace::_linear only support the dtypes float and bfloat16");
  TORCH_CHECK(
      dtype_supported(weight.scalar_type(), {at::kBFloat16, at::kFloat}),
      "pace::_linear only support the dtypes float and bfloat16");
  if (bias.dim() != 0) {
    TORCH_CHECK(
        (input.scalar_type() == bias.scalar_type()),
        "pace::_linear got mismatched types, input: ",
        input.scalar_type(),
        "bias: ",
        bias.scalar_type(),
        ".");
    TORCH_CHECK(
        dtype_supported(bias.scalar_type(), {at::kBFloat16, at::kFloat}),
        "pace::_linear only support the dtypes float and bfloat16");
  }

  // If the input dim is more than 2, multiply all the
  // sizes to ndim-1 dims and flatten it, then later
  // reconstruct the output.
  const at::SymIntArrayRef input_sizes = input.sym_sizes();
  at::Tensor input_reshaped = flatten_input(input);
  memory input_mem = view_tensor_as_memory(input_reshaped);

  std::array<int64_t, 2> shape_arr;
  shape_arr[0] = input_reshaped.sizes()[0];
  shape_arr[1] = weight.sizes()[0];
  c10::IntArrayRef shape = c10::IntArrayRef(&shape_arr[0], 2);
  at::Tensor output = at::empty(shape, input.scalar_type());
  memory output_mem = view_tensor_as_memory(output);

  primitive_attr attr;
  attr.set_post_ops(post_ops);

  memory weight_mem =
      view_tensor_as_memory(weight, /*transpose*/ transpose_dims(0, 1));
  memory bias_mem = get_bias_memory(bias);
  kernels::linear_kernel(input_mem, weight_mem, bias_mem, output_mem, attr);

  // Reconstruct output in original shape
  at::SymIntArrayRef new_size = input_sizes.slice(0, input_sizes.size() - 1);
  c10::SymDimVector sizes_vec(new_size.begin(), new_size.end());
  sizes_vec.push_back(output.sym_size(1));
  return output.view_symint(sizes_vec);
}

at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
  PROFILE_PACE_FUNCTION("linear");

  post_ops post_ops;

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _linear(input, weight, bias, post_ops);
  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {});
  return output;
}

at::Tensor linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
  PROFILE_PACE_FUNCTION("linear_relu");

  post_ops post_ops;
  post_ops.append_eltwise(algorithm::eltwise_relu, 1.0f, 0.0f);

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _linear(input, weight, bias, post_ops);
  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {"relu"});
  return output;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def("linear(Tensor input, Tensor weight, Tensor ? bias) -> Tensor");
  m.def("linear_relu(Tensor input, Tensor weight, Tensor ? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("linear", pace::linear);
  m.impl("linear_relu", pace::linear_relu);
}

} // namespace
