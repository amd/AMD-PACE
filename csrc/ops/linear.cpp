/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/quantized/Quantizer.h>
#include <core/logging.h>
#include <ops/cpu.h>
#include <ops/kernels/linear_kernel.h>
#include <ops/linear.h>
#include <pace_tensor/quantize_utils.h>
#include <torch/library.h>
#include <utils/utils.h>
#include <utils/zen_utils.h>
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
    const at::Tensor bias,
    bool addmm = false) {
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
  int64_t weight_K = addmm ? weight.size(0) : weight.size(1);
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
    int64_t weight_N = addmm ? weight.size(1) : weight.size(0);
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
    const post_ops& post_ops,
    bool addmm = false) {
  verify_linear_dims("_linear", input, weight, bias, addmm);
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
  if (addmm) {
    shape_arr[1] = weight.sizes()[1];
  }
  c10::IntArrayRef shape = c10::IntArrayRef(&shape_arr[0], 2);
  at::Tensor output = at::empty(shape, input.scalar_type());
  memory output_mem = view_tensor_as_memory(output);

  primitive_attr attr;
  attr.set_post_ops(post_ops);

  // If the tensor has storage, it is a normal aten tensor and if it does not,
  // the tensor would be a PACETensor (being an OpaqueTensor, it has no storage)
  if (weight.has_storage()) {
    transpose_dims trans_dim;
    if (!addmm) {
      trans_dim = transpose_dims(0, 1);
    }
    PACETensor weight_mem = PACETensor(
        view_tensor_as_memory(weight, /*transpose*/ trans_dim), {}, {});
    PACETensor bias_mem = get_bias_if_available(bias);
    kernels::linear_kernel(input_mem, weight_mem, bias_mem, output_mem, attr);
  } else {
    PACETensor& weight_mem = retrieve_pace_tensor_from_dense(weight);
    if (bias.has_storage()) {
      PACETensor bias_mem = get_bias_if_available(bias);
      kernels::linear_kernel(input_mem, weight_mem, bias_mem, output_mem, attr);
    } else {
      PACETensor& bias_mem = retrieve_pace_tensor_from_dense(bias);
      kernels::linear_kernel(input_mem, weight_mem, bias_mem, output_mem, attr);
    }
  }

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
#ifdef USE_ZENDNN
  post_ops.append_eltwise(1.0f, algorithm::eltwise_relu, 1.0f, 0.0f);
#else
  post_ops.append_eltwise(algorithm::eltwise_relu, 1.0f, 0.0f);
#endif

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _linear(input, weight, bias, post_ops);
  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {"relu"});
  return output;
}

at::Tensor pace_addmm(
    const at::Tensor bias_opt,
    const at::Tensor input,
    const at::Tensor weight) {
  PROFILE_PACE_FUNCTION("pace_addmm");

  post_ops post_ops;

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _linear(input, weight, bias, post_ops, true);
  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {});
  return output;
}

// Wrapper method which takes care of int8 inputs
at::Tensor _qlinear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    float output_scale,
    const int output_zpoint,
    const at::ScalarType& output_dtype,
    const post_ops& post_ops = post_ops(),
    const std::vector<memory>& post_ops_mem = {}) {
  verify_linear_dims("_qlinear", input, weight, bias);
  TORCH_CHECK(
      dtype_supported(input.scalar_type(), {at::kQUInt8, at::kQInt8}),
      "pace::_qlinear only support the dtypes Int8 types for input");
  TORCH_CHECK(
      dtype_supported(weight.scalar_type(), {at::kQInt8}),
      "pace::_qlinear only support the dtypes QInt8 types for weights");
  TORCH_CHECK(
      dtype_supported(bias.scalar_type(), {at::kFloat}),
      "pace::_qlinear only support the dtypes float types for bias");
  TORCH_CHECK(
      dtype_supported(output_dtype, {at::kFloat, at::kQUInt8, at::kQInt8}),
      "pace::_qlinear only support the dtypes Int8 and Float "
      "types for output");
  TORCH_CHECK(
      qscheme_supported(input, {at::kPerTensorAffine}),
      "pace::_qlinear only supports per tensor quantization for input");

  int binary_post_ops_count = count_post_ops(post_ops, primitive::kind::binary);
  TORCH_CHECK(
      binary_post_ops_count == post_ops_mem.size(),
      "pace::_qlinear post_ops and post_ops_mem must have the "
      "same length");

  // Get the scales for input and weight, and calculate the scales for bias and
  // output.
  quantize_scales q_scales = get_quantize_scales(input, weight, output_scale);

  // If the input dim is more than 2, multiply all the
  // sizes to ndim-1 dims and flatten it, then later
  // reconstruct the output.
  const at::SymIntArrayRef input_sizes = input.sym_sizes();
  at::Tensor input_reshaped = flatten_input(input);
  memory input_mem = view_tensor_as_memory(input_reshaped);

  at::Tensor output;
  if (output_dtype == at::kFloat) {
    // To support the sigmoid case, we need to support the output dtype to be
    // float In this case, we create a new empty tensor with the output dtype
    output =
        at::empty({input_reshaped.sizes()[0], weight.sizes()[0]}, output_dtype);
  } else {
    // For the other cases, we can use the output dtype to create the quantized
    // tensor
    at::QuantizerPtr output_quantizer = at::make_per_tensor_affine_quantizer(
        output_scale, output_zpoint, output_dtype);
    at::TensorOptions opt =
        c10::TensorOptions().dtype(output_dtype).device(at::kCPU);
    output = at::new_qtensor(
        /*sizes=*/{input_reshaped.sizes()[0], weight.sizes()[0]},
        opt,
        output_quantizer);
  }
  memory output_mem = view_tensor_as_memory(output);

  primitive_attr attr;
  attr.set_post_ops(post_ops);

  // Weight zero point is always 0
  quantize_zeropoint q_zp = {
      static_cast<int>(at::q_zero_point(input)), output_zpoint};

  // If the tensor has storage, it is a normal aten tensor and if it does not,
  // the tensor would be a PACETensor (being an OpaqueTensor, it has no storage)
  if (weight.has_storage()) {
    PACETensor weight_mem = PACETensor(
        view_tensor_as_memory(weight, /*transpose*/ transpose_dims(0, 1)),
        get_weight_scales(weight),
        {});
    PACETensor bias_mem = get_bias_if_available(bias);
    kernels::linear_kernel(
        input_mem,
        weight_mem,
        bias_mem,
        output_mem,
        attr,
        /*with reorder*/ true,
        q_scales,
        q_zp,
        post_ops_mem);
  } else {
    PACETensor& weight_mem = retrieve_pace_tensor_from_dense(weight);
    if (bias.has_storage()) {
      PACETensor bias_mem = get_bias_if_available(bias);
      kernels::linear_kernel(
          input_mem,
          weight_mem,
          bias_mem,
          output_mem,
          attr,
          /*with reorder*/ true,
          q_scales,
          q_zp,
          post_ops_mem);
    } else {
      PACETensor& bias_mem = retrieve_pace_tensor_from_dense(bias);
      kernels::linear_kernel(
          input_mem,
          weight_mem,
          bias_mem,
          output_mem,
          attr,
          /*with reorder*/ true,
          q_scales,
          q_zp,
          post_ops_mem);
    }
  }

  // Reconstruct output in original shape
  at::SymIntArrayRef new_size = input_sizes.slice(0, input_sizes.size() - 1);
  c10::SymDimVector sizes_vec(new_size.begin(), new_size.end());
  sizes_vec.push_back(output.sym_size(1));
  return output.view_symint(sizes_vec);
}

at::Tensor qlinear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Scalar& output_scale,
    const at::Scalar& output_zpoint,
    const at::ScalarType& output_dtype) {
  PROFILE_PACE_FUNCTION("qlinear");

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _qlinear(
      input,
      weight,
      bias,
      output_scale.toFloat(),
      output_zpoint.toInt(),
      output_dtype);

  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {});
  return output;
}

at::Tensor qlinear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Scalar& output_scale,
    const at::Scalar& output_zpoint,
    const at::ScalarType& output_dtype) {
  PROFILE_PACE_FUNCTION("qlinear_relu");

  post_ops post_ops;
  // Using ReLU is causing issue where values greater than 127 is being coming
  // up as negative values when using QInt8 dtype, thus using clip
  float relu_max = (output_dtype == at::kQInt8) ? 127.0f : 256.0f;
#ifdef USE_ZENDNN
  post_ops.append_eltwise(1.0f, algorithm::eltwise_clip, 0.0f, relu_max);
#else
  post_ops.append_eltwise(algorithm::eltwise_clip, 0.0f, relu_max);
#endif

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _qlinear(
      input,
      weight,
      bias,
      output_scale.toFloat(),
      output_zpoint.toInt(),
      output_dtype,
      post_ops);
  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {"relu"});
  return output;
}

at::Tensor qlinear_mul_add(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& multiplier,
    const at::Tensor& addend,
    const at::Scalar& alpha) {
  PROFILE_PACE_FUNCTION("qlinear_mul_add");

  TORCH_CHECK(
      alpha.toInt() == 1,
      "pace::qlinear_mul_add only supports alpha=1, got " +
          std::to_string(alpha.toInt()));
  TORCH_CHECK(
      dtype_supported(multiplier.scalar_type(), {at::kFloat}),
      "pace::qlinear_mul_add only supports float type for multiplier");
  TORCH_CHECK(
      dtype_supported(addend.scalar_type(), {at::kFloat}),
      "pace::qlinear_mul_add only supports float type for addend");
  TORCH_CHECK(
      multiplier.dim() == 2,
      "pace::qlinear_mul_add only supports 2D tensors for multiplier");
  TORCH_CHECK(
      addend.dim() == 2,
      "pace::qlinear_mul_add only supports 2D tensors for addend");

  memory multiplier_mem = view_tensor_as_memory(multiplier);
  memory addend_mem = view_tensor_as_memory(addend);

  post_ops post_ops;
  std::vector<memory> post_ops_mem;

  post_ops.append_binary(algorithm::binary_mul, multiplier_mem.get_desc());
  post_ops_mem.emplace_back(multiplier_mem);
  post_ops.append_binary(algorithm::binary_add, addend_mem.get_desc());
  post_ops_mem.emplace_back(addend_mem);

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _qlinear(
      input,
      weight,
      bias,
      /*output_scale*/ 1.0f,
      /*output_zpoint*/ 0,
      at::kFloat,
      post_ops,
      post_ops_mem);
  PROFILE_ADD_INFO_LINEAR(
      input,
      weight,
      bias,
      output,
      at::ArrayRef({multiplier, addend}),
      std::vector<std::string>({"mul", "add"}));
  return output;
}

at::Tensor qlinear_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
  PROFILE_PACE_FUNCTION("qlinear_sigmoid");

  post_ops post_ops;
#ifdef USE_ZENDNN
  post_ops.append_eltwise(1.0f, algorithm::eltwise_logistic, 1.0f, 0.0f);
#else
  post_ops.append_eltwise(algorithm::eltwise_logistic, 1.0f, 0.0f);
#endif

  at::Tensor bias = get_bias_from_opt(bias_opt);
  at::Tensor output = _qlinear(
      input,
      weight,
      bias,
      /*output_scale*/ 1.0f,
      /*output_zpoint*/ 0,
      at::kFloat,
      post_ops);
  PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {"sigmoid"});
  return output;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  // For the fp32/bf16 dtypes without scales
  m.def("linear(Tensor input, Tensor weight, Tensor ? bias) -> Tensor");
  m.def("linear_relu(Tensor input, Tensor weight, Tensor ? bias) -> Tensor");

  // For the int8 -> int8 and int8 -> fp32 dtypes with scales and zero points
  m.def(
      "qlinear(Tensor input, Tensor weight, Tensor ? bias, Scalar o_scale, Scalar o_zero_point, ScalarType o_dtype) -> Tensor");
  m.def(
      "qlinear_relu(Tensor input, Tensor weight, Tensor ? bias, Scalar o_scale, Scalar o_zero_point, ScalarType o_dtype) -> Tensor");
  // For the INT8 -> FP32 dtypes
  m.def(
      "qlinear_mul_add(Tensor input, Tensor weight, Tensor ? bias, Tensor multiplier, Tensor addend, Scalar alpha) -> Tensor");
  m.def(
      "qlinear_sigmoid(Tensor input, Tensor weight, Tensor ? bias) -> Tensor");

  // torch.compile experiment
  m.def("pace_addmm(Tensor bias, Tensor input, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("linear", pace::linear);
  m.impl("linear_relu", pace::linear_relu);
  m.impl("pace_addmm", pace::pace_addmm);
}

TORCH_LIBRARY_IMPL(pace, QuantizedCPU, m) {
  m.impl("qlinear", pace::qlinear);
  m.impl("qlinear_relu", pace::qlinear_relu);
  m.impl("qlinear_mul_add", pace::qlinear_mul_add);
  m.impl("qlinear_sigmoid", pace::qlinear_sigmoid);
}

} // namespace
