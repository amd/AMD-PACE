/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/cpu.h>
#include <ops/jit_helper.h>
#include <ops/kernels/mlp_kernel.h>
#include <ops/mlp.h>
#include <pace_tensor/pace_aten_interface.h>
#include <pace_tensor/pace_tensor_impl.h>
#include <torch/library.h>
#include <utils/utils.h>

using tag = memory::format_tag;

namespace pace {

at::Tensor mlp_mlp_fusion(
    const at::Tensor& src,
    const std::vector<at::Tensor>& weights1,
    const c10::optional<std::vector<at::Tensor>>& bias1,
    const std::vector<at::Tensor>& weights2,
    const c10::optional<at::Tensor>& bias2,
    std::string nlf,
    const c10::optional<std::vector<at::Tensor>>& weights_gateProj,
    const c10::optional<std::vector<at::Tensor>>& bias_gateProj) {
  PROFILE_PACE_FUNCTION("mlp_mlp_fusion");
  auto original_shape = src.sizes();
  at::Tensor src_reshaped = flatten_to_2D(src);
  std::vector<at::Tensor> bias1_vec;
  at::Tensor bias2_vec;
  std::vector<at::Tensor> bias_gateProj_vec;

  if (bias1.has_value()) {
    bias1_vec = get_optional_list_tensors(bias1);
    bias1_vec = unsqueeze_bias_2D(bias1_vec);
  }
  if (bias2.has_value()) {
    bias2_vec = get_bias_from_opt(bias2);
    bias2_vec = unsqueeze_bias_2D(bias2_vec);
  }
  std::vector<at::Tensor> weights_gateProj_vec;
  if (weights_gateProj.has_value()) {
    weights_gateProj_vec = get_optional_list_tensors(weights_gateProj);
  }
  if (bias_gateProj.has_value()) {
    bias_gateProj_vec = get_optional_list_tensors(bias_gateProj);
    bias_gateProj_vec = unsqueeze_bias_2D(bias_gateProj_vec);
  }

  auto src_dims = src_reshaped.sizes();
  at::ScalarType output_dtype = src_reshaped.scalar_type();
  TORCH_CHECK(
      src_reshaped.dim() == 2,
      "pace::mlp_mlp_fusion only supports 2D tensors for now, got " +
          std::to_string(src_reshaped.dim()) + "D tensor");
  TORCH_CHECK(
      dtype_supported(output_dtype, {at::kFloat, at::kBFloat16}),
      "pace::mlp_mlp_fusion only support the dtypes Bfloat16 and Float types for output");
  if (weights_gateProj.has_value()) {
    TORCH_CHECK(
        nlf == "silu" || nlf == "Silu",
        "pace::mlp_mlp_fusion only supports silu for llama type models, got " +
            nlf);
  } else {
    TORCH_CHECK(
        nlf == "gelu" || nlf == "Gelu" || nlf == "relu" || nlf == "Relu",
        "pace::mlp_mlp_fusion only supports gelu and relu for opt type models, got " +
            nlf);
  }
  std::array<int64_t, 2> shape_arr;
  shape_arr[0] = src_dims[0];
  shape_arr[1] = src_dims[1];
  c10::IntArrayRef shape = c10::IntArrayRef(&shape_arr[0], 2);
  at::Tensor output = at::zeros(shape, at::kFloat);
  memory dst2_mem = view_tensor_as_memory(output);
  // Calling the IMBPS_MLP kernel
  kernels::IMBPS_MLP(
      src_reshaped,
      weights1,
      (bias1.has_value()) ? bias1_vec : bias1,
      weights2,
      (bias2.has_value()) ? bias2_vec : bias2,
      nlf,
      dst2_mem,
      weights_gateProj,
      (bias_gateProj.has_value()) ? bias_gateProj_vec : bias_gateProj);
  output = output.to(output_dtype);
  PROFILE_ADD_INFO_MLP_MLP_FUSION(
      src_reshaped,
      weights1,
      (bias1.has_value()) ? bias1_vec : bias1,
      weights2,
      (bias2.has_value()) ? bias2_vec : bias2,
      nlf,
      weights_gateProj,
      (bias_gateProj.has_value()) ? bias_gateProj_vec : bias_gateProj,
      output);
  if (src.dim() != 2) {
    output = output.reshape(original_shape);
  }
  return output;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "mlp_mlp_fusion(Tensor src, Tensor[] weights1, Tensor[] ? bias1, Tensor[] weights2, Tensor ? bias2, str nlf, Tensor[] ? weights_gateProj, Tensor[] ? bias_gateProj)->Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("mlp_mlp_fusion", pace::mlp_mlp_fusion);
}

} // namespace
