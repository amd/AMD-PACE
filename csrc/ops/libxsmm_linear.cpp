/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/kernels/libxsmm_linear_kernel.h>
#include <torch/library.h>
#include <utils/utils.h>
#include <mutex>
namespace pace {
// Helper function to validate input and weight tensors
void _validate_inputs(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::string& op_name,
    c10::optional<at::Tensor> multiplier,
    const at::Tensor& bias = at::Tensor()) {
  int expected_input_dim = 3;
  TORCH_CHECK(
      (input.scalar_type() == weight.scalar_type()),
      "pace::",
      op_name,
      " got mismatched types, input: ",
      input.scalar_type(),
      ", weight: ",
      weight.scalar_type(),
      ".");
  TORCH_CHECK(
      dtype_supported(input.scalar_type(), {at::kBFloat16}),
      "pace::",
      op_name,
      " only supports bfloat16 types.");
  TORCH_CHECK(
      input.dim() == expected_input_dim,
      "pace::",
      op_name,
      " expected input to be ",
      expected_input_dim,
      "D, but got ",
      input.dim(),
      "D.");
  TORCH_CHECK( // TODO: dynamically reshape the weight to 5D
      weight.dim() == 5 || weight.dim() == 2,
      "pace::",
      op_name,
      " expected weight to be one of 2D or 5D, but got ",
      weight.dim(),
      "D.");
  if (multiplier.has_value() && multiplier.value().numel() > 0 &&
      multiplier.value().dim() != 0) {
    TORCH_CHECK( // TODO: dynamically reshape the input to 3D
        multiplier.value().dim() == expected_input_dim,
        "pace::",
        op_name,
        " expected input to be ",
        expected_input_dim,
        "D, but got ",
        multiplier.value().dim(),
        "D.");
  }
  if ((bias.numel() > 0 and bias.dim() != 0)) {
    TORCH_CHECK(
        (bias.scalar_type() == input.scalar_type()),
        "pace::",
        op_name,
        " got mismatched types, input: ",
        input.scalar_type(),
        ", bias: ",
        bias.scalar_type(),
        ".");
    TORCH_CHECK(
        bias.dim() == 1,
        "pace::",
        op_name,
        " expected bias to be 1D, but got ",
        bias.dim(),
        "D.");
    TORCH_CHECK(
        bias.size(0) ==
            (weight.dim() == 5 ? weight.size(0) * weight.size(3)
                               : weight.size(0)),
        "pace::",
        op_name,
        " expected bias size to match output size, but got ",
        bias.size(0),
        " and ",
        (weight.dim() == 5 ?: weight.size(0) * weight.size(3), weight.size(0)),
        ".");
  }
}

// Helper function for linear kernel (with and without activation)
at::Tensor _libxsmmlinear(
    at::Tensor& input,
    at::Tensor& weight,
    at::Tensor& bias_opt,
    const std::string& activation,
    c10::optional<at::Tensor> multiplier) {
  PROFILE_PACE_FUNCTION(activation);
  _validate_inputs(input, weight, activation, multiplier, bias_opt);
  at::Tensor output;

  auto multiplier_tensor = multiplier.has_value()
      ? multiplier.value()
      : at::empty({0}, input.options());

  if (weight.dim() == 5) {
    // Existing logic for 5D weight
    auto sizes = input.sizes().vec();
    auto wt_sizes = weight.sizes();
    sizes[sizes.size() - 1] = wt_sizes[0] * wt_sizes[3];
    output = input.new_empty(sizes);

    if (activation == "libxsmmlinear_relu") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::ReLUActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_gelu") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::GeluActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_silu") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::SiLUActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_mul") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::MulActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else if (activation == "libxsmmlinear_plain") {
      pace::kernels::libxsmmlinear_kernel<pace::kernels::NoOpActivation>(
          input, multiplier_tensor, weight, bias_opt, output);
    } else {
      TORCH_CHECK(false, "Unsupported activation type: ", activation);
    }
  } else {
    // Fallback for non-5D weight
    using namespace logging;
    static std::once_flag log_first_time;
    std::call_once(log_first_time, []() {
      logging::PACE_LOG_WARNING(
          "Using Unoptimized path for libxsmmlinear with 2D weight.");
    });
    weight = weight.transpose(0, 1);
    output = at::matmul(input, weight);
    if ((bias_opt.numel() > 0 and bias_opt.dim() != 0)) {
      output.add_(bias_opt);
    }

    // Apply activation manually
    if (activation == "libxsmmlinear_relu") {
      output = at::relu(output);
    } else if (activation == "libxsmmlinear_gelu") {
      output = at::gelu(output);
    } else if (activation == "libxsmmlinear_silu") {
      output = at::silu(output);
    } else if (activation == "libxsmmlinear_mul") {
      output.mul_(multiplier_tensor);
    } else if (activation != "libxsmmlinear_plain") {
      TORCH_CHECK(false, "Unsupported activation type: ", activation);
    }
  }

  if (activation != "libxsmmlinear_mul") {
    PROFILE_ADD_INFO_LINEAR(input, weight, bias_opt, output, {}, {activation});
  } else {
    PROFILE_ADD_INFO_LINEAR(
        input,
        weight,
        bias_opt,
        output,
        at::ArrayRef({multiplier_tensor}),
        {"mul"});
  }

  return output;
}
at::Tensor libxsmmlinear_silu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_silu", std::nullopt);
}

at::Tensor libxsmmlinear_relu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_relu", std::nullopt);
}

at::Tensor libxsmmlinear_gelu(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_gelu", std::nullopt);
}

at::Tensor libxsmmlinear_plain(
    at::Tensor& input,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(
      input, weight, bias, "libxsmmlinear_plain", std::nullopt);
}

at::Tensor libxsmmlinear_mul(
    at::Tensor& input,
    at::Tensor& multiplier,
    at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _libxsmmlinear(input, weight, bias, "libxsmmlinear_mul", multiplier);
}
at::Tensor libxsmm_fused_mlp(
    const at::Tensor& input,
    c10::optional<at::Tensor> wt_gate,
    const at::Tensor& wt_up,
    const at::Tensor& wt_down,
    c10::optional<at::Tensor> gate_bias,
    c10::optional<at::Tensor> up_bias,
    c10::optional<at::Tensor> down_bias,
    const std::string& activation) {
  PROFILE_PACE_FUNCTION("libxsmm_fused_mlp");
  static const std::string op = "pace::libxsmm_fused_mlp";
  TORCH_CHECK(
      dtype_supported(input.scalar_type(), {at::kBFloat16}),
      op,
      " only supports bfloat16, got ",
      input.scalar_type(),
      ".");
  TORCH_CHECK(
      input.dim() >= 2 && input.dim() <= 3,
      op,
      " expected input to be 2D or 3D, got ",
      input.dim(),
      "D.");
  TORCH_CHECK(
      wt_up.dim() == 5,
      op,
      " expected wt_up to be 5D (packed), got ",
      wt_up.dim(),
      "D.");
  TORCH_CHECK(
      wt_down.dim() == 5,
      op,
      " expected wt_down to be 5D (packed), got ",
      wt_down.dim(),
      "D.");
  TORCH_CHECK(
      activation == "silu" || activation == "gelu" || activation == "relu",
      op,
      " unsupported activation: ",
      activation);
  if (wt_gate.has_value() && wt_gate.value().numel() > 0) {
    TORCH_CHECK(
        wt_gate.value().dim() == 5,
        op,
        " expected wt_gate to be 5D (packed), got ",
        wt_gate.value().dim(),
        "D.");
    TORCH_CHECK(
        wt_gate.value().sizes() == wt_up.sizes(),
        op,
        " wt_gate and wt_up shape mismatch: ",
        wt_gate.value().sizes(),
        " vs ",
        wt_up.sizes());
  }

  auto C = input.size(-1);
  auto N_out = wt_up.size(0) * wt_up.size(3);
  auto K_out = wt_down.size(0) * wt_down.size(3);
  auto up_C = wt_up.size(1) * wt_up.size(2) * wt_up.size(4);
  auto down_C = wt_down.size(1) * wt_down.size(2) * wt_down.size(4);

  TORCH_CHECK(
      up_C == C,
      op,
      " input features (",
      C,
      ") != wt_up packed input dim (",
      up_C,
      ").");
  TORCH_CHECK(
      down_C == N_out,
      op,
      " wt_up output dim (",
      N_out,
      ") != wt_down packed input dim (",
      down_C,
      ").");

  auto validate_bias = [&](const c10::optional<at::Tensor>& b,
                           const char* name,
                           int64_t expected_size) {
    if (b.has_value() && b.value().numel() > 0 && b.value().dim() > 0) {
      TORCH_CHECK(
          b.value().scalar_type() == at::kBFloat16,
          op,
          " expected ",
          name,
          " to be bfloat16, got ",
          b.value().scalar_type(),
          ".");
      TORCH_CHECK(
          b.value().dim() == 1,
          op,
          " expected ",
          name,
          " to be 1D, got ",
          b.value().dim(),
          "D.");
      TORCH_CHECK(
          b.value().size(0) == expected_size,
          op,
          " expected ",
          name,
          " size ",
          expected_size,
          ", got ",
          b.value().size(0),
          ".");
    }
  };
  validate_bias(gate_bias, "gate_bias", N_out);
  validate_bias(up_bias, "up_bias", N_out);
  validate_bias(down_bias, "down_bias", K_out);

  auto orig_dim = input.dim();
  at::Tensor in_3d = input;
  if (orig_dim < 3) {
    for (int d = orig_dim; d < 3; ++d)
      in_3d = in_3d.unsqueeze(0);
  }
  auto result = pace::kernels::fused_mlp_dispatch(
      in_3d,
      wt_gate,
      wt_up,
      wt_down,
      gate_bias,
      up_bias,
      down_bias,
      activation);
  if (orig_dim == 2)
    result = result.squeeze(0);
  return result;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "libxsmmlinear_plain(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "libxsmmlinear_gelu(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "libxsmmlinear_relu(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "libxsmmlinear_silu(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "libxsmmlinear_mul(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "libxsmm_fused_mlp(Tensor src, Tensor? wt_gate, Tensor wt_up, "
      "Tensor wt_down, Tensor? gate_bias, Tensor? up_bias, "
      "Tensor? down_bias, str activation) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("libxsmmlinear_plain", pace::libxsmmlinear_plain);
  m.impl("libxsmmlinear_gelu", pace::libxsmmlinear_gelu);
  m.impl("libxsmmlinear_relu", pace::libxsmmlinear_relu);
  m.impl("libxsmmlinear_silu", pace::libxsmmlinear_silu);
  m.impl("libxsmmlinear_mul", pace::libxsmmlinear_mul);
  m.impl("libxsmm_fused_mlp", pace::libxsmm_fused_mlp);
}

} // namespace
