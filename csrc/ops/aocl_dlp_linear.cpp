/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <ops/kernels/aocl_dlp_linear_kernel.h>
#include <torch/library.h>
#include <utils/utils.h>
#include <algorithm>
#include <cctype>
#include <cstdlib> // for getenv
namespace pace {
using namespace logging;
// Helper function to validate input and weight tensors
void _validate_dlp_inputs(
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
  TORCH_CHECK(
      weight.dim() == 2,
      "pace::",
      op_name,
      " expected weight to be 2D, but got ",
      weight.dim(),
      "D.");
  if (multiplier.has_value() && multiplier.value().numel() > 0 &&
      multiplier.value().dim() != 0) {
    TORCH_CHECK(
        multiplier.value().scalar_type() == input.scalar_type(),
        "pace::",
        op_name,
        " got mismatched types, input: ",
        input.scalar_type(),
        ", multiplier: ",
        multiplier.value().scalar_type(),
        ".");
    TORCH_CHECK(
        multiplier.value().dim() == expected_input_dim,
        "pace::",
        op_name,
        " expected multiplier to be ",
        expected_input_dim,
        "D, but got ",
        multiplier.value().dim(),
        "D.");
    const int64_t batch_size = input.size(0);
    const int64_t seq_len = input.size(1);
    const int64_t N = weight.size(1);
    TORCH_CHECK(
        multiplier.value().numel() == batch_size * seq_len * N,
        "pace::",
        op_name,
        " expected multiplier size to match output size (batch*seq_len*N=",
        batch_size * seq_len * N,
        "), but got ",
        multiplier.value().numel(),
        " elements.");
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
        bias.size(0) == weight.size(1),
        "pace::",
        op_name,
        " expected bias size to match output size, but got ",
        bias.size(0),
        " and ",
        weight.size(1),
        ".");
  }
}

// Helper function to map activation string to PostOpType.
// activation should be the activation name only (e.g. "silu", "relu");
// op_name for errors is built as "dlp_linear_" + activation.
PostOpType get_post_op_type(const std::string& activation) {
  if (activation.find("plain") != std::string::npos) {
    return PostOpType::NONE;
  } else if (activation.find("relu") != std::string::npos) {
    return PostOpType::RELU;
  } else if (activation.find("gelu") != std::string::npos) {
    return PostOpType::GELU;
  } else if (activation.find("silu") != std::string::npos) {
    return PostOpType::SILU;
  } else if (activation.find("mul") != std::string::npos) {
    return PostOpType::MUL;
  }
  TORCH_CHECK(
      false,
      "pace::dlp_linear_",
      activation,
      " unsupported activation type: \"",
      activation,
      "\". Supported types are: plain, relu, gelu, silu, mul.");
  return PostOpType::NONE; // unreachable
}

// Returns true if PACE_USE_AOCL_DLP_RESHAPE env var is enabled.
// Default (unset): enabled. "1": enabled. "0": disabled.
// Any other value: print a message and enable.
static bool use_dlp_reshape_from_env() {
  static const bool kUseDlpReshape = []() {
    const char* env_val = std::getenv("PACE_USE_AOCL_DLP_RESHAPE");
    if (env_val == nullptr) {
      return true; // unset: default enabled
    }
    std::string val(env_val);
    const size_t start = val.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      val.clear();
    } else {
      const size_t end = val.find_last_not_of(" \t\r\n");
      val = val.substr(start, end - start + 1);
    }
    if (val == "0") {
      return false; // explicitly disabled
    }
    if (val == "1") {
      return true; // explicitly enabled
    }
    // Any other value: warn and enable
    PACE_LOG_WARNING(
        "PACE_USE_AOCL_DLP_RESHAPE has invalid value \"",
        val,
        "\"; expected \"0\" or \"1\". Defaulting to enabled (1).");
    return true;
  }();
  return kUseDlpReshape;
}

// Helper function for linear kernel (with and without activation).
// activation is the activation name only (e.g. "silu", "relu"); op_name for
// validation and profiling is "dlp_linear_" + activation.
at::Tensor _aocl_dlp_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const std::string& activation,
    c10::optional<at::Tensor> multiplier) {
  const std::string op_name = "dlp_linear_" + activation;
  PROFILE_PACE_FUNCTION(op_name);
  _validate_dlp_inputs(input, weight, op_name, multiplier, bias);

  // Determine post-op type
  PostOpType post_op = get_post_op_type(activation);

  // Get multiplier tensor if available
  auto multiplier_tensor = multiplier.has_value()
      ? multiplier.value()
      : at::empty({0}, input.options());

  // Handle weight tensor (no transposition needed; AOCL-DLP expects K x N)
  at::Tensor weight_transposed = weight;

  // Get dimensions: input is [batch, seq_len, K], weight is [K, N]
  // Flatten input to [M, K] where M = batch * seq_len
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t K = input.size(2);
  const int64_t N = weight_transposed.size(1);
  const int64_t M = batch_size * seq_len;

  // Flatten input to 2D for matmul: [M, K]
  at::Tensor input_2d = input.reshape({M, K}).contiguous();

  // Allocate output tensor: [M, N]
  at::Tensor output = at::empty({M, N}, input.options());

  // Get data pointers (p-prefix per coding standards)
  const bfloat16* pInput =
      reinterpret_cast<const bfloat16*>(input_2d.data_ptr<at::BFloat16>());
  const bfloat16* pWeight = reinterpret_cast<const bfloat16*>(
      weight_transposed.data_ptr<at::BFloat16>());
  bfloat16* pOutput =
      reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>());

  // Get bias pointer if provided
  const bfloat16* pBias = nullptr;
  if (bias.numel() > 0 && bias.dim() != 0) {
    pBias = reinterpret_cast<const bfloat16*>(bias.data_ptr<at::BFloat16>());
  }

  // Get multiplier pointer if provided (for MUL post-op)
  const bfloat16* pMul = nullptr;
  at::Tensor mul_2d; // Keep tensor alive throughout the function
  if (multiplier_tensor.numel() > 0 && multiplier_tensor.dim() != 0) {
    mul_2d = multiplier_tensor.reshape({M, N}).contiguous();
    pMul = reinterpret_cast<const bfloat16*>(mul_2d.data_ptr<at::BFloat16>());
  }

  MatmulConfig matmul_config;
  // Note: we pre-transpose the weight tensor into weight_transposed, so
  // matmul_config.transb remains at its default.

  // When AOCL-DLP reshape is used, weight is in reordered layout; we must set
  // mem_format_b='R' so the matmul kernel interprets the B matrix correctly.
  if (use_dlp_reshape_from_env()) {
    matmul_config.mem_format_b = 'R';
  }

  // Call AOCL-DLP kernel
  aocl_dlp_matmul_bf16(
      pInput,
      pWeight,
      pOutput,
      static_cast<unsigned int>(M),
      static_cast<unsigned int>(K),
      static_cast<unsigned int>(N),
      pBias,
      post_op,
      pMul,
      &matmul_config // Use default config
  );

  // Reshape output back to [batch, seq_len, N]
  output = output.reshape({batch_size, seq_len, N});

  if (activation == "mul" && multiplier_tensor.numel() > 0 &&
      multiplier_tensor.dim() != 0) {
    PROFILE_ADD_INFO_LINEAR(
        input,
        weight,
        bias,
        output,
        at::ArrayRef({multiplier_tensor}),
        {"mul"});
  } else {
    PROFILE_ADD_INFO_LINEAR(input, weight, bias, output, {}, {activation});
  }
  return output;
}
//----------------------------------------------------------PUBLIC API

at::Tensor aocl_dlp_linear_silu(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _aocl_dlp_linear(input, weight, bias, "silu", std::nullopt);
}

at::Tensor aocl_dlp_linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _aocl_dlp_linear(input, weight, bias, "relu", std::nullopt);
}

at::Tensor aocl_dlp_linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _aocl_dlp_linear(input, weight, bias, "gelu", std::nullopt);
}

at::Tensor aocl_dlp_linear_plain(
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _aocl_dlp_linear(input, weight, bias, "plain", std::nullopt);
}

at::Tensor aocl_dlp_linear_mul(
    const at::Tensor& input,
    const at::Tensor& multiplier,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt) {
  auto bias = get_bias_from_opt(bias_opt, input.options());
  return _aocl_dlp_linear(input, weight, bias, "mul", multiplier);
}

// Reshapes weight for AOCL-DLP matmul. Expects weight with shape [K, N]
// (size(0) = K, size(1) = N). Callers must pass already-transposed weights;
// the Python backend transposes to [K, N] before calling this op.
at::Tensor aocl_dlp_reshape_weights(const at::Tensor& weight) {
  TORCH_CHECK(
      weight.dim() == 2,
      "pace::aocl_dlp_reshape_weights expected weight to be 2D, but got ",
      weight.dim(),
      "D.");

  TORCH_CHECK(
      dtype_supported(weight.scalar_type(), {at::kBFloat16}),
      "pace::aocl_dlp_reshape_weights only supports bfloat16 types, got ",
      weight.scalar_type());

  // Make weight contiguous if it's not already
  at::Tensor weight_contig = weight.contiguous();

  // Get dimensions: weight layout is [K, N] (size(0)=K, size(1)=N)
  const int64_t K = weight_contig.size(0);
  const int64_t N = weight_contig.size(1);

  // Get input pointer (p-prefix per coding standards)
  bfloat16* pInput =
      reinterpret_cast<bfloat16*>(weight_contig.data_ptr<at::BFloat16>());

  // Get required output buffer size from AOCL-DLP (may differ from K*N for
  // reordered layout). Allocate one buffer and pass it to the reshape method.
  const size_t buf_size = aocl_dlp_reshape_bf16_buf_size(
      'R',
      'N',
      'B',
      static_cast<unsigned int>(K),
      static_cast<unsigned int>(N));
  TORCH_CHECK(
      buf_size % sizeof(at::BFloat16) == 0,
      "pace::aocl_dlp_reshape_weights: reorder buffer size ",
      buf_size,
      " is not a multiple of sizeof(BFloat16); would produce incorrect view");
  const int64_t buf_elems =
      static_cast<int64_t>(buf_size / sizeof(at::BFloat16));
  TORCH_CHECK(
      buf_elems > 0, "pace::aocl_dlp_reshape_weights invalid buffer size");

  at::Tensor output = at::empty(
      {buf_elems}, at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  bfloat16* pOutput =
      reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>());

  // Call AOCL-DLP reshape kernel to reorder the weight matrix (input is [K, N])
  // trans='N' for no transpose
  // matrix_type='B' for weight matrix (second operand in A*B)
  // order='R' for row-major layout
  aocl_dlp_reshape_bf16(
      pOutput,
      pInput,
      'N', // no transpose
      static_cast<unsigned int>(K), // M dimension
      static_cast<unsigned int>(N), // N dimension
      static_cast<int>(N), // leading dimension (number of columns in row-major)
      'R', // row-major order
      'B' // matrix type B (weight matrix)
  );

  // Output size from AOCL-DLP may not be exactly {K, N}; interpret buffer for
  // downstream matmul: logical shape is [K, N], with possible row padding.
  if (buf_elems == K * N) {
    return output.view({K, N});
  }
  TORCH_CHECK(
      buf_elems % K == 0,
      "pace::aocl_dlp_reshape_weights: buffer element count ",
      buf_elems,
      " is not divisible by K (",
      K,
      "); would produce incorrect strided view");
  const int64_t ld = buf_elems / K;
  TORCH_CHECK(
      ld >= N, "pace::aocl_dlp_reshape_weights buffer too small for [K, N]");
  return output.as_strided({K, N}, {ld, 1});
}

} // namespace pace

namespace {

// Schema only (required for torch.compile); impl is in TORCH_LIBRARY_IMPL.
TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "aocl_dlp_linear_plain(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "aocl_dlp_linear_gelu(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "aocl_dlp_linear_relu(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "aocl_dlp_linear_silu(Tensor input, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "aocl_dlp_linear_mul(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor");
  m.def("aocl_dlp_reshape_weights(Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl("aocl_dlp_linear_plain", pace::aocl_dlp_linear_plain);
  m.impl("aocl_dlp_linear_gelu", pace::aocl_dlp_linear_gelu);
  m.impl("aocl_dlp_linear_relu", pace::aocl_dlp_linear_relu);
  m.impl("aocl_dlp_linear_silu", pace::aocl_dlp_linear_silu);
  m.impl("aocl_dlp_linear_mul", pace::aocl_dlp_linear_mul);
  m.impl("aocl_dlp_reshape_weights", pace::aocl_dlp_reshape_weights);
}

} // namespace
